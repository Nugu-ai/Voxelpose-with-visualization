# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
from prettytable import PrettyTable
import copy

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import models

import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # h36m
LIMBS17 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
          [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

LIMBS15 = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def visualize(config, meta, preds, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'demo_image')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    preds = preds.cpu().numpy()

    xplot = 1
    yplot = 1
    width = 4.0 * xplot
    height = 4.0 * yplot
    
    fig = plt.figure(0, figsize=(width, height))
    ax = fig.add_subplot(111, projection="3d",auto_add_to_figure=False)
    ax.set_xlim3d(-config.MULTI_PERSON.SPACE_SIZE[0]//2, config.MULTI_PERSON.SPACE_SIZE[0]//2)
    ax.set_ylim3d(-config.MULTI_PERSON.SPACE_SIZE[1]//2, config.MULTI_PERSON.SPACE_SIZE[1]//2)
    ax.set_zlim3d(-config.MULTI_PERSON.SPACE_SIZE[2]//2, config.MULTI_PERSON.SPACE_SIZE[2]//2)
    colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
   
    # modify to set boundaries of number of entities
    for n in range(len(preds)):
        joint = preds[n]
        if joint[0, 3] >= 0:
            for k in eval("LIMBS{}".format(len(joint))):
                
                x = [float(joint[k[0], 0])-config.MULTI_PERSON.SPACE_CENTER[0], float(joint[k[1], 0])-config.MULTI_PERSON.SPACE_CENTER[0]]
                y = [float(joint[k[0], 1])-config.MULTI_PERSON.SPACE_CENTER[1], float(joint[k[1], 1])-config.MULTI_PERSON.SPACE_CENTER[1]]
                z = [float(joint[k[0], 2])-config.MULTI_PERSON.SPACE_CENTER[2], float(joint[k[1], 2])-config.MULTI_PERSON.SPACE_CENTER[2]]
                ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,markeredgewidth=1)
    plt.savefig(file_name)
    plt.close(0)

def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'vis_map')
    cfg_name = os.path.basename(args.cfg).split('.')[0]

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    # /home/ha/Documents/voxelpose/output/shelf_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/model_best.pth.tar
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = []
    final_output_dir
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
                 
            if 'panoptic' in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(views=inputs, meta=meta)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)

            
            # string prefix2 /home/ha/Documents/voxelpose/output/shelf_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/visualize_00000003
            batch_size = meta[0]['num_person'].shape[0]
            for x in range(batch_size):
                prefix2 = '{}_{:08}'.format(os.path.join(final_output_dir, 'visualize'), i*4+x)
                visualize(config, meta[0], pred[x], prefix2)
            
            


if __name__ == "__main__":
    main()
