# -*- coding: utf-8 -*-
"""
Image Sequence to Video
Created on Mon Oct 30 12:43:54 2017
@author: Dr.Geol Choi
"""
 
import os, errno
import cv2
import numpy as np
 
###############################################################################
# parameters defined by user
PATH_TO_INPUT_IMAGES_PATH_SHELF = './output/shelf_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/demo_image'#'vis_results_panoptic_test_1/vis_3d/'
PATH_TO_INPUT_IMAGES_PATH_CAMPUS = './output/campus_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/demo_image'
PATH_TO_INPUT_IMAGES_PATH_PANOPTIC  = './output/panoptic/multi_person_posenet_50/prn64_cpn80x80x20_960x512_cam5/demo_image'
PATH_TO_OUTPUT_VIDEO_DIR = 'demo_video/'
VIDEO_FILE = 'test_panoptic_model_ with cam5 on panoptic.mp4'
############################################################################### 
 
def main():
    ## make result directory if not exist
    print(PATH_TO_INPUT_IMAGES_PATH_PANOPTIC)
    try:
        if(os.path.isdir(PATH_TO_INPUT_IMAGES_PATH_PANOPTIC)):
            for root, dirs, files in os.walk(PATH_TO_INPUT_IMAGES_PATH_PANOPTIC, topdown=True):
                
                bIsFirst = True
                ll = []
                ext = []
                file=[]

                for f in files:
                    ll.append(f.split('.')[0])
                    ext.append(f.split('.')[1])
                ll.sort()
                for l,e in zip(ll,ext):
                    file.append(l +'.'+ e)

                for name in file:
                    cur_file = os.path.join(PATH_TO_INPUT_IMAGES_PATH_PANOPTIC, name)
                    cur_img = cv2.imread(cur_file)
                    
                    print("Currently %s being processed..." % (cur_file))
                    
                    if (type(cur_img) == np.ndarray):
                        if (bIsFirst):
                            frame_height = cur_img.shape[0]
                            frame_width = cur_img.shape[1]
                            
                            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                            video_file = os.path.join(PATH_TO_OUTPUT_VIDEO_DIR, VIDEO_FILE)
                            out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
                        
                        # record the current image frame to video file
                        out.write(cur_img)
                    
                    bIsFirst = False

            out.release()
                    
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
     
    # When everything done, release the video capture and video write objects
    
 
 
if __name__ == '__main__':
    main()
