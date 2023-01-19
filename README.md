# Voxelpose-with-visualization
## About
We found 2 methods to visualize <a href="https://github.com/microsoft/voxelpose-pytorch">voxelpose-pytorch</a>.
1. <a href="https://github.com/open-mmlab/mmpose">mmpose</a>
2. Voxelpose itself

## <a href="https://github.com/open-mmlab/mmpose">mmpose</a>
1. requirements <br>
You may follow <a href="https://github.com/open-mmlab/mmpose#installation">this</a> installation guide <br><br>
2. datasets <br>
Using mmpose to inference and visualize voxelpose only supports on CMU-Panoptic dataset yet. Prepare dataset by using <a href="https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox">panoptic-toolbox</a>. We tested "160905_pizza1" and "170221_haggling_m3" and found these works well. <br> <br>
3. Run demo <br>
Please follow <a href="https://github.com/open-mmlab/mmpose/blob/master/demo/docs/3d_multiview_human_pose_demo.md">this docs</a> to run demo. Add your datasets' path to run this demo. <br><br>
4. Concat images and make video again <br>
We provide tools to concat image files(2d and 3d) and make those image files into demo video. Please refer to mmpose/concatimg.py, mmpose/mkvid.py. Add your own path to test this.
