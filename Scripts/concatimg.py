import cv2
import numpy as np


    
NUM_IMAGES = 4055

for i in range(NUM_IMAGES):
    images = []
    save_img_path = 'vis_results_concat_pizza/' + str(i) + '.jpg'
    for j in range(6):
        if j != 5:
            img_path = 'vis_results_pizza/'+ str(i*5) + '_' + str(j) + '_2d.jpg'
        else:
            img_path = 'vis_results_pizza/vis_3d/' + str(i*5) + '_3d.jpg'
            
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 480))
        images.append(img)
        
    concat_image = np.hstack((images))
    cv2.imwrite(save_img_path, concat_image)