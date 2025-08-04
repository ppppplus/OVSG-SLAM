from vrb.vrb_afford_extract import VRBExtractor
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np


img = cv2.imread("/home/ubuntu/TJH/Work/aff_ws/SGS-SLAM/affordance/vrb/kitchen.jpeg")
print(img.shape)

vrb_extractor = VRBExtractor()
output = vrb_extractor.extract_afford(img)
plt.imshow(output)
plt.show()