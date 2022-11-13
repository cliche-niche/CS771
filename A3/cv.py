# import opencv
import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
import math
from utils import *

start = time.time()
for i in range(0,2000):
    img = cv.imread("train/"+str(i)+".png")
    images = process(img)
    # Write to file
    cv.imwrite("outs/"+str(i)+".png", img)
    for j in range(len(images)):
        cv.imwrite("outs/"+str(i)+"_"+str(j)+".png", images[j])
    # make_preds(images)
    print(i)

end = time.time()
print("Time taken: ", end-start)