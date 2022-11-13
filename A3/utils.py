import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
import math

def load_reference_images(folder):
    ref_dir = {}
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        ref = filename.split(".")[0]
        ref_dir[ref] = img
        # Make black background white
        ref_dir[ref][ref_dir[ref] == 0] = 255
    return ref_dir

def remove_stray_lines(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Dilate then erode to remove noise
    # Stronger dilation than erosion
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((4,4),np.uint8)
    img = cv.dilate(img,kernel2,iterations = 1)
    img = cv.erode(img,kernel1,iterations = 1)
    return img


def morph_grad_and_threshold(img):
    # Morphological gradient
    kernel3 = np.ones((6,6),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel3)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    return img


def find_largest_connected_component(img):
    # find connected components
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8, ltype=cv.CV_32S)
    # find the largest component
    largest = 0
    largest_area = 0
    for i in range(1, nlabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest = i
    # Make largest connected component black
    img[labels == largest] = 0
    return img
def get_parent(hierarchy, index):
    return hierarchy[0][index][3]

def contour_based_segmentation(img):

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    useful_contours = []
    outer_contours = []
    for i in range(len(contours)):
        level = 1
        parent = get_parent(hierarchy, i)
        while parent != -1:
            level  = level + 1
            parent = get_parent(hierarchy, parent)
        if level == 1:
            outer_contours.append(contours[i])
        elif level == 2:
            useful_contours.append(contours[i])
        else:
            # Fit polygon to contour
            epsilon = 0.12*cv.arcLength(contours[i],True)
            approx = cv.approxPolyDP(contours[i],epsilon,True)
            # If area of polygon is close to contour area, it is a useful contour
            if(approx is not None and cv.contourArea(approx) > 0.9*cv.contourArea(contours[i])):
                useful_contours.append(contours[i])
    
    
    # Solid fill the contours
    # Sort useful contours by area
    outer_contours = sorted(outer_contours, key=cv.contourArea, reverse=True)
    # Find the largest decreasing area
    useful_contours.append(outer_contours[0])
    for i in range(1,len(outer_contours)):
        if cv.contourArea(outer_contours[i]) < cv.contourArea(outer_contours[i-1]) * 0.25:
            break
        useful_contours.append(outer_contours[i])
    # Add inner contours
    empty_img = np.zeros(img.shape, np.uint8)

    cv.drawContours(empty_img, useful_contours, -1, (255,0,0), -1)
    img = empty_img
    return img

def show_image(img):
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_splitter(img):
    # Find contours after conversion to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Get outer contours
    bounding_boxes = []
    images = []
    for i in range(len(contours)):
        if(get_parent(hierarchy, i) == -1):
            bounding_boxes.append(cv.boundingRect(contours[i]))
    # Sort contours by x position
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
    # If x is too close then merge
    merged_bounding_boxes = [bounding_boxes[0]]
    for i in range(1, len(bounding_boxes)):
        x1, y1, w1, h1 = bounding_boxes[i]
        x2, y2, w2, h2 = merged_bounding_boxes[-1]
        if x1 - x2 < 25:
            # New center is average of centers
            c1 = (x1 + w1/2, y1 + h1/2)
            c2 = (x2 + w2/2, y2 + h2/2)
            c = ((c1[0] + c2[0])/2, (c1[1] + c2[1])/2)
            w = max(x1 + w1, x2 + w2) - min(x1, x2)
            h = max(y1 + h1, y2 + h2) - min(y1, y2)
            x = c[0] - w/2
            y = c[1] - h/2
            merged_bounding_boxes[-1] = (x, y, w, h)
        else:
            merged_bounding_boxes.append(bounding_boxes[i])
    # Split image
    
    for box in merged_bounding_boxes:
        x,y,w,h = box
        # Take floor for all values
        x = math.floor(x)
        y = math.floor(y)
        w = math.floor(w)
        h = math.floor(h)
        try:
            images.append(img[y:y+h, x:x+w])
        except:
            print("Error with box: " + str(box))
    # Convert to RGB
    # Make non black pixels blue
    for i in range(len(images)):
        images[i][images[i] != 0] = 255
        # Pad with black pixels to (140,140)
        images[i] = cv.copyMakeBorder(images[i], 20, 20, 20, 20, cv.BORDER_CONSTANT, value=[0])
    return images

def process(img):
    # Get time for each step
    img = remove_stray_lines(img)
    img = morph_grad_and_threshold(img)
    img = find_largest_connected_component(img)
    img = contour_based_segmentation(img)
    images = image_splitter(img)
    if(len(images) != 3):
        print("FOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    return images   

def better_image_splitter():
    pass