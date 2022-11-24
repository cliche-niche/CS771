import cv2 as cv
for i in range(2000):
    img = cv.imread("train/"+str(i)+".png")
    if(i<1500):
        cv.imwrite("curr_train/"+str(i)+".png", img)
    else:
        cv.imwrite("test/"+str(i-1500)+".png", img)