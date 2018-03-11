#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# activation map 인경우 0 ,255 로 이루어진 uint8 np array
actmap = np.zeros((300,300)).astype("uint8")

"""
def get_rect(actmap):
    (_,contours,_) = cv2.findContours(actmap.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours :
        rect = cv2.boundingRect(contour)
        print rect
        exit()
        plt.imshow(actmap)
        rect=patches.Rectangle((250,677) , 1,73)
        



        # rect의 x, y ,w ,h 의 조건에 따라 bounding box 출력
        if rect[2] > 10 and rect[3] > 10 :
            print rect
"""


#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# activation map 인경우 0 ,255 로 이루어진 uint8 np array


def get_rect(ori_img , actmap):
    (_,contours,_) = cv2.findContours(actmap.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ori_img)
    rects=[]
    for contour in contours :
        rect = cv2.boundingRect(contour)
        if rect[2] < 5 or rect[3]  < 5 :
            continue
        print rect
        rect=patches.Rectangle((rect[0],rect[1]) , rect[2],rect[3] , fill=False , edgecolor='r')
        ax.add_patch(rect)
        rects.append(rect)
        # rect의 x, y ,w ,h 의 조건에 따라 bounding box 출력
        #if rect[2] > 10 and rect[3] > 10 :
        #    print rect
    return rects

