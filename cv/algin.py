import cv2
import numpy as np
import sys
import os


def getCanny(image, ksize=5, sigmax_y=2, threshold1=12, threshold2=12, apertureSize=0, Canny_border=3):
    bordertype = [cv2.BORDER_CONSTANT,cv2.BORDER_REPLICATE,cv2.BORDER_REFLECT,
                  cv2.BORDER_WRAP,cv2.BORDER_REFLECT_101,cv2.BORDER_TRANSPARENT,cv2.BORDER_ISOLATED][Canny_border]

    if ksize % 2 == 0:
        ksize += 1
    if apertureSize == 0:
        apertureSize = 3
    else:
        apertureSize = 7
    threshold1 *= 5
    threshold2 *= 5
    # msg = "\rksize:[{}],sigmax_y:[{}],threshold1:[{}],threshold2:[{}],apertureSize:[{}], bordertype:[{}]".format(
    #     ksize,sigmax_y,threshold1,threshold2,apertureSize,bordertype)
    # if os.get_terminal_size().columns > len(msg):
    #     sys.stdout.write(msg)
    # else:
    #     print(msg)
    image_ret = cv2.GaussianBlur(image, (ksize, ksize), sigmax_y, sigmax_y, bordertype)
    image_ret = cv2.Canny(image_ret, threshold1, threshold2, apertureSize=apertureSize)
    kernel = np.ones((ksize, ksize), np.uint8)
    image_ret = cv2.dilate(image_ret, kernel, iterations=1)

    return image_ret


def getMaxContour(image, mode=1, method=1):

    mode_ = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE][mode]
    method_ = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS][method]
    try:
        #contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(image, mode_, method_)
    except ValueError:
        _, contours, _ = cv2.findContours(image, mode_, method_)
    areas = list(map(cv2.contourArea,contours))
    try:
        max_area = max(areas)
        contour = contours[areas.index(max_area)]
    except ValueError:
        contour = None
    return contour

def getBoxPoint(contour,epsilon_k=34 ,Box_close=2):
    close = ([True, True],[False, True],[True,False],[False,False])

    hull = cv2.convexHull(contour)
    # arcLength(curve, closed) -> retval
    if epsilon_k == 0:
        k = 0.1
    else:
        k = 0.1 / epsilon_k    
    epsilon = k * cv2.arcLength(contour, close[Box_close][0])

    approx = cv2.approxPolyDP(hull, epsilon, close[Box_close][1])
    approx = approx.reshape((len(approx), 2))
    return approx

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


def get_points(image):
    if image is None:
        return None
    return orderPoints(getBoxPoint(getMaxContour(getCanny(image))))