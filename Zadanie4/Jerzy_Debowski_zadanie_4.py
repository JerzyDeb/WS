import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def convert_to_gray(img_path, new_img_path):
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Grayscale', gray_img)
    cv2.imwrite(new_img_path, gray_img)


def gauss(img_path, new_img_path):
    blur_kernel_size = (15, 15)
    gray_img = cv2.imread(img_path)
    gray_blur = cv2.GaussianBlur(gray_img, blur_kernel_size, 0)
    cv2.imshow('Grayscale blur', gray_blur)
    cv2.imwrite(new_img_path, gray_blur)


def canny(img_path, new_img_path):
    canny_low_threshold = 20
    canny_high_threshold = 100
    gray_blur = cv2.imread(img_path)
    gray_blur_canny = cv2.Canny(gray_blur, canny_low_threshold, canny_high_threshold)
    cv2.imshow('Grayscale blur canny', gray_blur_canny)
    cv2.imwrite(new_img_path, gray_blur_canny)


def kenny(img_path, new_img_path):
    img = cv2.imread(img_path, 0)
    height, width = img.shape[:2]
    print(height, width)
    h, w, x, y = 200, 1247, 400, 0
    img1 = img[x:x + h, y:y + w]
    img2 = np.zeros_like(img)
    img2[x:x + h, y:y + w] = img1
    cv2.imshow('Kenny', img2)
    cv2.imwrite(new_img_path, img2)


def huff(img_path, new_img_path):
    src = cv2.imread(img_path)
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 100, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 100, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('Detected lines (in red)', cdstP)
    cv2.imwrite(new_img_path, cdstP)


def put_on_original(img_path, original_img_path, new_img_path):
    img = cv2.imread(original_img_path)
    img1 = cv2.imread(img_path)
    img2 = cv2.addWeighted(img, 0.8, img1, 1, 0)
    cv2.imshow('Sum', img2)
    cv2.imwrite(new_img_path, img2)



image = 'droga.png'
convert_to_gray(image, 'droga_gray.png')
gauss('droga_gray.png', 'droga_gray_blur.png')
canny('droga_gray_blur.png', 'droga_gray_blur_canny.png')
kenny('droga_gray_blur_canny.png', 'droga_gray_blur_canny_kenny.png')
huff('droga_gray_blur_canny_kenny.png', 'huff.png')
put_on_original('huff.png', 'droga.png', 'result.png')
cv2.waitKey()
cv2.destroyAllWindows()
