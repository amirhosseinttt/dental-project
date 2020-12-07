import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from scipy.signal import argrelextrema
from sklearn.metrics import r2_score


# todo: argrelexterma for finding local maximum and minimum
# todo: outgoing deletion from the line
# todo: first detect asiaab then niiiiish

def circle_around(img):
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=16, minRadius=20, maxRadius=80)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)
    return img

    # write red channel to greyscale image


def tresh(img):
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input images
    cv2.imshow('Binary Threshold', thresh1)
    cv2.imshow('Binary Threshold Inverted', thresh2)
    cv2.imshow('Truncated Threshold', thresh3)
    cv2.imshow('Set to 0', thresh4)
    cv2.imshow('Set to 0 Inverted', thresh5)


def mask(img):
    any_low = np.array([30, 25, 70])
    any_high = np.array([160, 120, 200])

    mask = cv2.inRange(img, any_low, any_high)

    img[mask > 0] = (0, 0, 0)

    return img


def gray_thresh(img, arg1=101, arg2=60):
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, arg1, arg2)
    return th3


def mask_alternative(img):
    x = -1
    for i in img:
        x += 1
        y = -1
        for j in i:
            y += 1
            print(j)
            if j[0] + j[1] + j[2] < 200:
                img[x:y] = (0, 0, 0)
    return img


def paksh(img):
    kernel = np.ones((15, 15), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def purify(img):
    img = mask(img)
    img = paksh(img)
    i = 0
    for pixel_row in img:
        j = 0
        for pixel in pixel_row:
            if pixel.all(0): img[i, j] = img[i, j]
            j += 1
        i += 1

    return img


def harris(img):
    # img should be grayscale
    dst = cv2.cornerHarris(img, 6, 3, 0.04)
    dst = cv2.dilate(dst, None)
    return dst


def canny(img, arg1=100, arg2=200):
    canny = cv2.Canny(img, arg1, arg2)
    return canny


def arch_detection(img):
    img = canny(img)
    x = []
    y = []
    i = 0
    for pixel_row in img:
        j = 0
        for pixel in pixel_row:
            if pixel.all(0):
                x.append(i)
                y.append(j)
            j += 1
        i += 1

    print(x)
    print(y)
    mymodel = (np.poly1d(np.polyfit(y, x, 2)))
    myline = np.linspace(0, len(img[1]), 20)
    plt.scatter(y, x)
    plt.plot(myline, mymodel(myline))
    plt.show()
    return mymodel


def find_curve_line(img):
    img = purify(img)
    cv2.imshow("purified", img)
    # cv2.imshow('arch detection', arch_detection(img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_thresh(gray)
    line = arch_detection(gray)
    print(line)
    cv2.imshow('gray1', gray)
    gray = np.float32(gray)
    cv2.imshow('canny', canny(img))
    return line


# ---------------------------------------------------------------


def outgoing_dlwtion(img, line_model):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i, j] > 0:
                # print(line_model(i)-j)
                pass


def detect_asiaab(img, line_model):
    ...


def second_method(img, line_model):
    print(line)
    cv2.imshow("ffffff", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_thresh(gray, 101, 5)
    outgoing_dlwtion(gray, line)
    cv2.imshow('gray2', gray)
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
    cv2.imshow("here", th3)


if __name__ == "__main__":
    img_path = 'data/image5.jpg'
    img = cv2.imread(img_path)
    cv2.imshow("original mage", img)
    line = find_curve_line(img)
    img = cv2.imread(img_path)
    second_method(img, line)
    cv2.imshow("ttt", canny(img, 0, 200))
    cv2.waitKey(0)
