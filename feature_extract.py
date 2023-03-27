import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def featureExtraction(imgPath,show):
    print(imgPath)
    image = cv2.imread(imgPath)
    if show:
        cv2.imshow('original',image)
    img_org = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Testign",gray.shape)
    if show:
        cv2.imshow('gray', gray)

    blur = cv2.GaussianBlur(gray, (1,1),0)

    ret,thresh1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

    if show:
        cv2.imshow('thresh1', thresh1)

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 5)
    if show:
        cv2.imshow('erosion',erosion)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

    if show:
        cv2.imshow('closing',closing)

    contours, img = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,(0,0,255),1)
    if show:
        cv2.imshow('gs latest',image)
        #cv2.imshow('img',img)
    hsv = cv2.cvtColor(img_org, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if show:
        cv2.imshow("H", h)
        cv2.imshow("S", s)
        cv2.imshow("V", v)
        cv2.imshow('hsv',hsv)

    print(gray.shape)
    stencil = np.zeros(gray.shape).astype(gray.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(gray, stencil)
    print("Testing",contours)

    if contours:
        area = 0
        perimeter = 0
        w=0
        h=0
        aspect_ratio=0
        rectangularity=0
        circularity=0
        
        for cnt in contours:
            cnt = contours[0]
            area = area+cv2.contourArea(cnt)
            perimeter = perimeter+cv2.arcLength(cnt,True)
            x,y,w1,h1 = cv2.boundingRect(cnt)
            
            if (h1):
                aspect_ratio = aspect_ratio+float(w1)/h1
            if area:
                rectangularity = rectangularity + (w*h/area)
                circularity = circularity + (((perimeter)**2)/area)
            w=w+w1
            h=h+h1
    else:
        area=0
        perimeter=0
        aspect_ratio=0
        rectangularity=0
        circularity=0
        w=0
        h=0
        

    red_channel = img_org[:,:,0]
    green_channel = img_org[:,:,1]
    blue_channel = img_org[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,red_mean,green_mean,blue_mean,red_std,green_std,blue_std]
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
    # left nearest neighbor
    glcm = greycomatrix(result, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        vector.append( greycoprops(glcm, f)[0,0] )
    # upper nearest neighbor
    glcm = greycomatrix(result, [1], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        vector.append( greycoprops(glcm, f)[0,0] )
    if show:
        cv2.waitKey()
    return vector

