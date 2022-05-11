from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2


def preprocess(img, imgSize, dataAugmentation=False):
    "pone las imagenes en el tamaño, les añade escala de grises y rellena si es que falta"

    #para ver si hay imagenes dañadsas
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # i
    if dataAugmentation:
        stretch = (random.random() - 0.5) # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    
    # copia la muestra de la imagen
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpone para TF
    img = cv2.transpose(target)

    # las normaliza
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

