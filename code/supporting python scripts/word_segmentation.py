import glob
import sys
import cv2
import numpy as np
import os
import tensorflow as tf
from tkinter import Tcl
from handshape_feature_extractor import HandShapeFeatureExtractor

test = glob.glob('./test/*')

for t in test:
  base = os.path.basename(t)
  true = os.path.splitext(base)[0]

  print("Segmenting word into",len(true),"parts")
  l = np.array_split(l, len(true))
  for i in range(len(l)):
    print(l[i])

  word=''
  for i in range(len(l)):
    u,c = np.unique(l[i], return_counts=True)
    word += u[np.argmax(c)]

  print("\nActual word :",true.upper(), "\nPredicted word :",word)