import glob
import sys
import cv2
import numpy as np
import os

test = glob.glob('./test/*')

for t in test:
  base = os.path.basename(t)
  true = os.path.splitext(base)[0]
  pathOut = r"./data/"
  vidcap = cv2.VideoCapture(t);
  count = 1
  success = True
  while success:
      success,image = vidcap.read()
      if count%10 == 0 :
          cv2.imwrite(pathOut + 'frame%d.png'%count,image)
      count+=1