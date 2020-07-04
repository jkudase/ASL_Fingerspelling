import glob
import sys
import cv2
import numpy as np
import os
from PIL import Image, ImageOps

files = []
frame_folder_path = os.path.join('data')
path = os.path.join(frame_folder_path, "*.png")
frames = glob.glob(path)
frames.sort()
files = frames

for f in files:
  base = os.path.basename(f)
  im = Image.open(f)
  width, height = im.size 
  left = x-100
  top = y-100
  right = x+100
  bottom = y+100
  im1 = im.crop((left, top, right, bottom)) 
  im1.save('./crop/crop_'+base)

cv2.waitKey(0)
cv2.destroyAllWindows() 