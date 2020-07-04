import glob
import sys
import cv2
import numpy as np
import os
import tensorflow as tf
from tkinter import Tcl
from handshape_feature_extractor import HandShapeFeatureExtractor

def get_inference_vector_one_frame_alphabet(files_list):
    # model trained based on https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    video_names = []
    step = int(len(files_list) / 100)
    if step == 0:
        step = 1

    count = 0
    for video_frame in files_list:
        # print(video_frames)
        # assert len(video_frames) == 6

        img = cv2.imread(video_frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = model.extract_feature(img)
        results = np.squeeze(results)

        vectors.append(results)
        video_names.append(os.path.basename(video_frame))

        count += 1
        if count % step == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    return vectors

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id

print("\n")

label_file = 'output_labels_alphabet.txt'
id_to_labels, labels_to_id = load_label_dicts(label_file)

files = []
crop_folder_path = os.path.join('crop')
path = os.path.join(crop_folder_path, "*.png")
frames = glob.glob(path)
files = frames
files = list(Tcl().call('lsort', '-dict', files))

prediction_vector = get_inference_vector_one_frame_alphabet(files)
prediction_vector = np.array(prediction_vector)

print("\n")

l=[]
for i in range(prediction_vector.shape[0]):
  l.append(id_to_labels[np.argmax(prediction_vector[i])])
print("Labels from Cropped images")
print(l)

files = []
data_folder_path = os.path.join('data')
path = os.path.join(data_folder_path, "*.png")
frames = glob.glob(path)
files = frames
files = list(Tcl().call('lsort', '-dict', files))

prediction_vector = get_inference_vector_one_frame_alphabet(files)
prediction_vector = np.array(prediction_vector)

print("\n")

l=[]
for i in range(prediction_vector.shape[0]):
  l.append(id_to_labels[np.argmax(prediction_vector[i])])

print("Labels from Uncropped images")
print(l)