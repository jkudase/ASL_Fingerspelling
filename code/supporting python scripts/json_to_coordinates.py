import json
import numpy as np
import pandas as pd
import os

columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
info = json.loads(open('key_points.json', 'r').read())  #Put the name of the .json here
csv = np.zeros((len(info), len(columns)))
for i in range(csv.shape[0]):
    one = []
    one.append(info[i]['score'])
    for object in info[i]['keypoints']:
        one.append(object['score'])
        one.append(object['position']['x'])
        one.append(object['position']['y'])
    csv[i] = np.array(one)
df=pd.DataFrame(csv, columns=columns) #dataframe
# pd.DataFrame(csv, columns=columns).to_csv('key_points1.csv', index_label='Frames#') #csv
x = df['leftWrist_x'][0]
y = df['leftWrist_y'][0]
x = int(x)
y = int(y)
print("Left Wrist x =",x,", Left wrist y =",y)