import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

nose_front_pts = np.array([4,   5,   6, 168, 195, 197])
nose_base_pts = np.array([2,  19,  20,  60,  94,  97,  99, 125,
                         141, 238, 241, 242, 250, 290, 326, 328, 354, 370, 458, 461, 462])
nose_base_pts = np.array([1, 2,  19,  94])

lower_lip_pts = np.array([61, 146, 91, 181, 84, 17, 314, 405, 321,
                         375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78])
upper_lip_pts = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270,
                         409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, ])

nose_pts = np.array([1,   2,   3,   4,   5,   6,  19,  20,  44,  45,  48,  49,  51,
                     59,  60,  64,  75,  79,  94,  97,  98,  99, 102, 114, 115, 122,
                     125, 128, 129, 131, 134, 141, 166, 168, 174, 188, 193, 195, 196,
                     197, 198, 203, 217, 218, 219, 220, 235, 236, 237, 238, 239, 240,
                     241, 242, 244, 245, 248, 250, 274, 275, 278, 279, 281, 289, 290,
                     294, 305, 309, 326, 327, 328, 331, 343, 344, 351, 354, 357, 358,
                     360, 363, 370, 392, 399, 412, 417, 419, 420, 423, 437, 438, 439,
                     440, 455, 456, 457, 458, 459, 460, 461, 462, 464, 465])
nose_left_pt = 102
nose_right_pt = 331

eye_left_left_pt = 33
eye_left_right_pt = 133
eye_left_up_pt = 159
eye_left_down_pt = 144
eye_right_left_pt = 362
eye_right_right_pt = 263
eye_right_up_pt = 386
eye_right_down_pt = 373


foldername = "FFHQ/dataset"

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

nose_angles = []
nose_areas = []
nose_widths = []
lips_areas = []
eyes_ratios = []
filenames = []

i = 0
for filename in os.listdir(foldername)[:]:
    i += 1
    if i % 1000 == 0:
        df = pd.DataFrame({'filenames': filenames,
                           'nose_angle': nose_angles,
                           'nose_area': nose_areas,
                           'nose_width': nose_widths,
                           'lips_area': lips_areas,
                           'eyes_ratio': eyes_ratios})
        df.to_csv('labelsFFHQ.csv')

    image = cv2.imread(foldername + '/' + filename)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        face_landmarks = results.multi_face_landmarks[0]
    except:
        # nose_angle.append(None)
        # filenames.append(filename)
        pass
    else:
        data = []
        for pt in face_landmarks.landmark:
            data.append([pt.x, pt.y, pt.z])
        data = np.array(data)
        if len(data) == 478:
            pca = PCA(n_components=3)
            pca.fit(data)
            trans_data = pca.transform(data)

            # nose angle
            nose_front_data = data[nose_front_pts]
            nose_base_data = data[nose_base_pts]
            uu, dd, vv = np.linalg.svd(
                nose_base_data - nose_base_data.mean(axis=0))
            b = vv[0]
            uu, dd, vv = np.linalg.svd(
                nose_front_data - nose_front_data.mean(axis=0))
            d = vv[0]
            nose_angle = np.arccos(
                np.dot(b, d) / (np.linalg.norm(d)*np.linalg.norm(b)))
            # print("nose angle:", nose_angle)
            nose_angles.append(nose_angle)

            # nose area
            nose_data = data[nose_pts]
            trans_nose_data = pca.transform(nose_data)
            nose_hull = ConvexHull(trans_nose_data[:, :2])
            nose_area = nose_hull.volume
            # print("nose area:", nose_area)
            nose_areas.append(nose_area)

            # nose width
            nose_left = trans_data[nose_left_pt]
            nose_right = trans_data[nose_right_pt]
            nose_width = np.linalg.norm(nose_left-nose_right)
            # print("nose_width:", nose_width)
            nose_widths.append(nose_width)

            # lips area
            lower_lip = data[lower_lip_pts]
            trans_lower_lip = pca.transform(lower_lip)
            lower_lip_hull = ConvexHull(trans_lower_lip[:, :2])
            lower_lip_area = lower_lip_hull.volume
            upper_lip = data[upper_lip_pts]
            trans_upper_lip = pca.transform(upper_lip)
            upper_lip_hull = ConvexHull(trans_upper_lip[:, :2])
            upper_lip_area = upper_lip_hull.volume
            lips_area = lower_lip_area + upper_lip_area
            # print("lips area:", lips_area)
            lips_areas.append(lips_area)

            # eyes ratio
            left_eye_width = np.linalg.norm(
                trans_data[eye_left_left_pt]-trans_data[eye_left_right_pt])
            left_eye_height = np.linalg.norm(
                trans_data[eye_left_up_pt]-trans_data[eye_left_down_pt])
            left_eye_ratio = left_eye_height/left_eye_width
            right_eye_width = np.linalg.norm(
                trans_data[eye_right_left_pt]-trans_data[eye_right_right_pt])
            right_eye_height = np.linalg.norm(
                trans_data[eye_right_up_pt]-trans_data[eye_right_down_pt])
            right_eye_ratio = right_eye_height/right_eye_width
            eyes_ratio = (left_eye_ratio+right_eye_ratio)/2
            # print("eyes_ratio:", eyes_ratio)
            eyes_ratios.append(eyes_ratio)

            # filename
            filenames.append(filename)

        else:
            # nose_angle.append(None)
            # filenames.append(filename)
            pass

df = pd.DataFrame({'filenames': filenames, 'nose_angle': nose_angle})
df.to_csv('labels.csv')
