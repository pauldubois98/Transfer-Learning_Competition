import os
import random
import cv2
import dlib
import numpy as np


detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

in_folder_name = 'raw'
out_folder_name = 'cropped'
img_list = os.listdir(in_folder_name)

i = 0
for img_filename in img_list[:]:
    if i % 100 == 0:
        print(i, img_filename)
    age, sex, race, _ = img_filename.split('_')
    # print(age, sex, race)

    image = cv2.imread(in_folder_name+'/'+img_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    for rect in rects:
        l = rect.left()
        r = rect.right()
        b = rect.bottom()
        t = rect.top()
        w = rect.width()
        h = rect.height()
        s = int((w+h)/4)
        if s < 100:
            continue
        l = rect.center().x-s
        r = rect.center().x+s
        t = rect.center().y-s
        b = rect.center().y+s
        margin = min(l, t, gray.shape[0]-r, gray.shape[1]-b)
        if margin < 0:
            continue

        l -= margin
        r += margin
        t -= margin
        b += margin

        if l < 0 or t < 0 or r >= gray.shape[0] or b >= gray.shape[1]:
            # print('out')
            continue

        # print('in')
        i += 1
        cropped = image[t:b, l:r, :]
        out_name = out_folder_name
        out_name += '/big'
        out_name += str(i)
        out_name += '_'
        if sex == '0':
            out_name += 'H'
        elif sex == '1':
            out_name += 'F'
        out_name += '_'
        out_name += age
        out_name += '_'
        out_name += race
        out_name += '.jpg'
        cv2.imwrite(out_name, cropped)
