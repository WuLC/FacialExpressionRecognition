# -*- coding: utf-8 -*-
# Created on Mon Nov 27 2017 11:41:32
# Author: WuLC
# EMail: liangchaowu5@gmail.com
import os
import time

import shutil
import cv2
import face_recognition


def load_known_face_encoding(img_dir):
    known_people, known_face_encodings = [], []
    files = os.listdir(img_dir)
    for file in files:
        name = file.split('.')[0]
        file_path = img_dir + file
        img = face_recognition.load_image_file(file_path)
        img_encoding = face_recognition.face_encodings(img)[0]
        known_people.append(name)
        known_face_encodings.append(img_encoding)
    return tuple(known_people), tuple(known_face_encodings)
        


img_dir = './known/'
KNOWN_PEOPLE, KNOWN_FACE_ENCODING = load_known_face_encoding(img_dir)

base_dir = 'G:/FacialExpressionRecognition/System/detected_records/'
img_dir = base_dir + 'all_faces/'
categorized_dir = base_dir +'categorized/'

for img_name in os.listdir(img_dir):
    image_path = img_dir + img_name

    np_img = cv2.imread(image_path)
    face_locations = face_recognition.face_locations(np_img)
    face_encodings = face_recognition.face_encodings(np_img, face_locations, num_jitters=1)


    for face_encoding in face_encodings:
        """
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(KNOWN_FACE_ENCODING, face_encoding, tolerance=0.4)
        name = "Unknown"
        print(match)
        for i in range(len(match)):
            if match[i]:
                name = KNOWN_PEOPLE[i]
                break
        """
        # recognize with euclidean distance directly
        distances = face_recognition.face_distance(KNOWN_FACE_ENCODING, face_encoding)
        #print(distances)
        name = "Uknown"
        tolerance = 0.4
        min_dist = min(distances)
        if min_dist <= tolerance:
            name = KNOWN_PEOPLE[list(distances).index(min_dist)]
    print(name)
    des_dir = categorized_dir + name
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    shutil.copy(image_path, des_dir)
    

