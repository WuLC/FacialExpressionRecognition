# -*- coding: utf-8 -*-
# Created on Mon Nov 27 2017 11:41:32
# Author: WuLC
# EMail: liangchaowu5@gmail.com
import os
import time

import numpy
import shutil
import cv2
import face_recognition


def load_known_face_encoding(img_dir):
    known_face_encodings = {}
    files = os.listdir(img_dir)
    for file in files:
        name = file.split('.')[0][:-1]
        known_face_encodings.setdefault(name, [])
        file_path = img_dir + file
        img = face_recognition.load_image_file(file_path)
        img_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings[name].append(img_encoding)
        print(name)
    return known_face_encodings
        

def calculate_face_distance(known_face_encodings, face_encoding):
    r_dist, r_name = None, None
    for name, encodings in known_face_encodings.items():
        min_dist = None
        for encoding in encodings:
            if min_dist != None:
                min_dist = min(min_dist, numpy.linalg.norm(encoding - face_encoding))
            else:
                min_dist = numpy.linalg.norm(encoding - face_encoding)
        if r_dist == None or min_dist < r_dist:
            r_dist = min_dist
            r_name = name
    return r_dist, r_name


img_dir = './known/known_faces/sample/'
KNOWN_FACE_ENCODING = load_known_face_encoding(img_dir)

base_dir = 'G:/FacialExpressionRecognition/System/detected_records/20171214/'
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
        name = "Uknown"
        tolerance = 0.4
        r_distance, r_name = calculate_face_distance(KNOWN_FACE_ENCODING, face_encoding)
        if r_distance < tolerance:
            name = r_name
        print(name)
        des_dir = categorized_dir + name
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
        shutil.copy(image_path, des_dir)
    

