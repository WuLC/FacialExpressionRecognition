import os
import numpy as np
import cv2
import dlib


def detect_face_with_opencv(img_path, face_detector):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected = False
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5)
    for (x, y, w, h) in faces:
        detected = True
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    if detected:
        cv2.imwrite(img_path.rstrip('.jpeg') + '_opencv.jpeg', img)
        return True
    else:
        return False


def detect_face_with_dlib(img_path, face_detector):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected = False
    faces = face_detector(gray, 0)
    for face in faces:
        left_top, right_bottom = (face.left(), face.top()), (face.right(), face.bottom())
        detected = True
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 255), 2)
    if detected:
        cropped_img = gray[face.top():face.bottom(), face.left():face.right()]
        cv2.imwrite(img_path.rstrip('.jpeg') + '_dlib.jpeg', cropped_img)
        return True
    else:
        return False


def crop_face(img_path):
    img = cv2.imread(img_path)
    width, height, channels = img.shape
    crop_size = 80
    left_top = (int((width-crop_size)/2), int((height-crop_size)/2))
    right_bottom = (left_top[0]+crop_size, left_top[1]+crop_size)
    cv2.rectangle(img, left_top, right_bottom, (255, 0, 255), 2)
    cv2.imwrite(img_path.rstrip('.jpeg') + '_crop.jpeg', img)


def remove_rectangle_files(img_dir):
    files = os.listdir(img_dir)
    for file in files:
        if file.endswith('dlib.jpeg') or file.endswith('opencv.jpeg') or file.endswith('crop.jpeg'):
            os.remove(img_dir + file)


def main():
    face_detector_model = '../OpencvFaceDetector/haarcascade_frontalcatface_extended.xml'
    opencv_face_detector = cv2.CascadeClassifier(face_detector_model)
    dlib_face_detector = dlib.get_frontal_face_detector()
    detect = True
    img_dir = '../crop_imgs/'
    img_dir = '../FaceRecognition/known/records/'
    total_count, crop_count = 0, 0
    if detect:
        for f in os.listdir(img_dir):
            if f.endswith('face++.jpeg'):
                continue
            total_count += 1
            img_path = img_dir + f
            if detect_face_with_dlib(img_path, dlib_face_detector):
                print('detect with dlib')
            elif detect_face_with_opencv(img_path, opencv_face_detector):
                print('detect with opencv')
            else:
                crop_count += 1
                crop_face(img_path)
                print('crop directly')
        print('total count {0}, crop count {1}, account for {2}'.format(total_count, crop_count, crop_count/total_count))
    else:
        remove_rectangle_files(img_dir)



if __name__ == '__main__':
    main()


"""
dlib: 213/332
face++: 228/332
"""