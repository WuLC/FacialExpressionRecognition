import os 

import dlib
import cv2
import numpy as np
import fire

img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/1_S010_004/S010_004_00000008.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/5_S010_006/S010_006_00000009.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/6_S011_002/S011_002_00000008.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/7_S010_002/S010_002_00000011.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g2/4_S050_001/S050_001_00000005.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g2/4_S050_001/S050_001_00000009.png'

cheek1 = ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9))
cheek2 = ((33, 0), (33, 1), (33, 2), (33, 3), (33, 4),(33, 5),(33, 6), (33, 7), (33, 8), (33, 9), (33, 10), (33, 11), (33, 12), (33, 13), (33, 14), (33, 15), (33, 16))
eyebrow1 = ((21, 22), (20, 23), (19, 24), (18, 25), (17, 26),(39, 42), (38, 43), (40, 47), (36, 45))
eyebrow2 = ((17, 27), (18, 27), (19, 27), (20, 27), (21, 27), (22, 27), (23, 27), (24, 27), (25, 27), (26, 27))
eyebrow3 = ((18, 20), (17, 21), (23, 25), (22, 26), (37, 38), (40, 41), (36, 39), (43, 44), (46, 47), (42, 45))
eyebrow4 = ((18, 36), (19, 41), (20, 40), (21, 39), (22, 42), (23, 47), (24, 46), (25, 45))
eyebrow5 = ((19, 24), (19, 33), (33, 24))
eye1 = ((21, 27), (22, 27), (27, 28))
eye2 = ((37, 41), (38, 40), (36, 39), (43, 47), (44, 46), (42, 45))
angry_eye = ()
disgust_eye = ((27, 21), (27, 22), (27, 28))
disgust_mouth = ((49, 59), (50, 58), (51, 57), (52, 56), (53, 55))
mouth = ((61, 67), (62, 66), (63, 65), (60, 64))


img_dir = 'E:/FaceExpression/TrainSet/CK+/10_fold_original/'
predictor_path = '../dlibmodel/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def draw_landmarks(src):
    for f in os.listdir(src): 
        img_path = src + f
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        print("Number of faces detected: {}".format(len(faces)))
        for k, d in enumerate(faces):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            landmarks = predictor(gray_img, d)
            landmarks = land2coords(landmarks)

            # draw points on face
            for (x, y) in landmarks:
                cv2.circle(gray_img, (x, y), 2, (255, 0, 0), -1)

            # draw lines between points
            for p1, p2 in cheek2:
               cv2.line(gray_img, tuple(landmarks[p1]), tuple(landmarks[p2]), (255, 0, 0))
            
            # for i in range(68):
            #     if i != 33:
            #         cv2.line(gray_img, tuple(landmarks[33]), tuple(landmarks[i]), (255, 0, 0))

            # draw circle
            # center, radius = draw_circle_with_3_points(landmarks[19], landmarks[20], landmarks[21])
            # print(center, radius)
            # cv2.circle(gray_img, (int(center[0]), int(center[1])), int(radius), (255, 255, 255))

            cv2.imwrite(img_path.split('.')[0] + '_landmark.jpg', gray_img)


def merge_landmark(src):
    #src = F:\FacialExpressionRecognition\Algorithm\Datasets\CK+\cohn-kanade-images\S010\002_landmark_movements\
    all_landmarks = []
    for f in ['S010_002_00000001.png', 'S010_002_00000014.png']:
        img_path = src + f
        img = cv2.imread(img_path)
        img_size = img.shape
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        print("Number of faces detected: {}".format(len(faces)))
        for k, d in enumerate(faces):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            landmarks = predictor(gray_img, d)
            landmarks = land2coords(landmarks)
            all_landmarks.append(landmarks)
    # Drawing points on canvas
    colors = ((255, 255, 255), (0, 0, 255))
    canvas = np.zeros(img_size, dtype='int')
    for j in range(68):
        x1, y1 = all_landmarks[0][j]
        x2, y2 = all_landmarks[1][j]
        cv2.circle(canvas, (x1, y1), 1.5, colors[0], -1)
        cv2.circle(canvas, (x2, y2), 1, colors[1], -1)
        cv2.arrowedLine(canvas, (x1, y1), (x2, y2), colors[0], 1)
    cv2.imwrite(src+'merge_landmark.jpg', canvas)    


def draw_circle_with_3_points(p1, p2, p3):
    x, y, z = complex(p1[0], p1[1]), complex(p2[0], p2[1]), complex(p3[0], p3[1]),
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    center = (c.real, c.imag)
    radius = abs(c+x)
    return center, radius


# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords


def main():
    pass


if __name__ == '__main__':
    fire.Fire()