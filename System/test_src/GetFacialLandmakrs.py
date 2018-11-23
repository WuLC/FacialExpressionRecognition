import dlib
import cv2
import numpy as np


img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/1_S010_004/S010_004_00000008.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/5_S010_006/S010_006_00000009.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/6_S011_002/S011_002_00000008.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/7_S010_002/S010_002_00000011.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g2/4_S050_001/S050_001_00000005.png'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g2/4_S050_001/S050_001_00000009.png'

mouth = ((48, 54))
cheek = ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9))
eyebrow = ((21, 22), (20, 23), (19, 24), (18, 25), (17, 26), (17, 27), (18, 27), (19, 27), (20, 27), (21, 27))
eyebrow = ((18, 20), (17, 21), (23, 25), (22, 26))
eye = ((37, 41), (38, 40), (36, 39), (43, 47), (44, 46), (42, 45))
mouth = ((61, 67), (62, 66), (63, 65), (60, 64))
sad_eyebrow = ((19, 24), (19, 33), (33, 24))
fear_eye_brow = ((18, 36), (19, 41), (20, 40), (21, 39), (22, 42), (23, 47), (24, 46), (25, 45))
angry_eye = ((39, 42), (38, 43), (40, 47), (36, 45))
disgust_eye = ((27, 21), (27, 22), (27, 28))
disgust_mouth = ((49, 59), (50, 58), (51, 57), (52, 56), (53, 55))


img_dir = 'E:/FaceExpression/TrainSet/CK+/10_fold_original/'
predictor_path = '../dlibmodel/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def draw_landmarks(img_path):
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
        # Drawing points on face
        for (x, y) in landmarks:
            cv2.circle(gray_img, (x, y), 2, (255, 0, 0), -1)
        cv2.circle(gray_img, (100, 200), 10,  (0, 255, 0), -1)
        # for p1, p2 in disgust_mouth:
        #    cv2.line(gray_img, tuple(landmarks[p1]), tuple(landmarks[p2]), (255, 0, 0))
        
        # for i in range(68):
        #     if i != 33:
        #         cv2.line(gray_img, tuple(landmarks[33]), tuple(landmarks[i]), (255, 0, 0))
        # cv2.imwrite(img_path.split('.')[0] + '_landmark.jpg', gray_img)

        # draw circle
        # center, radius = draw_circle_with_3_points(landmarks[19], landmarks[20], landmarks[21])
        # print(center, radius)
        # cv2.circle(gray_img, (int(center[0]), int(center[1])), int(radius), (255, 255, 255))
        return gray_img


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
    im1 = 'g1/3_S005_001/S005_001_00000002.png'
    im2 = 'g1/3_S005_001/S005_001_00000009.png'
    im1 = 'g1/1_S011_004/S011_004_00000004.png'
    im2 = 'g1/1_S011_004/S011_004_00000019.png'
    img1 = draw_landmarks(img_dir+im1)
    img2 = draw_landmarks(img_dir+im2)
    cv2.imshow('test', np.hstack((img1, img2)))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()