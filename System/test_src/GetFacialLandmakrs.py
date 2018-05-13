import dlib
import cv2
import numpy as np

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

predictor_path = '../dlibmodel/shape_predictor_68_face_landmarks.dat'
img_path = '/mnt/e/FaceExpression/TrainSet/CK+/10_fold_original/g1/1_S010_004/S010_004_00000008.png'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
mouth = ((48, 54))
cheek = ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9))
eyebrow = ((21, 22), (20, 23), (19, 24), (18, 25), (17, 26), (17, 27), (18, 27), (19, 27), (20, 27), (21, 27))

faces = detector(gray_img, 1)
print("Number of faces detected: {}".format(len(faces)))
for k, d in enumerate(faces):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    landmarks = predictor(gray_img, d)
    landmarks = land2coords(landmarks)
    for (x, y) in landmarks:
        # Drawing points on face
        cv2.circle(gray_img, (x, y), 2, (255, 0, 0), -1)
    for p1, p2 in eyebrow:
        cv2.line(gray_img, tuple(landmarks[p1]), tuple(landmarks[p2]), (255, 0, 0))
    cv2.imwrite(img_path.split('.')[0] + '_landmark.jpg', gray_img)
    