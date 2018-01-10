import os
import time
import logging
import json
import requests

from sklearn.metrics import confusion_matrix

headers = {
    # Request headers. Replace the placeholder key below with your subscription key.
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': 'XXXXXXXXX',
}

server = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
img_dir = 'F:/FaceExpression/processed/front_regroup_oneframe/all/'


LOG_FILE = 'F:/FaceExpression/processed/front_regroup_oneframe/ms_all.log'
logging.basicConfig(level=logging.DEBUG, filename=LOG_FILE, filemode="a+", format="")

def predict_label(img_path):
    MAPPING = {
                'sadness':'sad', 
                'anger':'angry', 
                'happiness':'happy',
                'disgust':'disgust',
                'fear':'fear',
                'neutral':'neutral',
                'surprise':'surprise',
                'contempt':'contempt'
               }
    with open(img_path, 'rb') as rf:
        img = rf.read()
    response = requests.post(url=server,
                            data=img,
                            headers = headers)

    faces = json.loads(response.content.decode('utf8'))
    if len(faces) == 0:
        print('detect no face in image {0}'.format(img_path))
        return
    else:
        distribution = faces[0]['scores']
        return MAPPING[max(distribution, key=distribution.get)]
    

def fetch_predicted_images(log_file):
    predicted_images = set()
    with open(log_file, 'r') as rf:
        for line in rf:
            predicted_images.add(line.split()[0])
    return predicted_images


def predict_emotion():
    global LOG_FILE
    img_dir = 'G:/FacialExpressionRecognition/System/detected_records/all/'
    img_dir = 'F:/KDEF/SideFaces/'
    # img_dir = 'F:/KDEF/FrontalizeSideFaces/'
    img_dir = 'F:/FaceExpression/processed/front_regroup_oneframe/all/'
    
    mapping = {
                'SA':'sad', 
                'AN':'angry', 
                'HA':'happy',
                'DI':'disgust',
                'FE':'fear',
                'NE':'neutral',
                'SU':'surprise'
               }
    predicted_images = fetch_predicted_images(LOG_FILE)
    files = os.listdir(img_dir)
    for img_name in files:
        if img_name in predicted_images:
            print('{0} has been predicted'.format(img_name))
            continue
        # true_label = img_name.split('_')[0]
        true_label = mapping[img_name.split('_')[1]]
        img_path = img_dir + img_name
        predicted_label = predict_label(img_path)
        print(img_name, true_label, predicted_label, true_label==predicted_label)
        logging.info('{0} {1} {2} {3}'.format(img_name, true_label, predicted_label, true_label==predicted_label))
        time.sleep(4)
    return True


def calculate_accuracy(log_file):
    correct_count, total_count = 0, 0
    with open(log_file, 'r') as rf:
        for line in rf:    
            image_name, true, predict, result = line.strip().split()
            if result == 'True':
                correct_count += 1
            total_count += 1
    print(correct_count, total_count, 1.0*correct_count/total_count)



    
if __name__ == '__main__':
    # detect_face()
    # for _ in range(10000):
    #     try:
    #         finished = predict_emotion()
    #         if finished:
    #             break
    #     except Exception as e:
    #         print('exception occure')
    #         print(e)
    
    # calculate_accuracy(LOG_FILE)