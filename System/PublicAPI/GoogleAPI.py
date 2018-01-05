# encoding=utf8
import os
import random
import base64
import json
import logging

import requests

key = 'AIzaSyC1XQigTOPCYkHGYkjA6QF-wur6sJsJJaw'
request_url =  'https://vision.googleapis.com/v1/images:annotate?key={0}'.format(key)


LOG_FILE = '/media/lc/F/FaceExpression/CollectedValidationSet/processed/front_regroup_oneframe/google_all.log'
logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode="a+", format="")

def predict_label(img_path):
    global request_url
    request_list = []
    with open(img_path, 'rb') as rf:
        encoded_img = base64.b64encode(rf.read()).decode('utf8')


    content_json_obj = {'content': encoded_img}
    feature_json_obj = [{'type':'FACE_DETECTION', 'maxResults':1}]
    request_list.append({'image':content_json_obj, 'features':feature_json_obj})


    req = {'requests': request_list}
    request_json = json.dumps(req)

    response = requests.post(request_url, 
                    data = request_json, 
                    headers = {'Content-Type': 'application/json'})
    result = json.loads(response.text)
    emotions = result['responses'][0]['faceAnnotations'][0]

    distri = {}
    keys = ['joyLikelihood', 'sorrowLikelihood', 'angerLikelihood', 'surpriseLikelihood']
    for k in keys:
        distri[k.rstrip('Likelihood')] = emotions[k].lower()
    return json.dumps(distri)


def fetch_predicted_images(log_file):
    predicted_images = set()
    with open(log_file, 'r') as rf:
        for line in rf:
            predicted_images.add(line.split()[0])
    return predicted_images


def predict_emotion():
    global LOG_FILE
    img_dir = '/media/lc/F/FaceExpression/CollectedValidationSet/processed/front_regroup_oneframe/all/original/'
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
        distribution = predict_label(img_path)
        print(img_name, true_label, distribution)
        logging.info('{0} {1} {2}'.format(img_name, true_label, distribution))
    return True


def calculate_accuracy(log_file):
    correct_count, total_count = 0, 0
    scores = {'very_likely':5, 'likely':4, 'possible':3, 'unlikely':2, 'very_unlikely':1}
    new_log_file = log_file+ '_new'
    with open(new_log_file, encoding='utf8', mode='w') as wf:
        with open(log_file, 'r') as rf:
            for line in rf:
                image_name, true, *distribution = line.strip().split()
                cates = json.loads(line.strip().lstrip(image_name + ' ' + true + ' '))
                emotions = {}
                emotions['angry'] = scores[cates['anger']]
                emotions['disgust'] = scores[cates['anger']]
                emotions['happy'] = scores[cates['joy']]
                emotions['sad'] = scores[cates['sorrow']]
                emotions['fear'] = scores[cates['surprise']]
                emotions['surprise'] = scores[cates['surprise']]
                # rule: disgust = anger, fear = surprise, neutral if other emotions equal
                max_score = max(emotions.values())
                candidates = set()
                for k, v in emotions.items():
                    if v == max_score:
                        candidates.add(k)
                predict = None
                if len(candidates) == 6:
                    predict = 'neutral'
                    if true == 'neutral':
                        correct_count += 1
                elif true in candidates:
                    correct_count += 1
                    predict = true
                else:
                    predict = random.sample(candidates, 1)[0]  
                total_count += 1
                print(predict)
                wf.write('{0} {1} {2} {3}\n'.format(image_name, true, predict, true==predict))
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
    calculate_accuracy(LOG_FILE)