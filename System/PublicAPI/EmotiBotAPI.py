# -*- coding: utf-8 -*-
# Created on Wed Jan 03 2018 15:29:34
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import time
import json
import logging
import requests

appid = "5a200ce8e6ec3a6506030e54ac3b970e"
userid = '0F31A7360C847D2563E0CFE5204080A9E'
url = "http://idc.emotibot.com/api/ApiKey/openapi.php"
params = {   
    "cmd":"getFace",
    "appid":appid,
    "userid":userid
    }

LOG_FILE = 'F:/FaceExpression/CollectedValidationSet/processed/front_regroup_oneframe/emotibot_all.log'
logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode="a+", format="")


def predict_label(img_path):
    global params
    upload_file = {'file': open(img_path, 'rb')}
    r = requests.post(url, params=params,files = upload_file)
    response = json.dumps(r.json(), ensure_ascii=False)
    jsondata = json.loads(response)
    success = True if jsondata.get('return') == 0 else False
    print(success)
    datas = jsondata.get('data')
    for data in datas:
        e_distribution = {}
        all_distribution = data.get('emotions')
        for e in ['angry', 'disgust', 'fear', 'neutral', 'happy', 'sad', 'surprise']:
            if e in all_distribution:
                e_distribution[e] = all_distribution[e]
        print(e_distribution, sum(e_distribution.values()))
        emotion = max(e_distribution, key=e_distribution.get)
        return emotion, e_distribution


def fetch_predicted_images(log_file):
    predicted_images = set()
    with open(log_file, 'r') as rf:
        for line in rf:
            predicted_images.add(line.split()[0])
    return predicted_images


def predict_emotion():
    global LOG_FILE
    img_dir = 'F:/FaceExpression/CollectedValidationSet/processed/front_regroup_oneframe/all/original/'
    
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
        predicted_label, distribution = predict_label(img_path)
        print(img_name, true_label, predicted_label, true_label==predicted_label)
        logging.info('{0} {1} {2} {3} {4}'.format(img_name, true_label, predicted_label, true_label==predicted_label, distribution))
        time.sleep(4)
    return True


def calculate_accuracy(log_file):
    correct_count, total_count = 0, 0
    with open(log_file, 'r') as rf:
        for line in rf:    
            image_name, true, predict, result, *distribution = line.strip().split()
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
    calculate_accuracy(LOG_FILE)