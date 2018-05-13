# -*- coding: utf-8 -*-
# Created on Tue Oct 10 2017 10:57:58
# Author: WuLC
# EMail: liangchaowu5@gmail.com


import os
import time
import base64
import json

from kafka import KafkaConsumer, KafkaProducer

SERVER = '125.216.242.158:9092'
TOPIC = 'test__'


class Producer():

    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers = [SERVER])

    def send_img(self, img_dir):
        count = 0
        for img in os.listdir(img_dir):         
            with open(img_dir + img, 'rb') as f:
                content = f.read()
            if content:
                count += 1
                print('======producer {0}'.format(count))
                self.producer.send(TOPIC, key = img.encode('utf8'), value = content)
                #time.sleep(1)

    def send_json(self, img_path, points):
        data = {}
        with open(img_path, 'rb') as rf:
            encoded_img = base64.encodebytes(rf.read())
        data['img'] = encoded_img.decode('utf8')
        data['points'] = points
        self.producer.send(TOPIC, value = json.dumps(data).encode('utf8'))


if __name__ == "__main__":
    img_dir = '../test_imgs/'
    producer = Producer()
    # producer.send_img(img_dir)
    img_path = img_dir + 'multi_faces.jpg'
    points = '10#20#30#40*60#70#80#90'
    producer.send_json(img_path, points)