# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 09:20:46
# @Last Modified by:   lc
# @Last Modified time: 2017-09-13 09:23:26

import os
import time

from kafka import KafkaConsumer, KafkaProducer

SERVER = '125.216.242.154:9092'
SERVER = '192.168.31.89:9092'
TOPIC = 'video'


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

if __name__ == "__main__":
    img_dir = './src_img/'
    producer = Producer()
    producer.send_img(img_dir)