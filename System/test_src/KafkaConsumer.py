# -*- coding: utf-8 -*-
# Created on Tue Oct 10 2017 10:58:17
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import time
import base64
import json

from kafka import KafkaConsumer, KafkaProducer

SERVER = '125.216.242.158:9092'
TOPIC = 'test__'

class Consumer():

    def __init__(self):
        self.consumer = KafkaConsumer(bootstrap_servers = [SERVER], auto_offset_reset = 'earliest') # earliest
        self.consumer.subscribe([TOPIC])


    def recv_json(self):
        for data in self.consumer:
            msg = json.loads(data.value.decode('utf8'))

            encoded_img = msg['img'].encode('utf8')
            points = msg['points']
            print('Received points {0}'.format(points))
            with open('test.jpg', 'wb') as wf:
                wf.write(base64.decodebytes(encoded_img))
        


if __name__ == "__main__":
    consumer = Consumer()
    consumer.recv_json()

