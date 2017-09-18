# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-12 21:06:45
# @Last Modified by:   lc
# @Last Modified time: 2017-09-18 20:31:38


from kafka import KafkaConsumer

SERVER = '127.0.0.1:9092'
IMAGE_TOPIC = 'image'
PROBABILITY_TOPIC = 'emotion_probability'

class ImageConsumer():
    def __init__(self):
        self.consumer = KafkaConsumer(bootstrap_servers = [SERVER], auto_offset_reset='latest') # earliest
        self.consumer.subscribe([IMAGE_TOPIC])
    
    def get_img(self):
        for img in self.consumer:
            if img.value:
                yield img.value


class ProbabilityConsumer():
    def __init__(self):
        self.consumer = KafkaConsumer(bootstrap_servers = [SERVER], auto_offset_reset='latest') # earliest
        self.consumer.subscribe([PROBABILITY_TOPIC])

    def get_msg(self):
        for msg in self.consumer:
            yield msg.value
