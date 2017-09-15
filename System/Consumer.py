# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-12 21:06:45
# @Last Modified by:   lc
# @Last Modified time: 2017-09-15 15:28:01


from kafka import KafkaConsumer


SERVER = '125.216.242.154:9092'
SERVER = '192.168.31.89:9092'
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
