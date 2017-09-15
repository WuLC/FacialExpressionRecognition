# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 09:20:58
# @Last Modified by:   lc
# @Last Modified time: 2017-09-15 17:02:07

import time
import json
from datetime import datetime

import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from FaceProcessUtil import preprocessImage
from AlexNet import AlexNet
from VGG import VGGModel


SERVER = '125.216.242.154:9092'
SERVER = '192.168.31.89:9092'
VIDEO_TOPIC = 'video'
IMAGE_TOPIC = 'image'
PROBABILITY_TOPIC = 'emotion_probability'
EMOTION = ['neural', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad']

class VideoConsumer():
    def __init__(self):
        self.consumer = KafkaConsumer(bootstrap_servers = [SERVER], auto_offset_reset='latest') # earliest

    def get_img(self):
        self.consumer.subscribe([VIDEO_TOPIC])
        for message in self.consumer:
            if message.value != None:
                yield message.value
            """
            # convert bytes to image in memory
            nparr = np.fromstring(message.value, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
            """


class ImageProducer():
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers = [SERVER])

    def send_img(self, img):
        self.producer.send(IMAGE_TOPIC, value = img)


class ProbabilityProducer():
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers = [SERVER])

    def send_probability_distribution(self, msg):
        self.producer.send(PROBABILITY_TOPIC, value = msg)



if __name__ == "__main__":
    consumer = VideoConsumer()
    img_producer = ImageProducer()
    pro_producer = ProbabilityProducer()
    # model = AlexNet()
    model = VGGModel()

    consume_count = 0
    produce_count = 0
    while True:
        for img in consumer.get_img():
            start_time = time.time()
            consume_count += 1
            print('========Consume {0} from video stream'.format(consume_count))

            # write original image to disk
            """
            raw_dest_img = './rev_img/original{0}.png'.format(consume_count)
            with open(raw_dest_img, 'wb') as f:
                f.write(img)
            """
            # transform image from bytes to ndarray
            np_arr = np.fromstring(img, np.uint8) # one dimension array
            np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            result = preprocessImage(np_img)
            print('**********time consumed by face detection: {0}s'.format(time.time() - start_time))
            start_time = time.time()
            if result['detected']: # detect human face
                produce_count += 1
                emotion, probability_distribution = model.predict(result['rescaleimg'])
                print('*****************probability_distribution: {0}'.format(probability_distribution))
                # add square and text to the human face in the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0,255,127)
                left_top, right_bottom = result['rectPoints']
                cv2.rectangle(np_img, left_top, right_bottom, color, 2)
                text_left_bottom = (right_bottom[0], right_bottom[1] + 20)
                cv2.putText(np_img, emotion, text_left_bottom, font, 1, color, 2)
                
                # cv2.imwrite('./text_imgs/img_{0}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), np_img)
                
                
                # send image to kafka
                img_producer.send_img(cv2.imencode('.jpeg', np_img)[1].tostring())
                print('#########produce {0} to image stream'.format(produce_count))

                # send emotion probability distribution to kafka
                distribution = dict(zip(EMOTION, probability_distribution.tolist()[0]))
                pro_producer.send_probability_distribution(json.dumps(distribution).encode('utf8'))
                print('#########produce {0} to probability stream'.format(distribution))
                """
                # send both image and distribution to kafka
                message = []
                message['img'] = np_img.tolist()
                message['distribution'] = json.dumps(dict(zip(EMOTION, probability_distribution.tolist()[0])))
                print('#########produce {0} to image stream, {1}'.format(produce_count, message['distribution']))
                """
            else:
                # message = {'img': img, 'distribution': None}
                img_producer.send_img(img)
                print('#########produce raw image to image stream')
                distribution = dict(zip(EMOTION, [0] * 7))
                pro_producer.send_probability_distribution(json.dumps(distribution).encode('utf8'))
            # img_producer.send_img(json.dumps(message).encode('utf8'))
            print('**********time consumed by prediction: {0}s'.format(time.time() - start_time))