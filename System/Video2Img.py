# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 09:20:58
# @Last Modified by:   lc
# @Last Modified time: 2017-09-18 21:50:04

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # decide to use CPU or GPU
import time
import json
from datetime import datetime

import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from FaceProcessUtilMultiFaces import preprocessImage
from AlexNet import AlexNet
from VGG import VGGModel


SERVER = '127.0.0.1:9092'
VIDEO_TOPIC = 'video'
IMAGE_TOPIC = 'image'
PROBABILITY_TOPIC = 'emotion_probability'
EMOTION = ('neural', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad')
COLOR_RGB = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))
COLOR_HEX = ('#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF')
FONT = cv2.FONT_HERSHEY_SIMPLEX

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



def predict_and_label_frame(video_consumer, img_producer, probability_producer, model, maximum_detect_face = 6):
    """fetch original frame from kafka
       detect whether there is human face in the frame
       predict the emotion of the human face, label it on the image
       then send it to kafka
    """
    consume_count = 0
    produce_count = 0
    while True:
        for img in video_consumer.get_img():
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
                # deal with multiple face in an image
                num_faces = min(maximum_detect_face, len(result['rescaleimg']))
                face_imgs, face_points = result['rescaleimg'], result['originalPoints']
                emotion_distributions = {}
                for i in range(num_faces):
                    emotion, probability_distribution = model.predict(face_imgs[i])
                    distribution = dict(zip(EMOTION, probability_distribution.tolist()[0]))
                    emotion_distributions[COLOR_HEX[i]] = distribution
                    print('*****************probability_distribution: {0}'.format(probability_distribution))
                    
                    # add square and text to the human face in the image
                    left_top, right_bottom = face_points[i]
                    cv2.rectangle(np_img, left_top, right_bottom, COLOR_RGB[i], 2)
                    text_left_bottom = (left_top[0], left_top[1] + 20)
                    cv2.putText(np_img, emotion, text_left_bottom, FONT, 1, COLOR_RGB[i], 2)

                    
                # cv2.imwrite('./text_imgs/img_{0}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), np_img)              
                    
                # send image to kafka
                img_producer.send_img(cv2.imencode('.jpeg', np_img)[1].tostring())
                print('#########produce {0} to image stream'.format(produce_count))

                # send emotion probability distribution to kafka
                    
                probability_producer.send_probability_distribution(json.dumps(emotion_distributions).encode('utf8'))
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
                empty_distribution = {COLOR_HEX[0] : dict(zip(EMOTION, [0] * 7))}
                probability_producer.send_probability_distribution(json.dumps(empty_distribution).encode('utf8'))
            # img_producer.send_img(json.dumps(message).encode('utf8'))
            print('**********time consumed by prediction: {0}s'.format(time.time() - start_time))


if __name__ == '__main__':
    video_consumer = VideoConsumer()
    img_producer = ImageProducer()
    probability_producer = ProbabilityProducer()
    # model = AlexNet()
    model = VGGModel()

    predict_and_label_frame(video_consumer, img_producer, probability_producer, model)