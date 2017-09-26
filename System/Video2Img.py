# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 09:20:58
# @Last Modified by:   lc
# @Last Modified time: 2017-09-26 15:17:20


#######################################################################
# 1. fetch original frame from kafka 
# 2. detect whether there are human faces in the frame
# 3. if detected, predict the emotions of  human faces, label them on the image
# 4. send the processed image to kafka   
# 5. send the emotion distribution to kafka
#######################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # decide to use CPU or GPU
import time
import json
from datetime import datetime
from multiprocessing import Pool

import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from FaceProcessUtilMultiFaces import preprocessImage
from Recorder import FileRecorder, RedisRecorder
from Models.ShallowModels import LogisticRegression

SERVER = '127.0.0.1:9092'
SERVER = '125.216.242.158:9092'
VIDEO_TOPIC = 'video'
IMAGE_TOPIC = 'image'
PROBABILITY_TOPIC = 'emotion_probability'
EMOTION = ('neutral', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad')
COLOR_RGB = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0))
COLOR_HEX = ('#00FF00', '#0000FF', '#FF0000', '#FFFF00', '#FF00FF', '#00FFFF')
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

"""
global_model = VGGModel()
def predict(face_img):
    global global_model
    return global_model.predict(face_img) # (emotion, probability_distribution)
"""

def predict_and_label_frame(video_consumer , img_producer, probability_producer, recorder, model, pool = None, maximum_detect_face = 6):
    """fetch original frame from kafka with video_consumer
       detect whether there are human faces in the frame
       predict the emotions of  human faces, label them on the image
       send the processed image to kafka with img_producer 
       send the emotion distribution to kafka with probability_producer
    """
    consume_count = 0
    produce_count = 0
    while True:
        for img in video_consumer.get_img():
            start_time = time.time()
            consume_count += 1
            # write original image to disk
            """
            raw_dest_img = './rev_img/original{0}.png'.format(consume_count)
            with open(raw_dest_img, 'wb') as f:
                f.write(img)
            """
            # transform image from bytes to ndarray
            np_arr = np.fromstring(img, dtype = np.uint8) # one dimension array
            np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            result = preprocessImage(np_img)
            print('**********time consumed by face detection: {0}s'.format(time.time() - start_time))
            
            start_time = time.time()
            if result['detected']: # detect human face
                produce_count += 1
                # deal with multiple face in an image
                num_faces = min(maximum_detect_face, len(result['rescaleimg']))
                face_imgs, geometric_features, face_points = result['rescaleimg'], result['geometricFeatures'], result['originalPoints']
                emotion_distributions = {}
                # use multiple processes to predict
                # predicted_results = pool.map(predict, face_imgs)
                # predicted_results = [pool.apply(predict, args = (face_imgs[i], )) for i in range(num_faces)]
                print(type(face_imgs), face_imgs.shape)
                print(type(geometric_features), geometric_features.shape)

                all_emotion, all_probability_distribution = model.predict(geometric_features)
                for i in range(num_faces):
                    emotion, probability_distribution = all_emotion[i], all_probability_distribution[i]
                    #emotion, probability_distribution = model.predict(face_imgs[i])
                    distribution = dict(zip(EMOTION, probability_distribution.tolist()))
                    emotion_distributions[COLOR_HEX[i]] = distribution
                    print('*****************probability_distribution: {0}'.format(probability_distribution))
                    
                    # write the record to redis     
                    recorder.write_record(face_imgs[i].tostring(), emotion)

                    # add square and text to the human face in the image
                    left_top, right_bottom = face_points[i]
                    cv2.rectangle(np_img, left_top, right_bottom, COLOR_RGB[i], 2)
                    text_left_bottom = (left_top[0], left_top[1] - 20)
                    cv2.putText(np_img, emotion, text_left_bottom, FONT, 1, COLOR_RGB[i], 2)

                print('**********time consumed by predicting, storing and texting image: {0}s'.format(time.time() - start_time))
                    
                # cv2.imwrite('./test_imgs/img_{0}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), np_img)              
                
                start_time = time.time()
                # send image to kafka
                img_producer.send_img(cv2.imencode('.jpeg', np_img)[1].tostring())
                print('#########produce {0} to image stream'.format(produce_count))
                # send emotion probability distribution to kafka
                probability_producer.send_probability_distribution(json.dumps(emotion_distributions).encode('utf8'))
                print('#########produce {0} to probability stream'.format(emotion_distributions))
                print('**********time consumed by sending image and distribution: {0}s'.format(time.time() - start_time))

            else:
                # message = {'img': img, 'distribution': None}
                img_producer.send_img(img)
                print('#########produce raw image to image stream')
                empty_distribution = {COLOR_HEX[0] : dict(zip(EMOTION, [0] * 7))}
                probability_producer.send_probability_distribution(json.dumps(empty_distribution).encode('utf8'))
            

if __name__ == '__main__':
    video_consumer = VideoConsumer()
    img_producer = ImageProducer()
    probability_producer = ProbabilityProducer()
    recorder = RedisRecorder()
    # pool = Pool(2)
    # record_dir = './detected_records/'
    # file_recorder = FileRecorder(record_dir)
    # model = AlexNet()
    model = LogisticRegression()
    print('model is loaded successfully, ready to start')

    predict_and_label_frame(video_consumer = video_consumer, 
                            img_producer = img_producer, 
                            probability_producer = probability_producer, 
                            recorder = recorder,
                            model = model)