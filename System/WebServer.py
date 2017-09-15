# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 13:08:51
# @Last Modified by:   lc
# @Last Modified time: 2017-09-15 15:27:56

import time 

import cv2
from flask import Flask, render_template, Response

from camera import Camera
from Consumer import ImageConsumer, ProbabilityConsumer


app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def fetch_image():
    consumer = ImageConsumer()
    count = 0
    for frame in consumer.get_img():
        count += 1
        print('#####get frame {0}, type:{1}'.format(count, type(frame)))
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def fetch_probability():
    consumer = ProbabilityConsumer()
    for msg in consumer.get_msg():
        print(msg.decode('utf8'))
        yield(msg)


@app.route('/video_feed')
def video_feed():
    #return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(fetch_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/historgram')
def historgram_feed():
    consumer = ProbabilityConsumer()
    for msg in consumer.get_msg():
        print(msg.decode('utf8'))
        return msg.decode('utf8')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)