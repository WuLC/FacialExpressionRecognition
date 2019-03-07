# -*- coding: utf-8 -*-
# Created on Wed Mar 06 2019 22:59:9
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import cv2
import urllib2
import time
import ast


def face_expression_recognition(frame):
    http_url='https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "7bf9zJLe912ZdClwKRDPUJqY3RwpENJ4"
    secret = "UaLGOBCfT64AOitUah8rk7rI1pb5nuFr"
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(cv2.imencode('.jpg', frame)[1].tostring())
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append("emotion")
    data.append('--%s--\r\n' % boundary)

    http_body='\r\n'.join(data)
    #buld http request
    req=urllib2.Request(http_url)
    #header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.add_data(http_body)
    try:
        #req.add_header('Referer','http://remotserver.com/')
        #post data to server
        resp = urllib2.urlopen(req, timeout=5)
        #get response
        content = resp.read()
        return position_and_emotion(content)
    except urllib2.HTTPError as e:
        print e.read()


def position_and_emotion(content):
    # response is string, change it to dictionary
    d = ast.literal_eval(content)
    if len(d['faces']) == 0:
        return False, None
    emotion, prob = "", 0
    for k, v in d['faces'][0]['attributes']['emotion'].items():
        if v > prob:
            emotion = k
            prob = v
    top, left = d['faces'][0]['face_rectangle']['left'], d['faces'][0]['face_rectangle']['top']
    width, height = d['faces'][0]['face_rectangle']['width'], d['faces'][0]['face_rectangle']['height']
    return True, (emotion, prob, top, left, width, height)


def main():
    capture = cv2.VideoCapture(0)
    _, frame = capture.read()
    count = 0
    while frame is not None:
        if count % 5 == 0:
            detected, info = face_expression_recognition(frame)
            if detected:
                print(info)
                emotion, prob, x, y, w, h = info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                cv2.putText(frame, emotion+'({0}%)'.format(prob), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)      
            cv2.imshow('Video', frame)
        _, frame = capture.read()
        count += 1
        # Hit 'q' on the keyboard to quit!   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # img_path = "D:/test.jpg"
    # frame = cv2.imread(img_path)
    # detected, info = face_expression_recognition(frame)
    # if detected:
    #     print(info)
    #     emotion, x, y, w, h = info
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    #     cv2.putText(frame, emotion, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    #     cv2.imwrite(img_path.rstrip('.jpg') + '_opencv.jpg', frame)
    # else:
    #     print('detect no face')
    main()