# -*- coding: utf-8 -*-
# Created on Tue Dec 05 2017 20:9:28
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import cv2

def video2Images(video_path, img_dir):
    capture = cv2.VideoCapture(video_path)
    emotion = video_path.split('/')[-1].split('.')[0]
    des_dir = img_dir+emotion
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    recv, image = capture.read()
    count = 0
    while recv:
        count += 1
        cv2.imwrite("{0}/{1:03d}.jpg".format(des_dir, count), image)     # save frame as JPEG file
        recv, image = capture.read()
    print('extract {0} frames'.format(count))

def main():
    angles = ['left', 'front', 'right']
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for angle in angles:
        video_dir = './constant_light/videos/{0}/'.format(angle)
        img_dir = './constant_light/frames/{0}/'.format(angle)
        for person_id in os.listdir(video_dir):
            for e in emotions:
                video_path = video_dir + '{0}/{1}.avi'.format(person_id, e)
                img_des_dir = img_dir + '{0}/'.format(person_id)
                if not os.path.exists(video_path):
                    print('{0} not exists'.format(video_path))
                    continue
                video2Images(video_path, img_des_dir)

if __name__ == '__main__':
    main()