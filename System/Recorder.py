# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-09-21 11:16:46
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-28 15:42:22

########################################################
# save the detected face and emotion on disk or database
########################################################

import os
from datetime import datetime

import redis
import cv2
import numpy as np

class FileRecorder():
    """write records in a file on disk
    Attributes:
        curr_id (int): id of the current record to write on disk 
        latest_file (str): path of the record file
        max_record (int): maximum records allowed to store in a file
        record_dir (str): directory containing the record files
    """
    def __init__(self, record_dir, max_record_pre_file = 1000):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
            print('Successfully creating directory {0}'.format(record_dir))
        self.record_dir = record_dir
        self.max_record = max_record_pre_file
        self.curr_id = 0
        self.latest_file = self._get_latest_file()


    def _get_latest_file(self):
        files = sorted(os.listdir(self.record_dir))
        if len(files) > 0:
            curr_latest = self.record_dir + files[-1]
            with open(curr_latest, 'r') as rf:
                lines_count = sum(1 for line in rf)
            if lines_count < self.max_record:
                self.curr_id = lines_count
                return curr_latest
        new_csv = self.record_dir + '{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
        with open(new_csv, 'w') as wf: # add csv header
            wf.write('id,pixels,label\n')
        return new_csv


    def write_record(self, img, label):
        self.curr_id += 1
        params = {'id' : self.curr_id,  'pixels' : img,  'label' : label}
        with open(self.latest_file, 'a') as wf:
            wf.write('{id:05d},{pixels},{label}\n'.format(**params))
        # create new csv file when reaching max records
        if self.curr_id == self.max_record:
            new_csv = self.record_dir + '{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(new_csv, 'w') as wf: # add csv header
                wf.write('id,pixels,label\n')
            self.latest_file = new_csv
            self.curr_id = 0


class RedisRecorder():
    def __init__(self,
                 HOST = '125.216.242.158',
                 PORT = 6379,
                 PASSWORD = 'XXXXX',
                 DB = 0):
        try:
            self.conn = redis.Redis(host = HOST, port = PORT, password = PASSWORD, db= DB)
        except Exception:
            print('Exception while connecting to redis')
            exit()

        try:
            self.count = int(self.conn.get('count'))
        except Exception:
            print('Exception while getting the count variable, set it to 0\nExit')
            self.count = 0
            self.conn.set('count', 0)


    def write_record(self, img, emotion):
        # update count in the db
        self.count += 1
        self.conn.set('count', str(self.count))
        # store image and label in the db
        record_name = 'face{0:06d}'.format(self.count)
        self.conn.hset(name = record_name, key = 'str_img', value = img)
        self.conn.hset(name = record_name, key = 'emotion', value = emotion)


    def restore_record(self, key_pattern, record_dir):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        face_ids = self.conn.keys(key_pattern)
        print('Totally get {0} records from redis database'.format(len(face_ids)))
        resize_shape = (128, 128)
        for i in range(len(face_ids)):
            try:
                key = face_ids[i]
                str_img = self.conn.hget(key, 'str_img')
                emotion = self.conn.hget(key, 'emotion')
                img = np.fromstring(str_img, dtype = np.uint8)
                file_path = record_dir + '{0}.jpg'.format(key.decode('utf8'))
                cv2.imwrite(file_path, img.reshape(resize_shape))
                if i % 1000 == 0:
                    print('Dumping {0} records so far'.format(i))
            except Exception:
                print('Exception while dumping record {0}'.format(str(key)))
                continue
        print('Finish totally {0} records'.format(len(face_ids)))
            


if __name__ == '__main__':
    """# test file recorder
    record_dir = './records_csv/'
    file_recorder = FileRecorder(record_dir)
    for i in range(52000):
        file_recorder.write_record(i, i*10)

    # test redis recorder
    redis_recorder = RedisRecorder()
    
    for i in range(10):
        redis_recorder.write_record(str(i), str(i*20))
    """

    key_pattern = 'face*'
    record_dir = './detected_records/'
    redis_recorder.restore_record(key_pattern, record_dir)
