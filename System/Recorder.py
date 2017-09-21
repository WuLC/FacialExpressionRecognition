# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-09-21 11:16:46
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-21 17:50:29

########################################################
# save the detected face and emotion on disk or database
########################################################

import os
import time
from datetime import datetime


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


if __name__ == '__main__':
    record_dir = './records_csv/'
    file_recorder = FileRecorder(record_dir)
    for i in range(52000):
        file_recorder.write_record(i, i*10)