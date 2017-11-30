# -*- coding: utf-8 -*-
# Created on Wed Nov 29 2017 22:45:55
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import argparse


parser = argparse.ArgumentParser(description='Sort Files in a directory')
parser.add_argument('--dir', help = 'the directory that needs to be sorted')
parser.add_argument('--start', help = 'the number used to start the naming of dirs')
args = parser.parse_args()
des_dir = args.dir
count = int(args.start)

if not os.path.exists(des_dir):
    print('path {0} not exists'.format(des_dir))

files = sorted(os.listdir(des_dir))
for i in range(len(files)):
    os.rename(des_dir+files[i], des_dir+str(count))
    count += 1
