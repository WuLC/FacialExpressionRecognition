# -*- coding: utf-8 -*-
# Created on Wed Nov 29 2017 22:45:55
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Sort Files in a directory')
parser.add_argument('--src_dir', help = 'the directory of the source files')
parser.add_argument('--des_dir', help = 'the directory of the destinate files')
parser.add_argument('--start', help = 'the number used to start the naming of dirs')
args = parser.parse_args()
src_dir = args.src_dir
des_dir = args.des_dir
count = int(args.start)

if not os.path.exists(src_dir):
    print('path {0} not exists'.format(src_dir))

if not os.path.exists(des_dir):
    os.makedirs(des_dir)

files = sorted(map(lambda x:int(x), [name for name in os.listdir(src_dir) if name.isdigit()]))
for i in range(len(files)):
    shutil.move(src_dir+str(files[i]), des_dir+str(count))
    count += 1
