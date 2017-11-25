# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-11-20 15:28:44
# @Last Modified by:   lc
# @Last Modified time: 2017-11-21 09:25:04

import os
import time
from datetime import datetime
import shutil

import dlib
import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk


# global variables
FACE_DECTOR = dlib.get_frontal_face_detector()
VIDEO_CAPTURE = cv2.VideoCapture(0)
VIDEO_WRITER = None
FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
DATA_DIR = os.getcwd().replace('\\', '/')
REF_IMAGE_DIR = '{0}/images/'.format(DATA_DIR)
VIDEO_DIR = '{0}/videos/'.format(DATA_DIR)
PERSONAL_VIDEO_DIR = None
EMOTIONS = ('neutral', 'angry', 'happy', 'disgust', 'sad', 'fear', 'surprise')
IDX = -1
# FRAME_SIZE = (640, 480)
FRAME_SIZE = (1280, 720)
RECORD_START_TIME = None


#Set up GUI
WINDOW = tk.Tk()  #Makes main WINDOW
WINDOW.wm_title("Emotion Data Collection")
WINDOW.config(background="#FFFFFF")

introduction_message = \
"""
您好, 欢迎参加华南理工大学移动软件开发环境团队实验室的表情采集实验。
在本次采集中，您需要根据提示依次做出七种表情，依次是

1. neutral(中性的)
2. angry(愤怒的)
3. happy(高兴的)
4. disgust(厌恶的)
5. sad(悲伤的)
6. fear(害怕的)
7. surprise(惊讶的)

采集每种表情时都会给出与这个表情相关的图片用于参考

我们承诺采集到图片仅作科研用途，不会用于商业，也不会发布到互联网上。

感谢您的配合！


                                                移动软件开发环境团队实验室
                                                {0}
""".format(datetime.now().strftime("%Y/%m/%d"))

# introduction INTRO_TEXT
INTRO_TEXT = Text(WINDOW)
INTRO_TEXT.configure(font=("microsoft yahei", 15, "bold"))
INTRO_TEXT.insert(INSERT, introduction_message)
INTRO_TEXT.grid(row=0, column=0)

# frame showing
imageFrame = tk.Frame(WINDOW)#, width=FRAME_SIZE[0]/3, height=FRAME_SIZE[1]/3)
imageFrame.grid(row=0, column=0, sticky=tk.NE)

# place video stream on the left and referred image
VIDEO_STREAM = tk.Label(imageFrame)
VIDEO_STREAM.grid(row=1, column=0, sticky=tk.NE)
REF_IMAGE = tk.Label(imageFrame)
REF_IMAGE.grid(row=0, column=0, sticky=tk.NE)

TEXT_HINT = Text(WINDOW)
TEXT_HINT.configure(font=("microsoft yahei", 13, "bold"), fg='blue')
TEXT_HINT.insert(INSERT, '未录制')
TEXT_HINT.grid(row=0, column=1)


def start_collect():
    global VIDEO_DIR, PERSONAL_VIDEO_DIR, INTRO_TEXT
    print('start to collect')
    INTRO_TEXT.delete('1.0', END)
    # create dir for the video
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    id = len(os.listdir(VIDEO_DIR))
    PERSONAL_VIDEO_DIR = VIDEO_DIR + '{0}/'.format(id)
    assert not os.path.exists(PERSONAL_VIDEO_DIR), 'ERROR, personal dir already exists'
    os.makedirs(PERSONAL_VIDEO_DIR)
    next_emotion()
    show_frame_stream()


def restart_collect():
    global VIDEO_WRITER
    if VIDEO_WRITER:
        VIDEO_WRITER.release()
        shutil.rmtree(PERSONAL_VIDEO_DIR)
    VIDEO_CAPTURE.release()
    VIDEO_WRITER.release()
    messagebox.showinfo(title='重新开始', message='请重新启动测试程序')
    exit(0)


def next_emotion():
    global IDX, VIDEO_WRITER, REF_IMAGE, REF_IMAGE_DIR, EMOTIONS, TEXT_HINT, RECORD_START_TIME
    
    RECORD_START_TIME = None
    if IDX >= 0:
        video_path = '{0}{1}.avi'.format(PERSONAL_VIDEO_DIR, EMOTIONS[IDX])
        print('{0} is saved'.format(video_path))
        if VIDEO_WRITER:
            VIDEO_WRITER.release()
        VIDEO_WRITER = None
    
    IDX += 1
    if IDX == len(EMOTIONS):
        print('finish collecting')
        VIDEO_CAPTURE.release()
        #VIDEO_WRITER.release()
        messagebox.showinfo(title='完成', message='采集完成，感谢您的参与')
        # time.sleep(5)
        exit(0)

    if TEXT_HINT:
        TEXT_HINT.delete('1.0', END)
        TEXT_HINT.insert(INSERT, '待采集第{0}种表情:{1}\n'.format(IDX+1, EMOTIONS[IDX]))
        TEXT_HINT.insert(INSERT, '未录制')

    ref_frame = REF_IMAGE_DIR + '{}.jpg'.format(EMOTIONS[IDX])
    ref_cv2image = cv2.cvtColor(cv2.imread(ref_frame), cv2.COLOR_BGR2RGBA)
    ref_img = Image.fromarray(ref_cv2image)
    ref_imgtk = ImageTk.PhotoImage(image=ref_img)
    REF_IMAGE.imgtk = ref_imgtk  
    REF_IMAGE.configure(image=ref_imgtk)


def start_record():
    global VIDEO_WRITER, EMOTIONS, IDX, PERSONAL_VIDEO_DIR, RECORD_START_TIME
    if TEXT_HINT:
        TEXT_HINT.delete('1.0', END)
        TEXT_HINT.insert(INSERT, '正在采集第{0}种表情:{1}\n'.format(IDX+1, EMOTIONS[IDX]))
        TEXT_HINT.insert(INSERT, '录制中..........')
    RECORD_START_TIME = int(datetime.now().strftime("%M%S"))
    video_path = '{0}{1}.avi'.format(PERSONAL_VIDEO_DIR, EMOTIONS[IDX])
    VIDEO_WRITER = cv2.VideoWriter(video_path, FOURCC, 20.0, FRAME_SIZE)


def restart_record():
    global IDX, VIDEO_WRITER, PERSONAL_VIDEO_DIR, RECORD_START_TIME
    RECORD_START_TIME = None
    if VIDEO_WRITER:
        VIDEO_WRITER.release()
        video_path = '{0}{1}.avi'.format(PERSONAL_VIDEO_DIR, EMOTIONS[IDX])
        os.remove(video_path)
    TEXT_HINT.delete('1.0', END)
    TEXT_HINT.insert(INSERT, '已取消录制，请点击 Start Recording 重新录制当前表情{0}'.format(EMOTIONS[IDX]))


def show_frame_stream():
    global VIDEO_CAPTURE, VIDEO_WRITER, EMOTIONS, REF_IMAGE_DIR, IDX
    ret, frame = VIDEO_CAPTURE.read()
    frame = cv2.flip(frame, 1)
    if VIDEO_WRITER:
        VIDEO_WRITER.write(frame)
    # rectangle the face 
    frame = cv2.resize(frame, (int(FRAME_SIZE[0]/4), int(FRAME_SIZE[1]/4)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DECTOR(gray, 0)
    if len(faces) > 0:
        face = faces[0]
        left_top, right_bottom = (face.left(), face.top()), (face.right(), face.bottom())
        cv2.rectangle(frame, left_top, right_bottom, (0, 255, 255), 2)
    # add timer when start recording
    if RECORD_START_TIME:
        curr = int(datetime.now().strftime("%M%S"))
        curr_min, curr_sec = curr//100, curr%100
        start_min, start_sec = RECORD_START_TIME//100, RECORD_START_TIME%100
        seconds = (curr_min - start_min) * 60 + curr_sec - start_sec
        time_count = '{0:02d}:{1:02d}'.format(seconds//60, seconds%60)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, time_count, (5, 25), font, 1, (255, 0, 255), 2)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    VIDEO_STREAM.imgtk = imgtk #Shows frame for display 1
    VIDEO_STREAM.configure(image=imgtk)
    WINDOW.after(1, show_frame_stream)


start = Button(WINDOW, text="Start Collecting", width=15, command=start_collect)
start.grid(row = 4, column=0, sticky=tk.NE)

nextButton = Button(WINDOW, text="Next Emotion", width=15,command=next_emotion)
nextButton.grid(row = 5, column=0, sticky=tk.NE)

nextButton = Button(WINDOW, text="Start Recording", width=15,command=start_record)
nextButton.grid(row = 6, column=0, sticky=tk.NE)

restartRecordingButton = Button(WINDOW, text="Restart Recording", width=15, command=restart_record)
restartRecordingButton.grid(row = 7, column=0, sticky=tk.NE)

# restartCollectingButton = Button(WINDOW, text="Restart Collecting", width=15, command=restart_collect)
# restartCollectingButton.grid(row = 8, column=0)

WINDOW.mainloop()  #Starts GUI