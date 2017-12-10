import cv2
import base64
import numpy as np
from kafka import KafkaProducer

SERVER = '127.0.0.1:9092'
# SERVER = '222.201.145.235:9092'
SERVER = '125.216.242.158:9092'
producer = KafkaProducer(bootstrap_servers= SERVER)
cv2.namedWindow('video')
capture = cv2.VideoCapture(0)
_, frame = capture.read()

count = 0
while frame is not None:
    """
    key = cv2.waitKey(10)
    if key == ord('s'):     # 当按下"s"键时，将保存当前画面
        cv2.imwrite('screenshot.bmp', frame)
    elif key == ord('q'):   # 当按下"q"键时，将退出循环
        break
    """
    count += 1
    if count % 30 == 0:
        print('send {0} frames, shape {1}'.format(count//10, frame.shape))
        img_str = cv2.imencode('.jpeg', frame)[1].tostring()
        encoded_image = base64.b64encode(img_str)
        producer.send('video', encoded_image)
    _, frame = capture.read()