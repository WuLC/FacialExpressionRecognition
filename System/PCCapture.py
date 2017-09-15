import cv2
import numpy as np
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers="localhost:9092")
cv2.namedWindow('video')
capture = cv2.VideoCapture(0)
_, frame = capture.read()
#size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#vw = cv2.VideoWriter(vfile,cv2.VideoWriter_fourcc('M','J','P','G'), 30, size, 1)

count = 0
while frame is not None:
    #cv2.imshow('Video', frame)
    #print(np.shape(frame))
    
    #vw.write(frame)
    """
    key = cv2.waitKey(10)
    if key == ord('s'):     # 当按下"s"键时，将保存当前画面
        cv2.imwrite('screenshot.bmp', frame)
    elif key == ord('q'):   # 当按下"q"键时，将退出循环
        break
    """
    count += 1
    if count % 10 == 0:
        producer.send('video', cv2.imencode('.jpeg', frame)[1].tostring())
    _, frame = capture.read()
cv2.destroyWindow('Video')