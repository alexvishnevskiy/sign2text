import cv2
import threading
from .detect import get_model, detect

class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0) # camera index
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    model = get_model('VideoStream/best_model.pytorch')
    while True:
        frame = camera.get_frame()
        print(detect(frame, model))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

