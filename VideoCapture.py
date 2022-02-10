import multiprocessing
import cv2
import time

class VideoCapture:
    def __init__(self, queue):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.frame = None
        
        self.fps = multiprocessing.Queue()
        
        # used to record the time when processed last frame
        self.prev_frame_time = 0
        # used to record the time at which processed current frame
        self.new_frame_time = 0
        
        # rtsp://<username>:<password>@<ip_address>:<port>/Streaming/Channels/<id>
        # initialize widget properties
        self.cap = None
        self.currentFrame = None
        
        self.process = multiprocessing.Process(target=self.video_capture)
        
        # queue for processing image
        self.queue = queue
        
        self.cap_queue = multiprocessing.Queue()
        
    def start(self):
        self.process.start()
        
    def join(self):
        self.process.join()
        
    def calculate_fps(self):
        # time when finished processing current frame
        self.new_frame_time = time.time()
        
        # calculating fps
        self.fps.put(int(1/(self.new_frame_time - self.prev_frame_time)))
        self.prev_frame_time = self.new_frame_time
        
        
    def video_capture(self):
        # video capture loop
        self.cap = cv2.VideoCapture("rtsp://192.168.1.2:8080/h264_ulaw.sdp")
        # self.cap = cv2.VideoCapture(0)
        self.cap_queue.put(self.cap)
        
        while True:
            _, frame = self.cap.read()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                self.queue.put(frame)
                self.calculate_fps()
        
    def destroy_window(self):
        self.process.terminate()
