import torch.multiprocessing as multiprocessing
import cv2
import time

class VideoCapture:
    def __init__(self, queue, video_link = 0):
        self.video_link = 0
        self.first_frame = None
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
        
    def set_videocapture(self, video_link, is_livestream=False):
        # get first frame of video
        if video_link == "0":
            # read webcam
            video_link = 0
        else:
            if is_livestream:
                video_link = "rtsp://" + video_link + "/h264_ulaw.sdp"
        self.video_link = video_link
        video_cap = cv2.VideoCapture(video_link)
        _, self.first_frame = video_cap.read()
        
    def video_capture(self):
        # video capture loop
        # self.cap = cv2.VideoCapture("rtsp://192.168.1.2:8080/h264_ulaw.sdp")
        self.cap = cv2.VideoCapture(self.video_link)
        
        while True:
            _, frame = self.cap.read()
            if frame is not None:
                self.queue.put(frame)
                self.calculate_fps()
        
    def destroy_window(self):
        self.process.terminate()
