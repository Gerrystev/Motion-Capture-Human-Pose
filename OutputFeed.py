from PIL import Image
from PIL import ImageTk
import threading
import tkinter as tk
import datetime
import imutils
import cv2
import os
import PIL.Image
import PIL.ImageTk
import time

class OutputFeed:
    def __init__(self, root, fpsLabel):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.root = root
        self.outputPath = "video/"
        self.frame = None
        self.stopEvent = None
        self.fps = 0
        self.fpsLabel = fpsLabel
        
        # used to record the time when processed last frame
        self.prev_frame_time = 0
        # used to record the time at which processed current frame
        self.new_frame_time = 0
        
        # initialize the root window and image panel
        self.label = tk.Label(root)
        
        # rtsp://<username>:<password>@<ip_address>:<port>/Streaming/Channels/<id>
        # cap = cv2.VideoCapture("rtsp://192.168.1.4:8080/h264_ulaw.sdp")
        # initialize widget properties
        self.width, self.height = 640, 360
        # self.cap = cv2.VideoCapture(0)
        self.currentFrame = None
        
        # waitingFrame for queue of image to be processed from VideoFeed
        self.waitingFrame = []
        
        # initialize thread
        self.stopped = True
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        
    def start(self):
        self.stopped = False
        self.thread.start()
        
    def calculateFps(self):
        # time when finished processing current frame
        self.new_frame_time = time.time()
        
        # calculating fps
        self.fps = int(1/(self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time
        
        # put fps into label
        self.fpsLabel.configure(text=str(self.fps) + " FPS")
        self.fpsLabel.update()
        
    def processFrame(self):
        time.sleep(.1)
        
    def videoLoop(self):
        # if waitingFrame is not empty render current object
        if len(self.waitingFrame) > 0:
            self.currentFrame = self.waitingFrame.pop(0)
                
            self.processFrame()
            self.calculateFps()
            img = PIL.Image.fromarray(self.currentFrame)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.label.after(10, self.videoLoop)
        
    def destroy_window(self):
        # self.stopEvent.set()
        print()