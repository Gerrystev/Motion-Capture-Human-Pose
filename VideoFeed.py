from PIL import Image
from PIL import ImageTk

import tkinter as tk
import datetime
import imutils
import cv2
import os
import PIL.Image
import PIL.ImageTk
import time

class VideoFeed:
    def __init__(self, root, fpsLabel, outputFeed):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.root = root
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.fps = 0
        self.fpsLabel = fpsLabel
        
        self.outputFeed = outputFeed
        
        # used to record the time when processed last frame
        self.prev_frame_time = 0
        # used to record the time at which processed current frame
        self.new_frame_time = 0
        
        # initialize the root window and image panel
        self.label = tk.Label(root)
        
        # rtsp://<username>:<password>@<ip_address>:<port>/Streaming/Channels/<id>
        # initialize widget properties
        self.width, self.height = 640, 360
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("rtsp://192.168.1.2:8080/h264_ulaw.sdp")
        self.currentFrame = None
        # self.cap = cv2.VideoCapture("rtsp://192.168.1.4:8080/h264_ulaw.sdp")
        
        self.initialize = False
        
    def calculateFps(self):
        # time when finished processing current frame
        self.new_frame_time = time.time()
        
        # calculating fps
        self.fps = int(1/(self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time
        
        # put fps into label
        self.fpsLabel.configure(text=str(self.fps) + " FPS")
        self.fpsLabel.update()
        
    def videoLoop(self):
        # video capture loop
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height))
        
        self.calculateFps()
    
        self.currentFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.outputFeed.waitingFrame.append(self.currentFrame)
        if not self.initialize:
            self.initialize = True
            self.outputFeed.start()
        
        img = PIL.Image.fromarray(self.currentFrame)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)        
        self.label.after(10, self.videoLoop)
        
    def destroy_window(self):
        self.cap.release()