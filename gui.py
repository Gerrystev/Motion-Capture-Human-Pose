import tkinter as tk
import cv2
import time
import numpy as np
import PIL.Image
import PIL.ImageTk
import torch.multiprocessing as multiprocessing
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog

from VideoCapture import VideoCapture
from OutputFeed import OutputFeed

import _init_paths
from core.config import config

def update_video(fps_queue, image_label, fps_label, queue):
   width, height = 640, 360
   if not queue.empty(): 
       # fps = fps_queue.get()
       fps = 0
       frame1 = np.copy(queue.get())
       frame = cv2.resize(frame1, (width, height))
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       img = PIL.Image.fromarray(frame)
       imgtk = PIL.ImageTk.PhotoImage(image=img)
       image_label.configure(image=imgtk)
       image_label._image_cache = imgtk  # avoid garbage collection
   
       # put fps into label
       fps_label.configure(text=str(fps) + " FPS")

       # if config.IS_LIVESTREAM:
       #     queue.put(np.copy(frame1))

       root.update()

def update_output(next_fps, image_label, fps_label, next_frame):
   try:
       width, height = 640, 360
       fps = next_fps
       frame = np.copy(next_frame)
       frame = cv2.resize(frame, (width, height))
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       img = PIL.Image.fromarray(frame)
       imgtk = PIL.ImageTk.PhotoImage(image=img)
       image_label.configure(image=imgtk)
       image_label._image_cache = imgtk  # avoid garbage collection

       # put fps into label
       fps_label.configure(text=str(fps) + " FPS")
       root.update()
   except:
       print('error updating feed')

def update_all(root, video_capture, output_feed, v_label, o_label, v_fps, v_fps_label,
               o_fps_label, v_queue, timeout_time=-1):
   if not v_queue.empty():
       update_video(v_fps, v_label, v_fps_label, v_queue)
       output_feed.process_frame()
       update_output(output_feed.fps, o_label, o_fps_label, output_feed.o_frame)
       timeout_time = time.time()
   else:
       if time.time() - timeout_time >= 5 and timeout_time != -1:
           # if timeout 3s disconnect and terminate multiprocess
           video_capture.destroy_window()
           if config.SAVE_TXT:
               output_feed.write_coord_txt()
           output_feed.destroy_window()
           return

   root.after(0, func=lambda: update_all(root, video_capture, output_feed, v_label, o_label, v_fps, v_fps_label,
           o_fps_label, v_queue, timeout_time))
   
def show_thumbnail(root, first_image, image_label):
    width, height = 640, 360 
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    first_image = cv2.resize(first_image, (width, height))
    img = PIL.Image.fromarray(first_image)
    imgtk = PIL.ImageTk.PhotoImage(image=img)
    image_label.configure(image=imgtk)
    image_label._image_cache = imgtk  # avoid garbage collection
    
    root.update()
   
def browse_files(root, video_capture, output_feed, video_display,
                 output_display, button_display):
    filename = filedialog.askopenfilename(filetypes = ( ("video files","*.*"), ) )
    
    # initialize videocapture with selected file
    video_capture.set_videocapture(filename)
    
    show_thumbnail(root, video_capture.first_frame, video_display)
    show_thumbnail(root, output_feed.first_frame, output_display)
    button_display['state'] = "normal"

def show_dialog(root, video_capture, output_feed, video_display,
                output_display, button_display):
    ip_address = simpledialog.askstring("Input", "Input the ip address of camera",
                                parent=root)
    
    # initialize videocapture with ip address
    video_capture.set_videocapture(ip_address, True)
    
    show_thumbnail(root, video_capture.first_frame, video_display)
    show_thumbnail(root, output_feed.first_frame, output_display)
    button_display['state'] = "normal"
    
def start_capture(root, video_display, output_display, video_capture, 
                  output_feed, video_fps_display, output_fps_display, v_queue):

    video_capture.start()

    update_all(root, video_capture, output_feed, video_display, output_display,
               video_capture.fps,
               video_fps_display, output_fps_display, v_queue)

if __name__ == '__main__':
    root = Tk()
    frame = ttk.Frame(root, padding=20)
    
    # labels
    video_display = Label(root)
    output_display = Label(root)
    title_display = Label(text="Human Pose Estimation", font=("Tahoma", 24))
    video_label_display = Label(text="Video Feed", font=("Tahoma", 20))
    output_label_display = Label(text="3D Estimation", font=("Tahoma", 20))
    video_fps_display = Label(text="0 FPS", font=("Tahoma", 12))
    output_fps_display = Label(text="0 FPS", font=("Tahoma", 12))    
    video_type_label_display = Label(text="Video Type", font=("Tahoma", 20))

    def text_checked():
        config.SAVE_TXT = is_save_txt.get()

    # check button
    is_save_txt = BooleanVar()
    check_button_display = ttk.Checkbutton(root, text='Save coordinate to .json', variable=is_save_txt,
     	    onvalue=True, offvalue=False, command=text_checked)
    
    # video type button
    # initalize queue for multiprocessing
    v_queue = multiprocessing.Queue()
    o_frame = np.zeros((360, 640, 3), dtype=np.int8)        # init as black screen
    
    video_capture = VideoCapture(v_queue)
    output_feed = OutputFeed(v_queue, o_frame)
    
    livestream_button_display = ttk.Button(root, text="Livestream Video", width=200,
                                command= lambda: show_dialog(root, video_capture, output_feed,
                                                             video_display, output_display,
                                                             button_display))
    video_button_display = ttk.Button(root, text="Choose video file", width=200,
                                       command= lambda: browse_files(root, video_capture, output_feed,
                                                             video_display, output_display, 
                                                             button_display))
    
    # start record button
    button_display = ttk.Button(root, text="Start Record", width=200, state=tk.DISABLED,
                                command= lambda: start_capture(
                                       root, video_display, output_display, 
                                       video_capture, output_feed, video_fps_display, 
                                       output_fps_display, v_queue))
    
    # display everything to grid
    title_display.grid(row=0, column=0, columnspan=2, pady=10, stick="nsew")
    video_label_display.grid(row=1, column=0, padx=10, sticky="nsew")
    output_label_display.grid(row=1, column=1, padx=10, sticky="nsew")
    video_display.grid(row=2, column=0, padx=10, sticky="nsew")
    output_display.grid(row=2, column=1, padx=10, sticky="nsew")
    video_fps_display.grid(row=3, column=0, pady=20, sticky="nsew")
    output_fps_display.grid(row=3, column=1, pady=20, sticky="nsew")
    video_type_label_display.grid(row=4, column=0, columnspan=2, pady=5)
    livestream_button_display.grid(row=5, column=0, columnspan=2, pady=5)
    video_button_display.grid(row=6, column=0, columnspan=2, pady=5)
    check_button_display.grid(row=7, column=0, columnspan=2, pady=10)
    button_display.grid(row=8, column=0, columnspan=2, pady=5)
    
    # # Grid for full screen
    # Grid.rowconfigure(root, 0, weight=1)
    # Grid.rowconfigure(root, 1, weight=1)
    # Grid.rowconfigure(root, 2, weight=1)
    # Grid.rowconfigure(root, 3, weight=1)
    # Grid.rowconfigure(root, 4, weight=1)
    # Grid.columnconfigure(root, 0, weight=1)
    # Grid.columnconfigure(root, 1, weight=1)

    output_feed.load_yolo_model()
    output_feed.load_simple_model()
    output_feed.load_videopose()
    
    root.mainloop()
    video_capture.destroy_window()
    output_feed.destroy_window()
    cv2.destroyAllWindows()

