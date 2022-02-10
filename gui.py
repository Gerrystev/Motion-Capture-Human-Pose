import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import multiprocessing
from tkinter import *
from tkinter import ttk

from VideoCapture import VideoCapture
from OutputFeed import OutputFeed

def update_video(fps_queue, image_label, fps_label, queue):
   width, height = 640, 360 
   if not queue.empty(): 
       fps = fps_queue.get()
       frame = queue.get()
       frame = cv2.flip(frame, 1)
       frame = cv2.resize(frame, (width, height))
       img = PIL.Image.fromarray(frame)
       imgtk = PIL.ImageTk.PhotoImage(image=img)
       image_label.configure(image=imgtk)
       image_label._image_cache = imgtk  # avoid garbage collection
   
       # put fps into label
       fps_label.configure(text=str(fps) + " FPS")
       root.update()

def update_all(root, v_label, o_label, v_fps, o_fps, v_fps_label, 
               o_fps_label, v_queue, o_queue):
   update_video(v_fps, v_label, v_fps_label, v_queue)
   update_video(o_fps, o_label, o_fps_label, o_queue)
   
   root.after(0, func=lambda: update_all(root, v_label, o_label, v_fps, o_fps, v_fps_label, 
               o_fps_label, v_queue, o_queue))

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
    
    # check button
    save_fbx = False
    check_button_display = ttk.Checkbutton(root, text='Save to .fbx', variable=save_fbx,
     	    onvalue=True, offvalue=False)
    
    # start record button
    button_display = ttk.Button(root, text="Start Record", width=200)
    
    # display everything to grid
    title_display.grid(row=0, column=0, columnspan=2, pady=10, stick="nsew")
    video_label_display.grid(row=1, column=0, padx=10, sticky="nsew")
    output_label_display.grid(row=1, column=1, padx=10, sticky="nsew")
    video_display.grid(row=2, column=0, padx=10, sticky="nsew")
    output_display.grid(row=2, column=1, padx=10, sticky="nsew")
    video_fps_display.grid(row=3, column=0, pady=20, sticky="nsew")
    output_fps_display.grid(row=3, column=1, pady=20, sticky="nsew")
    check_button_display.grid(row=4, column=0, columnspan=2, pady=10)
    button_display.grid(row=5, column=0, columnspan=2, pady=10)
    
    # # Grid for full screen
    # Grid.rowconfigure(root, 0, weight=1)
    # Grid.rowconfigure(root, 1, weight=1)
    # Grid.rowconfigure(root, 2, weight=1)
    # Grid.rowconfigure(root, 3, weight=1)
    # Grid.rowconfigure(root, 4, weight=1)
    # Grid.columnconfigure(root, 0, weight=1)
    # Grid.columnconfigure(root, 1, weight=1)

    # initalize multiprocessing
    v_queue = multiprocessing.Queue()
    o_queue = multiprocessing.Queue()
    
    video_capture = VideoCapture(v_queue)
    output_feed = OutputFeed(v_queue, o_queue)
    
    video_capture.start()
    output_feed.start()
    
    update_all(root, video_label_display, output_label_display, 
               video_capture.fps, output_feed.fps, 
               video_fps_display, output_fps_display, v_queue, o_queue)
    
    
    
    root.mainloop()
    video_capture.destroy_window()
    output_feed.destroy_window()
    cv2.destroyAllWindows()

