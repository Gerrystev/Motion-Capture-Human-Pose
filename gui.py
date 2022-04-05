import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import torch.multiprocessing as multiprocessing
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog

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
   
def show_thumbnail(root, first_image, image_label):
    width, height = 640, 360 
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGBA)
    first_image = cv2.flip(first_image, 1)
    first_image = cv2.resize(first_image, (width, height))
    img = PIL.Image.fromarray(first_image)
    imgtk = PIL.ImageTk.PhotoImage(image=img)
    image_label.configure(image=imgtk)
    image_label._image_cache = imgtk  # avoid garbage collection
    
    root.update()
   
def browse_files(root, video_capture, output_feed, v_queue, o_queue, video_display, 
                 output_display, button_display):
    filename = filedialog.askopenfilename(filetypes = ( ("video files","*.mp4, *.avi"), ) )
    
    # initialize videocapture with selected file
    video_capture.set_videocapture(filename)
    
    show_thumbnail(root, video_capture.first_frame, video_display)
    show_thumbnail(root, video_capture.first_frame, output_display)
    button_display['state'] = "normal"
    

def show_dialog(root, video_capture, output_feed, v_queue, o_queue, video_display, 
                output_display, button_display):
    ip_address = simpledialog.askstring("Input", "Input the ip address of camera",
                                parent=root)
    
    # initialize videocapture with ip address
    video_capture.set_videocapture(ip_address, True)
    
    show_thumbnail(root, video_capture.first_frame, video_display)
    show_thumbnail(root, video_capture.first_frame, output_display)
    button_display['state'] = "normal"
    
def start_capture(root, video_display, output_display, video_capture, 
                  output_feed, video_fps_display, output_fps_display, v_queue, o_queue):
    
    video_capture.start()
    output_feed.start()
    
    update_all(root, video_display, output_display, 
               video_capture.fps, output_feed.fps, 
               video_fps_display, output_fps_display, v_queue, o_queue)

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
    
    # check button
    save_fbx = False
    check_button_display = ttk.Checkbutton(root, text='Save to .fbx', variable=save_fbx,
     	    onvalue=True, offvalue=False)
    
    # video type button
    # initalize queue for multiprocessing
    v_queue = multiprocessing.Queue()
    o_queue = multiprocessing.Queue()
    
    video_capture = VideoCapture(v_queue)
    output_feed = OutputFeed(v_queue, o_queue)
    
    livestream_button_display = ttk.Button(root, text="Livestream Video", width=200,
                                command= lambda: show_dialog(root, video_capture, 
                                                             output_feed, v_queue, o_queue,
                                                             video_display, output_display,
                                                             button_display))
    video_button_display = ttk.Button(root, text="Choose video file", width=200,
                                       command= lambda: browse_files(root, video_capture, 
                                                             output_feed, v_queue, o_queue,
                                                             video_display, output_display, 
                                                             button_display))
    
    # start record button
    button_display = ttk.Button(root, text="Start Record", width=200, state=tk.DISABLED,
                                command= lambda: start_capture(
                                       root, video_display, output_display, 
                                       video_capture, output_feed, video_fps_display, 
                                       output_fps_display, v_queue, o_queue))
    
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
    
    root.mainloop()
    video_capture.destroy_window()
    output_feed.destroy_window()
    cv2.destroyAllWindows()

