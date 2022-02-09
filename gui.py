import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter import ttk
from multiprocessing import Process

from VideoFeed import VideoFeed
from OutputFeed import OutputFeed

root = Tk()
frame = ttk.Frame(root, padding=20)

# Labels
title_display = Label(text="Human Pose Estimation", font=("Tahoma", 24))
video_label_display = Label(text="Video Feed", font=("Tahoma", 20))
output_label_display = Label(text="3D Estimation", font=("Tahoma", 20))
video_fps_display = Label(text="0 FPS", font=("Tahoma", 12))
output_fps_display = Label(text="0 FPS", font=("Tahoma", 12))

# Capture video
output_feed = OutputFeed(root, output_fps_display)
video_feed = VideoFeed(root, video_fps_display, output_feed)

video_display = video_feed.label
output_display = output_feed.label

# Check button
save_fbx = False
check_button_display = ttk.Checkbutton(root, text='Save to .fbx', variable=save_fbx,
 	    onvalue=True, offvalue=False)

# Start record button
button_display = ttk.Button(root, text="Start Record", width=200)

# Display everything to grid
title_display.grid(row=0, column=0, columnspan=2, pady=10, stick="nsew")
video_label_display.grid(row=1, column=0, padx=10, sticky="nsew")
output_label_display.grid(row=1, column=1, padx=10, sticky="nsew")
video_display.grid(row=2, column=0, padx=10, sticky="nsew")
output_display.grid(row=2, column=1, padx=10, sticky="nsew")
video_fps_display.grid(row=3, column=0, pady=20, sticky="nsew")
output_fps_display.grid(row=3, column=1, pady=20, sticky="nsew")
check_button_display.grid(row=4, column=0, columnspan=2, pady=10)
button_display.grid(row=5, column=0, columnspan=2, pady=10)

video_feed.videoLoop()
output_feed.videoLoop()

# # Grid for full screen
# Grid.rowconfigure(root, 0, weight=1)
# Grid.rowconfigure(root, 1, weight=1)
# Grid.rowconfigure(root, 2, weight=1)
# Grid.rowconfigure(root, 3, weight=1)
# Grid.rowconfigure(root, 4, weight=1)
# Grid.columnconfigure(root, 0, weight=1)
# Grid.columnconfigure(root, 1, weight=1)

root.mainloop()
video_feed.destroy_window()
output_feed.destroy_window()
cv2.destroyAllWindows()

