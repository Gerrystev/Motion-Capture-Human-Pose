import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
from tkinter import *
# from tkinter import ttk

# root = Tk()
# frame = ttk.Frame(root, padding=20)

# video_feed = PIL.ImageTk.PhotoImage(PIL.Image.open('D:\College\Others\counterside.jpg').resize((640,360), PIL.Image.ANTIALIAS));
# output_feed = PIL.ImageTk.PhotoImage(PIL.Image.open('D:\College\Others\counterside.jpg').resize((640,360), PIL.Image.ANTIALIAS));

# # For FPS calculation
# video_fps = 0
# output_fps = 0

# # Labels
# title_display = Label(text="Human Pose Estimation", font=("Tahoma", 24))
# video_label_display = Label(text="Video Feed", font=("Tahoma", 20))
# video_display = Label(image=video_feed)
# output_label_display = Label(text="3D Estimation", font=("Tahoma", 20))
# output_display = Label(image=output_feed)
# video_fps_display = Label(text=str(video_fps) + " FPS", font=("Tahoma", 12))
# output_fps_display = Label(text=str(output_fps) + " FPS", font=("Tahoma", 12))

# # Check button
# save_fbx = False
# check_button_display = ttk.Checkbutton(root, text='Save to .fbx', variable=save_fbx,
#  	    onvalue=True, offvalue=False)

# # Start record button
# button_display = ttk.Button(root, text="Start Record", width=200)

# # Display everything to grid
# title_display.grid(row=0, column=0, columnspan=2, pady=10, stick="nsew")
# video_label_display.grid(row=1, column=0, padx=10, sticky="nsew")
# output_label_display.grid(row=1, column=1, padx=10, sticky="nsew")
# video_display.grid(row=2, column=0, padx=10, sticky="nsew")
# output_display.grid(row=2, column=1, padx=10, sticky="nsew")
# video_fps_display.grid(row=3, column=0, pady=20, sticky="nsew")
# output_fps_display.grid(row=3, column=1, pady=20, sticky="nsew")
# check_button_display.grid(row=4, column=0, columnspan=2, pady=10)
# button_display.grid(row=5, column=0, columnspan=2, pady=10)

# # Grid for full screen
# Grid.rowconfigure(root, 0, weight=1)
# Grid.rowconfigure(root, 1, weight=1)
# Grid.rowconfigure(root, 2, weight=1)
# Grid.rowconfigure(root, 3, weight=1)
# Grid.rowconfigure(root, 4, weight=1)
# Grid.columnconfigure(root, 0, weight=1)
# Grid.columnconfigure(root, 1, weight=1)

# root.mainloop()

width, height = 800, 600
# rtsp://<username>:<password>@<ip_address>:<port>/Streaming/Channels/<id>
# cap = cv2.VideoCapture("rtsp://192.168.1.4:8080/h264_ulaw.sdp")
cap = cv2.VideoCapture(0)   
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
video_feed = Label(root)
video_feed.grid()

def show_video():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = PIL.ImageTk.PhotoImage(image=img)
    video_feed.imgtk = imgtk
    video_feed.configure(image=imgtk)
    video_feed.after(10, show_video)

show_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
root.destroy()