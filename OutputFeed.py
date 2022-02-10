import multiprocessing
import time

class OutputFeed:
    def __init__(self, v_queue, o_queue):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.frame = None
        self.fps = multiprocessing.Queue()
        
        # used to record the time when processed last frame
        self.prev_frame_time = 0
        # used to record the time at which processed current frame
        self.new_frame_time = 0
        
        self.current_frame = None
        
        self.process = multiprocessing.Process(target=self.video_loop)
        
        self.v_queue = v_queue
        self.o_queue = o_queue        
        
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
        
    def process_frame(self):
        time.sleep(.1)
        self.o_queue.put(self.current_frame)
        
    def video_loop(self):
        # if waitingFrame is not empty render current object
        while True:
            if not self.v_queue.empty():
                self.current_frame = self.v_queue.get()
                    
                self.process_frame()
                self.calculate_fps()
        
    def destroy_window(self):
        self.process.terminate()