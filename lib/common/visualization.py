# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import multiprocessing as mp
import io

import multiprocessing as mp

import _init_paths
from core.config import config

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)
            
def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            
                
                
    
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

class Grid3D:
    def __init__(self, skeleton, azim, viewport, anim_queue, skel_queue):
        plt.ioff()
        self.fig = plt.figure(figsize=(viewport[0] / 100, viewport[1] / 100))

        self.ax_3d = []
        self.lines_3d = []
        self.trajectories = []
        self.radius = 1.7

        self.azim = azim

        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        self.ax.view_init(elev=15., azim=azim)
        self.ax.set_xlim3d([-self.radius / 2, self.radius / 2])
        self.ax.set_zlim3d([0, self.radius])
        self.ax.set_ylim3d([-self.radius / 2, self.radius / 2])
        try:
            self.ax.set_aspect('equal')
        except NotImplementedError:
            self.ax.set_aspect('auto')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.dist = 7.5
        self.ax_3d.append(self.ax)
        self.lines_3d.append([])

        self.initialized = False

        self.parents = skeleton.parents()
        self.skeleton = skeleton

        self.anim_queue = anim_queue
        self.skel_queue = skel_queue

        self.process = mp.Process(target=self.update_video)

    def start_process(self):
        self.process.start()

    def stop_process(self):
        self.process.terminate()

    def update_video(self):
        while True:
            poses = self.skel_queue.get()
            self.poses = list(poses.values())

            for index, (title, data) in enumerate(poses.items()):
                self.trajectories = [data[:, 0, [0, 1]]]

            for n, ax in enumerate(self.ax_3d):
                self.ax.set_xlim3d([-self.radius / 2 + self.trajectories[n][0, 0], self.radius / 2 + self.trajectories[n][0, 0]])
                self.ax.set_ylim3d([-self.radius / 2 + self.trajectories[n][0, 1], self.radius / 2 + self.trajectories[n][0, 1]])

            if not self.initialized:
                for j, j_parent in enumerate(self.parents):
                    if j_parent == -1:
                        continue

                    col = 'red' if j in self.skeleton.joints_right() else 'black'
                    for n, ax in enumerate(self.ax_3d):
                        pos = self.poses[n][0]
                        self.lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                   [pos[j, 1], pos[j_parent, 1]],
                                                   [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

                self.initialized = True
            else:
                for j, j_parent in enumerate(self.parents):
                    if j_parent == -1:
                        continue

                    for n, ax in enumerate(self.ax_3d):
                        pos = self.poses[n][0]
                        self.lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        self.lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        self.lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

            # return plot numpy array
            with io.BytesIO() as buff:
                self.ax.figure.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = self.ax.figure.canvas.get_width_height()

            self.anim_queue.put(data.reshape((int(h), int(w), -1)))

    def get_figure_numpy(self):
        # return plot numpy array
        with io.BytesIO() as buff:
            self.ax.figure.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = self.ax.figure.canvas.get_width_height()
        return data.reshape((int(h), int(w), -1))
