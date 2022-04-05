import torch.multiprocessing as multiprocessing
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.inference import get_final_preds
from utils.transforms import *

# import yolo
from yolov4.tool.utils import *
from yolov4.tool.torch_utils import *
from yolov4.tool.darknet2pytorch import Darknet
import models

import torch
import cv2
import numpy as np

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

        # model properties
        self.simple_model = None
        self.yolo_model = None
        self.videopose_model = None
        
        self.v_queue = v_queue
        self.o_queue = o_queue

    def _box2cs(self, box, image_width, image_height):
        x = box[0] * image_width
        y = box[1] * image_height
        x2 = box[2] * image_width
        y2 = box[3] * image_height

        w = x2 - x
        h = y2 - y

        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        pixel_std = 200.0

        max_wh = max([w / pixel_std, h / pixel_std])
        scale = np.array([max_wh, max_wh], dtype='float32')

        scale = scale * 1.25

        return center, scale

    def detect_bbox(self, img):
        """hyper parameters"""
        use_cuda = True

        if use_cuda:
            self.yolo_model.cuda()

        sized = cv2.resize(img, (self.yolo_model.width, self.yolo_model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for i in range(2):
            start = time.time()
            boxes = do_detect(self.yolo_model, sized, 0.4, 0.6, use_cuda)
            finish = time.time()
            if i == 1:
                print('Image: Predicted in %f seconds.' % ((finish - start)))

        return boxes[0]

    def draw_skeleton(self, preds, img):
        pred = preds[0, :, 0:2] + 1.0
        pred = np.round(pred).astype(int)

        head = np.where(config.HUMANEVA_KEYPOINTS == 'head')[0][0]
        pelv = np.where(config.HUMANEVA_KEYPOINTS == 'pelvis')[0][0]
        thor = np.where(config.HUMANEVA_KEYPOINTS == 'thorax')[0][0]
        lsho = np.where(config.HUMANEVA_KEYPOINTS == 'lsho')[0][0]
        lelb = np.where(config.HUMANEVA_KEYPOINTS == 'lelb')[0][0]
        lwri = np.where(config.HUMANEVA_KEYPOINTS == 'lwri')[0][0]
        lhip = np.where(config.HUMANEVA_KEYPOINTS == 'lhip')[0][0]
        lkne = np.where(config.HUMANEVA_KEYPOINTS == 'lkne')[0][0]
        lank = np.where(config.HUMANEVA_KEYPOINTS == 'lank')[0][0]

        rsho = np.where(config.HUMANEVA_KEYPOINTS == 'rsho')[0][0]
        relb = np.where(config.HUMANEVA_KEYPOINTS == 'relb')[0][0]
        rwri = np.where(config.HUMANEVA_KEYPOINTS == 'rwri')[0][0]
        rkne = np.where(config.HUMANEVA_KEYPOINTS == 'rkne')[0][0]
        rank = np.where(config.HUMANEVA_KEYPOINTS == 'rank')[0][0]
        rhip = np.where(config.HUMANEVA_KEYPOINTS == 'rhip')[0][0]

        # get keypoint (15 keypoints)
        pelvis_point = tuple(pred[pelv])
        thorax_point = tuple(pred[thor])
        left_shoulder_point = tuple(pred[lsho])
        left_elbow_point = tuple(pred[lelb])
        left_wrist_point = tuple(pred[lwri])
        right_shoulder_point = tuple(pred[rsho])
        right_elbow_point = tuple(pred[relb])
        right_wrist_point = tuple(pred[rwri])
        left_hip_point = tuple(pred[lhip])
        left_knee_point = tuple(pred[lkne])
        left_ankle_point = tuple(pred[lank])
        right_hip_point = tuple(pred[rhip])
        right_knee_point = tuple(pred[rkne])
        right_ankle_point = tuple(pred[rank])
        head_point = tuple(pred[head])

        # draw line to make a skeleton
        # color (argument 4 is BGR)
        # thickness in px
        thickness = 5

        img_skel = cv2.line(img, pelvis_point, thorax_point, (203, 192, 255), thickness)
        img_skel = cv2.line(img_skel, thorax_point, left_shoulder_point, (0, 165, 255), thickness)
        img_skel = cv2.line(img_skel, left_shoulder_point, left_elbow_point, (128, 0, 128), thickness)
        img_skel = cv2.line(img_skel, left_elbow_point, left_wrist_point, (0, 75, 150), thickness)
        img_skel = cv2.line(img_skel, thorax_point, right_shoulder_point, (0, 255, 255), thickness)
        img_skel = cv2.line(img_skel, right_shoulder_point, right_elbow_point, (0, 255, 0), thickness)
        img_skel = cv2.line(img_skel, right_elbow_point, right_wrist_point, (0, 0, 255), thickness)
        img_skel = cv2.line(img_skel, pelvis_point, left_hip_point, (33, 0, 133), thickness)
        img_skel = cv2.line(img_skel, left_hip_point, left_knee_point, (0, 76, 255), thickness)
        img_skel = cv2.line(img_skel, left_knee_point, left_ankle_point, (0, 255, 0), thickness)
        img_skel = cv2.line(img_skel, pelvis_point, right_hip_point, (248, 0, 252), thickness)
        img_skel = cv2.line(img_skel, right_hip_point, right_knee_point, (0, 196, 92), thickness)
        img_skel = cv2.line(img_skel, right_knee_point, right_ankle_point, (0, 238, 255), thickness)
        img_skel = cv2.line(img_skel, head_point, thorax_point, (255, 0, 0), thickness)

        cv2.imwrite('predictions.jpg', img_skel)

        
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
        # Human estimation
        img = self.current_frame

        # object detection
        bbox = self.detect_bbox(img)

        # 2d estimation
        cen, s = self._box2cs(bbox[0], img.shape[1], img.shape[0])
        r = 0

        trans = get_affine_transform(cen, s, r, [256, 256])
        input = cv2.warpAffine(
            img,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input = transform(input).unsqueeze(0)

        # switch to evaluate mode
        self.simple_model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = self.simple_model(input)
            # compute coordinate
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), np.asarray([cen]), np.asarray([s]))

            # plot
            image = img.copy()
            image = self.draw_skeleton(preds, image)

        # put processed image to processing queue
        self.o_queue.put(image)

    def load_yolo_model(self):
        cfgfile = './yolov4_cfg/yolov4-tiny.cfg'
        weightfile = './models/yolov4/yolov4-tiny.weights'

        self.yolo_model = Darknet(cfgfile)

        self.yolo_model.print_network()
        self.yolo_model.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))

    def load_simple_model(self):
        # load efficientnet simple baseline weights
        cfgfile = './simple_baseline_cfg/efficientnet/256x256_d256x3_adam_lr1e-3.yaml'
        weightfile = './models/efficientnet_simple_baseline/efficientnet_simple_baseline_15.pth.tar'
        update_config(cfgfile)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        self.simple_model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
            config, is_train=False
        )

        gpus = [int(i) for i in config.GPUS.split(',')]
        self.simple_model = torch.nn.DataParallel(self.simple_model, device_ids=gpus).cuda()

        print('=> loading model from {}'.format(weightfile))
        self.simple_model.load_state_dict(torch.load(weightfile))

    def video_loop(self):
        # if waitingFrame is not empty render current object
        while True:
            if not self.v_queue.empty():
                self.current_frame = self.v_queue.get()
                self.current_frame = self.current_frame[:, :, :3]
                    
                self.process_frame()

                self.calculate_fps()
        
    def destroy_window(self):
        self.process.terminate()