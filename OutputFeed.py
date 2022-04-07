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

from common.generators import UnchunkedGenerator
from common.model import TemporalModel
from common.humaneva_dataset import humaneva_skeleton
from common.custom_dataset import custom_camera_params
from common.camera import camera_to_world
from common.camera import normalize_screen_coordinates

from common.visualization import Grid3D


class OutputFeed:
    def __init__(self, v_queue, o_frame):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.frame = None
        self.fps = 0
        
        # used to record the time when processed last frame
        self.prev_frame_time = 0
        # used to record the time at which processed current frame
        self.new_frame_time = 0
        
        self.current_frame = None

        # model properties
        self.simple_model = None
        self.yolo_model = None
        self.videopose_model = None
        
        self.v_queue = v_queue
        self.o_frame = o_frame

        self.frame_processed = 0

        # ndarray of 2d predicted joints
        self.preds_2d = None

        # videopose3d properties
        self.metadata = None
        self.receptive_field = 1        # n frame used as receptive field

        # grid 3D properties
        azimuth = np.array(0., dtype='float32')
        viewport = (640, 360)
        self.grid_3d = Grid3D(humaneva_skeleton, azimuth, viewport)
        self.first_frame = self.grid_3d.get_figure_numpy()

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


        
    def start(self):
        self.process_frame()
        
    def join(self):
        self.process.join()
        
    def calculate_fps(self):
        # time when finished processing current frame
        self.new_frame_time = time.time()

        # calculating fps
        self.fps = int(1/(self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time

    def animate_3d(self, preds_3d):
        # animation properties
        azimuth = np.array(0., dtype='float32')
        viewport = (640, 360)
        cam = {
            'orientation': [0.4214752, -0.4961493, -0.5838273, 0.4851187],
            'translation': [4112.9121, 626.4929, 845.2988],
        }

        # cam = custom_camera_params
        prediction = camera_to_world(preds_3d, R=np.array(cam['orientation'], dtype='float32'),
                                     t=np.array(cam['translation'], dtype='float32')/1000)
        # rot = cam['orientation']
        # prediction = camera_to_world(prediction, R=rot, t=0)

        anim_output = {'Reconstruction': prediction}

        return self.grid_3d.update_video(anim_output)
        
    def process_frame(self):
        # Human estimation
        if not self.v_queue.empty():
            img = np.copy(self.v_queue.get())

            # object detection
            bbox = self.detect_bbox(img)

            # 2d estimation
            if len(bbox) > 0:
                # if person is detected
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
                self.videopose_model.eval()

                with torch.no_grad():
                    # compute output heatmap
                    output = self.simple_model(input)
                    # compute coordinate
                    preds, maxvals = get_final_preds(
                        config, output.clone().cpu().numpy(), np.asarray([cen]), np.asarray([s]))

                    # if self.preds_2d is None:
                    #     self.preds_2d = np.copy(preds)
                    # else:
                    #     self.preds_2d = np.concatenate((self.preds_2d, preds))

                    self.preds_2d = np.copy(preds)

                    if self.preds_2d.shape[0] >= self.receptive_field:
                        # if array of 2d pred is fullfilled receptive field criteria
                        # estimate 3D
                        # normalize screen with camera
                        w, h = 640, 360
                        kps = np.copy(self.preds_2d)
                        kps[..., :2] = normalize_screen_coordinates(kps[0][..., :2], w=w, h=h)

                        pad = (self.receptive_field - 1) // 2  # Padding on each side
                        causal = 0

                        keypoints_symmetry = self.metadata['keypoints_symmetry']
                        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
                        gen = UnchunkedGenerator(None, None, [kps],
                                                 pad=pad, causal_shift=causal,
                                                 augment=True,
                                                 kps_left=kps_left, kps_right=kps_right, joints_left=kps_left,
                                                 joints_right=kps_right)

                        for _, batch, batch_2d in gen.next_epoch():
                            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                            if torch.cuda.is_available():
                                inputs_2d = inputs_2d.cuda()

                                predicted_3d_pos = self.videopose_model(inputs_2d)

                                # Undo flipping and take average with non-flipped version
                                predicted_3d_pos[1, :, :, 0] *= -1
                                predicted_3d_pos[1, :, kps_left + kps_right] = predicted_3d_pos[1, :,
                                                                                     kps_right + kps_left]
                                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                                predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()

                                # start 3D visualization
                                self.o_frame = self.animate_3d(predicted_3d_pos)

                                self.frame_processed += 1

                                self.calculate_fps()
            else:
                self.o_frame = img

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

    def load_videopose(self):
        # generate metadata
        metadata = {}
        metadata['layout_name'] = 'humaneva15'
        metadata['num_joints'] = 15
        metadata['keypoints_symmetry'] = [[2, 3, 4, 8, 9, 10], [5, 6, 7, 11, 12, 13]]

        self.metadata = metadata

        # videopose architecture
        keypoints = metadata['num_joints']
        filter_widths = [1, 1, 1]
        causal = False
        dropout = 0.25
        channels = 1024
        dense = False

        model = TemporalModel(keypoints, 2,
                                  keypoints,
                                  filter_widths=filter_widths, causal=causal, dropout=dropout,
                                  channels=channels,
                                  dense=dense)

        # load pretrained
        pretrained_filename = './models/videopose3d/pretrained_humaneva15_rf-1.bin'
        pretrained = torch.load(pretrained_filename)
        model.load_state_dict(pretrained['model_pos'])
        model.cuda()

        self.receptive_field = model.receptive_field()

        self.videopose_model = model

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