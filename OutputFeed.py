import torch.multiprocessing as multiprocessing
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


from lib.core.config import config
from lib.core.config import update_config
from lib.core.inference import get_final_preds
from lib.utils.transforms import *

# import yolo
from lib.yolov4.tool.utils import *
from lib.yolov4.tool.torch_utils import *
from lib.yolov4.tool.darknet2pytorch import Darknet
from lib.yolov4.tool.utils import plot_boxes_cv2
import lib.models

import torch
import cv2
import numpy as np
import json
import codecs

from lib.common.generators import UnchunkedGenerator
from lib.common.model import TemporalModel
from lib.common.humaneva_dataset import humaneva_skeleton, HUMANEVA_KEYPOINTS
from lib.common.camera import camera_to_world
from lib.common.camera import normalize_screen_coordinates, reset_screen_coordinate

from lib.common.visualization import Grid3D

# keypoints index
# humaneva index
hhead = np.where(HUMANEVA_KEYPOINTS == 'head')[0][0]
hpelv = np.where(HUMANEVA_KEYPOINTS == 'pelvis')[0][0]
hthor = np.where(HUMANEVA_KEYPOINTS == 'thorax')[0][0]
hlsho = np.where(HUMANEVA_KEYPOINTS == 'lsho')[0][0]
hlelb = np.where(HUMANEVA_KEYPOINTS == 'lelb')[0][0]
hlwri = np.where(HUMANEVA_KEYPOINTS == 'lwri')[0][0]
hlhip = np.where(HUMANEVA_KEYPOINTS == 'lhip')[0][0]
hlkne = np.where(HUMANEVA_KEYPOINTS == 'lkne')[0][0]
hlank = np.where(HUMANEVA_KEYPOINTS == 'lank')[0][0]

hrsho = np.where(HUMANEVA_KEYPOINTS == 'rsho')[0][0]
hrelb = np.where(HUMANEVA_KEYPOINTS == 'relb')[0][0]
hrwri = np.where(HUMANEVA_KEYPOINTS == 'rwri')[0][0]
hrkne = np.where(HUMANEVA_KEYPOINTS == 'rkne')[0][0]
hrank = np.where(HUMANEVA_KEYPOINTS == 'rank')[0][0]
hrhip = np.where(HUMANEVA_KEYPOINTS == 'rhip')[0][0]

# mpii index
head = np.where(config.MPII_KEYPOINTS == 'head')[0][0]
pelv = np.where(config.MPII_KEYPOINTS == 'pelvis')[0][0]
thor = np.where(config.MPII_KEYPOINTS == 'thorax')[0][0]
lsho = np.where(config.MPII_KEYPOINTS == 'lsho')[0][0]
lelb = np.where(config.MPII_KEYPOINTS == 'lelb')[0][0]
lwri = np.where(config.MPII_KEYPOINTS == 'lwri')[0][0]
lhip = np.where(config.MPII_KEYPOINTS == 'lhip')[0][0]
lkne = np.where(config.MPII_KEYPOINTS == 'lkne')[0][0]
lank = np.where(config.MPII_KEYPOINTS == 'lank')[0][0]

rsho = np.where(config.MPII_KEYPOINTS == 'rsho')[0][0]
relb = np.where(config.MPII_KEYPOINTS == 'relb')[0][0]
rwri = np.where(config.MPII_KEYPOINTS == 'rwri')[0][0]
rkne = np.where(config.MPII_KEYPOINTS == 'rkne')[0][0]
rank = np.where(config.MPII_KEYPOINTS == 'rank')[0][0]
rhip = np.where(config.MPII_KEYPOINTS == 'rhip')[0][0]

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
        self.v_frame = None
        self.o_frame = o_frame
        self.anim_queue = multiprocessing.Queue()
        self.skel_queue = multiprocessing.Queue()

        self.frame_processed = 0

        # properties of bbox
        self.max_index_bbox = 0
        self.bbox = None

        # ndarray of 2d predicted joints
        self.preds_2d = None

        # videopose3d properties
        self.metadata = None
        self.receptive_field = 1        # n frame used as receptive field

        # grid 3D properties
        self.azimuth = np.array(0., dtype='float32')
        self.viewport = (640, 360)
        self.grid_3d = Grid3D(humaneva_skeleton, self.azimuth, self.viewport, self.anim_queue, self.skel_queue)
        self.first_frame = self.grid_3d.get_figure_numpy()

        self.filename = 'output/{}_3d_coord.json'.format(str(int(time.time())))
        self.joint_sequence = {
            'head': [],
            'pelv': [],
            'thor': [],
            'lsho': [],
            'lelb': [],
            'lwri': [],
            'lhip': [],
            'lkne': [],
            'lank': [],
            'rsho': [],
            'relb': [],
            'rwri': [],
            'rkne': [],
            'rank': [],
            'rhip': [],
        }

        self.started = False

    def reset_output(self):
        self.anim_queue = multiprocessing.Queue()
        self.skel_queue = multiprocessing.Queue()

        del self.grid_3d
        self.grid_3d = Grid3D(humaneva_skeleton, self.azimuth, self.viewport, self.anim_queue, self.skel_queue)
        self.first_frame = self.grid_3d.get_figure_numpy()

        self.preds_2d = None

        self.started = False

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
            boxes = do_detect(self.yolo_model, sized, 0.8, 0.6, use_cuda)

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
        cam = {
            'orientation': [0.4214752, -0.4961493, -0.5838273, 0.4851187],
            'translation': [4112.9121, 626.4929, 845.2988],
        }

        prediction = camera_to_world(preds_3d, R=np.array(cam['orientation'], dtype='float32'),
                                     t=np.array(cam['translation'], dtype='float32')/1000)

        anim_output = {'Reconstruction': prediction}
        self.skel_queue.put(anim_output)

        if not self.started:
            self.started = True
            self.grid_3d.start_process()

        if not self.anim_queue.empty():
            return self.anim_queue.get()

        if self.o_frame is None:
            return self.first_frame

        return self.o_frame

    def append_coord_dict(self, coord_3d):
        self.joint_sequence['head'].append(coord_3d[hhead].tolist())
        self.joint_sequence['pelv'].append(coord_3d[hpelv].tolist())
        self.joint_sequence['thor'].append(coord_3d[hthor].tolist())
        self.joint_sequence['lsho'].append(coord_3d[hlsho].tolist())
        self.joint_sequence['lelb'].append(coord_3d[hlelb].tolist())
        self.joint_sequence['lwri'].append(coord_3d[hlwri].tolist())
        self.joint_sequence['lhip'].append(coord_3d[hlhip].tolist())
        self.joint_sequence['lkne'].append(coord_3d[hlkne].tolist())
        self.joint_sequence['lank'].append(coord_3d[hlank].tolist())
        self.joint_sequence['rsho'].append(coord_3d[hrsho].tolist())
        self.joint_sequence['relb'].append(coord_3d[hrelb].tolist())
        self.joint_sequence['rwri'].append(coord_3d[hrwri].tolist())
        self.joint_sequence['rkne'].append(coord_3d[hrkne].tolist())
        self.joint_sequence['rank'].append(coord_3d[hrank].tolist())
        self.joint_sequence['rhip'].append(coord_3d[hrhip].tolist())

    def swap_keypoints(self, preds):
        new_preds = preds[:, :15, :].copy()
        new_preds[0][hhead] = preds[0][head]
        new_preds[0][hpelv] = preds[0][pelv]
        new_preds[0][hthor] = preds[0][thor]
        new_preds[0][hlsho] = preds[0][lsho]
        new_preds[0][hlelb] = preds[0][lelb]
        new_preds[0][hlwri] = preds[0][lwri]
        new_preds[0][hlhip] = preds[0][lhip]
        new_preds[0][hlkne] = preds[0][lkne]
        new_preds[0][hlank] = preds[0][lank]
        new_preds[0][hrsho] = preds[0][rsho]
        new_preds[0][hrelb] = preds[0][relb]
        new_preds[0][hrwri] = preds[0][rwri]
        new_preds[0][hrkne] = preds[0][rkne]
        new_preds[0][hrank] = preds[0][rank]
        new_preds[0][hrhip] = preds[0][rhip]

        return new_preds

    def process_bbox(self):
        if not self.v_queue.empty():
            self.current_frame = np.copy(self.v_queue.get())

            # object detection
            self.bbox = self.detect_bbox(self.current_frame)

            self.max_index_bbox, image = plot_boxes_cv2(self.current_frame, self.bbox)
        else:
            self.current_frame = None
            self.bbox = None

        return image

    def process_frame(self):
        # Human estimation
        if self.current_frame is not None and self.bbox is not None:
            img = self.current_frame

            # object detection
            bbox = self.bbox
            max_index_bbox = self.max_index_bbox

            if max_index_bbox > -1:
                # if bbox detected

                # 2d estimation
                # selected max height of bbox
                cen, s = self._box2cs(bbox[max_index_bbox], img.shape[1], img.shape[0])
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

                    # switch joint mpii index with humaneva index
                    # (1, 16, 2) => (1, 15, 2)
                    preds = self.swap_keypoints(preds)

                    if self.preds_2d is None:
                        self.preds_2d = np.copy(preds)
                    else:
                        self.preds_2d = np.concatenate((self.preds_2d, preds))

                    self.preds_2d = np.copy(preds)

                    # uncomment this to debug 2D
                    image = img.copy()
                    for mat in preds[0]:
                        x, y = int(mat[0]), int(mat[1])
                        cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

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
                                                 augment=False,
                                                 kps_left=kps_left, kps_right=kps_right, joints_left=kps_left,
                                                 joints_right=kps_right)

                        for _, batch, batch_2d in gen.next_epoch():
                            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                            if torch.cuda.is_available():
                                inputs_2d = inputs_2d.cuda()

                                predicted_3d_pos = self.videopose_model(inputs_2d)

                                predicted_3d_pos[0, :, kps_left + kps_right] = predicted_3d_pos[0, :,
                                                                                     kps_right + kps_left]
                                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                                predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()

                                if config.SAVE_TXT:
                                    pred_3d_coord = reset_screen_coordinate(predicted_3d_pos[0], w=w, h=h)
                                    self.append_coord_dict(pred_3d_coord)

                                # start 3D visualization
                                self.o_frame = self.animate_3d(predicted_3d_pos)

                                self.frame_processed += 1

                                self.calculate_fps()

    def load_yolo_model(self):
        cfgfile = './yolov4_cfg/yolov4-tiny.cfg'
        weightfile = './models/yolov4/yolov4-tiny.weights'

        self.yolo_model = Darknet(cfgfile)

        self.yolo_model.print_network()
        self.yolo_model.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))

    def load_simple_model(self):
        # load efficientnet simple baseline weights
        cfgfile = './simple_baseline_cfg/efficientnet/256x256_d256x3_adam_lr1e-3_k16.yaml'
        weightfile = './models/efficientnet_simple_baseline/efficientnet_simple_baseline_b0.pth.tar'
        update_config(cfgfile)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        self.simple_model = eval('lib.models.' + config.MODEL.NAME + '.get_pose_net')(
            config, is_train=False
        )

        gpus = [int(i) for i in config.GPUS.split(',')]
        self.simple_model = torch.nn.DataParallel(self.simple_model, device_ids=gpus).cuda()

        print('=> loading model from {}'.format(weightfile))
        state_dict = torch.load(weightfile)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.simple_model.load_state_dict(new_state_dict)
        # self.simple_model.load_state_dict(torch.load(weightfile))

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

    def write_coord_txt(self):
        json.dump(self.joint_sequence, codecs.open(self.filename, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)
        
    def destroy_window(self):
        try:
            self.grid_3d.stop_process()
            self.reset_output()
        except:
            print('output process finished')