from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time
import json_tricks

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
from core.config import update_dir
from core.inference import get_final_preds
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *

# import yolo
from yolov4.tool.utils import *
from yolov4.tool.torch_utils import *
from yolov4.tool.darknet2pytorch import Darknet
import torch

import cv2
import models
import numpy as np


DATASET_INDEX = {
    'S1': {
        'Walking 1': [(80, 590), (665, 1203)],      # +75
        'ThrowCatch 1': [(5, 473), (528, 984)],     # +55
        'Box 1': [(55, 435), (435, 789)],           # +50
        'Jog 1': [(50, 367), (412, 740)],           # +45
        'Gestures 1': [(45, 395), (435, 801)],      # +40
    },
    'S2': {
        'Jog 1': [(100, 398), (493, 795)],            # +92 // +88
        'Box 1': [(117, 382), (494, 734)],            # +112
        'Gestures 1': [(122, 500), (617, 901)],       # +617
        'ThrowCatch 1': [(130, 550), (675, 1128)],    # +125
        'Walking 1': [(115, 438), (548, 876)],        # +110
    },
    'S3': {
        'Box 1': [(1, 508), (508, 1017)],           # -4
        'Gestures 1': [(83, 533), (611, 1102)],     # +78
        'Jog 1': [(65, 401), (461, 842)],           # +60
    },
}

# these value for fixing imbalance dataset on test set
# Formula for these values are (30*(validate+train)/100)
TEST_N_FRAME = {
    'S1': {
        'Box 1': 88,
        'ThrowCatch 1': -1,
        'Walking 1': 204,
        'Jog 1': 73,
        'Gestures 1': 143,
    },
    'S2': {
        'Box 1': 70,
        'Gestures 1': 111,
        'ThrowCatch 1': 124,
        'Jog 1': 120,
        'Walking 1': 130,
    },
    'S3': {
        'Box 1': 186,
        'Gestures 1': 39,
        'Jog 1': 141,
    },
}

VALID_N_FRAME = {
    'S1': {
        'Box 1': 71,
        'ThrowCatch 1': 36,
        'Walking 1': 163,
        'Jog 1': 58,
        'Gestures 1': 115,
    },
    'S2': {
        'Box 1': 56,
        'Gestures 1': 89,
        'ThrowCatch 1': 99,
        'Jog 1': 96,
        'Walking 1': 104,
    },
    'S3': {
        'Box 1': 148,
        'Gestures 1': 31,
        'Jog 1': 113,
    },
}

# Frames to skip for each video (synchronization)
SYNC_DATA = {
    'S1': {
        'Walking 1': (82, 81, 82),
        'Jog 1': (51, 51, 50),
        'ThrowCatch 1': (61, 61, 60),
        'Gestures 1': (45, 45, 44),
        'Box 1': (57, 57, 56),
    },
    'S2': {
        'Walking 1': (115, 115, 114),
        'Jog 1': (100, 100, 99),
        'ThrowCatch 1': (127, 127, 127),
        'Gestures 1': (122, 122, 121),
        'Box 1': (119, 119, 117),
    },
    'S3': {
        'Walking 1': (80, 80, 80),
        'Jog 1': (65, 65, 65),
        'Gestures 1': (83, 83, 82),
        'Box 1': (1, 1, 1),
    }
}

HUMANEVA_KEYPOINTS = np.array([
    'pelvis',
    'thorax',
    'lsho',
    'lelb',
    'lwri',
    'rsho',
    'relb',
    'rwri',
    'lhip',
    'lkne',
    'lank',
    'rhip',
    'rkne',
    'rank',
    'head'
])

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def _box2cs(box, image_width, image_height):
    x = box[0] * image_width
    y = box[1] * image_height
    x2 = box[2] * image_width
    y2 = box[3] * image_height

    w = x2 - x
    h = y2 - y

    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h):

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    pixel_std = 200.0

    max_wh = max([w / pixel_std, h / pixel_std])
    scale = np.array([max_wh, max_wh], dtype='float32')

    scale = scale * 1.25

    return center, scale

def detect_bbox(cfgfile, weightfile, img):
    """hyper parameters"""
    use_cuda = True
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    namesfile = 'data/coco.names'

    class_names = load_class_names(namesfile)

    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('Image: Predicted in %f seconds.' % ((finish - start)))

    return boxes[0]

def draw_skeleton(preds, img):
    pred = preds[0,:, 0:2] + 1.0
    pred = np.round(pred).astype(int)

    head = np.where(HUMANEVA_KEYPOINTS == 'head')[0][0]
    pelv = np.where(HUMANEVA_KEYPOINTS == 'pelvis')[0][0]
    thor = np.where(HUMANEVA_KEYPOINTS == 'thorax')[0][0]
    lsho = np.where(HUMANEVA_KEYPOINTS == 'lsho')[0][0]
    lelb = np.where(HUMANEVA_KEYPOINTS == 'lelb')[0][0]
    lwri = np.where(HUMANEVA_KEYPOINTS == 'lwri')[0][0]
    lhip = np.where(HUMANEVA_KEYPOINTS == 'lhip')[0][0]
    lkne = np.where(HUMANEVA_KEYPOINTS == 'lkne')[0][0]
    lank = np.where(HUMANEVA_KEYPOINTS == 'lank')[0][0]

    rsho = np.where(HUMANEVA_KEYPOINTS == 'rsho')[0][0]
    relb = np.where(HUMANEVA_KEYPOINTS == 'relb')[0][0]
    rwri = np.where(HUMANEVA_KEYPOINTS == 'rwri')[0][0]
    rkne = np.where(HUMANEVA_KEYPOINTS == 'rkne')[0][0]
    rank = np.where(HUMANEVA_KEYPOINTS == 'rank')[0][0]
    rhip = np.where(HUMANEVA_KEYPOINTS == 'rhip')[0][0]

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

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False
    )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Load images from videos
    # configurate np.load allow_pickle to true
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # load dataset from .npz file and convert to dictionary
    data = np.load("data_2d_humaneva15_gt.npz")
    data_3d = np.load("data_3d_humaneva15.npz")

    metadata = data.f.metadata
    pos = data.f.positions_2d
    pos_3d = data_3d.f.positions_3d
    dataset = dict(enumerate(pos.flatten()))[0]
    dataset_3d = dict(enumerate(pos_3d.flatten()))[0]

    test_db = []

    test_3d_dict = dict()
    test_pred_dict = dict()

    json_test = json_tricks.dumps(test_db)

    for c in range(3):
        # loop through 3 camera
        # camera index to preview
        camera_index = c
        cam_num = camera_index + 1

        for subject, movement in DATASET_INDEX.items():
            # iterate train/ test subject dataset
            subject_name = subject
            for move, idx in movement.items():
                # iterate movement chunks
                video_index = -1
                start_index = -1
                end_index = -1

                train_index = False
                valid_index = False
                is_stop = False

                additional_frame = 0
                valid_frame = 0

                data_type = "Test"
                for i in range(len(idx)):
                    # iterate train/ validate start/ end frame
                    # start video from starting index
                    if subject_name == "S1" and move == "ThrowCatch 1" and i == 1:
                        continue
                    if video_index == -1 or train_index:
                        if train_index:
                            data_section = "Train/" + subject_name
                        else:
                            data_section = "Validate/" + subject_name

                        video_index = DATASET_INDEX[subject_name][move][i][0]
                        end_index = DATASET_INDEX[subject_name][move][i][1]

                        if subject_name == "S1" and move == "ThrowCatch 1":
                            data_section = "Train/" + subject_name
                            video_index = DATASET_INDEX[subject_name][move][1][0]
                            end_index = DATASET_INDEX[subject_name][move][1][1]

                        if start_index == -1:
                            # if switch movement/ start video
                            video_name = move.replace(" ", "_") + "_(C" + str(cam_num) + ")"
                            input_video_path = '../' + subject_name + '/Image_Data/' + video_name + '.avi'
                            cap = cv2.VideoCapture(input_video_path)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
                    frame_index = 0
                    chunk_index = 0
                    current_chunk = None
                    ret, frame = cap.read()
                    while video_index < end_index:
                        # show current frame from video_index
                        if not is_stop:
                            ret, frame = cap.read()

                        if ret:
                            if frame_index == 0:
                                # switch to next chunk
                                movement_name = move + " chunk" + str(chunk_index)
                                current_chunk = dataset[data_section][movement_name][camera_index]
                                new_chunk_3d = None
                                pred_chunk = None

                            img = frame
                            if np.isfinite(current_chunk).all():
                                # if current chunk is valid draw the skeleton
                                # get current keypoint from current frame
                                current_keypoint = current_chunk[frame_index]

                                # save image from valid frame
                                no_test = False
                                if subject_name == "S1" and move == "ThrowCatch 1":
                                    data_type = "Validate"
                                    no_test = True
                                    valid_index = True
                                    if valid_index:
                                        data_type = "Validate"
                                    if train_index:
                                        data_type = "Train"

                                if TEST_N_FRAME[subject_name][move] <= valid_frame and not valid_index and not no_test:
                                    valid_frame = 0
                                    valid_index = True
                                    data_type = "Validate"

                                if VALID_N_FRAME[subject_name][move] <= valid_frame and not train_index and valid_index:
                                    valid_frame = 0
                                    train_index = True
                                    data_type = "Train"

                                path = "images/C" + str(cam_num) + "/" + subject_name + "/" + move + "/" + data_type
                                image_name = str(valid_frame) + '.jpg'

                                # write to json
                                gt_coord = dict()
                                gt_coord['joints'] = current_keypoint
                                gt_coord['image'] = os.path.join(path, image_name)

                                if data_type == "Test":
                                    test_db.append(gt_coord)
                                    datatype_subject = 'Validate/' + subject
                                    current_chunk_3d = dataset_3d[data_section][movement_name]
                                    current_3d_coord = current_chunk_3d[frame_index]

                                    # object detection
                                    yolov4Tiny_cfgfile = './yolov4_cfg/yolov4-tiny.cfg'
                                    yolov4Tiny_weightfile = './models/yolov4/yolov4-tiny.weights'

                                    start_time = time.time()
                                    bbox = detect_bbox(yolov4Tiny_cfgfile, yolov4Tiny_weightfile, img)

                                    # 2d estimation
                                    cen, s = _box2cs(bbox[0], img.shape[1], img.shape[0])
                                    r = 0

                                    trans = get_affine_transform(cen, s, r, [256, 256])
                                    input = cv2.warpAffine(
                                        img,
                                        trans,
                                        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
                                        flags=cv2.INTER_LINEAR)

                                    # vis transformed image
                                    # cv2.imshow('image', input)
                                    # cv2.waitKey(3000)

                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                    ])
                                    input = transform(input).unsqueeze(0)

                                    # switch to evaluate mode
                                    model.eval()
                                    with torch.no_grad():
                                        # compute output heatmap
                                        output = model(input)
                                        # compute coordinate
                                        preds, maxvals = get_final_preds(
                                            config, output.clone().cpu().numpy(), np.asarray([cen]), np.asarray([s]))
                                        # plot
                                        image = img.copy()
                                        for mat in preds[0]:
                                            x, y = int(mat[0]), int(mat[1])
                                            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

                                        # vis result
                                        # cv2.imshow('res', image)
                                        # cv2.waitKey(10000)

                                    draw_skeleton(preds, img.copy())
                                    print("Program run at: --- %s seconds ---" % (time.time() - start_time))

                                    # 3D chunk
                                    if new_chunk_3d is None:
                                        new_chunk_3d = np.array([current_3d_coord])
                                    else:
                                        new_chunk_3d = np.vstack((new_chunk_3d, [current_3d_coord]))

                                    # pred chunk
                                    if pred_chunk is None:
                                        pred_chunk = preds
                                    else:
                                        pred_chunk = np.vstack((pred_chunk, preds))

                                    mdict = {movement_name: new_chunk_3d}
                                    cdict_2d = {c: preds}
                                    mdict_2d = {movement_name: cdict_2d}

                                    if datatype_subject not in test_3d_dict:
                                        test_3d_dict[datatype_subject] = mdict
                                        test_pred_dict[datatype_subject] = mdict_2d
                                    else:
                                        # 3d dict
                                        if movement_name not in test_3d_dict[datatype_subject].keys():
                                            test_3d_dict[datatype_subject].update(mdict)
                                        else:
                                            test_3d_dict[datatype_subject][movement_name] = new_chunk_3d

                                        # 2d dict
                                        if movement_name not in test_pred_dict[datatype_subject].keys():
                                            test_pred_dict[datatype_subject].update(mdict_2d)
                                        elif c not in test_pred_dict[datatype_subject][movement_name].keys():
                                            test_pred_dict[datatype_subject][movement_name].update(cdict_2d)
                                        else:
                                            test_pred_dict[datatype_subject][movement_name][c] = pred_chunk

                                # score valid frame
                                valid_frame += 1
                        else:
                            break

                        cv2.imshow(subject_name + " " + move + " C" + str(cam_num), img)

                        if not is_stop:
                            # continue to next video index & check sync_data index
                            video_index += 1
                            if video_index in SYNC_DATA[subject_name][move]:
                                # if next index is included in sync_data skip again
                                cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
                                video_index += 1

                            # continue to next frame in a chunk & check if frame index is more than len(chunk)
                            frame_index += 1
                            if frame_index >= len(current_chunk):
                                frame_index = 0
                                chunk_index += 1

                # release capture
                cap.release()
                cv2.destroyAllWindows()

    # save 2d pred npz
    np.savez_compressed('data_2d_preds_humaneva15.npz', positions_3d=test_pred_dict)

if __name__ == '__main__':
    main()