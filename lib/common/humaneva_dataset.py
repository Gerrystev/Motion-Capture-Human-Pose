# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from common.skeleton import Skeleton
       
humaneva_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
       joints_left=[2, 3, 4, 8, 9, 10],
       joints_right=[5, 6, 7, 11, 12, 13])

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