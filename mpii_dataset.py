import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)



KEYPOINT_NAMES = [
    'head_top',
    'upper_neck',
    'l_shoulder',
    'r_shoulder',
    'l_elbow',
    'r_elbow',
    'l_wrist',
    'r_wrist',
    'l_hip',
    'r_hip',
    'l_knee',
    'r_knee',
    'l_ankle',
    'r_ankle',
]

FLIP_CONVERTER = {'head_top': 'head_top',
                  'upper_neck': 'upper_neck',
                  'l_shoulder': 'r_shoulder',
                  'r_shoulder': 'l_shoulder',
                  'l_elbow': 'r_elbow',
                  'r_elbow': 'l_elbow',
                  'l_wrist': 'r_wrist',
                  'r_wrist': 'l_wrist',
                  'l_hip': 'r_hip',
                  'r_hip': 'l_hip',
                  'l_knee': 'r_knee',
                  'r_knee': 'l_knee',
                  'l_ankle': 'r_ankle',
                  'r_ankle': 'l_ankle',
                  }

FLIP_INDICES = [KEYPOINT_NAMES.index(FLIP_CONVERTER[k]) for k in KEYPOINT_NAMES]

KEYPOINT_NAMES = ['instance'] + KEYPOINT_NAMES

COLOR_MAP = {
    'instance': (225, 225, 225),
    'head_top': (255, 0, 0),
    'upper_neck': (255, 85, 0),
    'r_shoulder': (255, 170, 0),
    'r_elbow': (255, 255, 0),
    'r_wrist': (170, 255, 0),
    'l_shoulder': (85, 255, 0),
    'l_elbow': (0, 127, 0),
    'l_wrist': (0, 255, 85),
    'r_hip': (0, 170, 170),
    'r_knee': (0, 255, 255),
    'r_ankle': (0, 170, 255),
    'l_hip': (0, 85, 255),
    'l_knee': (0, 0, 255),
    'l_ankle': (85, 0, 255),
    'r_eye': (170, 0, 255),
    'l_eye': (255, 0, 255),
    'r_ear': (255, 0, 170),
    'l_ear': (255, 0, 85),
}

EDGES_BY_NAME = [
    ['instance', 'upper_neck'],
    ['upper_neck', 'head_top'],
    ['upper_neck', 'l_shoulder'],
    ['upper_neck', 'r_shoulder'],
    ['upper_neck', 'l_hip'],
    ['upper_neck', 'r_hip'],
    ['l_shoulder', 'l_elbow'],
    ['l_elbow', 'l_wrist'],
    ['r_shoulder', 'r_elbow'],
    ['r_elbow', 'r_wrist'],
    ['l_hip', 'l_knee'],
    ['l_knee', 'l_ankle'],
    ['r_hip', 'r_knee'],
    ['r_knee', 'r_ankle'],
]

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'upper_neck', 'head_top']
TRACK_ORDER_1 = ['instance', 'upper_neck', 'l_shoulder', 'l_elbow', 'l_wrist']
TRACK_ORDER_2 = ['instance', 'upper_neck', 'r_shoulder', 'r_elbow', 'r_wrist']
TRACK_ORDER_3 = ['instance', 'upper_neck', 'l_hip', 'l_knee', 'l_ankle']
TRACK_ORDER_4 = ['instance', 'upper_neck', 'r_hip', 'r_knee', 'r_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4]
DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])

# [[[0, 1], [2, 1]],
#  [[0, 2, 6, 7], [2, 3, 5, 7]],
#  [[0, 3, 8, 9], [2, 4, 6, 8]],
#  [[0, 4, 10, 11], [2, 9, 11, 13]],
#  [[0, 5, 12, 13], [2, 10, 12, 14]]]

# DEFAULT_KEYPOINT_NAMES = [
#     'nose',
#     'left_eye',
#     'right_eye',
#     'left_ear',
#     'right_ear',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
#     'left_hip',
#     'right_hip',
#     'left_knee',
#     'right_knee',
#     'left_ankle',
#     'right_ankle'
# ]
#
# FLIP_CONVERTER = {
#     'nose': 'nose',
#     'neck': 'neck',
#     'left_eye': 'right_eye',
#     'right_eye': 'left_eye',
#     'left_ear': 'right_ear',
#     'right_ear': 'left_ear',
#     'left_shoulder': 'right_shoulder',
#     'right_shoulder': 'left_shoulder',
#     'left_elbow': 'right_elbow',
#     'right_elbow': 'left_elbow',
#     'left_wrist': 'right_wrist',
#     'right_wrist': 'left_wrist',
#     'left_hip': 'right_hip',
#     'right_hip': 'left_hip',
#     'left_knee': 'right_knee',
#     'right_knee': 'left_knee',
#     'left_ankle': 'right_ankle',
#     'right_ankle': 'left_ankle',
# }
#
# # update keypoints
# KEYPOINT_NAMES = ['neck'] + DEFAULT_KEYPOINT_NAMES
# FLIP_INDICES = [KEYPOINT_NAMES.index(FLIP_CONVERTER[k]) for k in KEYPOINT_NAMES]  # 翻转时各点变换名字
# # update keypoints
# KEYPOINT_NAMES = ['instance'] + KEYPOINT_NAMES
#
# COLOR_MAP = {
#     'instance': (225, 225, 225),
#     'nose': (255, 0, 0),
#     'neck': (255, 85, 0),
#     'right_shoulder': (255, 170, 0),
#     'right_elbow': (255, 255, 0),
#     'right_wrist': (170, 255, 0),
#     'left_shoulder': (85, 255, 0),
#     'left_elbow': (0, 127, 0),
#     'left_wrist': (0, 255, 85),
#     'right_hip': (0, 170, 170),
#     'right_knee': (0, 255, 255),
#     'right_ankle': (0, 170, 255),
#     'left_hip': (0, 85, 255),
#     'left_knee': (0, 0, 255),
#     'left_ankle': (85, 0, 255),
#     'right_eye': (170, 0, 255),
#     'left_eye': (255, 0, 255),
#     'right_ear': (255, 0, 170),
#     'left_ear': (255, 0, 85),
# }
#
# EDGES_BY_NAME = [
#     ['instance', 'neck'],
#     ['neck', 'nose'],
#     ['nose', 'left_eye'],
#     ['left_eye', 'left_ear'],
#     ['nose', 'right_eye'],
#     ['right_eye', 'right_ear'],
#     ['neck', 'left_shoulder'],
#     ['left_shoulder', 'left_elbow'],
#     ['left_elbow', 'left_wrist'],
#     ['neck', 'right_shoulder'],
#     ['right_shoulder', 'right_elbow'],
#     ['right_elbow', 'right_wrist'],
#     ['neck', 'left_hip'],
#     ['left_hip', 'left_knee'],
#     ['left_knee', 'left_ankle'],
#     ['neck', 'right_hip'],
#     ['right_hip', 'right_knee'],
#     ['right_knee', 'right_ankle'],
# ]
#
# EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]
#
# TRACK_ORDER_0 = ['instance', 'neck', 'nose', 'left_eye', 'left_ear']
# TRACK_ORDER_1 = ['instance', 'neck', 'nose', 'right_eye', 'right_ear']
# TRACK_ORDER_2 = ['instance', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist']
# TRACK_ORDER_3 = ['instance', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist']
# TRACK_ORDER_4 = ['instance', 'neck', 'left_hip', 'left_knee', 'left_ankle']
# TRACK_ORDER_5 = ['instance', 'neck', 'right_hip', 'right_knee', 'right_ankle']
#
# TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5]
# DIRECTED_GRAPHS = []
#
# for keypoints in TRACK_ORDERS:
#     es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
#     ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
#     DIRECTED_GRAPHS.append([es, ts])