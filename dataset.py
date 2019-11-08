import torch
import json
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from utils import pairwise
import math
import os
from torchvision.transforms import ToTensor,Normalize
from dataaugment import resize, augment_image, random_crop, random_flip, random_rotate, Pad
from mpii_dataset import *
import torch.nn.functional as F

class KeypointDataset2D(Dataset):
    def __init__(self,insize,keypoint_names,edges,flip_indices,
                 keypoints,bbox,is_visible,is_labeled,image_paths,
                 image_root,do_augmentation,outsize,local_grid_size,parts_scale):
        super(KeypointDataset2D,self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.local_grid_size = local_grid_size
        self.parts_scale = parts_scale
        self.edges = edges
        self.do_augmentation = do_augmentation
        self.image_root = image_root
        self.flip_indices = flip_indices
        # nums
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.bbox = bbox
        self.is_visible = is_visible
        self.is_labeled = is_labeled
        #print(len(self.image_paths))


    def __len__(self):
        return len(self.image_paths)

    def transform(self,image, keypoints, bbox, is_labeled, is_visible):
        # Color augmentation
        image = augment_image(image)
        # Random rotate
        image, keypoints, bbox = random_rotate(image, keypoints, bbox)
        # Random flip
        #image, keypoints, bbox, is_labeled, is_visible = random_flip(image, keypoints, bbox, is_labeled, is_visible,
        #                                                             self.flip_indices)
        # Random crop
        image, keypoints, bbox = random_crop(image, keypoints, bbox, is_labeled)
        return image, keypoints, bbox, is_labeled, is_visible

    def __getitem__(self, index):
        h,w = self.insize
        image = Image.open(os.path.join(self.image_root,self.image_paths[index]))
        #print(self.image_paths[index])
        image = np.array(image) # hwc # uint8



        keypoints = self.keypoints[index]
        bbox = self.bbox[index]
        is_labeled = self.is_labeled[index]
        is_visible = self.is_visible[index]

        #image = image.copy()
        keypoints = keypoints.copy()
        bbox = bbox.copy()
        is_labeled = is_labeled.copy()
        is_visible = is_visible.copy()

        image, keypoints, bbox = Pad(image, keypoints, bbox)
        if self.do_augmentation:
            image, keypoints, bbox, is_labeled, is_visible = self.transform(image,keypoints,bbox,is_labeled,is_visible)

        image, keypoints, bbox = resize(image, keypoints, bbox, (h, w))


        image = torch.from_numpy(image).permute(2,0,1).contiguous().type(torch.float) # chw
        image = image / 255.
        image = Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])(image)



        ################encode##################################
        outH, outW = self.outsize
        inH, inW = self.insize
        gridW, gridH = int(inW / outW), int(inH / outH)

        K = len(KEYPOINT_NAMES)
        delta = torch.zeros((K, outH, outW))
        tx = torch.zeros((K, outH, outW))
        ty = torch.zeros((K, outH, outW))
        tw = torch.zeros((K, outH, outW))
        th = torch.zeros((K, outH, outW))
        te = torch.zeros((
            len(EDGES),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW))

        for (x, y, w, h), points, labeled in zip(bbox, keypoints, is_labeled):
            partsW = partsH = self.parts_scale * math.sqrt(w * w + h * h)  # 0.2*bbox
            instanceW, instanceH = w, h
            cy = y + h / 2
            cx = x + w / 2
            points = [[cy, cx]] + points.tolist()  # 列表的加法，把中心点加在最上面 19个
            labeled = [True] + labeled.tolist()

            for k, (yx, l) in enumerate(zip(points, labeled)):  # yx??
                if not l:
                    continue
                cy = yx[0] / gridH  # 求点在哪个grid中
                cx = yx[1] / gridW
                ix, iy = int(cx), int(cy)
                sizeW = instanceW if k == 0 else partsW  # 一个人的 关节框大小一样
                sizeH = instanceH if k == 0 else partsH
                if 0 <= iy < outH and 0 <= ix < outW:
                    delta[k, iy, ix] = 1  # 如果有标签，delta为1
                    tx[k, iy, ix] = cx - ix  # 实际位置和gird的偏移，也就是推理时应求的tx
                    ty[k, iy, ix] = cy - iy
                    tw[k, iy, ix] = sizeW / inW
                    th[k, iy, ix] = sizeH / inH

            for ei, (s, t) in enumerate(EDGES):
                if not labeled[s]:
                    continue
                if not labeled[t]:
                    continue
                src_yx = points[s]
                tar_yx = points[t]
                iyx = (int(src_yx[0] / gridH), int(src_yx[1] / gridW))
                jyx = (int(tar_yx[0] / gridH) - iyx[0] + self.local_grid_size[1] // 2,
                       int(tar_yx[1] / gridW) - iyx[1] + self.local_grid_size[0] // 2)  # +2 ？？

                if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                    continue
                if jyx[0] < 0 or jyx[1] < 0 or jyx[0] >= self.local_grid_size[1] or jyx[1] >= self.local_grid_size[0]:
                    continue

                te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1  # 【edges，h‘，w’，h，w】

            # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        max_delta_ij = torch.ones((len(self.edges),
                                   outH, outW,
                                   self.local_grid_size[1], self.local_grid_size[0]))  # 【edges，h,w,h',w'】
        or_delta = torch.zeros((len(self.edges), outH, outW))  # [edges，h,w】
        for ei, (s, t) in enumerate(self.edges):
            or_delta[ei] = torch.min(torch.Tensor(delta[s] + delta[t]), torch.Tensor([1.]))  #

        mask = F.max_pool2d(torch.unsqueeze(or_delta, 0),
                            (self.local_grid_size[1], self.local_grid_size[0]),
                            1,
                            (self.local_grid_size[1] // 2, self.local_grid_size[0] // 2))  # [1,edges,7,7]
        mask = torch.squeeze(mask, 0)
        mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)
        max_delta_ij = mask * max_delta_ij

        max_delta_ij = max_delta_ij.permute(0, 3, 4, 1, 2)



        return {"image":image,"delta":delta, "max_delta_ij":max_delta_ij,
                "tx":tx, "ty":ty, "tw":tw, "th":th, "te":te, "image_name":self.image_paths[index]}




def get_mpii_dataset(insize,outsize,local_grid_size,parts_scale, image_root, annotations,
                     min_num_keypoints=4,do_augmentation=False):
    cat_id = 1
    dataset = json.load(open(annotations, 'r',encoding='UTF-8'))
    cat = dataset['categories'][cat_id - 1]
    #assert cat['keypoints'] == DEFAULT_KEYPOINT_NAMES
    # image_id => filename, keypoints, bbox, is_visible, is_labeled
    images = {}

    for image in dataset['images']:  # 得到所有图片
        images[int(image['id'])] = image['file_name'], [], [], [], []

    for anno in dataset['annotations']:  # 遍历所有anno
        if anno['num_keypoints'] < min_num_keypoints:
            continue
        if int(anno['category_id']) != cat_id:
            continue
        if anno['iscrowd'] != 0:
            continue
        image_id = int(anno['image_id'])
        d = np.array(anno['keypoints'], dtype='float32').reshape(-1, 3)
        # define neck from left_shoulder and right_shoulder
        #left_shoulder_idx = DEFAULT_KEYPOINT_NAMES.index('left_shoulder')
        #right_shoulder_idx = DEFAULT_KEYPOINT_NAMES.index('right_shoulder')
        #left_shoulder, left_v = d[left_shoulder_idx][:2], d[left_shoulder_idx][2]
        #right_shoulder, right_v = d[right_shoulder_idx][:2], d[right_shoulder_idx][2]
        # if left_v >= 1 and right_v >= 1:
        #     neck = (left_shoulder + right_shoulder) / 2.
        #     labeled = 1
        #     d = np.vstack([np.array([*neck, labeled],dtype='float32'), d])
        # else:
        #     labeled = 0
        #     # insert dummy data correspond to `neck`  如果左右肩有一个不存在，则定义一个不存在的 脖子
        #     d = np.vstack([np.array([0.0, 0.0, labeled],dtype='float32'), d])

        keypoints = d[:, [1, 0]]  # array of y,x  获得每个点的坐标 # 为啥要反过来？
        bbox = anno['bbox']  # 人体框
        is_visible = d[:, 2] == 2 # 是否可见
        is_labeled = d[:, 2] >= 1 # 是否有标签

        entry = images[image_id] # 得到
        entry[1].append(np.asarray(keypoints))
        entry[2].append(np.asarray(bbox,dtype='float32'))
        entry[3].append(np.asarray(is_visible).astype(np.bool))
        entry[4].append(np.asarray(is_labeled).astype(np.bool))
        #  所有anno信息放入images的[{},{},{}]中
    # filter-out non annotated images
    image_paths = []
    keypoints = []
    bbox = []
    is_visible = []
    is_labeled = []

    for filename, k, b, v, l in images.values():
        if len(k) == 0:
            continue
        image_paths.append(filename)
        bbox.append(b)
        keypoints.append(k)
        is_visible.append(v)
        is_labeled.append(l)
        ## 放入list中
    print(len(image_paths))
    print(len(keypoints))
    # for test
    #return image_paths,keypoints

    return KeypointDataset2D(
        insize=insize,  # input image size
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES), # limb
        flip_indices=FLIP_INDICES,  # 翻转时名字转换
        keypoints=keypoints,
        bbox=bbox,
        is_visible=is_visible,
        is_labeled=is_labeled,
        image_paths=image_paths,
        image_root=image_root,
        do_augmentation=do_augmentation,
        outsize=outsize,
        local_grid_size=local_grid_size,
        parts_scale=parts_scale
    )





# if __name__=="__main__":
#     x,y = get_coco_dataset(insize=(224,224),image_root='path',annotations='test_data/my_valdata1.json')
#     print(x[0])
#     print(type(y[0]))

