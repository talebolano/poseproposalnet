import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.transforms.functional
from PIL import ImageDraw, Image
from mpii_dataset import *

EPSILON = 1e-6


def area(bbox):
    _, _, w, h = bbox
    return w * h


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    w = (torch.min(x0 + w0 / 2, x1 + w1 / 2) - torch.max(x0 - w0 / 2, x1 - w1 / 2)).clamp(0)
    h = (torch.min(y0 + h0 / 2, y1 + h1 / 2) - torch.max(y0 - h0 / 2, y1 - h1 / 2)).clamp(0)

    return w * h


def iou(bbox0, bbox1):
    area0 = area(bbox0)
    area1 = area(bbox1)
    intersect = intersection(bbox0, bbox1)

    return intersect / (area0 + area1 - intersect + EPSILON)

def restore_xy(x, y,gridsize,outsize):
    gridH, gridW = gridsize # 32
    outH, outW = outsize  ## 7
    Y,X = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
    Y = Y.type_as(y)
    X = X.type_as(x)
    return (x + X) * gridW, (y + Y) * gridH

def restore_size(w, h,insize):
    inH, inW = insize
    return inW * w, inH * h


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return torch.zeros((0,), dtype=torch.long)

    if score is not None:
        order = score.argsort().flip(0) # 从大到小
        bbox = bbox[order]
    bbox_area = torch.prod(bbox[:, 2:] - bbox[:, :2], 1)

    selec = torch.zeros(bbox.shape[0], dtype=torch.bool)
    for i, b in enumerate(bbox):
        tl = torch.max(b[:2], bbox[selec, :2])
        br = torch.min(b[2:], bbox[selec, 2:])
        area = torch.prod(br - tl, 1) * (tl < br).all(1).type_as(bbox)
        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None:
            break

    selec = torch.nonzero(selec).t().tolist()
    if score is not None:
        selec = order[selec]
    return selec


def encode(keypoints,bbox,is_labeled,insize,outsize,local_grid_size,parts_scale):
    #image = in_data['image']
    # keypoints = in_data['keypoints']
    # bbox = in_data['bbox']
    # is_labeled = in_data['is_labeled']
    #dataset_type = in_data['dataset_type']
    outH, outW = outsize
    inH, inW = insize
    gridW,gridH = int(inW / outW), int(inH / outH)

    K = len(KEYPOINT_NAMES)
    delta = torch.zeros((K, outH, outW)).type_as(bbox)
    tx = torch.zeros((K, outH, outW)).type_as(bbox)
    ty = torch.zeros((K, outH, outW)).type_as(bbox)
    tw = torch.zeros((K, outH, outW)).type_as(bbox)
    th = torch.zeros((K, outH, outW)).type_as(bbox)
    te = torch.zeros((
        len(torch["edges"]),
        local_grid_size[1], local_grid_size[0],
        outH, outW)).type_as(bbox)

    for (x, y, w, h), points, labeled in zip(bbox, keypoints, is_labeled):
        partsW, partsH = parts_scale * math.sqrt(w * w + h * h) #0.2*bbox
        instanceW, instanceH = w, h
        cy = y + h / 2
        cx = x + w / 2
        points = [[cy, cx]] + list(points)  # 列表的加法，把中心点加在最上面 19个
        labeled = [True] + list(labeled)


        for k, (yx, l) in enumerate(zip(points, labeled)):  #  yx??
            if not l:
                continue
            cy = yx[0] / gridH  # 求点在哪个grid中
            cx = yx[1] / gridW
            ix, iy = int(cx), int(cy)
            sizeW = instanceW if k == 0 else partsW # 一个人的 关节框大小一样
            sizeH = instanceH if k == 0 else partsH
            if 0 <= iy < outH and 0 <= ix < outW:
                delta[k, iy, ix] = 1 # 如果有标签，delta为1
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
            jyx = (int(tar_yx[0] / gridH) - iyx[0] + local_grid_size[1] // 2,
                    int(tar_yx[1] / gridW) - iyx[1] + local_grid_size[0] // 2) # +2 ？？

            if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                continue
            if jyx[0] < 0 or jyx[1] < 0 or jyx[0] >= local_grid_size[1] or jyx[1] >= local_grid_size[0]:
                continue

            te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1 #【edges，h‘，w’，h，w】

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
    max_delta_ij = torch.ones((len(EDGES),
                                outH, outW,
                                local_grid_size[1], local_grid_size[0])).type_as(bbox) # 【edges，h,w,h',w'】
    or_delta = torch.zeros((len(EDGES), outH, outW)).type_as(bbox) # [edges，h,w】
    for ei, (s, t) in enumerate(EDGES):
        or_delta[ei] = torch.min(torch.tensor(delta[s] + delta[t]), torch.tensor([1])) #

    mask = F.max_pool2d(torch.unsqueeze(or_delta, 0),
                                (local_grid_size[1], local_grid_size[0]),
                                1,
                                (local_grid_size[1] // 2, local_grid_size[0] // 2))  # [1,edges,7,7]
    mask = torch.squeeze(mask, 0)
    mask = mask.view(mask.shape[0],mask.shape[1],mask.shape[2],1,1)
    max_delta_ij = mask*max_delta_ij

    max_delta_ij = max_delta_ij.permute(0, 3, 4, 1, 2)


    return delta, max_delta_ij, tx, ty, tw, th, te


def cmpute_loss(pre,target,insize,outsize,lambdas):

    MSE = nn.MSELoss()


    resp, conf, x, y, w, h, e = pre
    delta, max_delta_ij, tx, ty, tw, th, te = target

    B = resp.shape[0]

    lambda_resp,lambda_iou,lambda_coor,lambda_size,lambda_limb = lambdas
    outH, outW = outsize
    inH, inW = insize
    gridsize = int(inW / outW), int(inH / outH)

    (rx, ry), (rw, rh) = restore_xy(x, y,gridsize,outsize), restore_size(w, h,insize)  # 预测的位置
    (rtx, rty), (rtw, rth) = restore_xy(tx, ty,gridsize,outsize), restore_size(tw, th,insize)
    ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))  # 预测位置和真实位置的iou

    zero_place = torch.zeros(max_delta_ij.shape).type_as(max_delta_ij)
    zero_place[max_delta_ij < 0.5] = 0.0005
    weight_ij = torch.min(max_delta_ij + zero_place, torch.tensor([1.0]).type_as(delta))

    # add weight where can't find keypoint
    zero_place = torch.zeros(delta.shape).type_as(delta)
    zero_place[delta < 0.5] = 0.0005
    weight = torch.min(delta + zero_place, torch.tensor([1.0]).type_as(delta))

    half = torch.zeros(delta.shape).type_as(delta)  # 没有的地方填0.5
    half[delta < 0.5] = 0.5

    loss_resp = 1/B*torch.sum(torch.pow(resp - delta,2)) ## [delta-resp]
    loss_iou = 1/B*torch.sum(delta * torch.pow(conf - ious,2))
    loss_coor = 1/B*torch.sum(weight * (torch.pow(x - tx - half,2) + torch.pow(y - ty - half,2)))
    loss_size = 1/B*torch.sum(weight * (torch.pow(torch.sqrt(w + EPSILON) - torch.sqrt(tw + EPSILON),2) +
                                torch.pow(torch.sqrt(h + EPSILON) - torch.sqrt(th + EPSILON),2)))
    loss_limb = 1/B*torch.sum(weight_ij * torch.pow(e - te,2))

    loss = lambda_resp * loss_resp + \
           lambda_iou * loss_iou + \
           lambda_coor * loss_coor + \
           lambda_size * loss_size + \
           lambda_limb * loss_limb

    return {'loss':loss,
            'loss_resp': loss_resp,
            'loss_iou': loss_iou,
            'loss_coor': loss_coor,
            'loss_size': loss_size,
            'loss_limb': loss_limb}


def get_humans_by_feature( feature_map, insize,outsize,local_grid_size,detection_thresh=0.15,min_num_keypoints=-1):
    resp, conf, x, y, w, h, e = feature_map

    resp = resp.squeeze(0)
    conf = conf.squeeze(0)
    x = x.squeeze(0)
    y = y.squeeze(0)
    w = w.squeeze(0)
    h = h.squeeze(0)
    e = e.squeeze(0)

    #start = time.time()
    delta = resp * conf # 19,7,7
    ROOT_NODE = 0  # instance
    #start = time.time()
    outH, outW = outsize
    inH, inW = insize
    gridsize = int(inW / outW), int(inH / outH)

    rx, ry = restore_xy(x, y,gridsize,outsize)
    rw, rh = restore_size(w, h,insize)
    ymin, ymax = ry - rh / 2, ry + rh / 2 # 19.7.7
    xmin, xmax = rx - rw / 2, rx + rw / 2

    ymax = ymax.unsqueeze(0)
    ymin = ymin.unsqueeze(0)
    xmax = xmax.unsqueeze(0)
    xmin = xmin.unsqueeze(0)

    bbox = torch.cat([ymin, xmin, ymax, xmax]) #4,19,7,7
    bbox = bbox.permute(1, 2, 3, 0)
    root_bbox = bbox[ROOT_NODE] #
    score = delta[ROOT_NODE] # 7,7
    candidate = (score>detection_thresh).nonzero().t().tolist() # 实体的序号 两列 x，y
    #candidate = np.where(score > detection_thresh)
    score = score[candidate]
    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    #selected = selected.tolist()
    root_bbox = root_bbox[selected]
    #logger.info('detect instance {:.5f}'.format(time.time() - start))
    #start = time.time()
    candidate = torch.tensor(candidate)
    humans = []

    humans = []
    e = e.permute(0, 3, 4, 1, 2)  # 18 5 5 7 7 # 18 7 7 5 5
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # DIRECTED_GRAPHS = [[[0, 1, 2, 3], [1, 2, 3, 5]],
    #[[0, 1, 4, 5], [1, 2, 4, 6]],
    #[[0, 6, 7, 8], [1, 7, 9, 11]],
    #[[0, 9, 10, 11], [1, 8, 10, 12]],
    #[[0, 12, 13, 14], [1, 13, 15, 17]],
    #[[0, 15, 16, 17], [1, 14, 16, 18]]]
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])].cpu().numpy()}  # initial # 18 7 7 4  # 4 # 实例点
        for graph in DIRECTED_GRAPHS:
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                # u_ind = np.unravel_index(torch.argmax(e[index]), e[index].shape) # e[index] 5,5  求坐标
                u_ind = torch.argmax(e[index])/e[index].shape[1],torch.argmax(e[index])%e[index].shape[1]
                j_h = i_h + u_ind[0] - local_grid_size[1] // 2
                j_w = i_w + u_ind[1] - local_grid_size[0] // 2
                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)].cpu().numpy()
                i_h, i_w = j_h, j_w
        if min_num_keypoints <= len(human) - 1:
            humans.append(human)
    return humans


def draw_humans(keypoint_names, edges, pil_image, humans, mask=None, visbbox=True):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    #start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
            ymin, xmin, ymax, xmax = b
            if k == 0:  # human instance
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
            else:
                if visbbox:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
                else:
                    r = 2
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    drawer.ellipse((x - r, y - r, x + r, y + r),
                                   fill=COLOR_MAP[keypoint_names[k]])

        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=COLOR_MAP[keypoint_names[s]], width=3)

    #logger.info('draw humans {: .5f}'.format(time.time() - start))
    return pil_image

