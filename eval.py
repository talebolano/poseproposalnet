import json
from mpii_dataset import *
import model
import numpy as np
import config
import torch
import cv2
import PIL.Image as Image
import utils
from torchvision.transforms import Resize,ToTensor,Normalize,Pad
from dataaugment import resize_point,pad_keypoint


def evaluation( list_):
    #dataset_type = config.get('dataset', 'type')
    # gt_key points
    gt_kps_list = list_[0]
    # humans
    humans_list = list_[1]
    # gt bboxs
    gt_bboxs_list = list_[2]

    # is_visible
    is_visible_list = list_[3]



    # prediction bboxes list
    pred_bboxs_list = []

    kps_names = ['head_top', 'upper_neck', 'l_shoulder', 'r_shoulder', 'l_elbow',
                 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    # 将humans转化成gt_kps的形式
    pred_kps_list = []
    for humans in humans_list:
        pred_bboxs = []
        pred_kps = []
        # humans maybe have several person
        for person in humans:
            item_pred = []
            pred_bboxs.append(person[0]) # 预测的头部/身体边框
            for num in range(1,15):
                if num in person:
                    y = (person[num][0] + person[num][2]) / 2
                    x = (person[num][1] + person[num][3]) / 2
                    item_pred.append([y, x])
                else:
                    item_pred.append([0,0])
            pred_kps.append(item_pred) # 表示一张图片上的所有人的关键点
        pred_kps_list.append(pred_kps)
        pred_bboxs_list.append(pred_bboxs)



    factor = 0.5 * 0.6

    # pred_boxs_list and gt_boxs_list's center point
    pred_cp_list = []
    gt_cp_list = []
    for item in pred_bboxs_list:
        pred_cp = []
        for per_people in item:
            ymin, xmin, ymax, xmax = per_people
            cp_x = xmin + (xmax-xmin)/2
            cp_y = ymin + (ymax-ymin)/2
            pred_cp.append((cp_x,cp_y))
        pred_cp_list.append(pred_cp)
    for item in gt_bboxs_list:
        gt_cp = []
        for per_people in item:
            rx, ry, rw, rh = per_people
            cp_x = rx+rw/2
            cp_y = ry+rh/2
            gt_cp.append((cp_x,cp_y))
        gt_cp_list.append(gt_cp)

    results = {}

    for k in range(len(gt_cp_list)):
        pred_cp, gt_cp, pred_kps, gt_kps, gt_boxs,is_visible = pred_cp_list[k], gt_cp_list[k], pred_kps_list[k], gt_kps_list[k], gt_bboxs_list[k], is_visible_list[k]

        # find out a matched person if image have several person
        for p_index in range(len(pred_cp)):
            p_cp = pred_cp[p_index]
            dists = [np.linalg.norm(np.array(p_cp) - np.array(g_cp)) for g_cp in gt_cp]
            if dists == []:
                continue

            g_index = dists.index(min(dists))

            # select matched person
            pred_kps_item = pred_kps[p_index]
            gt_kps_item = gt_kps[g_index]
            is_visible_item = is_visible[g_index]


            gt_box = gt_boxs[g_index]
            h,w = gt_box[2:]
            length = np.sqrt((pow(h,2) + pow(w, 2))) * factor


            for i in range(len(kps_names)):
                name = kps_names[i]

                if not results.get(name):
                    results[name] = []

                is_v = is_visible_item[i]

                if is_v == 0 and i!=0 :
                    continue

                pred_point = pred_kps_item[i]
                gt_point = gt_kps_item[i]
                # compute pred gt point distance
                distance_pt = np.linalg.norm(np.array(pred_point)-np.array(gt_point))


                if distance_pt < length:
                    results[name].append(1)
                else:
                    results[name].append(0)

            # 单人检测，detect single person in a image
            break



    # accuracy
    total_results = []
    head = []
    shoulder = []
    ankle = []
    elbow = []
    wrist = []
    hip = []
    knee = []
    for item in kps_names:
        total_results += results[item]
        if 'knee' in item:
            knee += results[item]
        elif 'shoulder' in item:
            shoulder += results[name]
        elif 'ankle' in item:
            ankle += results[item]
        elif 'elbow' in item:
            elbow += results[item]
        elif 'wrist' in item:
            wrist += results[item]
        elif 'hip' in item:
            hip += results[item]
        else:
            head += results[item]


    print(len(head), len(hip))
    pck = np.sum(total_results)/len(total_results)
    pck_head = np.sum(head)/len(head)
    pck_shoulder = np.sum(shoulder)/len(shoulder)
    pck_ankle = np.sum(ankle)/len(ankle)
    pck_elbow = np.sum(elbow)/len(elbow)
    pck_wrist = np.sum(wrist)/len(wrist)
    pck_hip = np.sum(hip)/len(hip)
    pck_knee = np.sum(knee)/len(knee)

    pck, pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip, pck_knee = round(pck,2), round(pck_head,2), round(pck_shoulder, 2), round(pck_ankle, 2), round(pck_elbow, 2), round(pck_wrist, 2), round(pck_hip, 2), round(pck_knee, 2)

    print('the total pck is: {}'.format(pck))
    print('head: {}\tshoulder: {}\tankle: {}\telbow: {}\twrist: {}\thip: {}\n\n'.format(pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip))




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--insize",default=config.insize,type=tuple)
    parser.add_argument("--local_grid_size",default=config.local_grid_size,type=tuple)
    parser.add_argument("-n", "--testnum", help="the number of test image", type=int, default=1000, dest='test_num')
    parser.add_argument("--val_json",default=config.val_json,type=str)
    parser.add_argument("--image_root",default=config.image_root,type=str)
    parser.add_argument("--checkpointy",type=str)
    args = parser.parse_args()


    val_json = json.load(open(args.val_json,"r"))
    images = {}
    for image in val_json["images"]:
        images[int(image['id'])] = image['file_name'], [], [], [], []
    for anno in val_json['annotations']:  # 遍历所有anno
        if anno['num_keypoints'] < 1:
            continue
        if anno['iscrowd'] != 0:
            continue
        image_id = int(anno['image_id'])
        d = np.array(anno['keypoints'], dtype='float32').reshape(-1, 3)

        keypoints = d[:, [1, 0]]  # array of y,x  获得每个点的坐标 # 为啥要反过来？
        bbox = anno['bbox']  # 人体框
        is_visible = d[:, 2] == 2  # 是否可见
        is_labeled = d[:, 2] >= 1  # 是否有标签

        entry = images[image_id]  # 得到
        entry[1].append(np.asarray(keypoints))
        entry[2].append(np.asarray(bbox, dtype='float32'))
        entry[3].append(np.asarray(is_visible).astype(np.bool))
        entry[4].append(np.asarray(is_labeled).astype(np.bool))
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

    pose_proposal_net = model.poseprosalnet(KEYPOINT_NAMES,EDGES,args.local_grid_size,args.insize,
                                            args.checkpoint)
    pose_proposal_net.eval()
    CUDA = torch.cuda.is_available()
    if CUDA:
        pose_proposal_net.cuda()
    pck_object = [[], [], [], []]
    for i in range(len(image_paths)):
        image = Image.open(args.image_root+image_paths[i])
        oriW,oriH = image.size

        h_pad = int(np.clip(((max(oriH, oriW) - oriH) + 1) // 2, 0, 1e6))  # 填充在两边
        w_pad = int(np.clip(((max(oriH, oriW) - oriW) + 1) // 2, 0, 1e6))
        image = Pad((w_pad,h_pad))(image)

        padedWH = max(oriH,oriW)

        image = Resize(args.insize)(image)
        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0)
        pre = pose_proposal_net(image)



        ####################pad#######################
        paded_keypoints = [
            pad_keypoint(points, h_pad, w_pad)
            for points in keypoints[i]
        ]

        paded_bbox = []
        for x, y, bw, bh in bbox[i]:
            [[y, x]] = pad_keypoint(np.array([[y, x]]), h_pad, w_pad)
            paded_bbox.append(np.array([x, y, bw, bh]))
        ###################resize#############################
        new_keypoints = [
            resize_point(points, (padedWH, padedWH), args.insize)
            for points in paded_keypoints
        ]
        new_bbox = []
        for x, y, bw, bh in paded_bbox:
            [[y, x]] = resize_point(np.array([[y, x]]), (padedWH, padedWH), args.insize)
            bw *= args.insize[1] / padedWH
            bh *= args.insize[0] / padedWH
            new_bbox.append(np.array([x, y, bw, bh], dtype='float32'))


        humans = utils.get_humans_by_feature(pre, args.insize, pose_proposal_net.outsize, args.local_grid_size, )
        pck_object[0].append(new_keypoints)
        pck_object[1].append(humans)
        pck_object[2].append(new_bbox)
        pck_object[3].append(is_visible[i])
    evaluation(pck_object)