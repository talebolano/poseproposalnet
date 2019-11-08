import config
import argparse
import torch
import model
import utils
import torch.nn as nn
import dataset
import mpii_dataset
import numpy as np
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import apex
from apex import amp
import json

def parse():
    argparses = argparse.ArgumentParser(description="pose proposal network")
    argparses.add_argument("--train_json",default=config.train_json,type=str)
    argparses.add_argument("--val_json",default=config.val_json,type=str)
    argparses.add_argument("--image_root", type=str, default=config.image_root)
    argparses.add_argument("--epochs",default=config.epochs,type=int)
    argparses.add_argument("--batchsize", default=config.batchsize, type=int)
    argparses.add_argument("--lr",default=config.lreaning_rate,type=float)
    argparses.add_argument("--momentum", default=config.momentum, type=float)
    argparses.add_argument("--weight_decay", default=config.weight_decay, type=float)
    argparses.add_argument("--local_grid_size",default=config.local_grid_size,type=tuple)
    argparses.add_argument("--seed",type=int,default=config.seed)
    argparses.add_argument("--insize", type=tuple, default=config.insize)
    argparses.add_argument("--parts_scale", default=config.parts_scale, type=float)
    argparses.add_argument("--checkpoint", type=str, default=config.checkpoint)
    argparses.add_argument("--num_workers",type=int,default=config.num_workers)
    argparses.add_argument("--val_epochs", type=int, default=config.val_epochs)
    argparses.add_argument("--pretrain_model", type=str, default=None)


    return argparses.parse_args()


def main():
    opt = parse()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = True
    CUDA = torch.cuda.is_available()

    pose_model = model.poseprosalnet(mpii_dataset.KEYPOINT_NAMES,
                                     mpii_dataset.EDGES, opt.local_grid_size,
                                     opt.insize,
                                     pretrained=True)

    traindataset = dataset.get_mpii_dataset(opt.insize,pose_model.outsize,
                                            opt.local_grid_size,
                                            opt.parts_scale,opt.image_root,opt.train_json,
                                            do_augmentation=True)

    valdataset = dataset.get_mpii_dataset(opt.insize,pose_model.outsize,
                                            opt.local_grid_size,
                                            opt.parts_scale,opt.image_root,opt.val_json,
                                            do_augmentation=False)
    val_json = json.load(open(opt.val_json,"r"))

    traindataloader = DataLoader(traindataset,opt.batchsize,shuffle=True,num_workers=opt.num_workers,
                                 pin_memory=True,drop_last=True)


    valdataloader = DataLoader(valdataset,1,shuffle=False,num_workers=opt.num_workers)


    optim = torch.optim.SGD(pose_model.parameters(),opt.lr,momentum=opt.momentum,weight_decay=opt.weight_decay)


    lf = lambda x: (1-x/opt.epochs)
    scheduler = lr_scheduler.LambdaLR(optim,lr_lambda=lf)

    if opt.pretrain_model:
        pose_model.local_grid_size(torch.load(opt.pretrain_model))

    if CUDA:
        pose_model.cuda()

    lambdas = (config.lambda_resp,config.lambda_iou,config.lambda_coor,config.lambda_size,config.lambda_limb)
    #outsize = opt.insize[0]//32, opt.insize[1]//32
    #############################################

    #pose_model,optim = amp.initialize(pose_model,optim,opt_level="O1")
    write = SummaryWriter()

    best_loss = float('inf')
    for epoch in range(opt.epochs):
        pose_model.train()
        scheduler.step()
        # for x in optim.param_groups:
        #     print(x["lr"])
        for iter,in_data in enumerate(traindataloader):

            if CUDA:
                in_data["image"] = in_data["image"].cuda()
                in_data["delta"] = in_data["delta"].cuda()
                in_data["max_delta_ij"] = in_data["max_delta_ij"].cuda()
                in_data["tx"] = in_data["tx"].cuda()
                in_data["ty"] = in_data["ty"].cuda()
                in_data["tw"] = in_data["tw"].cuda()
                in_data["th"] = in_data["th"].cuda()
                in_data["te"] = in_data["te"].cuda()

            pre = pose_model(in_data["image"])
            target = (in_data["delta"],in_data["max_delta_ij"], in_data["tx"],
                      in_data["ty"], in_data["tw"], in_data["th"], in_data["te"] )
            losses = utils.cmpute_loss(pre,target,opt.insize,pose_model.outsize,lambdas)


            # with amp.scale_loss(losses["loss"],optim) as scaled_loss:
            #     scaled_loss.backward()

            losses["loss"].backward()
            optim.step()
            optim.zero_grad()
            print("epoch:{}, iter:{}, all_loss:{}, loss_resp:{},loss_iou:{},loss_coor:{},loss_size:{},loss_limb:{}".format(
                epoch+1,iter+1,float(losses["loss"]),float(losses["loss_resp"]),float(losses["loss_iou"]),float(losses["loss_coor"]),
                float(losses["loss_size"]),float(losses["loss_limb"])
            ))
            write.add_scalar('scalar/loss',float(losses["loss"]),len(traindataloader)*epoch+iter)

        if (epoch+1)%opt.val_epochs==0:
            all_humans = {}
            with torch.no_grad():
                losses_all = 0
                pose_model.eval()
                for iter,in_data in enumerate(valdataloader):
                    if CUDA:
                        in_data["image"] = in_data["image"].cuda()
                        in_data["delta"] = in_data["delta"].cuda()
                        in_data["max_delta_ij"] = in_data["max_delta_ij"].cuda()
                        in_data["tx"] = in_data["tx"].cuda()
                        in_data["ty"] = in_data["ty"].cuda()
                        in_data["tw"] = in_data["tw"].cuda()
                        in_data["th"] = in_data["th"].cuda()
                        in_data["te"] = in_data["te"].cuda()

                    pre = pose_model(in_data["image"])


                    target = (in_data["delta"], in_data["max_delta_ij"], in_data["tx"],
                                  in_data["ty"], in_data["tw"], in_data["th"], in_data["te"])
                    losses = utils.cmpute_loss(pre, target, opt.insize, pose_model.outsize, lambdas)
                    losses_all =losses_all+float(losses["loss"])

                    # resp, conf, x, y, w, h, e = pre
                    # resp = resp.squeeze(0)
                    # conf = conf.squeeze(0)
                    # x = x.squeeze(0)
                    # y = y.squeeze(0)
                    # w = w.squeeze(0)
                    # h = h.squeeze(0)
                    # e = e.squeeze(0)
                    # pre = resp, conf, x, y, w, h, e
                    # humans = utils.get_humans_by_feature(pre, opt.insize,
                    #                     pose_model.outsize,opt.local_grid_size)
                    # #utils.draw_humans(coco_dataset.KEYPOINT_NAMES,coco_dataset.EDGES,)
                    # all_humans[in_data["image_name"]] = humans
                losses_all = losses_all/(iter+1)

                if losses_all<best_loss:
                    best_loss = losses_all
                    torch.save(pose_model.state_dict(),opt.checkpoint+"best.pth")
                print("val_loss:{}".format(losses_all))
                write.add_scalar('scalar/val_loss', float(losses_all), epoch)

                #delta, "max_delta_ij": max_delta_ij,
                #"tx": tx, "ty": ty, "tw": tw, "th": th, "te": te






if __name__ =='__main__':
    main()