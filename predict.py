import model
import torch
import argparse
import mpii_dataset
import config
import utils
from PIL import Image
from torchvision.transforms import ToTensor,Resize,Normalize


def opt():
    arg = argparse.ArgumentParser()
    arg.add_argument("--insize",type=tuple,default=config.insize)
    arg.add_argument("--img",type=str,
        default="/home/mllabs/NewDisk/yww/VOC2000_LYC_20181025_16/train2017/HXD1C6082_济南若临_10_二端司机室_20181016_170041_06951.jpg")
    arg.add_argument("--local_grid_size",type=tuple,default=config.local_grid_size)
    arg.add_argument("--checkpoint",type=str,default="model/best.pth")

    return arg.parse_args()

def main():
    CUDA = torch.cuda.is_available()
    opts = opt()
    pose_model = model.poseprosalnet(mpii_dataset.KEYPOINT_NAMES,
                                     mpii_dataset.EDGES,
                                     opts.local_grid_size,
                                     opts.insize)
    pose_model.load_state_dict(torch.load(opts.checkpoint))

    pil_image = Image.open(opts.img)
    pil_image = Resize(opts.insize)(pil_image)
    img = ToTensor()(pil_image)
    img = Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])(img)

    img = img.unsqueeze(0)

    pose_model.eval()
    print(mpii_dataset.KEYPOINT_NAMES)
    with torch.no_grad():
        if CUDA:
            pose_model.cuda()
            img = img.cuda()

        pre = pose_model(img)

        humans = utils.get_humans_by_feature(pre,opts.insize,pose_model.outsize,opts.local_grid_size,)
        # print(humans[0])
        # print(humans[1])
        pil_image = utils.draw_humans(mpii_dataset.KEYPOINT_NAMES, mpii_dataset.EDGES, pil_image, humans, visbbox=False)

        pil_image.save("result.png","PNG")

if __name__=="__main__":
    main()