#!/usr/bin/python3
import torch, os, cv2, sys
import argparse 
from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
# from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

def lane_detect_func(model, img_transform, src_img_f, des_img_f):
    
    src_img_0 = cv2.imread(src_img_f)
    img_w, img_h = src_img_0.shape[1], src_img_0.shape[0]
    src_img_1 = cv2.cvtColor(src_img_0, cv2.COLOR_BGR2RGB)
    src_img_1 = Image.fromarray(src_img_1)
    src_img_t = img_transform(src_img_1).cuda().unsqueeze(0)
    with torch.no_grad():
        out = model(src_img_t)
    
    col_sample = np.linspace(0, 800 - 1, args.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(args.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == args.griding_num] = 0
    out_j = loc
    
    vis_img = cv2.imread(src_img_f)
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 4:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    if i == 0:
                        cv2.circle(vis_img, ppp, 5, (255,0,0),-1)
                    elif i == 1:
                        cv2.circle(vis_img, ppp, 5, (0,255,0),-1)
                    elif i == 2:
                        cv2.circle(vis_img, ppp, 5, (0,0,255),-1)
                    elif i == 3:
                        cv2.circle(vis_img, ppp, 5, (255,255,255),-1)
                        
    vis_img = cv2.resize(vis_img, (img_w//2, img_h//2))
    cv2.imwrite(des_img_f, vis_img)

if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark = True
    
    script_dir = sys.path[0]
    
    opts = argparse.ArgumentParser("Parameter for UFLD method based lane detection.")
    # opts.add_argument("--dataset", default="Tusimple")
    # opts.add_argument("--model_path", default=os.path.join(script_dir, "weight", "tusimple_18.pth"))
    opts.add_argument("--dataset", default="CULane")
    opts.add_argument("--model_path", default=os.path.join(script_dir, "weight", "culane_18.pth"))
    opts.add_argument("--backbone", default='18')
    opts.add_argument("--griding_num", default=100)
    opts.add_argument("--src_img_f", default="/home/ztp/Downloads/Test_ColorImage_road05/ColorImage_road05/171206_064746439_Camera_6.jpg")
    # opts.add_argument("--src_img_f", default="/home/ztp/Downloads/benchmark_velocity_supp/supp_img/0042.jpg")

    args = opts.parse_args()

    if args.dataset == 'CULane':
        cls_num_per_lane = 18
        args.griding_num = 200
        row_anchor = culane_row_anchor
    elif args.dataset == 'Tusimple':
        cls_num_per_lane = 56
        args.griding_num = 100
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    
    net = parsingNet(pretrained = False, backbone=args.backbone,cls_dim = (args.griding_num+1,cls_num_per_lane,4),
                use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(args.model_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
            
    net.load_state_dict(compatible_state_dict, strict=False)
    net.cuda()
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    des_img_dir = "/home/ztp/Projects/qingdao/lane_test/"
    src_img_dir = "/home/ztp/Downloads/lane_marking_examples/road02/ColorImage/Record002/Camera_5"
    img_list = os.listdir(src_img_dir)
    print(len(img_list))
    src_img_list = []
    
    with torch.no_grad():
        for img_f in tqdm.tqdm(img_list):
            src_img_f = os.path.join(src_img_dir, img_f)
            des_img_f = os.path.join(des_img_dir, img_f)
            lane_detect_func(net, img_transforms, src_img_f, des_img_f)

            
            
    