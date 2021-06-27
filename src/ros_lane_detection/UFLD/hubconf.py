"""UFLD PyTorch Hub models
Usage:
    import torch
    model = torch.hub.load('')
"""
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
from data.constant import culane_row_anchor, tusimple_row_anchor

def _create(dataset="CULane", backbone='18', pretrained=True, model_path=None):
    """Creates a specified UFLD model

    Arguments:
        name (str): name of model, i.e. 'UFLD'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters
    """
    pass

    if dataset == 'CULane':
        cls_num_per_lane = 18
        griding_num = 200
        row_anchor = culane_row_anchor
    elif dataset == 'Tusimple':
        cls_num_per_lane = 56
        griding_num = 100
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    
    model = parsingNet(pretrained = False, backbone=backbone, cls_dim = (griding_num+1,cls_num_per_lane,4),
                use_aux=False).cuda()
    
    if pretrained and model_path is not None:
        state_dict = torch.load(model_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
                
        model.load_state_dict(compatible_state_dict, strict=False)
        
        return model, cls_num_per_lane, griding_num, row_anchor
    else:
        return model, cls_num_per_lane, griding_num, row_anchor
    
def CULane(dataset="CULane", backbone='18', pretrained=True, model_path=None):
    
    return _create(dataset, backbone, pretrained, model_path)

def Tusimple(dataset="Tusimple", backbone='18', pretrained=True, model_path=None):

    return _create(dataset, backbone, pretrained, model_path)

if __name__ == '__main__':
    model, row_anchor = _create(dataset="CULane", backbone='18', gridding_num=200, pretrained=True, model_path="./weight/culane_18.pth")
    print(row_anchor)