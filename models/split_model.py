from distutils.log import error
import torch
import torch.nn as nn
import copy
import cv2

from models.experimental import attempt_load
from utils.torch_utils import select_device
from torchinfo import summary
import models.yolo as yolo
from models.yolo import Detect, IDetect, IAuxDetect, IBin
import sys
from utils.plots import plot_one_box

import numpy as np
from PIL import Image
import thop
from numpy import random
import yaml
from utils.torch_utils import time_synchronized, copy_attr
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

class SplitModel(yolo.Model):
    def __init__(self, pretrained = None, cfg='cfg/deploy/yolov7.yaml', ch=3, nc=None, anchors=None, num_layers = 106 ) -> None:
        with open('forward_optimization.yaml', 'r') as file:
            self.forward_optimization = yaml.load(file, Loader=yaml.FullLoader)
        self.layers = num_layers
        super(SplitModel, self).__init__(cfg, ch, nc, anchors)
        if pretrained:
            self.load_weights_of_model(pretrained)

    def load_weights_of_model(self, input_model):
        copy_attr(input_model, self, include=('yaml', 'nc', 'names', 'stride', 'save'), exclude=())  # copy attributes
        self.fuse() # unify conv+bn and removes training weights
        self.model.load_state_dict(input_model.model.state_dict())

    def autoshape(self):
        raise Exception("Not implemented.")

    def forward(self, x, augment=False, profile=False, y_from_edge = None, start_layer = None, end_layer = None, split = None):
        st = 0 if start_layer is None else start_layer # default entry
        en = self.layers if end_layer is None else end_layer # default exit
        augment = False #lock this out for now, not prepared for split network 
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, st, en, y_from_edge, profile, split)  # single-scale inference, train

    def forward_once(self, x, st, en, y_from_edge, profile=False, split = None):
        y, dt = [], []  # outputs
        y = y_from_edge if y_from_edge is not None else []
        enter_val = st
        exit_val = en
        counter = 0
        for m in self.model:
            if exit_val < counter:
                break # exit early
            if enter_val <= counter:
                needed = set() # this could be preconstructed in forward_optimization.yaml for slight speed increase
                for i in self.forward_optimization[counter-1:]:
                    if i is not None:
                        for e in i:
                            needed.add(e)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if not hasattr(self, 'traced'):
                    self.traced=False

                if self.traced:
                    if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                        break

                if profile:
                    c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                # reduce size of y list for transfer to server                
                for i, y_i in enumerate(y):
                    if i not in needed:
                        y[i] = None
            counter += 1

        if profile:
            print('%.1fms total' % sum(dt))
        if split:
            return x, y # y must be available for reference if split, but we do not want to interfere with normal operation
        else:
            return x
