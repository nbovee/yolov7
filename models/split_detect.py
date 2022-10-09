
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torchinfo import summary

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import split_model


opt = None # global options

def initialize():
    global opt
    opt.trace = not opt.no_trace
    opt.save_img = not opt.nosave and not opt.source.endswith('.txt')  # save inference images
    # Initialize
    # set_logging()
    opt.device = select_device(str(opt.device))
    opt.half = opt.device != 'cpu'  # half precision only supported on CUDA
    pretrained = attempt_load('yolov7.pt', map_location=opt.device) # need pretrained weights
    model = split_model.SplitModel(pretrained) # load pretrained weights into split capable model
    model.eval()
    model.to(opt.device)
    opt.stride = int(model.stride.max())  # model stride
    opt.img_size = check_img_size(opt.img_size, s=opt.stride)  # check img_size

    if opt.trace:
        model = TracedModel(model, device, opt.img_size)

    if opt.half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    webcam = False
    source = 'inference/images/horses.jpg'
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=opt.img_size, stride=opt.stride)
    else:
        dataset = LoadImages(source, img_size=opt.img_size, stride=opt.stride)

    # Get names and colors
    opt.names = model.module.names if hasattr(model, 'module') else model.names
    opt.colors = [[random.randint(0, 255) for _ in range(3)] for _ in opt.names]

    return model, dataset

def warmup_model(model, iterations):
    global opt
    if opt.device.type != 'cpu':
        for i in range(iterations):
            model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(opt.device).type_as(next(model.parameters())))  # run once

def process_image(dataset):
    global opt
    t = None
    for i in dataset:
        t = i # hacky but works here
    path, img, im0s, vid_cap = t
    img = torch.from_numpy(img).to(opt.device)
    img = img.half() if opt.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # for path, img, im0s, vid_cap in dataset :
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    return img

def inference(model, img, start_index, exit_index, s, y_n = None):
    global opt
    # Predict
    pred = model(img, augment=opt.augment, start_layer = start_index, end_layer = exit_index, split = s, y_from_edge = y_n)
    # Apply NMS
    return pred

def NMS(pred):
    global opt
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    # Apply Classifier
    classify = False
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
    return pred

def process_detections(pred):    
    global opt
    opt.webcam = False
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if opt.webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            s, frame = '', getattr(dataset, 'frame', 0)
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                s += (('%g ' * len(line)).rstrip() % line + '\n')
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{opt.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    return s


def full_detect(opt):
    # initialize required values
    m, d = initialize()

    # warmup inferences for GPU
    warmup_model(m, 10)

    # load image information
    img = process_image(d) #for this test, we only have a single image in our dataset. In future we will stream tensors.

    # Inference
    preds = inference(m, img)

    # Process Detections
    # relies on having source image, client only for now
    # reply_val = process_detections(preds)

    # return stuff
    # for full - print pre-processesing, inference, & post-processing time, pickled size of response

def edge_detect(opt):
    # load img
    # preprocesss
    # infer
    # pickle and upload
    # return stuff
    
    # for partial-edge - print pre-processing, inference, transfer time, pickled size of x, pickled size of y
    pass

def cloud_detect(opt):
    # unpickle input
    # infer
    # process outputs
    # return stuff
    # for partial-cloud - print inference, post-processing, response time, pickled size of response
    pass

if __name__ == '__main__':
    # global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default = False, help='display results')
    parser.add_argument('--save-txt', default = False, help='save results to *.txt')
    parser.add_argument('--save-conf', default = False, help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default = True, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default = False, help='augmented inference')
    parser.add_argument('--update', default = True, help='update all models')
    parser.add_argument('--split', default = False, help='is this model split into parts?')
    parser.add_argument('--split-side', type = str, default = 'edge', help='what side of the model is this?')
    parser.add_argument('--split-index', default = 0, help='where is the model split')
    # parser.add_argument('--edge-input-tensor', default = True, help='does the edge computer process the input image')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='ex', help='save results to project/name')
    parser.add_argument('--exist-ok', default = True, help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', default = True, help='don`t trace model')
    opt = parser.parse_args()
    opt.trials = 50
    opt.edge_processing = 0
    opt.edge_inference = 0
    opt.edge_transfer = 0

    opt.cloud_inference = 0
    opt.cloud_processing = 0
    opt.clound_transfer = 0
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        # local only
        # uploaded image
        # uploaded input tensor
        # iterate over various split layers
        m, d = initialize()
        print(m.save)
        # warmup inferences for GPU
        warmup_model(m, 10)
        import pickle
        import copy
        for split_layer in range(106):
            opt.split = True
            opt.split_index = split_layer
            t_process, t_infer1, t_infer2 = 0, 0, 0
            x_size, y_size = 0, 0
            iter = 50
            for i in range(iter):
                # load image information
                t0 = time_synchronized()
                img = process_image(d) #for this test, we only have a single image in our dataset. In future we will stream tensors.
                t1 = time_synchronized()
                # Inference
                x, y = inference(m, img, 0, split_layer, True)
                t2 = time_synchronized()
                preds = inference(m, x, split_layer + 1, 9999, False, y_n = copy.deepcopy(y))[0]
                preds = NMS(preds)
                t3 = time_synchronized()
                t_process += (t1 - t0)
                t_infer1 += (t2 - t1)
                t_infer2 += (t3 - t2)
                x_size = len(pickle.dumps(x))
                y_size = len(pickle.dumps(y))
                # print(len(pickle.dumps(preds)))
                # entirely server detected
            print(f"{x_size}\t{y_size}\t{t_process/iter:.04f}\t{t_infer1/iter:.04f}\t{t_infer2/iter:.04f}")










# summary(model, depth = 3, input_size = (1, 3, 640, 640), col_names = ['input_size', "kernel_size", "output_size"])

# python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg