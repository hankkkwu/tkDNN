from ctypes import *
import cv2
import numpy as np
import argparse
import os
from threading import Thread
import time

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("cl", c_int),
                ("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float),
                ("prob", c_float),
                ("name", c_char*20)]

class RESULT(Structure):
    _fields_ = [("dets", POINTER(DETECTION)),
                ("nboxes", c_int)]

lib = CDLL("./build/libdarknetTR.so", RTLD_GLOBAL)

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int, c_float]
load_network.restype = c_void_p

# copy the image from c_char_p to IMAGE
copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

# make a empty image array with the size of (w, h, c)
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, IMAGE]

do_batch_inference = lib.do_batch_inference
do_batch_inference.argtypes = [c_void_p, c_void_p, c_int]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

get_batch_boxes = lib.get_batch_boxes
get_batch_boxes.argtypes = [c_void_p]
get_batch_boxes.restype = POINTER(RESULT)


class Vector(object):
    lib = CDLL("./build/libdarknetTR.so", RTLD_GLOBAL)
    lib.new_vector.restype = c_void_p
    lib.new_vector.argtypes = []
    lib.delete_vector.argtypes = [c_void_p]
    lib.vector_push_back.argtypes = [c_void_p, IMAGE]

    def __init__(self):
        self.vector = Vector.lib.new_vector()  # pointer to new vector

    def __del__(self):  # when reference count hits 0 in Python
        # print("destructor called in c++ for deleting vector")
        Vector.lib.delete_vector(self.vector)  # call C++ vector destructor

    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def push(self, img):  # push calls vector's push_back
        Vector.lib.vector_push_back(self.vector, img)

    def get_vector(self):
        '''return the vector'''
        return self.vector

def resizePadding(image, height, width):
    '''not used currently'''
    desized_size = height, width
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] = new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image

def detect_image(net, darknet_image):
    '''not used currently'''
    num = c_int(0)
    pnum = pointer(num)
    
    do_inference(net, darknet_image)
    dets = get_network_boxes(net, 0, pnum)

    boxes = []
    scores = []
    classes = []
    for i in range(pnum[0]):
        b = dets[i].bbox
        boxes.append([b.x, b.y, b.x + b.w, b.y + b.h] * np.array([640,480,640,480], dtype=np.float32))  # (x0, y0, x1, y1)
        scores.append(dets[i].prob)
        classes.append(dets[i].cl)
        # res.append((dets[i].name.decode("ascii"), dets[i].prob, (b.x, b.y, b.w, b.h)))
    return boxes, scores, classes


def loop_detect(detect_m, video_path, batch_size):
    stream = cv2.VideoCapture(video_path)
    start = time.time()
    cnt = 0
    ret=True
    while stream.isOpened():
        image_batch = Vector()
        for _ in range(batch_size):
            ret, image = stream.read()
            if ret is False:
                break
            image = detect_m.preprocess(image)
            image_batch.push(image)
            cnt += 1
        if ret is False:
            break
        batch_boxes, batch_scores, batch_classes = detect_m.batch_detection(image_batch.get_vector())
        
    end = time.time()
    print("frame: {}, time: {:.3f}, FPS: {:.2f}".format(cnt, end-start, cnt/(end-start)))
    stream.release()


class YOLO4RT(object):
    def __init__(self,
                 input_size=608,    # input size for model
                 image_width=640,  # image size from camera
                 image_height=480,
                 weight_file='./yolo4_custom_fp16.rt',
                 batch_size=1,
                 num_classes = 80,
                 conf_thres=0.3):
        self.input_szie = input_size
        self.image_width = image_width
        self.image_height = image_height
        self.model = load_network(weight_file.encode("ascii"), num_classes, batch_size, conf_thres)
        self.n_batch = batch_size
        self.darknet_image = make_image(input_size, input_size, 3)
        self.thresh = conf_thres

    def detect(self, image):
        '''
        image: the input image has already resize to the size for the model (e.g. 608x608)
        
        output:
        boxes, scores, classes
        '''
        try:         
            frame_data = image.ctypes.data_as(c_char_p)
            copy_image_from_bytes(self.darknet_image, frame_data)

            boxes, scores, classes = detect_image(self.model, self.darknet_image)
            return boxes, scores, classes
        except Exception as e_s:
            print(e_s)

    def preprocess(self, image):
        '''
        resize the image for model input, and convert it for passing to C++
        '''
        darknet_image = make_image(self.input_szie, self.input_szie, 3)
        image = cv2.resize(image, (self.input_szie, self.input_szie), interpolation=cv2.INTER_LINEAR)
        frame_data = image.ctypes.data_as(c_char_p)
        copy_image_from_bytes(darknet_image, frame_data)
        return darknet_image

    def batch_detection(self, image_batch):
        do_batch_inference(self.model, image_batch, self.n_batch)
        result = get_batch_boxes(self.model)

        batch_boxes = []
        batch_scores = []
        batch_classes = []
        for bi in range(self.n_batch):
            boxes = []
            scores = []
            classes = []
            dets = result[bi].dets
            nboxes = result[bi].nboxes
            for i in range(nboxes):
                boxes.append([dets[i].x, 
                            dets[i].y, 
                            dets[i].x + dets[i].w, 
                            dets[i].y + dets[i].h] * np.array([self.image_width, self.image_height, self.image_width, self.image_height], dtype=np.float32))  # (x0, y0, x1, y1)
                scores.append(dets[i].prob)
                classes.append(dets[i].cl)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_classes.append(classes)
        # print("batch_boxes: ", batch_boxes)
        return batch_boxes, batch_scores, batch_classes

def parse_args():
    parser = argparse.ArgumentParser(description='tkDNN detect')
    parser.add_argument('weight', help='rt file path')
    parser.add_argument('--model_input', type=int, default=608, help='model input size')
    parser.add_argument('--image_width', type=int, default=640, help='width of image')
    parser.add_argument('--image_height', type=int, default=480, help='height of image')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_classes', type=int, default=80, help='number of classes')
    parser.add_argument('--video', type=str, help='video path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # python trt_inference.py build/yolo4_batch4_fp16.rt --batch_size=4 --num_classes=3 --video=demo/yolo_test.mp4

    args = parse_args()
    detect_m = YOLO4RT(input_size=args.model_input, image_width=args.image_width, image_height=args.image_height, 
                        weight_file=args.weight, batch_size=args.batch_size, num_classes=args.num_classes)
    t = Thread(target=loop_detect, args=(detect_m, args.video, args.batch_size), daemon=True)

    t.start()
    t.join()