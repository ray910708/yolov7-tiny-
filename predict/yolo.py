import colorsys
import os
import threading 
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import pyttsx3
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/best_epoch_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape"       : [640, 640],
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)
        
        self.speak_thread = None
        self.speak_queue = Queue()
        self.lock = threading.Lock()
        self.speak_thread = threading.Thread(target=self._speak)
        self.speak_thread.daemon = True
        self.speak_thread.start()
        
    def generate(self, onnx=False):
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #-----------預測功能------------#
    def detect_image(self, image, s, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_width = image.size[0]
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        try:
            with self.lock:
                image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
                with torch.no_grad():
                    images = torch.from_numpy(image_data)
                    if self.cuda:
                        images = images.cuda()
                    outputs = self.net(images)
                    outputs = self.bbox_util.decode_box(outputs)

                    results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                                image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                            
                    if results[0] is None: 
                        if s == 1:
                            self.speak("您未在人行道上")
                        return image

                    top_label   = np.array(results[0][:, 6], dtype = 'int32')
                    top_conf    = results[0][:, 4] * results[0][:, 5]
                    top_boxes   = results[0][:, :4]
                    
                font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
                #----------count和crop為圖片檢測功能--------------#
                if count:
                    print("top_label:", top_label)
                    classes_nums    = np.zeros([self.num_classes])
                    for i in range(self.num_classes):
                        num = np.sum(top_label == i)
                        if num > 0:
                            print(self.class_names[i], " : ", num)
                        classes_nums[i] = num
                    print("classes_nums:", classes_nums)
                if crop:
                    for i, c in list(enumerate(top_boxes)):
                        top, left, bottom, right = top_boxes[i]
                        top     = max(0, np.floor(top).astype('int32'))
                        left    = max(0, np.floor(left).astype('int32'))
                        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                        right   = min(image.size[0], np.floor(right).astype('int32'))
                        
                        dir_save_path = "img_crop"
                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        crop_image = image.crop([left, top, right, bottom])
                        crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                        print("save crop_" + str(i) + ".png to " + dir_save_path)
                #預測框繪製及語音提示
                labelzh={"person":"人", "car":"車輛", "cat":"貓", "dog":"狗", 
                         "bicycle":"腳踏車", "motorcycle":"機車", "sidewalk":"人行道"}
                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box             = top_boxes[i]
                    score           = top_conf[i]

                    top, left, bottom, right = box

                    top     = max(0, np.floor(top).astype('int32'))
                    left    = max(0, np.floor(left).astype('int32'))
                    bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                    right   = min(image.size[0], np.floor(right).astype('int32'))

                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')
                    
                    box_width_ratio = (right - left) / image_width
                    box_center_x = (left + right) / 2  
                    if box_width_ratio > 0.8:
                        location = '前方'
                    elif box_center_x > image_width * 0.66:
                        location = '右方'  
                    elif box_center_x < image_width * 0.33:
                        location = '左方'
                    else:
                        location = '前方'

                    if s == 1:
                        self.speak("檢測到"+labelzh[predicted_class]+"位於"+location)
                        print("檢測到"+labelzh[predicted_class]+"位於"+location)
                        if predicted_class == 'sidewalk':
                            if location == '右方':
                                print("提示人行道已偏移至右方")
                                self.speak("提示人行道已偏移至右方") 
                            elif location == '左方':
                                print("提示人行道已偏移至左方")
                                self.speak("提示人行道已偏移至左方")
                        else:
                            print(f"您未于人行道上")
                            self.speak(f"您未于人行道上")

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                    draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                    del draw
        except Exception as e:
            pass
        return image

    def speak(self, text):
        self.speak_queue.put(text)

    def _speak(self):
        engine = pyttsx3.init() 
        engine.setProperty('volume', 0.8)
        engine.setProperty('rate', 150)
        while True:
            text = self.speak_queue.get()
            engine.say(text)
            engine.runAndWait()