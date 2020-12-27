%reload_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
from fastai.callbacks.hooks import num_features_model
from torch.nn import L1Loss
import pandas as pd
from torch import optim
import re
import json
import cv2
import types
import fastai
#from fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
#master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
from PIL import Image
from utils import *
import os


path1 = Path('/content/drive/MyDrive/FaceDetection')
path2 = Path('/content/drive/MyDrive/emotion')

class StubbedObjectCategoryList(ObjectCategoryList):
    def analyze_pred(self, pred): return [pred.unsqueeze(0), torch.ones(1).long()]

class FaceDetector(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__() 
        self.cnn = create_body(arch)
        self.head = create_head(num_features_model(self.cnn) * 2, 4)
        
    def forward(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return 2 * (x.sigmoid_() - 0.5)

def loss_fn(preds, targs, class_idxs):
    return L1Loss()(preds, targs.squeeze())

def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

def area(boxes): 
    return ((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))

def union(preds, targs):
    return area(preds) + area(targs) - intersection(preds, targs)

def IoU(preds, targs):
    return intersection(preds, targs) / union(preds, targs)

def acc_detection(preds, targs, _):
    return IoU(preds, targs.squeeze()).mean()

metrics = acc_detection

learn1 = load_learner(path1,file='facedetec.pkl')
learn2 = load_learner(path2,file='emotion.pkl')

def draw_bbox(img, bbox, target=None, color=(255, 0, 0), thickness=2):
     y_min, x_min, y_max, x_max = map(int, bbox)
     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
     #if target is not None:
      #   y_min, x_min, y_max, x_max = map(int, target)
      #   cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0,255,0), thickness=thickness)
     return img
def cv_read(path):    
     im = cv2.imread(path, cv2.IMREAD_COLOR)
     im1 = cv2.resize(im,(224,224))
     return cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

def img_fastai(img):
   return vision.Image(pil2tensor(img,np.float32).div_(255))

def find_bbox(img,learn):
   input = img_fastai(img)
   lb,pred_idx,preds = learn.predict(input)
   SZ=224
   predicted_bboxes = ((preds + 1) / 2 * SZ).numpy()
   ims = np.stack([draw_bbox(img, predicted_bboxes)])
   return ims

def emotion_detect(img,learn):
   lbel= ['happy','neutral','sad']
   input = img_fastai(montage(np.stack(img), multichannel=True))
   lb,pred_idx,preds = learn.predict(input)
   return lbel[np.argmax(lb,axis=0)]

img1 = cv_read('/content/pem1.jpg')

img1.shape

bbox = find_bbox(img1,learn1)

emotion = emotion_detect(bbox,learn2)

plt.title(emotion)
plt.imshow(montage(np.stack(bbox), multichannel=True))