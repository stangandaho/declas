# Copyright CNRS 2023

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

YOLO_WIDTH = 960 # image width
YOLO_THRES = 0.6
YOLOHUMAN_THRES = 0.4 # boxes with human above this threshold are saved
YOLOCOUNT_THRES = 0.6
model = 'deepfaune-yolov8s_960.pt'

####################################################################################
### BEST BOX DETECTION 
####################################################################################
class Detector:
    def __init__(self):
        self.yolo = YOLO(model)
        
    def bestBoxDetection(self, filename_or_imagecv, threshold=YOLO_THRES):
        try:
            results = self.yolo(filename_or_imagecv, verbose=False, imgsz=YOLO_WIDTH)
        except FileNotFoundError:
            return None, 0, np.zeros(4), 0, None
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            #raise
        # orig_img a numpy array (cv2) in BGR
        imagecv = results[0].cpu().orig_img
        detection = results[0].cpu().numpy().boxes
        if not len(detection.cls) or detection.conf[0] < threshold:
            # category = 0
            return None, 0, np.zeros(4), 0, None
        ## best box
        category = detection.cls[0] + 1
        count = sum(detection.conf>YOLOCOUNT_THRES) # only if best box > YOLOTHRES
        box = detection.xyxy[0] # xmin, ymin, xmax, ymax
        # is an animal detected ?
        if category != 1:
            croppedimage = None # indeed, not required for further classification
        # if yes, cropping the bounding box
        else:
            croppedimage = cropSquareCVtoPIL(imagecv, box.copy())
        ## count
        count = sum(detection.conf>YOLOCOUNT_THRES) # only if best box > YOLOTHRES
        ## human boxes
        ishuman = (detection.cls==1) & (detection.conf>=YOLOHUMAN_THRES)
        if any(ishuman==True):
            humanboxes = detection.xyxy[ishuman,]
        else:
            humanboxes = None
        return croppedimage, category, box, count, humanboxes

    def merge(self, detector):
        pass
    

####################################################################################
### TOOLS
####################################################################################      
'''
:return: cropped PIL image, as squared as possible (rectangle if close to the borders)
'''
def cropSquareCVtoPIL(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0,int(y1)):min(int(y2),height),max(0,int(x1)):min(int(x2),width)]
    croppedimage = Image.fromarray(croppedimagecv[:,:,(2,1,0)]) # converted to PIL BGR image
    return croppedimage
