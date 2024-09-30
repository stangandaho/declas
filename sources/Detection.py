from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from pathlib import Path
import os, sys
from PIL import Image
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from Bases import *
from model_type.yolov8.detection.ultralytics_based import MegaDetectorV6


# Function to get result
def single_detections(weights = "", image_path = "", conf_thres = 0.55, 
                model_type = 'MegaDetectorV5', imgsz = (1920, 1440),
                device="cpu", class_of_interest = "animal", save_json = True):
    
    img = np.array(Image.open(image_path).convert("RGB"))
    if model_type == 'YoloV5':
    
        detection_model = pw_detection.MegaDetectorV5(weights=weights, device=device, pretrained=False)
        transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                    stride=detection_model.STRIDE)
        results = detection_model.single_image_detection(transform(img), img.shape, 
                                                    image_path, conf_thres=conf_thres)
        
    elif model_type in ["YoloV8/9"]:
        detection_model = MegaDetectorV6(weights=weights, device=device, pretrained=False)
        transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                stride=detection_model.STRIDE)
        results = detection_model.single_image_detection(img_path=image_path, 
                                                         conf_thres=conf_thres)

        
    output_dir = Path(Path(image_path).parent, "detections")
    output_dir.mkdir(exist_ok=True)
    
    # Save annotated image
    save_detection_images(results=results, output_dir=str(output_dir))
    
    image_id = Path(results.get('img_id')).name # image name
    cn = class_number(results = results, category=class_of_interest)

    to_save = dect_or_clf_dict(image_path=results.get('img_id'), image_id=image_id, count=cn)
    
    if save_json:
        save_detection_json(save_dir=str(Path(image_path).parent), to_save=to_save)

    return to_save



# BATCH
import PytorchWildlife.data as pw_data
from torch.utils.data import DataLoader
import PytorchWildlife.utils as pw_utils
from model_type.bases.detection import PW_baseDetector as pw_detection

def batch_detections(weights = "", data_path = "", conf_thres = 0.55, 
                model_type = 'MegaDetectorV5', imgsz = (1920, 1440),
                device="cpu", class_of_interest = "animal", save_json = True,
                log_queue = None):
    
    if model_type == 'YoloV5':
        detection_model = pw_detection.MegaD5(weights=weights, device=device, pretrained=False)
        transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                    stride=detection_model.STRIDE)
        
        det_dataset = pw_data.DetectionImageFolder(data_path, transform=transform)
        det_loader = DataLoader(det_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)

        det_results = detection_model.batch_detections(det_loader, conf_thres=conf_thres, log_queue=log_queue)

        # Save
        
        Path(data_path, "detections").mkdir(exist_ok=True)
        out_dir = str(Path(data_path, "detections"))
        pw_utils.save_detection_images(results=det_results, 
                                       output_dir=out_dir, 
                                       input_dir=data_path,
                                       overwrite=False)

        # JSON file to save
        to_save = {}
        main_dir = load_json()
        for rsl in det_results:
            keys = Path(rsl["img_id"]).stem
            animal_count = sum([1 for label in rsl['labels'] if class_of_interest in label])
            det_r = dect_or_clf_dict(image_path=rsl["img_id"], 
                             image_id=Path(rsl["img_id"]).name, 
                             count=animal_count)
            ## Add station and species from directory
            if not main_dir["run_on_main_dir"]:
                det_r["Station"] = Path(rsl["img_id"]).parent.parent.name
                det_r["Species"] = Path(rsl["img_id"]).parent.name
            to_save[keys] = det_r

        save_detection_json(save_dir=data_path, to_save=to_save)

    elif model_type == "YoloV8/9":
            detection_model = MegaDetectorV6(weights=weights,
                                             device=device, 
                                            pretrained=False)

            to_save = detection_model.batch_image_detection(data_path = data_path,
                                                batch_size=16, 
                                                conf_thres=conf_thres,
                                                class_of_interest=class_of_interest,
                                                save_json=save_json,
                                                extension="JPG",
                                                log_queue = log_queue)
            
    else:
        pass
    
    return to_save


# Function to get unique category from detection class
def get_unique(list):
    l = []
    for x in list:
        if x not in l:
            l.append(x)
    return l

