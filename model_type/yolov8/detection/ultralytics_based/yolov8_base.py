# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" YoloV8 base detector class. """

# Importing basic libraries
import os
from queue import Queue
import supervision as sv

from ultralytics.models import yolo
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_detector import BaseDetector
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data

from pathlib import Path
from PytorchWildlife import utils as pw_utils 
from sources.Bases import *

class YOLOV8Base(BaseDetector):
    """
    Base detector class for the new ultralytics YOLOV8 framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    This base detector class is also compatible with all the new ultralytics models including YOLOV9, 
    RTDetr, and more.
    """
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLOV8 detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        self.transform = transform
        super(YOLOV8Base, self).__init__(weights=weights, device=device, url=url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLOV8 model weights.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """

        self.predictor = yolo.detect.DetectionPredictor()
        # self.predictor.args.device = device # Will uncomment later
        self.predictor.args.imgsz = self.IMAGE_SIZE
        self.predictor.args.save = False # Will see if we want to use ultralytics native inference saving functions.

        if weights:
            self.predictor.setup_model(weights)
        elif url:
            raise Exception("URL weights loading not ready for beta testing.")
        else:
            raise Exception("Need weights for inference.")
        
        if not self.transform:
            self.transform = pw_trans.MegaDetector_v5_Transform(target_size=self.IMAGE_SIZE,
                                                                stride=self.STRIDE)

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (ultralytics.engine.results.Results): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        xyxy = preds.boxes.xyxy.cpu().numpy()
        confidence = preds.boxes.conf.cpu().numpy()
        class_id = preds.boxes.cls.cpu().numpy().astype(int)

        results = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"  
            for _, _, confidence, class_id, _ in results["detections"] 
        ]
        
        return results
        

    def single_image_detection(self, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img_path (str): 
                Image path or identifier.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        self.predictor.args.batch = 1
        self.predictor.args.conf = conf_thres
        det_results = list(self.predictor.stream_inference([img_path]))
        return self.results_generation(det_results[0], img_path, id_strip)



    def batch_image_detection(self, data_path, 
                              batch_size=16, 
                              conf_thres=0.2, 
                              id_strip=None, 
                              extension="JPG",
                              class_of_interest = "animal",
                              save_json = True,
                              log_queue = Queue()):
        """
        Perform detection on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference.
            batch_size (int, optional):
                Batch size for inference. Defaults to 16.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
            extension (str, optional):
                Image extension to search for. Defaults to "JPG"

        Returns:
            list: List of detection results for all images.
        """
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = conf_thres

        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transform,
            extension=extension
        )

        # Creating a DataLoader for batching and parallel processing of the images
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=0, drop_last=False
                            )
        
        results = []
        with tqdm(total=len(loader)) as pbar:
            for batch_index, (imgs, paths, sizes) in enumerate(loader):
                det_results = self.predictor.stream_inference(paths)
                batch_results = []
                for idx, preds in enumerate(det_results):
                    res = self.results_generation(preds, paths[idx], id_strip)
                    size = preds.orig_shape
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in res["detections"].xyxy]
                    res["normalized_coords"] = normalized_coords
                    results.append(res)
                pbar.update(1)

                if log_queue:
                    log_queue.put(f"Detection progress: {round((pbar.n/pbar.total)*100, 2)}%")
                results.extend(batch_results)


        
            result = {}
            # Iterate through the detection results
            for detection in results:
                img_path = detection['img_id']
                img_stem = Path(img_path).stem
                
                animal_count = sum([1 for label in detection['labels'] if class_of_interest in label])
                
                # Extract exif data
                image_id = Path(img_path).name # image name
                main_dir = load_json()
                to_save = dect_or_clf_dict(image_path=img_path, 
                                           image_id=image_id, 
                                           count = animal_count)
                if not main_dir["run_on_main_dir"]:
                    to_save["Station"] = Path(img_path).parent.parent.name
                    to_save["Species"] = Path(img_path).parent.name
  
        
                result[img_stem] = to_save

        Path(data_path, "detections").mkdir(exist_ok=True)
        out_dir = str(Path(data_path, "detections"))
        pw_utils.save_detection_images(results=results, 
                                       output_dir=out_dir, 
                                       input_dir=data_path,
                                       overwrite=False)
        
        if save_json:
            save_detection_json(save_dir=data_path, to_save=result)

        return result
