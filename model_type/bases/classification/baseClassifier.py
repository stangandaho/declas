
# Importing basic libraries
from queue import Queue
import os, sys
from pathlib import Path
import cv2
from PIL import Image
import pandas
import supervision as sv
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

#from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 
from PytorchWildlife.models.classification import PlainResNetInference

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root_dir)
# Importing the models, dataset, transformations, and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from model_type.bases.classification import PW_baseClassifier as pw_classification
from model_type.bases.detection import PW_baseDetector as pw_detection

from sources.Bases import dect_or_clf_dict, exif_table, save_detection_images, save_detection_json

class baseClassifier:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.detection_model = None
        self.classification_model = None
        self.trans_det = None
        self.trans_clf = None

    def load_models(self, det_weight, clf_weight, device):
        # Load detection model
        self.detection_model = pw_detection.MegaD5(weights=det_weight, device=device, pretrained=False)
        
        # Try loading the classification models one by one
        try:
            self.classification_model = pw_classification.Amazon(weights=clf_weight, device=device, pretrained=False)
        except Exception as e_amazon:
            try:
                self.classification_model = pw_classification.Serengeti(weights=clf_weight, device=device, pretrained=False)
            except Exception as e_serengeti:
                raise RuntimeError(f"Failed to load classification models: {e_amazon}, {e_serengeti}")

        # Load transformations for detection and classification
        self.trans_det = pw_trans.MegaDetector_v5_Transform(target_size=self.detection_model.IMAGE_SIZE, stride=self.detection_model.STRIDE)
        self.trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

        return "Loaded models successfully"


    @staticmethod
    def np_image(img_path):
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    @staticmethod
    def summarise_classification(clf_result, image_path):
        class_dict = {}
        for item in clf_result:
            class_name = item.split()[0]
            if class_name in class_dict:
                class_dict[class_name] += 1
            else:
                class_dict[class_name] = 1

        all_cfl = {}
        image_id = Path(image_path).stem
        for jtem in class_dict:
            to_save = dect_or_clf_dict(image_path=image_path, image_id=image_id, count=class_dict[jtem])
            to_save["Species"] = jtem
            all_cfl[f"{image_id}_{jtem}"] = to_save

        return all_cfl

    def annotate_and_save_image(self, detection, save_dir="detections"):
        image_path = detection["img_id"]
        scene = self.np_image(image_path)
        labels = detection["labels"]
        annotated_img = self.box_annotator.annotate(scene=scene, detections=detection["detections"], labels=labels)

        Path(Path(image_path).parent, save_dir).mkdir(exist_ok=True)
        with sv.ImageSink(target_dir_path=str(Path(Path(image_path).parent, save_dir)), overwrite=False) as sink:
            sink.save_image(image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=Path(image_path).name)
        return annotated_img

    @staticmethod
    def update_labels_with_classification(first_output, second_output):
        classification_dict = {}
        for entry in second_output:
            img_path, species, score = entry.rsplit(' ', 2)
            if img_path not in classification_dict:
                classification_dict[img_path] = []
            classification_dict[img_path].append(f"{species} {score}")

        for entry in first_output:
            img_id = entry['img_id']
            if img_id in classification_dict:
                entry['labels'] = classification_dict[img_id]

        return first_output

    def single_classification(self, img_path, det_conf_thres, clf_conf_thres, img_index=None):
        input_img = self.np_image(img_path)
        results_det = self.detection_model.single_image_detection(self.trans_det(input_img), input_img.shape, img_path=img_index, conf_thres=det_conf_thres)
        labels = []
        if self.classification_model is not None:
            for xyxy, det_id in zip(results_det["detections"].xyxy, results_det["detections"].class_id):
                if det_id == 0:
                    cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
                    results_clf = self.classification_model.single_image_classification(self.trans_clf(Image.fromarray(cropped_image)))
                    labels.append(f"{results_clf['prediction'] if results_clf['confidence'] > clf_conf_thres else 'Unknown'} {results_clf['confidence']:.2f}")
                else:
                    labels = results_det["labels"]
        else:
            labels = results_det["labels"]

        annotated_img = self.box_annotator.annotate(scene=input_img, detections=results_det["detections"], labels=labels)
        Path(Path(img_path).parent, "detections").mkdir(exist_ok=True)
        with sv.ImageSink(target_dir_path=str(Path(Path(img_path).parent, "detections")), overwrite=False) as sink:
            sink.save_image(image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=Path(img_path).name)

        summarised_clf = self.summarise_classification(labels, image_path=img_path)
        save_detection_json(save_dir=Path(img_path).parent, to_save=summarised_clf)
        
        return summarised_clf

    def batch_classification(self, data_path, det_thres, clf_thres, extension, device):
        det_dataset = pw_data.DetectionImageFolder(data_path, transform=self.trans_det)
        det_loader = DataLoader(det_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)

        det_results = self.detection_model.batch_image_detection(det_loader, conf_thres=det_thres)

        if self.classification_model is not None:
            clf_dataset = pw_data.DetectionCrops(det_results, transform=pw_trans.Classification_Inference_Transform(target_size=224), path_head=data_path)
            clf_loader = DataLoader(clf_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
            clf_results = self.classification_model.batch_classification(clf_loader, log_queue = Queue())
            clf_labels = []
            for i, j in enumerate(clf_results):
                aset = clf_results[i]
                clf_labels.append(f"{aset['img_id']} {aset['prediction'] if aset['confidence'] > clf_thres else 'Unknown'} {aset['confidence']:.2f}")

            imgs = []
            species = []
            result_dict = {}
            for i in clf_labels:
                img_species = i.split()
                imgs.append(img_species[0])
                species.append(img_species[1])

            result_dict["Image Path"] = imgs
            result_dict["Species"] = species
            tbl = pandas.DataFrame(result_dict).groupby(['Image Path', 'Species']).size().reset_index(name='Count')
            print(tbl)
            to_save = {}
            dict_list = [row.to_dict() for _, row in tbl.iterrows()]
            for jtem in dict_list:
                img_id = Path(jtem['Image Path']).stem
                species = jtem["Species"]
                ts_ = dect_or_clf_dict(image_path=jtem['Image Path'], 
                                       image_id=Path(jtem['Image Path']).name, 
                                       count=jtem['Count'])
                ts_["Species"] = species
                keys = f"{img_id}_{species}"
                to_save[keys] = ts_

            save_detection_json(data_path, to_save=to_save)
            updated_detection = self.update_labels_with_classification(det_results, clf_labels)
            for i in updated_detection:
                self.annotate_and_save_image(i)

        return to_save


# Test
if __name__ == "__main__":
    print(dir(PlainResNetInference))
    classifier = baseClassifier()
    classifier.load_models(
                           det_weight="MegaDetector_v5b.0.0.pt", 
                           #clf="AI4GSnapshotSerengeti", 
                           clf_weight="AI4GSnapshotSerengeti.ckpt",
                           device="cpu")
    data_path = r"C:\Users\ganda\Downloads\ee"#r"D:\100EK113\sub\inagain\subsub\B"
    results = classifier.batch_classification(data_path, 0.5, 0.2, "JPG", "cpu")