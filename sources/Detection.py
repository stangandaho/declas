from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from pathlib import Path
import json
from PIL import Image
import numpy as np
from Bases import *


# Set model weight (for MegaDetector6 - YOLOv9)
# DOWNLOAD_WEIGHT = 'https://zenodo.org/records/11192829/files/MDV6b-yolov9c.pt?download=1'
# MODEL_PATH = r'D:\Python Projects\megadetector\v6\MDV6b-yolov9c.pt'

def class_number(results = dict, category = str):
  l = []
  detect_category = [x.split() for x in results.get('labels')]
  detect_category = [x[0] for x in detect_category]
  for x in detect_category:
    if x == category:
      l.append(x)

  return len(l)

# Function to get result
def get_results(weights = "", image_path = "", conf_thres = 0.55, 
                device="cpu", class_of_interest = "animal", save_json = True):
  
  detection_model = pw_detection.MegaDetectorV5(weights=weights, device=device)
  img = np.array(Image.open(image_path).convert("RGB"))

  transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                stride=detection_model.STRIDE)

  results = detection_model.single_image_detection(transform(img), img.shape, 
                                                   image_path, conf_thres=conf_thres)
  Path(Path(image_path).parent, "detections").mkdir(exist_ok=True)
  output_dir = Path(Path(image_path).parent, "detections")
  
  save_detection_images(results=results, output_dir=str(output_dir))

  try:
    exif_data = exif_table(image_path=image_path)
  except Exception as e:
      print(f"ERROR ! - {e}")
      exif_data = {}
  
  image_id = Path(results.get('img_id')).name # image name
  cn = class_number(results = results, category=class_of_interest)

  to_save = {'Image path': f'{image_path}', 
              'Image': image_id, 
              'Count':cn,
              "Longitude": exif_data["Longitude"] if exif_data != {} else None,
              "Latitude": exif_data["Latitude"] if exif_data != {} else None,
              "Altitude": exif_data["Altitude"] if exif_data != {} else None,
              "Make": exif_data["Make"] if exif_data != {} else None,
              #"Model": device_model,
              "Date": exif_data["Date"] if exif_data != {} else None,
              "Time": exif_data["Time"] if exif_data != {} else None,
              "Time in Radian": exif_data["Time in Radian"] if exif_data != {} else None}
  
  if save_json:
    save_path = Path(Path(image_path).parent, "detections.json")
    if not save_path.exists():
      save_path.touch(exist_ok=True)
      with open(str(save_path), "w") as to_s:
        json.dump(obj={}, fp = to_s)

    with open(str(save_path), "r") as e:
      detection = json.load(e)
      detection[Path(image_path).stem] = to_save

    with open(str(save_path), "w") as to_s:
      json.dump(detection, to_s)

  return to_save


# Function to get unique category from detection class
def get_unique(list):
  l = []
  for x in list:
    if x not in l:
      l.append(x)
  return l


# Batch detection
def batch_detection(dir_path, conf_thres = 0.55, has_child = True):
  global on_dir, progress_on_dir
  
  all_detection = {}
  dir_path = Path(dir_path)

  all_files = [fls for fls in list(dir_path.iterdir()) if fls.suffix in [".JPG", ".JPEG", ".jpg", ".jpeg"]]
  pos = range(1, len(all_files) + 1)

  on_dir = f"\nProcessing dir: {dir_path}"
  for idx, dr in enumerate(all_files):
    progress_on_dir = f"Image number {pos[idx]} - ({round(pos[idx]*100/len(all_files), 2)}%)"

    result = get_results(image_path = dr, conf_thres = conf_thres)

    image_id = Path(result.get('img_id')).name # image name
    cn = class_number(results = result, category='animal')
    to_save = {'image_path': f'{dr}', 'image': image_id, 'number':cn}

    all_detection[Path(image_id).stem] = to_save
    
    # Saving the batch detection results as annotated images
    batc_output = Path(dr.parent, 'detections')
    if not batc_output.exists():
      batc_output.mkdir(exist_ok=True)

    save_detection_images(result, output_dir=batc_output)

    with open(Path(dir_path, "detections.json"), "w") as outfile: 
      json.dump(all_detection, outfile)

  return all_detection
