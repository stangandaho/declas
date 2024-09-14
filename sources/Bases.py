from PIL import Image, ExifTags
from datetime import datetime
import os, re, cv2, json, piexif
import numpy as np
from PIL import Image
import supervision as sv
from pathlib import Path
import pandas as pd
import numpy as np

#from PyQt5.QtWidgets import QTextEdit

def class_number(results = dict, category = str):
    l = []
    detect_category = [x.split() for x in results.get('labels')]
    detect_category = [x[0] for x in detect_category]
    for x in detect_category:
        if x == category:
            l.append(x)

    return len(l)


def get_metadata(image_path):
    # Extract and display metadata using Pillow
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data:
            metadata_str = ""
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)

                if tag_name in ["MakerNote", "ComponentsConfiguration", "FileSource", "SceneType"]:
                    continue
                
                if isinstance(value, bytes):
                    # Decode bytes to a readable string if possible
                    try:
                        value = value.decode('utf-8', 'ignore')
                    except UnicodeDecodeError:
                        value = "Binary data (not displayable)"

                if tag_name == "GPSInfo":
                    metadata_str += format_gps_info(value)
                    gps_data = metadata_str
                else:
                    metadata_str += f"{tag_name}: {value}\n"

            return metadata_str, gps_data
        else:
            return "No metadata found."
    except Exception as e:
        return  f"Failed to retrieve metadata: {e}"


def format_gps_info(gps_info):
        """
        Convert the GPSInfo tag data to a more readable format.
        """
        gps_str = "GPS Information:\n"
        
        def convert_to_degrees(value):
            """
            Helper function to convert the GPS coordinates stored in the EXIF to degrees in float format.
            The GPS coordinates are stored as [degrees, minutes, seconds] in EXIF.
            """
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)

        # Latitude
        if 1 in gps_info and 2 in gps_info:
            lat = convert_to_degrees(gps_info[2])
            if gps_info[1] == 'S':  # South latitude
                lat = -lat
            gps_str += f"  Latitude: {round(lat, 3)}\n"
        
        # Longitude
        if 3 in gps_info and 4 in gps_info:
            lon = convert_to_degrees(gps_info[4])
            if gps_info[3] == 'W':  # West longitude
                lon = -lon
            gps_str += f"  Longitude: {round(lon, 3)}\n"
        
        # Altitude
        if 6 in gps_info:
            alt = gps_info[6]
            gps_str += f"  Altitude: {alt} meters\n"
        
        # Other GPS info
        for key in gps_info:
            if key not in [1, 2, 3, 4, 6]:  # Skip already processed tags
                gps_str += f"  {ExifTags.GPSTAGS.get(key, key)}: {gps_info[key]}\n"

        return gps_str


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.



# !!! Output paths need to be optimized !!!
def save_detection_images(results, output_dir, input_dir = None, overwrite=False):
    """
    Save detected images with bounding boxes and labels annotated.

    Args:
        results (list or dict):
            Detection results containing image ID, detections, and labels.
        output_dir (str):
            Directory to save the annotated images.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    """
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(results, list):
        for entry in results:
            annotated_img = box_annotator.annotate(
                scene=np.array(Image.open(entry["img_id"]).convert("RGB")),
                detections=entry["detections"],
                labels=entry["labels"],
            )

            img_id_parts=Path(entry["img_id"]).parts
            last_input_dir=Path(input_dir).parts[-1]
            relative_dir=Path(*img_id_parts[img_id_parts.index(last_input_dir)+1:-1])
            full_output_dir = os.path.join(output_dir, relative_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            with sv.ImageSink(target_dir_path=full_output_dir, overwrite=overwrite) as sink: 
                sink.save_image(
                    image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=entry["img_id"].rsplit(os.sep, 1)[1]
                )
    else:
        annotated_img = box_annotator.annotate(
            scene=np.array(Image.open(results["img_id"]).convert("RGB")),
            detections=results["detections"],
            labels=results["labels"],
        )

        with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink: 
            sink.save_image(
                image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=Path(results["img_id"]).name
            )


import ast

def is_valid_dict(output):
    try:
        # Attempt to parse the string as a dictionary
        parsed_output = ast.literal_eval(output)
        # Check if the parsed output is a dictionary
        return isinstance(parsed_output, dict)
    except (ValueError, SyntaxError):
        return False
    
# Date to id
def date_to_id():
    dt = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    dt = re.sub(r'\s+|:|-', '', dt)
    return dt


def time_to_radians(time_str):
    """
    Converts a time string in the format %H:%M:%S to a vector of radians scaled to [0, 2π].

    Args:
        time_str: The time string to convert.

    Returns:
        A vector of radians representing the time.
    """

    # Convert the time string to a datetime object
    time_obj = pd.to_datetime(time_str, format='%H:%M:%S')

    # Extract the hour, minute, and second components
    hour = time_obj.hour
    minute = time_obj.minute
    second = time_obj.second

    # Calculate the total seconds
    total_seconds = hour * 3600 + minute * 60 + second

    # Convert seconds to radians
    radians = total_seconds / (24 * 3600) * 2 * np.pi

    return radians


def exif_table(image_path):
    exif = piexif.load(image_path)
    gps_info = exif.get("GPS")

    north_or_south = str(gps_info[1])
    west_or_est = str(gps_info[3])

    lon = gps_info[4][0][0] + (gps_info[4][1][0])/60.0 + (gps_info[4][2][0])/3600.0
    lat = gps_info[2][0][0] + (gps_info[2][1][0])/60.0 + (gps_info[2][2][0])/3600.0

    lon = -lon if "S" in north_or_south else lon
    lat = -lon if "W" in west_or_est else lat
    lon = round(lon, 3); lat = round(lat, 3)

    alti = gps_info[6][0]

    device_make = str(exif["0th"][271]).replace("b", "").replace("'", "")
    #device_model = exif["0th"][272].decode('utf-8')

    original_datetime = str(exif["Exif"][36867]).replace("b'", "").replace("'", "")
    original_date = original_datetime.split()[0].strip()
    original_time = original_datetime.split()[1].strip()

    time_diel = time_to_radians(original_time)
    #time_diel = float(time_diel[0]) + (float(time_diel[1])/60) + (float(time_diel[2])/3600)
    time_diel = round(time_diel, 3)

    out_table = {"Longitude": lon,
                 "Latitude": lat,
                 "Altitude": alti,
                 "Make": device_make,
                 #"Model": device_model,
                 "Date": original_date,
                 "Time": original_time,
                 "Time in Radian": time_diel}
    
    return out_table

CONFIG_ROOT = Path(__file__).resolve().parent.parent
def load_json(fp = Path(CONFIG_ROOT, "config/inference_param.json")):
    with open(str(fp), "r") as json_obj:
        json_obj = json.load(json_obj)
    
    return json_obj

def dump_json(dict_obj):
    fp = Path(CONFIG_ROOT, "config/inference_param.json")
    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    if not fp.exists():
        fp.touch(exist_ok=True)

    if not fp.is_dir():
        with open(str(fp), "w") as to_dump:
            json.dump(obj=dict_obj, fp=to_dump)


# Save detection json
def save_detection_json(save_dir, to_save):
    
    save_path = Path(Path(save_dir), "detections.json")
    if not save_path.exists():
        save_path.touch(exist_ok=True)
        with open(str(save_path), "w") as to_s:
            json.dump(obj={}, fp = to_s)

    # with open(str(save_path), "r") as e:
    #     detection = json.load(e)
    #     detection[Path(save_dir).stem] = to_save

    with open(str(save_path), "w") as to_s:
        json.dump(to_save, to_s, indent=4)

def load_weight():
    DECLAS_ROOT = Path(__file__).resolve().parent.parent
    Path(DECLAS_ROOT, "config/model_/mdls").mkdir(parents=True, exist_ok=True)
    if not Path(DECLAS_ROOT, "config/model_/mdls/mdl.json").exists():
        Path(DECLAS_ROOT, "config/model_/mdls/mdl.json").touch(exist_ok=True)
        with open(str(Path(DECLAS_ROOT, "config/model_/mdls/mdl.json")), "w") as tdump:
            json.dump(obj={}, fp=tdump)

    fp = Path(DECLAS_ROOT, "config/model_/mdls/mdl.json")

    available_weight = []
    try:
        with open(str(fp), "r") as weight:
            weights = json.load(fp=weight)
            for weight_name in weights:
                path = weights[weight_name]["Path"]
                available_weight.append(path)

    except:
        available_weight = []

    aw = "" if len(available_weight) == 0 else available_weight[-1]
    return aw