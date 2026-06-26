from PIL import Image, ExifTags
from datetime import datetime, timedelta
import os, re, cv2, json, piexif, struct, calendar
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

def get_video_metadata(video_path):
    """Extract metadata from a video file using OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return "Failed to open video file."

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        cap.release()

        duration_sec = (frame_count / fps) if fps > 0 else 0
        duration_str = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)

        return (
            f"File: {Path(video_path).name}\n"
            f"Size: {file_size_mb:.2f} MB\n"
            f"Resolution: {width} × {height}\n"
            f"Duration: {duration_str}\n"
            f"FPS: {fps:.2f}\n"
            f"Frames: {frame_count}\n"
            f"Codec: {codec}\n"
        )
    except Exception as e:
        return f"Failed to retrieve video metadata:\n {e}"


def extract_video_frames(video_path, vid_stride=5):
    """Extract every vid_stride-th frame from a video as JPEGs.

    Frames are saved to  <video_dir>/frames/  named  <video_stem>_f<idx:06d>.jpg.
    Returns the list of saved frame paths (strings).
    """
    video_path = Path(video_path)
    frames_dir = Path(video_path.parent, "frames")
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    extracted = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % vid_stride == 0:
            fname = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            fpath = str(Path(frames_dir, fname))
            cv2.imwrite(fpath, frame)
            extracted.append(fpath)
        frame_idx += 1

    cap.release()
    return extracted


def get_metadata(image_path, to_dict = False):
    # Extract and display metadata using Pillow
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data:
            metadata_str = ""
            gps_data = None
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                
                if tag_name in ["MakerNote", "ComponentsConfiguration",
                                "UserComment", "CompressedBitsPerPixel", 
                                "ImageDescription", "FileSource", "SceneType"]:
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
        return  f"Failed to retrieve metadata:\n {e}"


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
        else:
            gps_str += f"  Altitude: {None}\n"
        
        # Other GPS info
        # for key in gps_info:
        #     if key not in [1, 2, 3, 4, 6]:  # Skip already processed tags
        #         gps_str += f"  {ExifTags.GPSTAGS.get(key, key)}: {gps_info[key]}\n"

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

    def _decode_tag(raw):
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore").rstrip('\x00').strip()
        return str(raw).rstrip('\x00').strip()

    def _r(rational):
        """Convert a piexif (numerator, denominator) rational to float."""
        num, den = rational
        return num / den if den != 0 else 0.0

    gps_info = exif.get("GPS") or {}
    lon = lat = alti = None
    if gps_info:
        try:
            lat = _r(gps_info[2][0]) + _r(gps_info[2][1]) / 60.0 + _r(gps_info[2][2]) / 3600.0
            lon = _r(gps_info[4][0]) + _r(gps_info[4][1]) / 60.0 + _r(gps_info[4][2]) / 3600.0

            # Apply hemisphere signs (LatitudeRef N/S → lat, LongitudeRef E/W → lon)
            lat_ref = _decode_tag(gps_info.get(1, b'N'))
            lon_ref = _decode_tag(gps_info.get(3, b'E'))
            if 'S' in lat_ref:
                lat = -lat
            if 'W' in lon_ref:
                lon = -lon

            lat = round(lat, 7)
            lon = round(lon, 7)
            alti = round(_r(gps_info[6]), 1) if 6 in gps_info else None
        except (KeyError, TypeError, ZeroDivisionError, IndexError):
            lon = lat = alti = None

    device_make  = _decode_tag(exif["0th"].get(271, b""))
    device_model = _decode_tag(exif["0th"].get(272, b""))

    original_datetime = _decode_tag(exif["Exif"].get(36867, b""))
    dt_parts = original_datetime.split()
    original_date = dt_parts[0] if dt_parts else ""
    original_time = dt_parts[1] if len(dt_parts) > 1 else ""
    time_diel = round(time_to_radians(original_time), 3) if original_time else None

    out_table = {
        "longitude": lon,
        "latitude": lat,
        "altitude": alti,
        "make": device_make,
        "model": device_model,
        "date": original_date,
        "time": original_time,
        "time_radian": time_diel,
    }
    return out_table


# Video datetime helpers

def _get_mp4_creation_datetime(video_path):
    """Parse creation_time from an MP4 file's moov/mvhd box.

    Returns a naive local datetime, or None if unavailable / unreadable.
    The MP4 epoch is 1904-01-01 00:00:00 UTC.
    """
    MP4_EPOCH = datetime(1904, 1, 1)
    try:
        size = Path(video_path).stat().st_size
        read_bytes = min(2 * 1024 * 1024, size)   # first 2 MB is enough
        with open(str(video_path), 'rb') as f:
            data = f.read(read_bytes)
        idx = data.find(b'mvhd')
        if idx == -1:
            return None
        version = data[idx + 4]
        if version == 0: # 32-bit timestamps
            ct_raw = struct.unpack('>I', data[idx + 5: idx + 9])[0]
        else: # version 1: 64-bit timestamps
            ct_raw = struct.unpack('>Q', data[idx + 5: idx + 13])[0]
        if ct_raw == 0:
            return None
        dt_utc = MP4_EPOCH + timedelta(seconds=int(ct_raw))
        # Convert UTC → local naive datetime
        utc_ts = calendar.timegm(dt_utc.timetuple())
        return datetime.fromtimestamp(utc_ts)
    except Exception:
        return None


def get_video_start_datetime(video_path):
    """Return the best-available recording start datetime for a video.

    Priority:
      1. MP4 container creation_time (moov/mvhd box) — only for .mp4 files.
      2. File modification time (mtime) as fallback.
    """
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.mp4':
        dt = _get_mp4_creation_datetime(video_path)
        if dt and dt.year > 1980:   # sanity-check: reject obviously wrong dates
            return dt
    return datetime.fromtimestamp(video_path.stat().st_mtime)


def get_video_gps(video_path):
    """Try to extract GPS coordinates from a video file.

    Reads the MP4 binary and looks for the ©xyz atom (ISO 6709 location string)
    used by most modern cameras (iOS, Android, GoPro, some trail cameras).
    Format examples: "+25.1234+025.5678/" or "+25.1234+025.5678+1100.0/"

    Returns (lat, lon, alt) — floats, or (None, None, None) on failure.
    """
    import re as _re
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            return None, None, None
        with open(str(video_path), 'rb') as f:
            data = f.read(min(4 * 1024 * 1024, video_path.stat().st_size))
        for marker in (b'\xa9xyz', b'\xc2\xa9xyz'):
            idx = data.find(marker)
            if idx == -1:
                continue
            # Atom payload: skip 2-byte data-length + 2-byte language code
            chunk = data[idx + len(marker) + 4: idx + len(marker) + 4 + 64]
            text  = chunk.split(b'\x00')[0].decode('ascii', errors='ignore')
            m = _re.match(r'([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?', text)
            if m:
                lat = float(m.group(1))
                lon = float(m.group(2))
                alt = float(m.group(3)) if m.group(3) else None
                return round(lat, 7), round(lon, 7), alt
    except Exception:
        pass
    return None, None, None


def get_video_frame_datetime(frame_path, source_video):
    """Compute the exact recording datetime for an extracted video frame.

    Frame filename convention: ``<video_stem>_f<frame_idx:06d>.jpg``
    The offset is  frame_idx / FPS  seconds after the video's start time,
    so every frame gets a unique, accurate timestamp.

    Returns a datetime or None on failure.
    """
    try:
        stem = Path(frame_path).stem # e.g. IMAG0015_f000150
        frame_idx = int(stem.rsplit('_f', 1)[-1])  # → 150
        cap = cv2.VideoCapture(str(source_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        base_dt = get_video_start_datetime(source_video)
        return base_dt + timedelta(seconds=frame_idx / fps if fps > 0 else 0)
    except Exception:
        return None


def dect_or_clf_dict(image_path, image_id, count, source_video=None, category=None):
    """Build the per-media result dict saved to detections.json.

    Parameters
    ----------
    image_path : path of the image (or extracted video frame) being recorded.
    image_id : filename / display name.
    count : number of detections / animals of interest.
    source_video : original video path when image_path is an extracted frame.
                   Used to derive date/time from the video's mtime when the
                   frame itself carries no EXIF data.
    """
    # Frames extracted from videos live in a 'frames' sub-folder
    media_type = 'video' if 'frames' in Path(image_path).parts else 'image'

    try:
        exif_data = exif_table(image_path=image_path)
    except:
        exif_data = {}

    # Fallback for video frames: compute per-frame datetime from FPS + frame index,
    # and propagate GPS from the video container if available.
    if not exif_data and source_video is not None and source_video != "__batch__":
        try:
            frame_dt = get_video_frame_datetime(image_path, source_video)
            if frame_dt is None:
                frame_dt = get_video_start_datetime(source_video)
            frame_time_str = frame_dt.strftime('%H:%M:%S')
            v_lat, v_lon, v_alt = get_video_gps(source_video)
            exif_data = {
                "longitude": v_lon,
                "latitude": v_lat,
                "altitude": v_alt,
                "make": None,
                "model": None,
                "date": frame_dt.strftime('%Y:%m:%d'),
                "time": frame_time_str,
                "time_radian": round(time_to_radians(frame_time_str), 3),
            }
        except Exception:
            pass

    to_save = {
        'media_path':  str(image_path),
        'media': image_id,
        'media_type':  media_type,
        'category': category,
        'count': count,
        'longitude': exif_data.get('longitude'),
        'latitude': exif_data.get('latitude'),
        'altitude': exif_data.get('altitude'),
        'make': exif_data.get('make'),
        'model': exif_data.get('model'),
        'date': exif_data.get('date'),
        'time': exif_data.get('time'),
        'time_radian': exif_data.get('time_radian'),
    }

    return to_save



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


    with open(str(save_path), "w") as to_s:
        json.dump(to_save, to_s, indent=4)


def split_json_from_path(json_path, image_path, spliter = "\n\n###\n\n"):
    with open(json_path, "r") as det:
        detections = json.load(fp=det)

    # Legacy flat-dict format (old single-image saves): match only if Image path agrees
    if "media_path" in detections:
        if Path(detections.get("media_path", "")) == Path(image_path):
            return str(detections)
        return None  # Flat dict belongs to a different image

    # Nested dict: find keys that belong exactly to this file.
    # Keys may be  <stem>  (image),  <stem>_<species>  (classification),
    # or  <stem>_f<frame>  (video frame).
    stem = Path(image_path).stem
    right_key = [
        ky for ky in detections.keys()
        if ky == stem or ky.startswith(stem + "_") or ky.startswith(stem + "-")
    ]

    if not right_key:
        return None  # No detection recorded for this media
    if len(right_key) > 1:
        return spliter.join([str(detections[ky]) for ky in right_key])
    return str(detections[right_key[0]])

def split_json_obj(json_obj, spliter = "\n\n###\n\n", classif = True):
    right_key = [ky for _, ky in enumerate(list(json_obj.keys())) ]

    if not right_key:
        return None  # no detections/classifications to display

    if classif:
        if len(right_key) > 1:
            txt_split = spliter.join([str(json_obj[ky]) for ky in right_key])
        else:
            txt_split = json_obj[right_key[0]]
    else:
        txt_split = json_obj

    return str(txt_split)

def load_weight():
    DECLAS_ROOT = Path(__file__).resolve().parent.parent
    Path(DECLAS_ROOT, "config/model_/mdls").mkdir(parents=True, exist_ok=True)
    if not Path(DECLAS_ROOT, "config/model_/mdls/mdl.json").exists():
        Path(DECLAS_ROOT, "config/model_/mdls/mdl.json").touch(exist_ok=True)
        with open(str(Path(DECLAS_ROOT, "config/model_/mdls/mdl.json")), "w") as tdump:
            json.dump(obj={}, fp=tdump)

    fp = Path(DECLAS_ROOT, "config/model_/mdls/mdl.json")

    try:
        with open(str(fp), "r") as weight:
            weights = json.load(fp=weight)
    except:
        weights = None
        pass

    aw = "" if not weights else weights
    return aw


def get_unique(list: list):
    """
    Remove duplicate elements from a list while preserving order.

    Args:
        list (list): The list from which duplicates need to be removed.

    Returns:
        list: A new list containing only unique elements from the original list, in the same order.
    """
    inner_list = []
    for l in list:
        if l not in inner_list:
            inner_list.append(l)

    return inner_list
