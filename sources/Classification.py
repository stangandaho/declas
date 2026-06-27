import sys, os, json
from pathlib import Path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from Bases import (extract_video_frames, save_detection_json, dect_or_clf_dict)

# Model-extension inference

def _summarise_extension_detections(detections: list,
                                     image_path: str,
                                     source_video=None) -> dict:
    """Convert an adapter's detection list into the Declas JSON format.

    Groups detections by species (counting multiple individuals), then wraps
    each group in a dect_or_clf_dict entry keyed by  <stem>_<species>.
    """
    class_dict: dict[str, int] = {}
    for det in detections:
        sp = det.get("species", "Unknown")
        class_dict[sp] = class_dict.get(sp, 0) + 1

    image_id = Path(image_path).stem
    all_cfl = {}
    for species, count in class_dict.items():
        entry = dect_or_clf_dict(image_path=image_path,
                                 image_id=image_id,
                                 count=count,
                                 source_video=source_video,
                                 category=species)
        entry["species"] = species
        all_cfl[f"{image_id}_{species}"] = entry
    return all_cfl


def _draw_and_save_annotated(image_path: str, detections: list) -> None:
    """Draw bounding boxes from an adapter's output and save the annotated image."""
    import cv2
    import numpy as np
    from PIL import Image as _Image

    img = np.array(_Image.open(image_path).convert("RGB"))
    for det in detections:
        bbox = det.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 50), 3)
            label = f"{det.get('species', '')} {det.get('confidence', 0):.2f}"
            cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 50), 2)

    out_dir = Path(Path(image_path).parent, "detections")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / Path(image_path).name),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def extension_single_classification(image_path: str, adapter,
                                     conf_thres: float,
                                     source_video=None,
                                     class_filter=None) -> dict:
    """Run a model-extension adapter on one image.

    Saves the annotated image to  <img_dir>/detections/<img_name>  and,
    when source_video is None, writes the result to  <img_dir>/detections.json.

    Returns the result dict (same format as single_classifications).
    """
    detections = adapter.predict_single(image_path, conf_thres)
    if class_filter:
        detections = [d for d in detections if d.get("species") in class_filter]
    _draw_and_save_annotated(image_path, detections)

    result = _summarise_extension_detections(detections, image_path,
                                              source_video=source_video)
    if source_video is None:
        save_detection_json(save_dir=Path(image_path).parent, to_save=result)
    return result


def extension_batch_classification(data_path: str, adapter,
                                    conf_thres: float,
                                    extension: str = ".JPG",
                                    class_filter=None) -> dict:
    """Run a model-extension adapter on every image in data_path.

    Saves annotated images and writes a single merged detections.json.
    """
    to_save: dict = {}
    results = adapter.predict_batch(data_path, conf_thres, extension)
    for stem, detections in results.items():
        img_path = next(
            (str(p) for p in Path(data_path).iterdir()
             if p.stem == stem and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            None
        )
        if img_path is None:
            continue
        if class_filter:
            detections = [d for d in detections if d.get("species") in class_filter]
        _draw_and_save_annotated(img_path, detections)
        entry = _summarise_extension_detections(detections, img_path,
                                                 source_video="__batch__")
        to_save.update(entry)

    save_detection_json(save_dir=data_path, to_save=to_save)
    return to_save


def extension_video_classification(video_path: str, adapter,
                                    conf_thres: float,
                                    vid_stride: int = 5,
                                    log_queue=None,
                                    class_filter=None) -> dict:
    """Extract frames from a video and run an extension adapter on each frame."""
    video_path = Path(video_path)
    frames = extract_video_frames(video_path, vid_stride=vid_stride)

    if not frames:
        if log_queue:
            log_queue.put("❌ No frames extracted.")
        return {}

    if log_queue:
        log_queue.put(f"✅ Running inference on {len(frames)} frames…")

    to_save: dict = {}
    for fpath in frames:
        detections = adapter.predict_single(fpath, conf_thres)
        if class_filter:
            detections = [d for d in detections if d.get("species") in class_filter]
        _draw_and_save_annotated(fpath, detections)
        result = _summarise_extension_detections(detections, fpath,
                                                  source_video=str(video_path))
        to_save.update(result)

    # Merge into the video-directory detections.json
    if to_save:
        json_path = Path(video_path.parent, "detections.json")
        existing = {}
        if json_path.exists():
            try:
                with open(str(json_path), "r") as f:
                    existing = json.load(f)
                if "Image path" in existing:
                    existing = {}
            except Exception:
                pass
        existing.update(to_save)
        save_detection_json(save_dir=str(video_path.parent), to_save=existing)

    if log_queue:
        log_queue.put(f"🎉 Video processed.")

    return to_save