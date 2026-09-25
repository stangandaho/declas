# Changelog

## v1.4.0

### New Features
- draw bounding boxes around animals the model missed, and delete wrong ones, directly on the image; each drawn box is saved with its species to `detections.json` and counted in the report
- image and video thumbnails in the sidebar, with the open media highlighted; the folder tree now shows folders only
- French interface (Setting > General > Language, applied after restart)
- search field in the Extensions window to find a model by name, region, author, task or species
- new model extension: Sub-Saharan Africa Wildlife Detector v1 (36 classes, ONNX)

### Improvements
- narrower sidebar with a thin separator, giving more room to the image
- Previous / Next now go through files in name order, matching the thumbnails

## v1.3.1

### Bug Fixes
- distance estimation now works on single images (not only batch folders)
- distance sampled at the bottom-centre of the bounding box (ground contact) instead of the bbox centre
- FOV correction now applies to all stations when a single FOV entry is defined; a warning is shown in the status bar when no station name matches
- fixed station name resolution: `Run on main directory` mode uses the selected folder as the station; nested mode (`project/station/species/`) uses the correct station level
- fixed stacked floating windows appearing when switching task type between Detection and Classification
- depth model image loader now force-reads pixels before passing to the processor, preventing crashes on camera-trap images with unusual colour profiles or ICC metadata
- manual `distance_m` tags no longer overwrite auto-estimated distances in the report; they are appended as additional rows for undetected individuals
- report CSV filename changed to `task_YYYYMMDD_HHMMSS.csv` to avoid overwriting previous results

### Improvements
- new toolbar button to clear all detection/classification reports (`.json` and `.csv`) under the current folder
- batch run with distance estimation enabled skips re-detection if `detections.json` files already exist and goes straight to distance estimation

## v1.3.0

### New Features
- depth model integration for estimating subject distance from the camera; selectable depth model and per-station field-of-view (FOV) table in inference parameters
- replaced raw JSON editor with editable species/count cards; each card shows a species name field and a count spinner, supports adding and removing entries, and auto-saves to the JSON file on every change
- download and manage model weights at runtime via the Extensions dialog; bundled and online registries are merged so locally bundled adapters always appear even without internet
- notification sound informing operation end