from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from pathlib import Path
import sys, torch

from sources.Bases import get_unique

_DECLAS_ROOT = Path(__file__).resolve().parent.parent
if str(_DECLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECLAS_ROOT))

try:
    from model_extensions._loader import scan_extensions as _scan_extensions
    _INSTALLED_EXTENSIONS = _scan_extensions()
except Exception:
    _INSTALLED_EXTENSIONS = {}


DECLAS_ROOT = Path(__file__).resolve().parent.parent

class ModelParameter(QDialog):
   def __init__(self) -> None:
      super(ModelParameter, self).__init__()
      loadUi(f"{DECLAS_ROOT}/ui/ModelParameters.ui", self)
      icon_file = str(Path( Path(__file__).parent.parent, 'icons', 'logo.png'))
      self.setWindowIcon(QIcon(icon_file))
      self.setWindowFlags(Qt.WindowCloseButtonHint)
      self.setWindowTitle("Inference parameters")
      # Set font size for the entire window

      self.inference_param = None

      # TASK
      self.task.addItems(["Detection", "Classification"])
      self.task.setCurrentIndex(0)
      self.select_clf_model.hide()
      self.clf_model_label.hide()
      self.task.currentTextChanged.connect(self.update_model_type_show)

      # CLASSIF OR DETECTION MODEL — populated from installed extensions, filtered by task.
      self.model_type.setDuplicatesEnabled(False)
      self._populate_model_type(self.task.currentText())
      self.task.currentTextChanged.connect(self._on_task_changed)
      self.model_type.setCurrentIndex(0)

      self.select_det_model.setDuplicatesEnabled(False)
      self.select_det_model.setCurrentIndex(0)


      self.buttonBox.accepted.connect(self.save_inference_parameters)
      self.yolo_imgsz.textChanged.connect(self.yolo_imgsz_parse)
      
      ## Device
      DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
      if DEVICE == "cpu":
          self.yolo_device.addItem('cpu')
      else:
          self.yolo_device.addItem("cpu")
          self.yolo_device.addItem("cuda")

      self.yolo_classes.addItem("animal")
      self.yolo_classes.addItem("person")
      self.yolo_classes.addItem("vehicle")


   def yolo_imgsz_parse(self):
          yolo_imgsz = self.yolo_imgsz.text().split()
          try:
             if len(yolo_imgsz) == 2:
                yolo_imgsz = (int(yolo_imgsz[0]), int(yolo_imgsz[1]))
             elif len(yolo_imgsz) == 1:
                yolo_imgsz = int(yolo_imgsz[0])
             return(yolo_imgsz)
          except:
             return(640)
          
   def _populate_model_type(self, task: str) -> None:
       """Fill model_type combo with extensions that match *task* and are ready."""
       self.model_type.blockSignals(True)
       self.model_type.clear()
       task_key = task.lower()   # "detection" or "classification"
       for ext_name, ext_info in _INSTALLED_EXTENSIONS.items():
           if ext_info.get("status") != "ready":
               continue
           m = ext_info.get("manifest", {})
           if m.get("task", "").lower() != task_key:
               continue
           display_name = m.get("display_name", ext_name)
           self.model_type.addItem(display_name, userData=ext_name)
       if self.model_type.count() == 0:
           self.model_type.addItem("No models installed")
       self.model_type.blockSignals(False)

   def _on_task_changed(self, task: str) -> None:
       self._populate_model_type(task)
       self.model_type.setCurrentIndex(0)

   def update_model_type_show(self):
       task = self.task.currentText()
       if task == "Classification":
           self.select_det_model.hide()
           self.det_model_label.hide()
       else:
           self.select_det_model.show()
           self.det_model_label.show()


   def move_to_first_position(self, lst, selected_item):
      # Check if the selected item is in the list
      if selected_item in lst:
         lst.remove(selected_item)  # Remove the selected item from its current position
         lst.insert(0, selected_item)  # Insert it at the beginning (index 0)
      return get_unique(lst)

   def save_inference_parameters(self):
      yolo_conf = self.yolo_conf.value()
      yolo_imgsz = self.yolo_imgsz_parse()
      yolo_device = self.yolo_device.currentText()
      yolo_max_det = self.yolo_max_det.value()
      yolo_vid_stride = self.yolo_vid_stride.value()
      yolo_classes = self.yolo_classes.currentText()
      yolo_half = self.yolo_half.isChecked()
      run_on_main_dir = self.run_on_main_dir.isChecked()
      process_video = self.process_video.isChecked()
      task = self.task.currentText()
      # For extension entries the UserData stores the internal name; fall back to text.
      model_type = (self.model_type.currentData() or self.model_type.currentText())

      select_det_model = [self.select_det_model.itemText(i) for i in range(self.select_det_model.count())]
      select_det_model = self.move_to_first_position(select_det_model, self.select_det_model.currentText())

      self.inference_param = {"conf": yolo_conf,
                              "imgsz": yolo_imgsz,
                              "device": yolo_device,
                              "max_det": yolo_max_det,
                              "vid_stride": yolo_vid_stride,
                              "class_of_interest": yolo_classes,
                              "half": yolo_half,
                              "run_on_main_dir": run_on_main_dir,
                              "process_video": process_video,
                              "task": task,
                              "model_type": model_type,
                              "select_det_model": get_unique(select_det_model),
                              }
      
      self.accept()



