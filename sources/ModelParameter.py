from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt 
from PyQt5.QtGui import QIcon
import torch, os
from pathlib import Path


DECLAS_ROOT = Path(__file__).resolve().parent.parent

class ModelParameter(QDialog):
   def __init__(self) -> None:
      super(ModelParameter, self).__init__()
      loadUi(f"{DECLAS_ROOT}/ui/ModelParameters.ui", self)
      self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
      self.icon_file = str(Path( Path(__file__).parent.parent, 'icons', 'logo.jpg'))
      self.setWindowIcon(QIcon(self.icon_file))
      # Set font size for the entire window

      self.inference_param = None

      # TASK
      self.task.addItems(["Detection", "Classification"])
      self.task.setCurrentIndex(0)
      self.task.currentTextChanged.connect(self.update_model_type)

      # CLASSIF OR DETECTION MODEL
      self.detect_model = ["MegaDetectorV5", "MegaDetectorV6", "DeepFauna"]
      self.classif_model = ["DeepFauna", "AI4GAmazonRainforest", "AI4GSnapshotSerengeti",
                                 "Namibian Desert", "New Zealand Invasives", "Peruvian Amazon",
                                 "Iran", "Southwest USA"]
      
      self.model_type.addItems(["MegaDetectorV5", "MegaDetectorV6"])
      self.model_type.setCurrentIndex(0)
      self.model_type.setDuplicatesEnabled(False)

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


   def update_model_type(self):
      self.model_type.clear()

      if self.task.currentText() == "Classification":
            self.model_type.addItems(self.classif_model)

      if self.task.currentText() == "Detection":
            self.model_type.addItems(self.detect_model)


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

   def save_inference_parameters(self):
      yolo_conf = self.yolo_conf.value()
      yolo_imgsz = self.yolo_imgsz_parse()
      yolo_device = self.yolo_device.currentText()
      yolo_max_det = self.yolo_max_det.value()
      yolo_vid_stride = self.yolo_vid_stride.value()
      yolo_classes = self.yolo_classes.currentText()
      yolo_half = self.yolo_half.isChecked()
      run_on_main_dir = self.run_on_main_dir.isChecked()
      task = self.task.currentText()
      model_type = self.model_type.currentText()
      
      self.inference_param = {"conf": yolo_conf,
                         "imgsz": yolo_imgsz,
                         "device": yolo_device,
                         "max_det": yolo_max_det,
                         "vid_stride": yolo_vid_stride,
                         "class_of_interest": yolo_classes,
                         "half": yolo_half,
                         "run_on_main_dir": run_on_main_dir,
                         "task": task, 
                         "model_type": model_type}
      
      self.accept()



