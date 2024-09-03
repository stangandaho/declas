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
      self.icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.jpg') )
      icon_file = self.icon_file.replace("sources", "")
      self.setWindowIcon(QIcon(icon_file))
      # Set font size for the entire window

      self.inference_param = None

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

   def save_inference_parameters(self):
      yolo_conf = self.yolo_conf.value()
      yolo_imgsz = self.yolo_imgsz_parse()
      yolo_device = self.yolo_device.currentText()
      yolo_max_det = self.yolo_max_det.value()
      yolo_vid_stride = self.yolo_vid_stride.value()
      yolo_classes = self.yolo_classes.currentText()
      yolo_half = self.yolo_half.isChecked()
      run_on_main_dir = self.run_on_main_dir.isChecked()
      
      self.inference_param = {"conf": yolo_conf,
                         "imgsz": yolo_imgsz,
                         "device": yolo_device,
                         "max_det": yolo_max_det,
                         "vid_stride": yolo_vid_stride,
                         "class_of_interest": yolo_classes,
                         "half": yolo_half,
                         "run_on_main_dir": run_on_main_dir}
      
      self.accept()



