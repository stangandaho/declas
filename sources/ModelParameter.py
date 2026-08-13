from PyQt5.QtWidgets import (QDialog, QListWidgetItem, QVBoxLayout, QHBoxLayout,
                              QTableWidget, QTableWidgetItem, QHeaderView,
                              QPushButton, QDialogButtonBox, QFileDialog, QMessageBox)
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from pathlib import Path
import sys, torch, csv

from sources.Bases import get_unique

DECLAS_ROOT = Path(__file__).resolve().parent.parent
if str(DECLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(DECLAS_ROOT))

try:
    from model_extensions.loader import scan_extensions
    INSTALLED_EXTENSIONS = scan_extensions()
except Exception:
    INSTALLED_EXTENSIONS = {}


DECLAS_ROOT = Path(__file__).resolve().parent.parent


class FovDialog(QDialog):
    """Small dialog for entering horizontal Field-Of-View degrees per station."""

    def __init__(self, fov_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Field Of View per Station")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(600)
        self.setMinimumHeight(320)
        self.setMaximumWidth(600)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Station", "FOV (degrees)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(0, 380)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

        btn_bar = QHBoxLayout()

        add_btn = QPushButton(QIcon("icons/add.png"), "Add")
        add_btn.clicked.connect(lambda: self.add_row())
        rm_btn = QPushButton(QIcon("icons/minus.png"), "Remove")
        rm_btn.clicked.connect(self.remove_row)
        btn_bar.addWidget(add_btn)
        btn_bar.addWidget(rm_btn)
        layout.addLayout(btn_bar)

        btn_bar.addStretch()
        upload_btn = QPushButton(QIcon("icons/publish.png"), "Upload CSV")
        btn_bar.addWidget(upload_btn)
        upload_btn.clicked.connect(self.upload_csv) 

        note_bar = QHBoxLayout()
        from PyQt5.QtWidgets import QLabel
        note = QLabel("CSV format with one row per station.")
        note.setStyleSheet("color: gray; font-size: 14px; text-align:right;")
        note_bar.addWidget(note)
        layout.addLayout(note_bar)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        for station, fov in fov_data.items():
            self.add_row(str(station), str(fov))

    def add_row(self, station: str = "", fov: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(station))
        self.table.setItem(row, 1, QTableWidgetItem(fov))

    def remove_row(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def upload_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv *.txt)")
        if not path:
            return
        try:
            # utf-8-sig strips the Windows BOM (﻿) automatically
            with open(path, newline="", encoding="utf-8-sig") as f:
                sample = f.read(4096)
            # Auto-detect delimiter (comma or semicolon)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel  # default comma
            imported = 0
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, dialect=dialect)
                # Normalize header names to lowercase for matching
                for raw_row in reader:
                    row = {k.strip().lower(): v for k, v in raw_row.items() if k}
                    # Named column matching (case-insensitive)
                    station = (row.get("station") or row.get("site") or
                               row.get("camera") or "").strip()
                    fov = (row.get("fov") or row.get("fov_degrees") or
                           row.get("field_of_view") or row.get("hfov") or "").strip()
                    # Positional fallback: col 0 = station, col 1 = fov
                    if not station:
                        vals = list(raw_row.values())
                        if vals:
                            station = str(vals[0]).strip()
                        if len(vals) > 1:
                            fov = str(vals[1]).strip()
                    if station:
                        self.add_row(station, fov)
                        imported += 1
            if imported:
                QMessageBox.information(self, "CSV imported",
                                        f"{imported} station(s) loaded.")
            else:
                QMessageBox.warning(self, "Nothing imported",
                                    "No rows were found.\n\n"
                                    "Expected columns: station, fov  (or any two columns: first = station, second = FOV).")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read CSV:\n{e}")

    def get_fov_data(self) -> dict:
        data = {}
        for i in range(self.table.rowCount()):
            s_item = self.table.item(i, 0)
            f_item = self.table.item(i, 1)
            station = s_item.text().strip() if s_item else ""
            fov_txt = f_item.text().strip() if f_item else ""
            if station and fov_txt:
                try:
                    data[station] = float(fov_txt)
                except ValueError:
                    pass
        return data


class ModelParameter(QDialog):
   def __init__(self) -> None:
      super(ModelParameter, self).__init__()
      loadUi(f"{DECLAS_ROOT}/ui/ModelParameters.ui", self)
      global INSTALLED_EXTENSIONS
      INSTALLED_EXTENSIONS = scan_extensions()
      icon_file = str(Path( Path(__file__).parent.parent, 'icons', 'logo.png'))
      self.setWindowIcon(QIcon(icon_file))
      self.setWindowFlags(Qt.WindowCloseButtonHint)
      self.setWindowTitle("Inference parameters")

      self.inference_param = None
      self.fov_data: dict = {}

      # TASK
      self.task.addItems(["Detection", "Classification"])
      self.task.setCurrentIndex(0)
      self.select_clf_model.hide()
      self.clf_model_label.hide()
      self.task.currentTextChanged.connect(self.update_model_type_show)

      # CLASSIF OR DETECTION MODEL
      self.model_type.setDuplicatesEnabled(False)
      self.populate_model_type(self.task.currentText())
      self.task.currentTextChanged.connect(self.on_task_changed)
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

      self.populate_classes(self.model_type.currentData())
      self.model_type.currentIndexChanged.connect(
          lambda _: self.populate_classes(self.model_type.currentData())
      )

      # DISTANCE ESTIMATION
      self.populate_depth_models()
      self.estimate_distance.stateChanged.connect(self.toggle_distance_controls)
      self.toggle_distance_controls(Qt.Unchecked)
      self.fov_btn.setIcon(QIcon("icons/table2.png"))
      self.fov_btn.clicked.connect(self.open_fov_dialog)
      self.fov_btn.setFixedWidth(200)
      self.fov_btn.setMinimumWidth(0)
      self.fov_btn.setMaximumWidth(16777215)


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

   def populate_model_type(self, task: str) -> None:
       self.model_type.blockSignals(True)
       self.model_type.clear()
       task_key = task.lower()
       for ext_name, ext_info in INSTALLED_EXTENSIONS.items():
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

   def populate_depth_models(self) -> None:
       self.depth_model_combo.clear()
       for ext_name, ext_info in INSTALLED_EXTENSIONS.items():
           if ext_info.get("status") != "ready":
               continue
           if ext_info.get("manifest", {}).get("task", "").lower() == "depth":
               display = ext_info["manifest"].get("display_name", ext_name)
               self.depth_model_combo.addItem(display, userData=ext_name)
       if self.depth_model_combo.count() == 0:
           self.depth_model_combo.addItem("No depth models installed")

   def toggle_distance_controls(self, state) -> None:
       visible = (state == Qt.Checked)
       self.depth_model_label.setVisible(visible)
       self.depth_model_combo.setVisible(visible)
       self.fov_btn.setVisible(visible)

   def open_fov_dialog(self) -> None:
       dlg = FovDialog(self.fov_data, parent=self)
       if dlg.exec_() == QDialog.Accepted:
           self.fov_data = dlg.get_fov_data()

   def on_task_changed(self, task: str) -> None:
       self.populate_model_type(task)
       self.model_type.setCurrentIndex(0)
       self.populate_classes(self.model_type.currentData())

   def populate_classes(self, ext_name: str | None) -> None:
       classes = []
       if ext_name and ext_name in INSTALLED_EXTENSIONS:
           classes = INSTALLED_EXTENSIONS[ext_name].get("manifest", {}).get("classes", [])
       if not classes:
           classes = ["animal", "person", "vehicle"]
       self.yolo_classes.clear()
       for cls in classes:
           item = QListWidgetItem(cls)
           item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
           item.setCheckState(Qt.Checked)
           self.yolo_classes.addItem(item)

   def set_class_of_interest(self, value) -> None:
       for i in range(self.yolo_classes.count()):
           item = self.yolo_classes.item(i)
           state = Qt.Checked if (not value or item.text() in value) else Qt.Unchecked
           item.setCheckState(state)

   def update_model_type_show(self):
       task = self.task.currentText()
       if task == "Classification":
           self.select_det_model.hide()
           self.det_model_label.hide()
       else:
           self.select_det_model.show()
           self.det_model_label.show()

   def move_to_first_position(self, lst, selected_item):
      if selected_item in lst:
         lst.remove(selected_item)
         lst.insert(0, selected_item)
      return get_unique(lst)

   def save_inference_parameters(self):
      yolo_conf = self.yolo_conf.value()
      yolo_imgsz = self.yolo_imgsz_parse()
      yolo_device = self.yolo_device.currentText()
      yolo_max_det = self.yolo_max_det.value()
      yolo_vid_stride = self.yolo_vid_stride.value()
      checked = [
          self.yolo_classes.item(i).text()
          for i in range(self.yolo_classes.count())
          if self.yolo_classes.item(i).checkState() == Qt.Checked
      ]
      yolo_classes = None if len(checked) == self.yolo_classes.count() else checked
      yolo_half = self.yolo_half.isChecked()
      run_on_main_dir = self.run_on_main_dir.isChecked()
      process_video = self.process_video.isChecked()
      task = self.task.currentText()
      model_type = (self.model_type.currentData() or self.model_type.currentText())

      select_det_model = [self.select_det_model.itemText(i) for i in range(self.select_det_model.count())]
      select_det_model = self.move_to_first_position(select_det_model, self.select_det_model.currentText())

      estimate_distance = self.estimate_distance.isChecked()
      depth_model = self.depth_model_combo.currentData() or ""

      self.inference_param = {
          "conf": yolo_conf,
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
          "estimate_distance": estimate_distance,
          "depth_model": depth_model,
          "fov_table": self.fov_data,
      }

      self.accept()
