from random import choice
import shutil
from turtle import st
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDir, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QIcon, QPixmap, QFontDatabase
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QFileSystemModel, QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineSettings, QWebEnginePage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import folium
import sys, torch, os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
import pandas as pd


##
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))
os.environ['YOLO_VERBOSE'] = 'False'
from Classification import (extension_single_classification,
                            extension_batch_classification,
                            extension_video_classification)
from ModelParameter import ModelParameter
from Bases import *
from ErrorWarning import *
from ExtensionDialog import ExtensionManagerDialog, PublishGuidelinesDialog

##
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scan installed model extensions at startup
try:
    from model_extensions._loader import scan_extensions as _scan_extensions, load_adapter as _load_adapter
    INSTALLED_EXTENSIONS = _scan_extensions()
except Exception:
    INSTALLED_EXTENSIONS = {}

IMAGE_EXT  = {".JPG", ".JPEG", ".jpg", ".jpeg"}
VIDEO_EXT  = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

MEDIA_SUFFIX = [f"*{ext}" for ext in IMAGE_EXT | VIDEO_EXT] #["*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.mp4", "*.avi"]
MEDIA_EXT  = {p.lstrip("*") for p in MEDIA_SUFFIX} # suffix comparison set

IMG_PATH = []
SINGLE_DETECTION = {}
DECLAS_ROOT = Path(__file__).resolve().parent.parent

MESSAGE_DELAY = 7000

try:
    from ctypes import windll  # Only exists on Windows.
    declas_id = 'declas.1.1.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(declas_id)
except ImportError:
    pass

# 
class Declas(QMainWindow):
    def __init__(self) -> None:
        super(Declas, self).__init__()
        loadUi(f"{DECLAS_ROOT}/ui/Declas.ui", self)

        self.icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.png') )
        icon_file = self.icon_file.replace("sources", "")
        self.setWindowIcon(QIcon(icon_file))
        self.setWindowTitle("Declas 1.1.0")

        # Load a custom font
        font_path = str(Path(DECLAS_ROOT, "sources/styles/Montserrat-Regular.ttf"))
        fontid = QFontDatabase.addApplicationFont(font_path)
        font_family = QFontDatabase.applicationFontFamilies(fontid)
        #self.setFont(QFont(font_family, 12))

        self.action_models_parameter.triggered.connect(self.models_parameters)
        # Initialize default inference parameters
        if Path(DECLAS_ROOT, "config", "inference_param.json").exists():
            inf_p = load_json()
            if inf_p:
                self.inference_param = inf_p
        else:
            self.inference_param = {
                "conf": 0.55,
                "imgsz": (1920, 1440),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "max_det": 300,
                "vid_stride": 5,
                "class_of_interest": None,
                "half": False,
                "run_on_main_dir": False,
                "process_video": True,
                "task": "Detection",
                "model_type": "",
            }

        dump_json(dict_obj=self.inference_param)
        self.image_or_dir = None
        self.model_weight = load_weight()

        # IMPORT FILE
        self.action_import_dc_file.triggered.connect(self.import_dc_file) # dc = declas
        self.action_quit.triggered.connect(self.quit_declas)
        # MODE
        #self.action_dark.triggered.connect(self.set_dark_mode)
        self.action_light.triggered.connect(self.set_light_mode)
        # TOOLBAR
        select_dir = QAction(QIcon(f"{DECLAS_ROOT}/icons/folder.png"), "Select directory", self)
        select_dir.triggered.connect(self.select_folder)

        select_image = QAction(QIcon(f"{DECLAS_ROOT}/icons/image.png"), "Select media", self)
        select_image.setToolTip("Select a single image or video")
        select_image.triggered.connect(self.select_an_image)

        globe = QAction(QIcon(f"{DECLAS_ROOT}/icons/globe.png"), "Show on map", self)
        globe.triggered.connect(self.show_on_map)

        run_inference = QAction(QIcon(f"{DECLAS_ROOT}/icons/run.png"), "Run on current media", self)
        run_inference.triggered.connect(self.single_detection)

        batch_inference = QAction(QIcon(f"{DECLAS_ROOT}/icons/batch.png"), "Run folder(s)", self)
        batch_inference.triggered.connect(self.multiple_detection)

        buil_tables = QAction(QIcon(f"{DECLAS_ROOT}/icons/table.png"), "Construct table from detection/classification", self)
        buil_tables.triggered.connect(self.build_table)

        split_detection = QAction(QIcon(f"{DECLAS_ROOT}/icons/split.png"), "'Target | No target' split", self)
        split_detection.triggered.connect(self.filter_detection)

        self.tool_bar1.addAction(select_dir)
        self.tool_bar1.addAction(select_image)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(globe)
        self.tool_bar1.addAction(run_inference)
        self.tool_bar1.addAction(batch_inference)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(buil_tables)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(split_detection)


        # FOLDER TREE VIEW
        self.file_model = QFileSystemModel()

        self.file_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        self.file_model.setNameFilters(MEDIA_SUFFIX)
        self.file_model.setNameFilterDisables(False)

        self.dir_tree_view.setModel(self.file_model)
        self.dir_tree_view.hideColumn(1)
        self.dir_tree_view.hideColumn(2)
        self.dir_tree_view.hideColumn(3)

        self.dir_tree_view.selectionModel().selectionChanged.connect(self.on_image_selected)
        self.dir_tree_view.selectionModel().selectionChanged.connect(self.get_next_and_previous_media)

        # DISPLAY IMAGE / VIDEO
        self.previous_media.hide()
        self.next_media.hide()
        self.play_media.hide()
        self.play_media.setEnabled(False)
        self.previous_media.clicked.connect(self.show_previous_media)
        self.next_media.clicked.connect(self.show_next_media)
        self.current_selected_media = None

        # VIDEO PLAYER
        self._slider_dragging = False
        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_display)

        self.play_media.clicked.connect(self.toggle_play_pause)
        self.jump_back_btn.clicked.connect(self.jump_backward)
        self.jump_forward_btn.clicked.connect(self.jump_forward)

        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.stateChanged.connect(self._on_player_state_changed)

        self.video_seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.video_seek_slider.sliderReleased.connect(self._on_slider_released)

        self._set_video_controls_visible(False)

        # Ensure image page is shown at startup and video area is never black
        self.media_stack.setCurrentIndex(0)
        self.video_page.setStyleSheet("background-color: white;")
        self.video_display.setStyleSheet("background-color: white;")

        # DISPLAY IMAGE METADATA
        self.metadata_text.setReadOnly(True)

        # # DISPLAY MAP
        # Create an off-the-record QWebEngineProfile (no storage name provided)
        off_record_profile = QWebEngineProfile(self)
        # Create a QWebEnginePage with the off-the-record profile
        off_record_page = QWebEnginePage(off_record_profile, self)
        # Set the off-the-record page on the browser
        self.display_map.setPage(off_record_page)
        # Disable persistent storage and cookies for this profile
        off_record_profile.setHttpCacheType(QWebEngineProfile.NoCache)
        off_record_profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)

        # Modify settings to disable local storage and file URL access
        settings = off_record_profile.settings()
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, False)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, False)

        self.leaflet_map(11.108, 2.335, init=True)

        # INFERENCE
        self.inference_result.setReadOnly(True)
        self.view_detection.clicked.connect(self.show_detection)
        self.edit_inference.clicked.connect(self.edit_detection_result)
        self.apply_inference_edit.clicked.connect(self.save_inference_edit)

        # STATUS BAR
        self.statusbar = self.statusBar

        # LOG
        self.batch_detection_log.setReadOnly(True)

        # MODEL EXTENSIONS
        ext_action = QAction(QIcon(f"{DECLAS_ROOT}/icons/extensions.png"), "Extensions", self)
        ext_action.setToolTip("Browse and install model extensions from the online registry")
        ext_action.triggered.connect(self.open_extensions)
        self.menuModels.addAction(ext_action)

        # PUBLISH GUIDELINES
        pub_action = QAction(QIcon(f"{DECLAS_ROOT}/icons/publish.png"), "Publish", self)
        pub_action.setToolTip("Learn how to publish your own model extension")
        pub_action.triggered.connect(self.open_publish_guidelines)
        self.menuModels.addAction(pub_action)

        ## BUILD DETECTION TABLE
        #self.action_build_table.triggered.connect(self.build_table)

    def leaflet_map(self, lon, lat, zoom_start = 12, init = False):
        
        if init:
            self.display_map.setUrl(QUrl.fromLocalFile(f"{DECLAS_ROOT}/sources/tile.html"))
        else:
            m = folium.Map(
            location=[lat, lon],
            zoom_start=zoom_start,
            tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
            attr='&copy; OpenStreetMap France | &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            )
           
            folium.Marker([lat, lon], popup=f"{lon} | {lat}").add_to(m)
            # Generate HTML for the map in memory (without saving)
            map_html = m.get_root().render()
            self.display_map.setHtml(map_html)

    def import_dc_file(self):
        selected_json = QFileDialog.getOpenFileName(self, "Select json file", ".", "JSON (*json)")
        selected_json = selected_json[0]

        if selected_json:
            self.image_or_dir = selected_json
            self.statusbar.showMessage("File imported ✅", MESSAGE_DELAY)

    def quit_declas(self):
            sys.exit()

    def set_light_mode(self):
        with open(f"{DECLAS_ROOT}/sources/styles/light.qss", "r") as file:
            self.setStyleSheet(file.read())

    def models_parameters(self):
        current_set = load_json()

        mp = ModelParameter()
        mp.yolo_conf.setValue(current_set["conf"])

        imgsz = [str(i) for i in current_set["imgsz"]]
        imgsz = " ".join(imgsz) if len(imgsz) == 2 else imgsz

        mp.yolo_imgsz.setText(imgsz) #imgsz
        mp.yolo_device.setItemText(0, current_set["device"])
        mp.yolo_max_det.setValue(current_set["max_det"]) # max_det
        mp.yolo_vid_stride.setValue(current_set["vid_stride"]) # vid_stride
        mp.set_class_of_interest(current_set["class_of_interest"])
        mp.yolo_half.setChecked(current_set["half"])
        mp.run_on_main_dir.setChecked(current_set["run_on_main_dir"])
        mp.process_video.setChecked(current_set.get("process_video", True))
        mp.task.setCurrentText(current_set["task"])
        saved_mt = current_set["model_type"]
        idx = mp.model_type.findData(saved_mt)   # extension: match by userData
        if idx >= 0:
            mp.model_type.setCurrentIndex(idx)
        else:
            mp.model_type.setCurrentText(saved_mt)  # built-in: match by text
        
        mp.setWindowModality(Qt.ApplicationModal)
        if mp.exec_() == mp.Accepted:  # If the dialog is accepted
            self.inference_param = mp.inference_param
            dump_json(dict_obj=self.inference_param)

    def select_folder(self):
        global selected_folder
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if selected_folder :
            IMG_PATH.clear()
            self.image_or_dir = selected_folder
            self.file_model.setRootPath('.')
            self.dir_tree_view.setRootIndex(self.file_model.index(selected_folder))
            self.dir_tree_view.expandAll()

        return selected_folder
    

    def filter_detection(self):
        if selected_folder:
            detection_json = Path(selected_folder).rglob("detections.json")
            for _ in detection_json:
                Path(Path(_).parent, "has_target").mkdir(exist_ok=True)
                Path(Path(_).parent, "no_target").mkdir(exist_ok=True)

                try:
                    dfile = load_json(fp = _)
                    for d in dfile:
                        
                        if dfile[d]['Count'] > 0:
                            has_target_path = dfile[d]['Image path']
                            has_target_des_path = Path(Path(has_target_path).parent, 
                                                        "has_target", Path(has_target_path).name)
                            shutil.copy(has_target_path, str(has_target_des_path))
                        else:
                            no_target_path = dfile[d]['Image path']
                            no_target_des_path = Path(Path(has_target_path).parent, 
                                                        "no_target", Path(has_target_path).name)
                            shutil.copy(no_target_path, str(no_target_des_path))
                except Exception as e:
                    f"ERROR: {e}"

        self.statusbar.showMessage("Split applied \u2705", 1000)


    def select_an_image(self):
        IMG_PATH.clear()
        selected_media, _ = QFileDialog.getOpenFileName(
            self, "Select media", ".",
            "All media (*.jpg *.JPG *.jpeg *.JPEG *.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV);;"
            "Images (*.jpg *.JPG *.jpeg *.JPEG);;"
            "Videos (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV)"
        )
        if selected_media:
            self.image_or_dir = selected_media
            self._display_media(selected_media)
            self._update_metadata(selected_media)
            self._update_inference_result(selected_media)
            # Show nav buttons so the user can browse siblings
            self.previous_media.show()
            self.next_media.show()
            self.play_media.show()
            IMG_PATH.append(selected_media)
            return selected_media
        

    def on_image_selected(self, selected):
        # Get the index of the selected item
        indexes = selected.indexes()

        if indexes:
            index = indexes[0]  # We are interested in the first (and only) selected index
            file_path = self.file_model.filePath(index)

            global selected_file_path
            selected_file_path = file_path
            
            # Show navigation button
            self.previous_media.show()
            self.next_media.show()
            self.play_media.show()

            if not self.file_model.isDir(index):
                self._display_media(file_path)
                self._update_metadata(file_path)

                # Add inference result specific to this file
                self._update_inference_result(file_path)

                IMG_PATH.append(file_path)
                return file_path


    def display_image(self, image_path, message = "Unable to load image"):
        # Load and display the image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_display.setText(message)
            self.view_detection.setEnabled(False)
        else:
            self.image_display.setPixmap(pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def get_next_and_previous_media(self):

        try:
            if selected_folder:
                if selected_file_path:
                    sf = str(Path(selected_file_path).parent)
                    global all_files
                    all_files = [str(fl) for fl in Path(sf).iterdir() if not fl.is_dir() and fl.suffix in MEDIA_EXT]
                    
                else:
                    sf = selected_folder

                idx = all_files.index(str(Path(selected_file_path)))# selected_file_path from on_image_selected()
                self.current_selected_media = idx
                return idx, all_files
                
        except:
            pass

            
    def show_previous_media(self):
        try:
            if self.current_selected_media > 0:
                self.current_selected_media -= 1
                path = all_files[self.current_selected_media]
                self._display_media(path)
                self._update_metadata(path)
                self._update_inference_result(path)
                IMG_PATH.append(path)
        except:
            pass

    def show_next_media(self):
        try:
            if self.current_selected_media >= 0 and self.current_selected_media < (len(all_files) - 1):
                self.current_selected_media += 1
                path = all_files[self.current_selected_media]
                self._display_media(path)
                self._update_metadata(path)
                self._update_inference_result(path)
                IMG_PATH.append(path)
        except:
            pass


    # Media helpers 
    def _update_metadata(self, path):
        if self._is_video(path):
            self.metadata_text.setText(get_video_metadata(path))
        else:
            try:
                metadata_i, _ = get_metadata(path)
                self.metadata_text.setText(metadata_i)
            except:
                self.metadata_text.setText(str(get_metadata(path)))

    def _update_inference_result(self, file_path):
        """Show detection/classification results specific to file_path, or clear if none."""
        fp = Path(file_path)
        candidates = [
            fp.parent / "detections.json",
            fp.parent.parent / "detections.json",
        ]
        # Annotated video frames live inside frames/detections/
        # to reach <video_dir>/detections.json.
        if fp.parent.name == "detections":
            candidates.append(fp.parent.parent.parent / "detections.json")
        json_path = next((str(p) for p in candidates if p.exists()), None)

        if json_path:
            try:
                txt = split_json_from_path(json_path=json_path, image_path=file_path)
                if txt is not None:
                    self.inference_result.setText(str(txt))
                    self.edit_inference.setEnabled(True)
                    return
            except Exception:
                pass

        # No result found for this specific media
        self.inference_result.clear()
        self.edit_inference.setEnabled(False)

    def _is_video(self, path):
        return Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    def _set_video_controls_visible(self, visible):
        self.video_seek_slider.setVisible(visible)
        self.video_current_time.setVisible(visible)
        self.video_total_time.setVisible(visible)
        #self.stop_media.setVisible(visible)
        self.jump_back_btn.setVisible(visible)
        self.jump_forward_btn.setVisible(visible)

    def _display_media(self, path):
        if self._is_video(path):
            self._enter_video_mode(path)
        else:
            self._enter_image_mode()
            self.display_image(path)

    def _enter_image_mode(self):
        self.media_player.stop()
        self.media_stack.setCurrentIndex(0)
        self.play_media.setEnabled(False)
        self._set_video_controls_visible(False)

    def _enter_video_mode(self, video_path):
        self.media_stack.setCurrentIndex(1)
        self.play_media.setEnabled(True)
        self._set_video_controls_visible(True)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()

    # Video controls

    def toggle_play_pause(self):
        if self.media_stack.currentIndex() != 1:
            return
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def jump_backward(self):
        self.media_player.setPosition(max(0, self.media_player.position() - 10000))

    def jump_forward(self):
        self.media_player.setPosition(
            min(self.media_player.duration(), self.media_player.position() + 10000)
        )

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self.media_player.setPosition(self.video_seek_slider.value())

    def _on_position_changed(self, position):
        if not self._slider_dragging:
            self.video_seek_slider.setValue(position)
        self.video_current_time.setText(self._format_time(position))

    def _on_duration_changed(self, duration):
        self.video_seek_slider.setRange(0, duration)
        self.video_total_time.setText(self._format_time(duration))

    def _on_player_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_media.setIcon(
                self.style().standardIcon(self.style().SP_MediaPause)
            )
        else:
            self.play_media.setIcon(QIcon(f"{DECLAS_ROOT}/icons/play.png"))

    @staticmethod
    def _format_time(ms):
        s = ms // 1000
        return f"{s // 60:02d}:{s % 60:02d}"

    # Map

    def show_on_map(self):
            # Initialize a folium map centered on the provided coordinates
            try:
                image_path = IMG_PATH[-1]
            except:
                missed_path()
                return

            try:
                _, gps_i = get_metadata(image_path)
                gps_i = gps_i.split(sep="\n")
                lat = float(gps_i[1].split(sep=":")[1].strip().split(sep=" ")[0].strip())
                lon = float(gps_i[2].split(sep=":")[1].strip().split(sep=" ")[0].strip())
                #alt = float(gps_i[3].split(sep=":")[1].strip().split(sep=" ")[0].strip())
                
                self.leaflet_map(lat = lat, lon = lon)

            except:
                missed_gps()

    def single_detection(self):
        try:
            parameters  = load_json()
            model_weight = load_weight()

            try:
                media_path = IMG_PATH[-1]
            except:
                missed_path()
                return

            # Extensions are self-contained
            _mt = parameters.get("model_type", "")
            _is_ext = _mt in INSTALLED_EXTENSIONS
            if not _is_ext:
                no_weight()
                return

            is_vid = self._is_video(media_path)
            vid_stride = parameters.get("vid_stride", 5)
            to_save = None

            if is_vid and not parameters.get("process_video", True):
                self.statusbar.showMessage("Video processing is disabled. Enable it in Inference Parameters.", MESSAGE_DELAY)
                return

            self.statusbar.showMessage("Running…")
            QApplication.processEvents()

            if parameters["task"] == "Detection":
                # Detection-extension path
                if _is_ext:
                    ext_info = INSTALLED_EXTENSIONS[_mt]
                    if ext_info["status"] != "ready":
                        self.statusbar.showMessage(
                            f"Extension '{_mt}' weights not downloaded.", MESSAGE_DELAY)
                        return
                    adapter = _load_adapter(ext_info, device=parameters["device"])
                    if is_vid:
                        result_dict = extension_video_classification(
                            video_path=media_path,
                            adapter=adapter,
                            conf_thres=parameters["conf"],
                            vid_stride=vid_stride,
                            class_filter=parameters.get("class_of_interest"),
                        )
                        to_save = split_json_obj(json_obj=result_dict, classif=True) if result_dict else None
                    else:
                        entry = extension_single_classification(
                            image_path=media_path,
                            adapter=adapter,
                            conf_thres=parameters["conf"],
                            class_filter=parameters.get("class_of_interest"),
                        )
                        to_save = str(entry)

            elif parameters["task"] == "Classification":
                model_type = parameters["model_type"]

                # Model-extension path
                if model_type in INSTALLED_EXTENSIONS:
                    ext_info = INSTALLED_EXTENSIONS[model_type]
                    if ext_info["status"] != "ready":
                        self.statusbar.showMessage(
                            f"Extension '{model_type}' weights not downloaded.", MESSAGE_DELAY)
                        return
                    adapter = _load_adapter(ext_info, device=parameters["device"])
                    if is_vid:
                        result_dict = extension_video_classification(
                            video_path=media_path,
                            adapter=adapter,
                            conf_thres=parameters["conf"],
                            vid_stride=vid_stride,
                            class_filter=parameters.get("class_of_interest"),
                        )
                        to_save = split_json_obj(json_obj=result_dict, classif=True) if result_dict else None
                    else:
                        result_dict = extension_single_classification(
                            image_path=media_path,
                            adapter=adapter,
                            conf_thres=parameters["conf"],
                            class_filter=parameters.get("class_of_interest"),
                        )
                        to_save = split_json_obj(json_obj=result_dict)

                # Built-in YoloV5 / YoloV8/9 path
            self.statusbar.showMessage("Done ✅", MESSAGE_DELAY)
            if to_save:
                self.view_detection.setEnabled(True)
                self.edit_inference.setEnabled(True)
                self.inference_result.setText(to_save)

        except Exception as e:
            general_error(e)


    def show_detection(self, image_path):
        try:
            image_path = IMG_PATH[-1]
        except:
            missed_path()
            return
        image_path = Path(Path(image_path).parent, "detections", Path(image_path).name)
        self._enter_image_mode()  # annotated result is always a JPEG, show image page
        self.display_image(str(image_path), message="Run detection or classification before to click 'View'")


    def edit_detection_result(self):
        self.inference_result.setReadOnly(False)
        self.apply_inference_edit.setEnabled(True)


    def save_inference_edit(self):
        new_value = self.inference_result.toPlainText()
        try:
            image_path = IMG_PATH[-1]
        except:
            missed_path()
            return

        try:#if is_valid_dict(new_value):
            fpath = [Path(Path(image_path).parent, "detections.json"),
                            Path(Path(image_path).parent.parent, "detections.json")]
                
            fpath_exist = [Path(Path(image_path).parent, "detections.json").exists(),
                        Path(Path(image_path).parent.parent, "detections.json").exists()]
            
            if any(fpath_exist):
                json_path = str(Path(fpath[fpath_exist.index(True)].parent, "detections.json"))

                #json_path = str(Path(Path(image_path).parent, "detections.json"))
                with open(json_path, "r") as det:
                    detections = json.load(fp=det)
                    right_key = [ky for _, ky in enumerate(list(detections.keys())) if ky.startswith(Path(image_path).stem)]
                    
                    if right_key == []: # single detection
                        detections = ast.literal_eval(new_value)
                    elif len(right_key) > 1: #classification
                        txt_split = new_value.split("\n\n###\n\n")
                
                        for ky, txt in zip(right_key, txt_split):
                            detections[ky] = ast.literal_eval(txt)
                    else:
                        keys = right_key[0]
                        detections[keys] = ast.literal_eval(new_value)

                with open(json_path, "w") as out_file:
                    json.dump(obj=detections, fp=out_file, indent=4)

            self.statusbar.showMessage("Change applied \u2705", MESSAGE_DELAY)

        except:#else:
            invalid_edit()

    def multiple_detection(self):
        try:
            parameters   = load_json()
            model_weight = load_weight()
            _mt = parameters.get("model_type", "")
            _is_ext = _mt in INSTALLED_EXTENSIONS
            if not _is_ext and (model_weight == "" or not Path(model_weight).exists()):
                no_weight()
                return
        except Exception as e:
            f"ERROR: {e}"

        try:
            main_subdir = load_json()
            folder_path = selected_folder
            if folder_path:
                media_inside = [fls for fls in Path(folder_path).iterdir()
                                if fls.is_file() and fls.suffix in (IMAGE_EXT | VIDEO_EXT)]
                if len(media_inside) == 0 and main_subdir["run_on_main_dir"]:
                    missed_folder()
                    return
                # Create the worker and pass the folder path
                self.worker = DetectionWorker(folder_path, main_subdir=main_subdir["run_on_main_dir"],
                                                 conf_thres= main_subdir["conf"])
                # Connect the signals to the appropriate slots
                self.worker.detection_done.connect(self.on_detection_done)
                self.worker.error_occurred.connect(self.on_detection_error)
                self.worker.log_message.connect(self.update_log)
                # Start the worker thread
                self.worker.start()
            else:
                missed_folder()
        except:
            missed_folder()

    def update_log(self, message):
        self.batch_detection_log.append(message)

    def on_detection_done(self, message):
        self.statusbar.showMessage(message, MESSAGE_DELAY)

    def on_detection_error(self, message):
        self.statusbar.showMessage(message, MESSAGE_DELAY)

    def open_extensions(self):
        """Open the Extension Manager dialog (non-modal — Declas stays interactive)."""
        # Re-use an already-open dialog instead of stacking multiple instances.
        if hasattr(self, "_ext_dlg") and self._ext_dlg.isVisible():
            self._ext_dlg.raise_()
            self._ext_dlg.activateWindow()
            return
        self._ext_dlg = ExtensionManagerDialog(parent=self)
        self._ext_dlg.extension_changed.connect(self._reload_extensions)
        self._ext_dlg.show()
        self._ext_dlg.raise_()
        self._ext_dlg.activateWindow()

    def open_publish_guidelines(self):
        """Open the Publish guidelines dialog."""
        if hasattr(self, "_pub_dlg") and self._pub_dlg.isVisible():
            self._pub_dlg.raise_()
            self._pub_dlg.activateWindow()
            return
        self._pub_dlg = PublishGuidelinesDialog(parent=self)
        self._pub_dlg.show()
        self._pub_dlg.raise_()
        self._pub_dlg.activateWindow()

    def _reload_extensions(self):
        """Refresh the global extension registry after install / remove."""
        global INSTALLED_EXTENSIONS
        try:
            INSTALLED_EXTENSIONS = _scan_extensions()
        except Exception:
            INSTALLED_EXTENSIONS = {}
        self.statusbar.showMessage("Extensions reloaded.", MESSAGE_DELAY)

    def build_table(self):
        image_or_dir = self.image_or_dir
        content_data = []
        inference_param = load_json()

        if image_or_dir:
            root_path = Path(image_or_dir)
            folder = root_path if root_path.is_dir() else root_path.parent
            try:
                run_on_main_dir = inference_param["run_on_main_dir"]
                if Path(image_or_dir).is_dir() and run_on_main_dir:
                    json_files = [str(Path(image_or_dir, "detections.json"))]
                elif Path(image_or_dir).is_dir() and not run_on_main_dir:
                    # Exclude detections.json inside 'frames/' subdirectories —
                    # those are intermediate per-frame files; the complete merged
                    # result is always in the parent directory's detections.json.
                    json_files = [str(p) for p in root_path.rglob('detections.json')
                                  if 'frames' not in p.parts]
                elif not Path(image_or_dir).is_dir():
                    json_files = [str(Path(Path(image_or_dir).parent, "detections.json"))]
                for jsf in json_files:
                    with open(jsf, "r") as content:
                        content = json.load(content)
                        if content == {}:
                            continue
                        for each_c in content:
                            content_data.append(content[each_c])

                success_table_build(f"Table built successfully and saved at {folder}")

                pd.DataFrame(content_data).to_csv(str(Path(folder, "detections.csv")), index=False)

            except Exception as e:
                unsuccess_table_build(f"{e}")

        else:
            missed_path()


## QThread
def process_directory(dp, log_queue):
    parameters  = load_json()
    model_weight = load_weight()

    _mt = parameters.get("model_type", "")
    _is_ext = _mt in INSTALLED_EXTENSIONS
    if not _is_ext:
        no_weight()
        return

    # Separate image files from video files
    all_files = [f for f in Path(dp).iterdir() if f.is_file()]
    image_files = [f for f in all_files if f.suffix in IMAGE_EXT]
    video_files = [f for f in all_files if f.suffix in VIDEO_EXT]

    if not image_files and not video_files:
        return "No media files found in directory."

    vid_stride = parameters.get("vid_stride", 5)
    process_video = parameters.get("process_video", True)

    # When video processing is disabled, ignore video files entirely
    if not process_video:
        video_files = []

    try:
        if parameters["task"] == "Detection":
            # Detection-extension path
            if _is_ext:
                ext_info = INSTALLED_EXTENSIONS[_mt]
                if ext_info["status"] != "ready":
                    if log_queue:
                        log_queue.put(f"Extension '{_mt}' weights not downloaded.")
                    return "Extension weights missing."
                adapter = _load_adapter(ext_info, device=parameters["device"])
                if image_files:
                    ext_suffix = image_files[0].suffix
                    extension_batch_classification(data_path=dp,
                                                   adapter=adapter,
                                                   conf_thres=parameters["conf"],
                                                   extension=ext_suffix,
                                                   class_filter=parameters.get("class_of_interest"))
                for vf in video_files:
                    if log_queue:
                        log_queue.put(f"Processing video: {vf.name}")
                    extension_video_classification(video_path=str(vf),
                                                   adapter=adapter,
                                                   conf_thres=parameters["conf"],
                                                   vid_stride=vid_stride,
                                                   log_queue=log_queue,
                                                   class_filter=parameters.get("class_of_interest"))

        elif parameters["task"] == "Classification":
            model_type = parameters["model_type"]

            # Model-extension batch path
            if model_type in INSTALLED_EXTENSIONS:
                ext_info = INSTALLED_EXTENSIONS[model_type]
                if ext_info["status"] != "ready":
                    if log_queue:
                        log_queue.put(f"❌ Extension '{model_type}' weights not downloaded.")
                    return "Extension weights missing."
                adapter = _load_adapter(ext_info, device=parameters["device"])

                if image_files:
                    extension = image_files[0].suffix
                    extension_batch_classification(data_path=dp,
                                                   adapter=adapter,
                                                   conf_thres=parameters["conf"],
                                                   extension=extension,
                                                   class_filter=parameters.get("class_of_interest"))

                for vf in video_files:
                    if log_queue:
                        log_queue.put(f"🎬 Processing video: {vf.name}")
                    extension_video_classification(video_path=str(vf),
                                                   adapter=adapter,
                                                   conf_thres=parameters["conf"],
                                                   vid_stride=vid_stride,
                                                   log_queue=log_queue,
                                                   class_filter=parameters.get("class_of_interest"))


        emoji = ['\U0001F38A', '\U0001F389', '\u2705', '\U0001F917']
        return f"Completed successfully {choice(emoji)}"

    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def process_directory_wrapper(args):
    dp, log_queue = args
    return process_directory(dp, log_queue)


class LogEmitter(QThread):
    log_message = pyqtSignal(str)

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                message = self.log_queue.get(timeout=0.1)
                self.log_message.emit(message)
            except:
                continue

    def stop(self):
        self.running = False


class DetectionWorker(QThread):
    detection_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, folder_path, main_subdir, conf_thres=0.55):
        super().__init__()
        self.folder_path = folder_path
        self.main_subdir = main_subdir
        self.conf_thres = conf_thres
        self.manager = Manager()  # Create a manager
        self.log_queue = self.manager.Queue()  # Use the manager to create a Queue
        self.log_emitter = LogEmitter(self.log_queue)
        self.log_emitter.log_message.connect(self.log_message.emit)

    def run(self):
        self.log_emitter.start()

        try:
            #self.status_update.emit("Running ...")
            dir_path = Path(self.folder_path)

            if self.main_subdir:
                results = process_directory(dir_path, self.log_queue)
               
                if "error" in results.lower():
                    self.error_occurred.emit(results)
                else:
                    self.detection_done.emit(results)
                

            else:
                all_dirs = [dir for dir in Path(dir_path).iterdir() if dir.is_dir()]
                if len(all_dirs) == 0:
                    self.error_occurred.emit("\u274C Choose a directory with structure 'deployment/station/species'")
                    return
                
                for sub_dir in all_dirs:
                    self.log_queue.put(f"\U0001F504 RUNNING ON DIR: {sub_dir}\n")
                    species_dirs = [dir for dir in Path(sub_dir).iterdir() if dir.is_dir()]
                    if len(species_dirs) == 0:
                        self.error_occurred.emit("\u274C Choose a directory with structure 'deployment/station/species'")
                        return

                # Use ThreadPoolExecutor instead of ProcessPoolExecutor
                    try:
                        with ThreadPoolExecutor() as executor:
                            args_list = [(species_dir, self.log_queue) for species_dir in species_dirs]
                            results = executor.map(process_directory_wrapper, args_list)

                            emoji = ['\U0001F38A', '\U0001F389', '\u2705', '\U0001F917']
                            msg = f"Completed successfully {choice(emoji)}"

                            has_error = ['error' in x.lower() for x in results]
                            if any(has_error):
                                error_index = [x for x, y in enumerate(has_error) if y == True]
                                self.error_occurred.emit(results[error_index[0]])
                            else:
                                self.detection_done.emit(msg)
                    except Exception as e:
                        self.log_queue.put(f"Error processing directory {sub_dir}: {e}")
                        continue
                            
        finally:
            self.log_emitter.stop()
            self.log_emitter.wait()