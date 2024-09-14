import shutil
from turtle import st
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDir, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QIcon, QPixmap, QFontDatabase
from PyQt5.QtWidgets import QMainWindow, QAction, \
    QFileDialog, QFileSystemModel, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineView, QWebEngineSettings, QWebEnginePage
import folium
import sys, torch, os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
import pandas as pd
os.environ['QT_API'] = 'pyqt5'

##
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))
from ModelParameter import ModelParameter
from WeightTable import WeightTable
from Bases import *
from Detection import *
from ErrorWarning import *

##
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_END = ["*.JPG", "*.JPEG", "*.jpg", "*.jpeg"]
MEDIA_SUFFIX = [".JPG", ".JPEG", ".jpg", ".jpeg", ".mp4"]
IMG_PATH = []
SINGLE_DETECTION = {}
DECLAS_ROOT = Path(__file__).resolve().parent.parent

try:
    from ctypes import windll  # Only exists on Windows.
    declas_id = 'declas.1.0.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(declas_id)
except ImportError:
    pass

# 

# 

class Declas(QMainWindow):
    def __init__(self) -> None:
        super(Declas, self).__init__()
        loadUi(f"{DECLAS_ROOT}/ui/Declas.ui", self)

        self.icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.jpg') )
        icon_file = self.icon_file.replace("sources", "")
        self.setWindowIcon(QIcon(icon_file))
        self.setWindowTitle("Declas 1.0.0")

        # Load a custom font
        font_path = str(Path(DECLAS_ROOT, "sources/styles/Montserrat-Regular.ttf"))
        fontid = QFontDatabase.addApplicationFont(font_path)
        font_family = QFontDatabase.applicationFontFamilies(fontid)
        #self.setFont(QFont(font_family, 12))

        self.action_models_parameter.triggered.connect(self.models_parameters)
        # Initialize default inference parameters
        self.inference_param = {
            "conf": 0.55,
            "imgsz": (1920, 1440),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_det": 300,
            "vid_stride": 1,
            "class_of_interest": "animal",
            "half": False,
            "run_on_main_dir": False,
            "task": "Detection", 
            "model_type": "MegaDetectorV6"
        }

        dump_json(dict_obj=self.inference_param)
        self.image_or_dir = None
        self.model_weight = load_weight()

        # IMPORT FILE
        self.action_import_dc_file.triggered.connect(self.import_dc_file)
        self.action_quit.triggered.connect(self.quit_declas)
        # MODE
        #self.action_dark.triggered.connect(self.set_dark_mode)
        self.action_light.triggered.connect(self.set_light_mode)
        # TOOLBAR
        select_dir = QAction(QIcon(f"{DECLAS_ROOT}/icons/folder.png"), "Select directory", self)
        select_dir.triggered.connect(self.select_folder)

        select_image = QAction(QIcon(f"{DECLAS_ROOT}/icons/image.png"), "Select image", self)
        select_image.triggered.connect(self.select_an_image)

        globe = QAction(QIcon(f"{DECLAS_ROOT}/icons/globe.png"), "Show on map", self)
        globe.triggered.connect(self.show_on_map)

        run_inference = QAction(QIcon(f"{DECLAS_ROOT}/icons/run.png"), "Run Declas on single image", self)
        run_inference.triggered.connect(self.single_detection)

        batch_inference = QAction(QIcon(f"{DECLAS_ROOT}/icons/batch.png"), "Run Declas on folder(s)", self)
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
        self.file_model.setNameFilters(IMG_END)
        self.file_model.setNameFilterDisables(False)

        self.dir_tree_view.setModel(self.file_model)
        self.dir_tree_view.hideColumn(1)
        self.dir_tree_view.hideColumn(2)
        self.dir_tree_view.hideColumn(3)

        self.dir_tree_view.selectionModel().selectionChanged.connect(self.on_image_selected)
        self.dir_tree_view.selectionModel().selectionChanged.connect(self.get_next_and_previous_media)

        # DISPLAY IMAGE
        self.previous_media.hide()
        self.next_media.hide()
        self.play_media.hide()
        self.play_media.setEnabled(False)
        self.previous_media.clicked.connect(self.show_previous_media)
        self.next_media.clicked.connect(self.show_next_media)
        self.current_selected_media = None

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

        # ADD MODEL
        self.action_add_models.triggered.connect(self.add_model)
        self.action_show_models.triggered.connect(self.show_model)

        ## BUILD DETECTION TABLE
        #self.action_build_table.triggered.connect(self.build_table)

    def leaflet_map(self, lon, lat, init = False):
        # Create a QWebEngineView to display the map
        if init:
            self.display_map.setUrl(QUrl.fromLocalFile(f"{DECLAS_ROOT}/sources/tile.html"))
        else:
            folium_map = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], popup=f"{lon} | {lat}").add_to(folium_map)
            # Save the map to an HTML file
            map_path = os.path.abspath(f"{DECLAS_ROOT}/sources/updated_map.html")
            folium_map.save(map_path)
            # Load the updated map into QWebEngineView
            self.display_map.setUrl(QUrl.fromLocalFile(map_path))

    def import_dc_file(self):
        selected_json = QFileDialog.getOpenFileName(self, "Select json file", ".", "JSON (*json)")
        selected_json = selected_json[0]

        if selected_json:
            self.image_or_dir = selected_json
            self.statusbar.showMessage("File imported ✅", 5000)

    def quit_declas(self):
            sys.exit()

    def set_light_mode(self):
        with open(f"{DECLAS_ROOT}/sources/styles/light.qss", "r") as file:
            self.setStyleSheet(file.read())

    def models_parameters(self):
        current_set = load_json()

        mp = ModelParameter()
        mp.yolo_conf.setValue(current_set["conf"]) # conf

        imgsz = [str(i) for i in current_set["imgsz"]]
        imgsz = " ".join(imgsz) if len(imgsz) == 2 else imgsz

        mp.yolo_imgsz.setText(imgsz) #imgsz
        mp.yolo_device.setItemText(0, current_set["device"])
        mp.yolo_max_det.setValue(current_set["max_det"]) # max_det
        mp.yolo_vid_stride.setValue(current_set["vid_stride"]) # vid_stride
        mp.yolo_classes.setItemText(0, current_set["class_of_interest"])
        mp.yolo_half.setChecked(current_set["half"]) #run_on_main_dir
        mp.run_on_main_dir.setChecked(current_set["run_on_main_dir"]) #run_on_main_dir
        mp.task.setCurrentText(current_set["task"])
        mp.model_type.setCurrentText(current_set["model_type"])

        mp.setWindowModality(Qt.ApplicationModal)
        if mp.exec_() == mp.Accepted:  # If the dialog is accepted
            self.inference_param = mp.inference_param
            dump_json(dict_obj=self.inference_param)

    def select_folder(self):
        global selected_folder
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if selected_folder :
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

        self.statusbar.showMessage("Split applied ✅", 1000)


    def select_an_image(self):
        selected_image = QFileDialog.getOpenFileName(self, "Select an image", ".", "Images (*jpg *JPG *jpeg *JPEG)")
        selected_image = selected_image[0]
        if selected_image :
            self.display_image(selected_image)
            self.image_or_dir = selected_image

            try:
                metadata_i, _ = get_metadata(selected_image)
                self.metadata_text.setText(metadata_i)
            except:
                self.metadata_text.setText(get_metadata(selected_image))

            IMG_PATH.append(selected_image)
            return selected_image
        

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
                self.display_image(file_path)

                try:
                    metadata_i, _ = get_metadata(file_path)
                    self.metadata_text.setText(metadata_i)
                except:
                    self.metadata_text.setText(get_metadata(file_path))

                # Add inference if avalaible
                fpath = [Path(Path(file_path).parent, "detections.json"),
                         Path(Path(file_path).parent.parent, "detections.json")]
                
                fpath_exist = [Path(Path(file_path).parent, "detections.json").exists(),
                         Path(Path(file_path).parent.parent, "detections.json").exists()]
                
                if any(fpath_exist):
                    json_path = str(Path(fpath[fpath_exist.index(True)].parent, "detections.json"))
                    
                    try:
                        with open(json_path, "r") as det:
                            detections = json.load(fp=det)
                            detections = detections[Path(file_path).stem]
                            self.inference_result.setText(str(detections))
                            self.edit_inference.setEnabled(True)
                    except:
                        pass

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

        if selected_folder:
            if selected_file_path:
                sf = str(Path(selected_file_path).parent)
            else:
                sf = selected_folder

            global all_files
            all_files = [str(fl) for fl in Path(sf).iterdir() if not fl.is_dir() and fl.suffix in MEDIA_SUFFIX]
            
            try:
                idx = all_files.index(str(Path(selected_file_path)))# selected_file_path from on_image_selected()
                self.current_selected_media = idx
                return idx, all_files
            
            except Exception as e:
                f"ERROR {e}"

            
    def show_previous_media(self):
        if self.current_selected_media > 0:
            self.current_selected_media -= 1 

            try:
                previous_med = all_files[self.current_selected_media]
                self.display_image(previous_med)
            except Exception as e:
                pass
        
    
    def show_next_media(self):
        if self.current_selected_media >= 0 and self.current_selected_media < (len(all_files) - 1):
            self.current_selected_media += 1
            try:
                previous_med = all_files[self.current_selected_media]
                self.display_image(previous_med)
            except Exception as e:
                pass



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
                alt = float(gps_i[3].split(sep=":")[1].strip().split(sep=" ")[0].strip())

                self.leaflet_map(lat = lat, lon = lon)

            except:
                missed_gps()

    def single_detection(self):
        try:
            parameters = load_json()
            model_weight = load_weight()

            try:
                image_path = IMG_PATH[-1]
            except:
                missed_path()
                return

            if model_weight == "" or not Path(model_weight).exists():
                no_weight()
                return

            to_save = get_results(weights=model_weight, 
                                  image_path = image_path, 
                                  model_type=parameters["model_type"],
                                  conf_thres = parameters["conf"], 
                                  device=parameters["device"],
                                  class_of_interest=parameters["class_of_interest"], 
                                  save_json=True)
            print(parameters["conf"])

            if to_save:
                self.view_detection.setEnabled(True)
                self.edit_inference.setEnabled(True)

            self.inference_result.setText(f"{to_save}")

        except Exception as e:
            general_error(e)


    def show_detection(self, image_path):
        try:
            image_path = IMG_PATH[-1]
        except:
            missed_path()
            return
        image_path = Path(Path(image_path).parent, "detections", Path(image_path).name)
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

        if is_valid_dict(new_value):
            fpath = [Path(Path(image_path).parent, "detections.json"),
                         Path(Path(image_path).parent.parent, "detections.json")]
                
            fpath_exist = [Path(Path(image_path).parent, "detections.json").exists(),
                        Path(Path(image_path).parent.parent, "detections.json").exists()]
            
            if any(fpath_exist):
                json_path = str(Path(fpath[fpath_exist.index(True)].parent, "detections.json"))

                json_path = str(Path(Path(image_path).parent, "detections.json"))
                with open(json_path, "r") as det:
                    detections = json.load(fp=det)
                    detections[Path(image_path).stem] = ast.literal_eval(new_value)

                with open(json_path, "w") as out_file:
                    json.dump(obj=detections, fp=out_file, indent=4)

            else:
                SINGLE_DETECTION[str(Path(image_path).stem)] = ast.literal_eval(new_value)

                out_file = open(str(Path(Path(image_path).parent, "detections.json")), "w")
                json.dump(SINGLE_DETECTION, out_file, indent = 4)
                out_file.close()
            self.statusbar.showMessage("Change applied ✅", 1000)

        else:
            invalid_edit()


    def multiple_detection(self):
        try:
            model_weight = load_weight()
            if model_weight == "" or not Path(model_weight).exists():
                no_weight()
                return
        except Exception as e:
            f"ERROR: {e}"

        main_subdir = load_json()
        try:
            folder_path = selected_folder
            if folder_path:
                image_inside = [fls for fls in list(Path(folder_path).iterdir()) if fls.suffix in [".JPG", ".JPEG", ".jpg", ".jpeg"]]
                if len(image_inside) == 0 and main_subdir["run_on_main_dir"]:
                    missed_folder()
                    return
                self.statusbar.showMessage("Running detection ...", 1000)
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
        except Exception as e:
            missed_folder()

    def update_log(self, message):
        self.batch_detection_log.append(message)

    def on_detection_done(self, message):
        self.statusbar.showMessage(message, 5000)  # Show message for 5 seconds

    def on_detection_error(self, message):
        self.statusbar.showMessage(message, 5000)  # Show error message for 5 seconds

    def add_model(self):
        model_path = QFileDialog.getOpenFileName(caption="Select a model", directory=".", filter="Models (*.pt *.onnx)")
        model_path = model_path[0]
        if model_path:
            file_path = Path(f"{DECLAS_ROOT}/config/model_/mdls/mdl.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if not file_path.exists():
                file_path.touch(exist_ok=True)
                with open(str(file_path), "w") as initiate:
                    json.dump({}, initiate)

            with open(str(file_path), "r") as current_state:
                current_state = json.load(current_state)
                if Path(model_path).stem in current_state:
                    self.statusbar.showMessage(f"Weight {Path(model_path).stem.lower()} exists", 8*1000)
                    return
                current_state[Path(model_path).stem] = {
                    "Id" : f"{date_to_id()}",
                    "Path": str(Path(model_path)),
                    "Size": f"{round((Path(model_path).stat().st_size)/(1024 * 1024))} MB",
                    "Add date": datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                }

            with open(str(file_path), "w") as mdl:
                json.dump(current_state, mdl)
                self.statusbar.showMessage(f"Weight {Path(model_path).stem.lower()} added", 8*1000)

        else:
            self.statusbar.showMessage(f"Weight cannot be added", 8*1000)

    def remove_model(self, weights):
        weight_path = Path(f"{DECLAS_ROOT}/config/model_/mdls/mdl.json")
        if not weight_path.exists():
            no_weight()
        else:
            with open(weight_path, "r") as wei:
                weights = json.load(wei)
                if weights == {}:
                    no_weight()
                else:
                    weight_table = WeightTable(weight_data=weights)
                    weight_table.remove_row(weight_data=weights)

    def show_model(self):
        weight_path = Path(f"{DECLAS_ROOT}/config/model_/mdls/mdl.json")
        if not weight_path.exists():
            no_weight()
        else:
            try:
                with open(weight_path, "r") as wei:
                    weights = json.load(wei)
                    if weights == {} or weights == "":
                        no_weight()
                        return
                    weight_table = WeightTable(weight_data=weights)
                    for wgt in weights:
                        individual = weights[str(wgt)]
                        weight_table.add_row(id = individual["Id"],
                                                 weight=f"{wgt}",
                                                 path=individual["Path"],
                                                 size=individual["Size"],
                                                 add_date = individual["Add date"]
                                                 )
                    weight_table.show()

            except Exception as e:
                no_weight()

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
                    json_files =  list(root_path.rglob('detections.json'))
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
    parameters = load_json()

    model_weight = load_weight()

    if model_weight == "":
        no_weight()
        return

    try:
        detection_model = MegaDetectorV6(device=parameters["device"], 
                                         weights=model_weight, 
                                         pretrained=False)

        detection_model.batch_image_detection(dp,
                                            batch_size=16, 
                                            conf_thres=parameters["conf"],
                                            class_of_interest=parameters["class_of_interest"],
                                            extension="JPG",
                                            log_queue = log_queue)

        return "Completed successfully ✅"

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
    #progress_signal = pyqtSignal(int)  # Signal to send progress updates

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
            dir_path = Path(self.folder_path)

            if self.main_subdir:
                results = process_directory(dir_path, self.log_queue)

                for result in results:
                        if "error" in result.lower():
                            self.error_occurred.emit(result)
                        else:
                            self.detection_done.emit(result)
                

            else:
                all_dirs = [dir for dir in Path(dir_path).iterdir() if dir.is_dir()]
                for sub_dir in all_dirs:
                    self.log_queue.put(f"ON DIR: {sub_dir}")
                    species_dirs = [dir for dir in Path(sub_dir).iterdir() if dir.is_dir()]

                # Use ThreadPoolExecutor instead of ProcessPoolExecutor
                with ThreadPoolExecutor() as executor:
                    args_list = [(species_dir, self.log_queue) for species_dir in species_dirs]
                    results = executor.map(process_directory_wrapper, args_list)

                    for result in results:
                        if "error" in result.lower():
                            self.error_occurred.emit(result)
                        else:
                            self.detection_done.emit(result)
                            
        finally:
            self.log_emitter.stop()
            self.log_emitter.wait()