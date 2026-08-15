from random import choice
import shutil
from turtle import st
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDir, QThread, pyqtSignal, QUrl, QTimer, QSettings
from PyQt5.QtGui import QIcon, QPixmap, QFontDatabase, QPainter, QColor
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QFileSystemModel,
                             QApplication, QWidget, QDialog, QLineEdit, QComboBox, QCheckBox,
                             QDateEdit, QScrollArea, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QFormLayout, QGroupBox, QMenu,
                             QTableWidget, QHeaderView, QSpinBox)
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
from MagnifierOverlay import MagnifierOverlay, MagnifierFilter
from TagsDialog import (TagsDialog, load_tag_definitions,
                        load_media_tags, save_media_tags)

##
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scan installed model extensions at startup
try:
    from model_extensions.loader import scan_extensions, load_adapter
    INSTALLED_EXTENSIONS = scan_extensions()
except Exception:
    INSTALLED_EXTENSIONS = {}

IMAGE_EXT  = {".JPG", ".JPEG", ".jpg", ".jpeg"}
VIDEO_EXT  = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

MEDIA_SUFFIX = [f"*{ext}" for ext in IMAGE_EXT | VIDEO_EXT] #["*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.mp4", "*.avi"]
MEDIA_EXT  = {p.lstrip("*") for p in MEDIA_SUFFIX} # suffix comparison set

IMG_PATH = []
SINGLE_DETECTION = {}
DECLAS_ROOT = Path(__file__).resolve().parent.parent

MESSAGE_DELAY = 8000


def themed_icon(name: str, dark: bool) -> QIcon:
    """Return a QIcon for *name*, white-tinted when *dark* is True.

    Looks for icons/svg/<name>.svg first; falls back to icons/<name>.png.
    White tinting uses SourceIn composition so the alpha channel is preserved
    and only the colour of each pixel changes.
    """
    png = DECLAS_ROOT / "icons" / f"{name}.png"
    path = str(png)
    if not dark:
        return QIcon(path)
    px = QPixmap(path)
    if px.isNull():
        return QIcon()
    out = QPixmap(px.size())
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, px)
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.fillRect(out.rect(), QColor(255, 255, 255))
    p.end()
    return QIcon(out)

try:
    from ctypes import windll  # Only exists on Windows.
    declas_id = 'com.declas.app'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(declas_id)
except ImportError:
    pass


def play_notification_sound(sound_file=""):
    path = (sound_file or "").strip()
    if not path or not Path(path).exists():
        default = DECLAS_ROOT / "notifications" / "bell-notification-933.wav"
        path = str(default) if default.exists() else ""
    if not path:
        return
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except ImportError:
        import subprocess
        try:
            subprocess.Popen(["afplay", path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass
    except Exception:
        pass


class GeneralSettingsDialog(QDialog):
    def __init__(self, app_settings, notifications_dir, parent=None):
        super().__init__(parent)
        self.notifications_dir = Path(notifications_dir)
        self.setWindowTitle("General Settings")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(440)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Notification
        notif_group = QGroupBox("Notification")
        notif_layout = QVBoxLayout(notif_group)
        self.notify_check = QCheckBox("Enable notification sound on completion")
        self.notify_check.setChecked(app_settings.value("notification_sound", True, type=bool))
        notif_layout.addWidget(self.notify_check)

        sound_row = QHBoxLayout()
        sound_row.addWidget(QLabel("Sound:"))
        self.sound_combo = QComboBox()
        saved = app_settings.value("notification_sound_file", "", type=str)
        if self.notifications_dir.exists():
            for wav in sorted(self.notifications_dir.glob("*.wav")):
                label = wav.stem.replace("-", " ").replace("_", " ").title()
                self.sound_combo.addItem(label, wav.name)
        idx = self.sound_combo.findData(saved)
        self.sound_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.sound_combo.currentIndexChanged.connect(self.preview_sound)
        sound_row.addWidget(self.sound_combo, 1)
        notif_layout.addLayout(sound_row)
        layout.addWidget(notif_group)

        # Language
        lang_group = QGroupBox("Language")
        lang_layout = QFormLayout(lang_group)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English (en)", "Français (fr)"])
        current_lang = app_settings.value("language", "en", type=str)
        self.lang_combo.setCurrentIndex(0 if current_lang == "en" else 1)
        self.lang_combo.setEnabled(False)
        lang_layout.addRow("Language:", self.lang_combo)
        layout.addWidget(lang_group)

        # Appearance
        appear_group = QGroupBox("Appearance")
        appear_layout = QFormLayout(appear_group)
        self.appear_combo = QComboBox()
        self.appear_combo.addItems(["Light", "Dark", "System"])
        self.appear_combo.setCurrentText(app_settings.value("theme", "System", type=str))
        appear_layout.addRow("Theme:", self.appear_combo)
        layout.addWidget(appear_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def preview_sound(self):
        name = self.sound_combo.currentData()
        if name:
            play_notification_sound(str(self.notifications_dir / name))

    def get_settings(self):
        return {
            "notification_sound": self.notify_check.isChecked(),
            "notification_sound_file": self.sound_combo.currentData(),
            "language": "en" if self.lang_combo.currentIndex() == 0 else "fr",
            "theme": self.appear_combo.currentText(),
        }


class SpeciesCard(QFrame):
    """One editable card showing species name + count for an inference result entry."""
    def __init__(self, json_key, species, count, entry=None, parent=None):
        super().__init__(parent)
        self.json_key = json_key
        self.entry = entry or {}
        self.setFrameShape(QFrame.StyledPanel)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(8)

        sp_col = QVBoxLayout()
        sp_col.setSpacing(2)
        sp_col.addWidget(QLabel("Species"))
        self.species_edit = QLineEdit(str(species))
        self.species_edit.setPlaceholderText("Species name")
        sp_col.addWidget(self.species_edit)

        cnt_col = QVBoxLayout()
        cnt_col.setSpacing(2)
        cnt_col.addWidget(QLabel("Count"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 9999)
        self.count_spin.setValue(int(count))
        cnt_col.addWidget(self.count_spin)

        outer.addLayout(sp_col, 2)
        outer.addLayout(cnt_col, 1)

        self.rm_btn = QPushButton("−")
        self.rm_btn.setFixedSize(22, 22)
        self.rm_btn.setStyleSheet("padding: 0;")
        self.rm_btn.setToolTip("Remove this entry")
        outer.addWidget(self.rm_btn, 0, Qt.AlignTop)


class Declas(QMainWindow):
    def __init__(self) -> None:
        super(Declas, self).__init__()
        loadUi(f"{DECLAS_ROOT}/ui/Declas.ui", self)

        self.icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.png') )
        icon_file = self.icon_file.replace("sources", "")
        self.setWindowIcon(QIcon(icon_file))
        self.setWindowTitle("Declas 1.3.0")

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
        self.action_light.triggered.connect(self.set_light_mode)
        if hasattr(self, 'action_dark'):
            self.action_dark.triggered.connect(self.set_dark_mode)
        if hasattr(self, 'action_system'):
            self.action_system.triggered.connect(self.set_system_mode)
        # TOOLBAR
        self._dark_mode = False  # updated by set_light/dark_mode; used by refresh_icons
        self.add_entry_btn = None  # created later during tags-tab setup

        self.action_select_dir = QAction(themed_icon("folder", False), "Select directory", self)
        self.action_select_dir.triggered.connect(self.select_folder)

        self.action_select_image = QAction(themed_icon("image", False), "Select media", self)
        self.action_select_image.setToolTip("Select a single image or video")
        self.action_select_image.triggered.connect(self.select_an_image)

        self.zoom_action = QAction(themed_icon("zoom", False), "Zoom lens", self)
        self.zoom_action.setToolTip("Hover over the image to magnify")
        self.zoom_action.setCheckable(True)

        self.action_globe = QAction(themed_icon("globe", False), "Show on map", self)
        self.action_globe.triggered.connect(self.show_on_map)

        self.action_run = QAction(themed_icon("run", False), "Run on current media", self)
        self.action_run.triggered.connect(self.single_detection)

        self.action_batch = QAction(themed_icon("batch", False), "Run folder(s)", self)
        self.action_batch.triggered.connect(self.multiple_detection)

        self.action_build_tables = QAction(themed_icon("table", False), "Build table from detection/classification", self)
        self.action_build_tables.triggered.connect(self.build_table)

        self.action_split = QAction(themed_icon("split", False), "Target or No target split", self)
        self.action_split.triggered.connect(self.filter_detection)

        self.tool_bar1.addAction(self.action_select_dir)
        self.tool_bar1.addAction(self.action_select_image)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(self.action_globe)
        self.tool_bar1.addAction(self.action_run)
        self.tool_bar1.addAction(self.action_batch)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(self.action_build_tables)
        self.tool_bar1.addSeparator()
        self.tool_bar1.addAction(self.action_split)
        self.tool_bar1.addAction(self.zoom_action)


        # MAGNIFIER ZOOM LENS
        self.mag_overlay = MagnifierOverlay()
        self.mag_filter  = MagnifierFilter(self.image_display, self.mag_overlay)
        self.image_display.setMouseTracking(True)
        self.image_display.installEventFilter(self.mag_filter)
        self.zoom_action.toggled.connect(self.mag_filter.set_active)

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
        self.slider_dragging = False
        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_display)

        self.play_media.clicked.connect(self.toggle_play_pause)
        self.jump_back_btn.clicked.connect(self.jump_backward)
        self.jump_forward_btn.clicked.connect(self.jump_forward)

        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.stateChanged.connect(self.on_player_state_changed)

        self.video_seek_slider.sliderPressed.connect(self.on_slider_pressed)
        self.video_seek_slider.sliderReleased.connect(self.on_slider_released)

        self.set_video_controls_visible(False)

        # Ensure image page is shown at startup and video area is never black
        self.media_stack.setCurrentIndex(0)
        self.video_page.setStyleSheet("background-color: black;")
        self.video_display.setStyleSheet("background-color: black;")

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

        self.apply_inference_edit.hide()

        # Replace the plain QTextEdit with a scrollable cards form
        self.updating_inference = False
        self.inference_cards = []
        self.current_json_path = None
        self.current_image_path_for_form = None

        inference_inner = QWidget()
        self.inference_form_vlay = QVBoxLayout(inference_inner)
        self.inference_form_vlay.setAlignment(Qt.AlignTop)
        self.inference_form_vlay.setContentsMargins(4, 4, 4, 4)
        self.inference_form_vlay.setSpacing(6)

        self.inference_scroll = QScrollArea()
        self.inference_scroll.setWidgetResizable(True)
        self.inference_scroll.setFrameShape(QFrame.NoFrame)
        self.inference_scroll.setWidget(inference_inner)

        tab_grid = self.inference_result.parent().layout()
        tab_grid.removeWidget(self.inference_result)
        self.inference_result.hide()
        tab_grid.addWidget(self.inference_scroll, 0, 0, 1, 4)

        self.edit_inference.setText("Add species")

        # STATUS BAR
        self.statusbar = self.statusBar

        # SPINNER (bottom-left, shown during batch detection)
        self.spinner_label = QLabel("")
        self.spinner_label.setVisible(False)
        self.statusbar.addWidget(self.spinner_label)
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(100)
        self.spinner_timer.timeout.connect(self.tick_spinner)

        # LOG
        self.batch_detection_log.setReadOnly(True)

        # MODEL EXTENSIONS
        self.action_ext = QAction(themed_icon("extensions", False), "Extensions", self)
        self.action_ext.setToolTip("Browse and install model extensions from the online registry")
        self.action_ext.triggered.connect(self.open_extensions)
        self.menuModels.addAction(self.action_ext)

        # PUBLISH GUIDELINES
        self.action_pub = QAction(themed_icon("publish", False), "Publish", self)
        self.action_pub.setToolTip("Learn how to publish your own model extension")
        self.action_pub.triggered.connect(self.open_publish_guidelines)
        self.menuModels.addAction(self.action_pub)

        # CUSTOM TAGS — settings menu entry
        self.action_tags = QAction(themed_icon("tags", False), "Tags", self)
        self.action_tags.setToolTip("Define custom tags")
        self.action_tags.triggered.connect(self.open_tags_dialog)
        self.menuSetting.addAction(self.action_tags)

        # GENERAL SETTINGS
        self.app_settings = QSettings("Declas", "Declas")
        self.action_general = QAction(themed_icon("general_settings", False), "General", self)
        self.action_general.triggered.connect(self.open_general_settings)
        self.menuSetting.addAction(self.action_general)

        # Apply saved theme at startup
        saved_theme = self.app_settings.value("theme", "System", type=str)
        if saved_theme == "Light":
            self.set_light_mode()
        elif saved_theme == "Dark":
            self.set_dark_mode()
        else:
            self.set_system_mode()

        # CUSTOM TAGS — tab next to Inference
        self.tags_tab = QWidget()
        self.tag_cards = [] # list of (card_widget, widgets_dict)
        self.tag_definitions = [] # list of tag defs
        self.updating_tags = False # guard: suppress auto-save during load

        self.tags_container = QWidget()
        self.tags_vbox = QVBoxLayout(self.tags_container)
        self.tags_vbox.setContentsMargins(4, 4, 4, 4)
        self.tags_vbox.setSpacing(6)

        tags_scroll = QScrollArea()
        tags_scroll.setWidgetResizable(True)
        tags_scroll.setFrameShape(tags_scroll.NoFrame)
        tags_scroll.setWidget(self.tags_container)

        self.add_entry_btn = QPushButton(themed_icon("add", self._dark_mode), " Add entry ")
        self.add_entry_btn.clicked.connect(lambda: self.add_tag_card())

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.add_entry_btn)
        btn_row.addStretch()

        tab_layout = QVBoxLayout(self.tags_tab)
        tab_layout.addWidget(tags_scroll)
        tab_layout.addLayout(btn_row)

        self.tabWidget.addTab(self.tags_tab, "Custom Tags")
        self.build_tags_form()

        ## BUILD DETECTION TABLE
        #self.action_build_table.triggered.connect(self.build_table)

    def open_general_settings(self):
        dlg = GeneralSettingsDialog(
            self.app_settings,
            DECLAS_ROOT / "notifications",
            parent=self
        )
        if dlg.exec_() == dlg.Accepted:
            s = dlg.get_settings()
            for key, val in s.items():
                self.app_settings.setValue(key, val)
            theme = s["theme"]
            if theme == "Light":
                self.set_light_mode()
            elif theme == "Dark":
                self.set_dark_mode()
            else:
                self.set_system_mode()

    # ── Custom Tags
    def open_tags_dialog(self):
        dlg = TagsDialog(parent=self)
        if dlg.exec_() == dlg.Accepted:
            self.build_tags_form()
            # Reload values for whatever media is currently displayed
            try:
                self.update_custom_tags(IMG_PATH[-1])
            except IndexError:
                pass

    def build_tags_form(self):
        self.updating_tags = True
        while self.tags_vbox.count():
            item = self.tags_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_cards = []

        self.tag_definitions = load_tag_definitions()
        if not self.tag_definitions:
            lbl = QLabel("No tags defined. Go to Setting > Tags to add some.")
            self.tags_vbox.addWidget(lbl)
            self.updating_tags = False
            return

        self.tags_vbox.addStretch()
        self.updating_tags = False
        self.add_tag_card()

    def add_tag_card(self, values=None):
        from PyQt5.QtGui import QDoubleValidator, QIntValidator
        from PyQt5.QtCore import QDate

        if not self.tag_definitions:
            return

        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Entry {len(self.tag_cards) + 1}"))
        header.addStretch()
        rm_btn = QPushButton("−")
        rm_btn.setFixedSize(22, 22)
        rm_btn.setStyleSheet("padding: 0;")
        rm_btn.setToolTip("Remove this entry")
        rm_btn.clicked.connect(lambda: self.remove_tag_card(card))
        header.addWidget(rm_btn)
        card_layout.addLayout(header)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        widgets = {}
        for tag in self.tag_definitions:
            title  = tag["title"]
            type_  = tag["type"]
            predef = tag.get("values", [])
            raw    = (values or {}).get(title, "")

            if predef:
                w = QComboBox()
                w.addItems(predef)
                if raw in predef:
                    w.setCurrentText(str(raw))
                w.currentIndexChanged.connect(self.auto_save_tags)
            elif type_ == "boolean":
                w = QCheckBox()
                w.setChecked(bool(raw))
                w.stateChanged.connect(self.auto_save_tags)
            elif type_ == "date":
                w = QDateEdit()
                w.setCalendarPopup(True)
                w.setDate(QDate.fromString(str(raw), "yyyy-MM-dd") if raw
                          else QDate.currentDate())
                w.dateChanged.connect(self.auto_save_tags)
            elif type_ in ("float", "double"):
                w = QLineEdit(str(raw) if raw is not None else "")
                w.setValidator(QDoubleValidator())
                w.textChanged.connect(self.auto_save_tags)
            elif type_ == "integer":
                w = QLineEdit(str(raw) if raw is not None else "")
                w.setValidator(QIntValidator())
                w.textChanged.connect(self.auto_save_tags)
            else:
                w = QLineEdit(str(raw) if raw is not None else "")
                w.textChanged.connect(self.auto_save_tags)

            form.addRow(title + ":", w)
            widgets[title] = w

        card_layout.addLayout(form)

        insert_pos = max(0, self.tags_vbox.count() - 1)
        self.tags_vbox.insertWidget(insert_pos, card)
        self.tag_cards.append((card, widgets))

    def remove_tag_card(self, card):
        self.tags_vbox.removeWidget(card)
        self.tag_cards = [(c, w) for c, w in self.tag_cards if c is not card]
        card.deleteLater()
        self.auto_save_tags()

    def det_folder(self, file_path):
        """Return the detections/ folder for a given media path.
        Works whether file_path points to the original or to an image already
        inside a detections/ subfolder."""
        p = Path(file_path)
        if p.parent.name == "detections":
            return p.parent # already inside detections/
        return p.parent / "detections"

    def update_custom_tags(self, file_path):
        if not self.tag_definitions:
            return
        self.updating_tags = True
        for card, _ in self.tag_cards:
            self.tags_vbox.removeWidget(card)
            card.deleteLater()
        self.tag_cards = []
        det_folder = self.det_folder(file_path)
        entries = load_media_tags(det_folder, Path(file_path).name)
        self.updating_tags = False
        if not entries:
            self.add_tag_card()
        else:
            for entry in entries:
                self.add_tag_card(values=entry)

    def auto_save_tags(self, *args):
        if self.updating_tags:
            return
        self.save_current_media_tags(silent=True)

    def save_current_media_tags(self, silent=False):
        try:
            file_path = IMG_PATH[-1]
        except IndexError:
            return
        if not self.tag_definitions:
            return

        from PyQt5.QtCore import QDate

        entries = []
        for card, widgets in self.tag_cards:
            entry = {}
            for tag in self.tag_definitions:
                title  = tag["title"]
                type_  = tag["type"]
                predef = tag.get("values", [])
                w = widgets.get(title)
                if w is None:
                    continue
                if predef and isinstance(w, QComboBox):
                    val = w.currentText()
                    entry[title] = val if val else None
                elif type_ == "boolean" and isinstance(w, QCheckBox):
                    entry[title] = w.isChecked()
                elif type_ == "date" and isinstance(w, QDateEdit):
                    entry[title] = w.date().toString("yyyy-MM-dd")
                elif isinstance(w, QLineEdit):
                    text = w.text().strip()
                    if text == "":
                        entry[title] = None
                    elif type_ in ("float", "double"):
                        try:    entry[title] = float(text)
                        except ValueError: entry[title] = text
                    elif type_ == "integer":
                        try:    entry[title] = int(text)
                        except ValueError: entry[title] = text
                    else:
                        entry[title] = text
            if any(v is not None and v != "" for v in entry.values()):
                entries.append(entry)

        det_folder = self.det_folder(file_path)
        save_media_tags(det_folder, Path(file_path).name, entries)
        if not silent:
            self.statusbar.showMessage("Tags saved", MESSAGE_DELAY)

    # ───────────────────────────

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

    def refresh_icons(self, dark: bool) -> None:
        self._dark_mode = dark
        # Toolbar actions (Python-created)
        self.action_select_dir.setIcon(themed_icon("folder", dark))
        self.action_select_image.setIcon(themed_icon("image", dark))
        self.zoom_action.setIcon(themed_icon("zoom", dark))
        self.action_globe.setIcon(themed_icon("globe", dark))
        self.action_run.setIcon(themed_icon("run", dark))
        self.action_batch.setIcon(themed_icon("batch", dark))
        self.action_build_tables.setIcon(themed_icon("table", dark))
        self.action_split.setIcon(themed_icon("split", dark))
        self.action_ext.setIcon(themed_icon("extensions", dark))
        self.action_pub.setIcon(themed_icon("publish", dark))
        self.action_tags.setIcon(themed_icon("tags", dark))
        self.action_general.setIcon(themed_icon("general_settings", dark))
        # Menu/toolbar actions loaded from .ui file
        self.action_models_parameter.setIcon(themed_icon("inference", dark))
        self.action_import_dc_file.setIcon(themed_icon("upload", dark))
        self.action_quit.setIcon(themed_icon("quit", dark))
        self.menuAppearance.setIcon(themed_icon("appearance", dark))
        # Buttons loaded from .ui file
        self.previous_media.setIcon(themed_icon("previous", dark))
        self.next_media.setIcon(themed_icon("next", dark))
        self.view_detection.setIcon(themed_icon("view", dark))
        self.edit_inference.setIcon(themed_icon("edit", dark))
        self.apply_inference_edit.setIcon(themed_icon("apply", dark))
        # Buttons created in Python (may not exist yet during startup theme apply)
        if self.add_entry_btn:
            self.add_entry_btn.setIcon(themed_icon("add", dark))
        if self.media_player.state() != QMediaPlayer.PlayingState:
            self.play_media.setIcon(themed_icon("play", dark))

    def _apply_macos_appearance(self, dark) -> None:
        try:
            from AppKit import NSApp, NSAppearance
            if dark is None:
                NSApp.setAppearance_(None)
            else:
                name = "NSAppearanceNameDarkAqua" if dark else "NSAppearanceNameAqua"
                NSApp.setAppearance_(NSAppearance.appearanceNamed_(name))
        except Exception:
            pass

    def set_light_mode(self):
        styles_dir = str(DECLAS_ROOT / "sources" / "styles").replace("\\", "/")
        with open(f"{DECLAS_ROOT}/sources/styles/light.qss", "r") as file:
            self.setStyleSheet(file.read().replace("{STYLES_DIR}", styles_dir))
        self._apply_macos_appearance(False)
        self.refresh_icons(dark=False)

    def set_dark_mode(self):
        with open(f"{DECLAS_ROOT}/sources/styles/dark.qss", "r") as file:
            self.setStyleSheet(file.read())
        self._apply_macos_appearance(True)
        self.refresh_icons(dark=True)

    def set_system_mode(self):
        # Windows: check registry
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            if value == 0:
                self.set_dark_mode()
            else:
                self.set_light_mode()
            return
        except Exception:
            pass
        # macOS: check system preference
        try:
            import subprocess
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True
            )
            if result.stdout.strip().lower() == "dark":
                self.set_dark_mode()
                self._apply_macos_appearance(None)
                return
        except Exception:
            pass
        self.set_light_mode()

    def models_parameters(self):
        current_set = load_json()

        mp = ModelParameter()
        mp.setStyleSheet(self.styleSheet())
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
        
        # Restore distance settings
        mp.fov_data = current_set.get("fov_table", {})
        mp.estimate_distance.setChecked(current_set.get("estimate_distance", False))
        saved_depth = current_set.get("depth_model", "")
        if saved_depth:
            idx = mp.depth_model_combo.findData(saved_depth)
            if idx >= 0:
                mp.depth_model_combo.setCurrentIndex(idx)
                mp.toggle_distance_controls(
            Qt.Checked if current_set.get("estimate_distance") else Qt.Unchecked)

        mp.setWindowModality(Qt.ApplicationModal)
        if mp.exec_() == mp.Accepted:
            self.inference_param = mp.inference_param
            dump_json(dict_obj=self.inference_param)

    def select_folder(self):
        global selected_folder
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if selected_folder :
            IMG_PATH.clear()
            self.image_or_dir = selected_folder
            self.file_model.setRootPath('.')
            root_index = self.file_model.index(selected_folder)
            self.dir_tree_view.setRootIndex(root_index)
            self.dir_tree_view.expand(root_index)

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
                        
                        if dfile[d]['count'] > 0:
                            has_target_path = dfile[d]['Image path']
                            has_target_des_path = Path(Path(has_target_path).parent, 
                                                        "has_target", Path(has_target_path).name)
                            shutil.copy(has_target_path, str(has_target_des_path))
                        else:
                            no_target_path = dfile[d]['Image path']
                            no_target_des_path = Path(Path(no_target_path).parent,
                                                        "no_target", Path(no_target_path).name)
                            shutil.copy(no_target_path, str(no_target_des_path))
                except Exception as e:
                    self.statusbar.showMessage(f"Error: {e}", 3000)

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
            global selected_file_path
            selected_file_path = selected_media
            self.get_next_and_previous_media()

            # Clear the tree view — a file node has no children so the tree
            # renders empty, matching the state at app start.
            sm = self.dir_tree_view.selectionModel()
            sm.blockSignals(True)
            sm.clearSelection()
            self.file_model.setRootPath(str(Path(selected_media).parent))
            self.dir_tree_view.setRootIndex(self.file_model.index(selected_media))
            sm.blockSignals(False)

            self.display_media(selected_media)
            self.update_metadata(selected_media)
            self.update_inference_result(selected_media)
            self.update_custom_tags(selected_media)
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
                self.display_media(file_path)
                self.update_metadata(file_path)
                self.update_inference_result(file_path)
                self.update_custom_tags(file_path)

                IMG_PATH.append(file_path)
                return file_path


    def display_image(self, image_path, message = "Unable to load image"):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_display.setText(message)
            self.view_detection.setEnabled(False)
            self.mag_filter.set_pixmap(None)
        else:
            self.mag_filter.set_pixmap(pixmap)
            self.image_display.setPixmap(pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def get_next_and_previous_media(self):

        try:
            if selected_file_path:
                sf = str(Path(selected_file_path).parent)
                global all_files
                all_files = [str(fl) for fl in Path(sf).iterdir() if not fl.is_dir() and fl.suffix in MEDIA_EXT]
                idx = all_files.index(str(Path(selected_file_path)))
                self.current_selected_media = idx
                return idx, all_files
        except:
            pass

            
    def show_previous_media(self):
        try:
            if self.current_selected_media > 0:
                self.current_selected_media -= 1
                path = all_files[self.current_selected_media]
                self.display_media(path)
                self.update_metadata(path)
                self.update_inference_result(path)
                self.update_custom_tags(path)
                IMG_PATH.append(path)
        except:
            pass

    def show_next_media(self):
        try:
            if self.current_selected_media >= 0 and self.current_selected_media < (len(all_files) - 1):
                self.current_selected_media += 1
                path = all_files[self.current_selected_media]
                self.display_media(path)
                self.update_metadata(path)
                self.update_inference_result(path)
                self.update_custom_tags(path)
                IMG_PATH.append(path)
        except:
            pass


    # Media helpers 
    def update_metadata(self, path):
        if self.is_video(path):
            self.metadata_text.setText(get_video_metadata(path))
        else:
            try:
                metadata_i, _ = get_metadata(path)
                self.metadata_text.setText(metadata_i)
            except:
                self.metadata_text.setText(str(get_metadata(path)))

    def clear_inference_form(self):
        for i in reversed(range(self.inference_form_vlay.count())):
            item = self.inference_form_vlay.itemAt(i)
            w = item.widget() if item else None
            if w:
                w.deleteLater()
        self.inference_cards.clear()

    def update_inference_result(self, file_path):
        fp = Path(file_path)
        candidates = [
            fp.parent / "detections.json",
            fp.parent.parent / "detections.json",
        ]
        if fp.parent.name == "detections":
            candidates.append(fp.parent.parent.parent / "detections.json")
        json_path = next((str(p) for p in candidates if p.exists()), None)

        self.clear_inference_form()
        self.current_json_path = None
        self.current_image_path_for_form = None

        if json_path:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_detections = json.load(f)

                stem = fp.stem
                if "media_path" in all_detections:
                    # Legacy flat-dict format
                    if Path(all_detections.get("media_path", "")) == fp:
                        self.updating_inference = True
                        card = SpeciesCard(
                            json_key="",
                            species=all_detections.get("species", ""),
                            count=int(all_detections.get("count", 0)),
                            entry=all_detections,
                        )
                        self.inference_form_vlay.addWidget(card)
                        self.inference_cards.append(card)
                        self.updating_inference = False
                        self.connect_inference_card(card)
                        self.current_json_path = json_path
                        self.current_image_path_for_form = str(file_path)
                        self.edit_inference.setEnabled(True)
                        self.view_detection.setEnabled(True)
                        return
                else:
                    entries = {
                        k: v for k, v in all_detections.items()
                        if isinstance(v, dict) and (
                            k == stem
                            or k.startswith(stem + "_")
                            or k.startswith(stem + "-")
                        )
                    }
                    if entries:
                        self.updating_inference = True
                        self.current_json_path = json_path
                        self.current_image_path_for_form = str(file_path)
                        for key, entry in entries.items():
                            species = entry.get("species", "")
                            count = int(entry.get("count", 0))
                            card = SpeciesCard(json_key=key, species=str(species),
                                               count=count, entry=entry)
                            self.inference_form_vlay.addWidget(card)
                            self.inference_cards.append(card)
                        self.updating_inference = False
                        for card in self.inference_cards:
                            self.connect_inference_card(card)
                        self.edit_inference.setEnabled(True)
                        self.view_detection.setEnabled(True)
                        return
            except Exception:
                pass

        self.updating_inference = False
        self.edit_inference.setEnabled(False)
        self.view_detection.setEnabled(False)

    def is_video(self, path):
        return Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    def set_video_controls_visible(self, visible):
        self.video_seek_slider.setVisible(visible)
        self.video_current_time.setVisible(visible)
        self.video_total_time.setVisible(visible)
        #self.stop_media.setVisible(visible)
        self.jump_back_btn.setVisible(visible)
        self.jump_forward_btn.setVisible(visible)

    def display_media(self, path):
        if self.is_video(path):
            self.enter_video_mode(path)
        else:
            self.enter_image_mode()
            self.display_image(path)

    def enter_image_mode(self):
        self.media_player.stop()
        self.media_stack.setCurrentIndex(0)
        self.play_media.setEnabled(False)
        self.set_video_controls_visible(False)

    def enter_video_mode(self, video_path):
        self.mag_overlay.hide()   # lens irrelevant while video plays
        self.media_stack.setCurrentIndex(1)
        self.play_media.setEnabled(True)
        self.set_video_controls_visible(True)
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

    def on_slider_pressed(self):
        self.slider_dragging = True

    def on_slider_released(self):
        self.slider_dragging = False
        self.media_player.setPosition(self.video_seek_slider.value())

    def on_position_changed(self, position):
        if not self.slider_dragging:
            self.video_seek_slider.setValue(position)
        self.video_current_time.setText(self.format_time(position))

    def on_duration_changed(self, duration):
        self.video_seek_slider.setRange(0, duration)
        self.video_total_time.setText(self.format_time(duration))

    def on_player_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_media.setIcon(
                self.style().standardIcon(self.style().SP_MediaPause)
            )
        else:
            self.play_media.setIcon(themed_icon("play", self._dark_mode))

    @staticmethod
    def format_time(ms):
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
            mt = parameters.get("model_type", "")
            is_ext = mt in INSTALLED_EXTENSIONS
            if not is_ext:
                no_weight()
                return

            is_vid = self.is_video(media_path)
            vid_stride = parameters.get("vid_stride", 5)
            to_save = None

            if is_vid and not parameters.get("process_video", True):
                self.statusbar.showMessage("Video processing is disabled. Enable it in Inference Parameters.", MESSAGE_DELAY)
                return

            self.statusbar.showMessage("Running…")
            QApplication.processEvents()

            if parameters["task"] == "Detection":
                # Detection-extension path
                if is_ext:
                    ext_info = INSTALLED_EXTENSIONS[mt]
                    if ext_info["status"] != "ready":
                        self.statusbar.showMessage(
                            f"Extension '{mt}' weights not downloaded.", MESSAGE_DELAY)
                        return
                    adapter = load_adapter(ext_info, device=parameters["device"])
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
                    adapter = load_adapter(ext_info, device=parameters["device"])
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
                if not is_vid:
                    self.update_inference_result(media_path)
                else:
                    self.view_detection.setEnabled(True)
                    self.edit_inference.setEnabled(True)

        except Exception as e:
            general_error(e)


    def show_detection(self, image_path):
        try:
            image_path = IMG_PATH[-1]
        except:
            missed_path()
            return
        image_path = Path(Path(image_path).parent, "detections", Path(image_path).name)
        self.enter_image_mode()  # annotated result is always a JPEG, show image page
        self.display_image(str(image_path), message="Run detection or classification before to click 'View'")


    def connect_inference_card(self, card):
        card.species_edit.editingFinished.connect(self.auto_save_inference)
        card.count_spin.valueChanged.connect(self.auto_save_inference)
        card.rm_btn.clicked.connect(lambda: self.remove_inference_card(card))

    def remove_inference_card(self, card):
        self.inference_form_vlay.removeWidget(card)
        self.inference_cards = [c for c in self.inference_cards if c is not card]
        card.deleteLater()
        self.auto_save_inference()

    def auto_save_inference(self, *args):
        if self.updating_inference:
            return
        self.save_inference_edit(silent=True)

    def edit_detection_result(self):
        card = SpeciesCard(json_key="", species="", count=0)
        self.inference_form_vlay.addWidget(card)
        self.inference_cards.append(card)
        self.connect_inference_card(card)


    def save_inference_edit(self, silent=False):
        json_path = self.current_json_path
        image_path = self.current_image_path_for_form
        if not json_path or not image_path or not self.inference_cards:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_detections = json.load(f)

            stem = Path(image_path).stem

            # Remove old keys for the cards we are about to rewrite
            for card in self.inference_cards:
                if card.json_key and card.json_key in all_detections:
                    all_detections.pop(card.json_key)

            for card in self.inference_cards:
                new_species = card.species_edit.text().strip()
                new_count = card.count_spin.value()
                if not new_species:
                    continue
                entry = dict(card.entry)
                entry["species"] = new_species
                entry["count"] = new_count
                new_key = f"{stem}_{new_species}"
                all_detections[new_key] = entry
                card.json_key = new_key

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_detections, f, indent=4)

            if not silent:
                self.statusbar.showMessage("Change applied \u2705", MESSAGE_DELAY)

        except Exception:
            if not silent:
                invalid_edit()

    def multiple_detection(self):
        try:
            parameters   = load_json()
            model_weight = load_weight()
            mt = parameters.get("model_type", "")
            is_ext = mt in INSTALLED_EXTENSIONS
            if not is_ext and (model_weight == "" or not Path(model_weight).exists()):
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
                self.last_detection_folder = folder_path
                self.worker = DetectionWorker(folder_path, main_subdir=main_subdir["run_on_main_dir"],
                                                 conf_thres= main_subdir["conf"])
                self.worker.detection_done.connect(self.on_detection_done)
                self.worker.error_occurred.connect(self.on_detection_error)
                self.worker.error_occurred.connect(self.stop_spinner)
                self.worker.log_message.connect(self.update_log)
                self.start_spinner()
                self.worker.start()
            else:
                missed_folder()
        except:
            missed_folder()

    def update_log(self, message):
        self.batch_detection_log.append(message)

    def on_detection_done(self, message):
        self.statusbar.showMessage(message, MESSAGE_DELAY)

        if self.inference_param.get("estimate_distance") and getattr(self, "last_detection_folder", None):
            fov_table = self.inference_param.get("fov_table", {})
            if not fov_table:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "No FOV configured",
                    "No Field-Of-View data is set for any station.\n\n"
                    "Distance will be estimated as raw line-of-sight depth (no angular correction).\n\n"
                    "To improve accuracy, open Inference Parameters → Set Field Of View per Station."
                )
            self.dist_worker = DistanceWorker(self.last_detection_folder, self.inference_param)
            self.dist_worker.progress.connect(
                lambda msg: self.statusbar.showMessage(msg))
            self.dist_worker.distance_done.connect(
                lambda msg: (
                    self.stop_spinner(),
                    self.statusbar.showMessage(msg, MESSAGE_DELAY),
                    self.play_completion_sound(),
                ))
            self.dist_worker.error_occurred.connect(
                lambda err: (self.stop_spinner(), self.on_distance_error(err)))
            self.start_spinner()
            self.dist_worker.start()
        else:
            self.stop_spinner()
            self.play_completion_sound()

    def on_detection_error(self, message):
        self.statusbar.showMessage(message, MESSAGE_DELAY)

    def on_distance_error(self, message):
        self.statusbar.showMessage("Distance estimation failed", MESSAGE_DELAY)
        general_error(message)

    def play_completion_sound(self):
        if self.app_settings.value("notification_sound", True, type=bool):
            sound_name = self.app_settings.value("notification_sound_file", "", type=str)
            sound_file = str(DECLAS_ROOT / "notifications" / sound_name) if sound_name else ""
            play_notification_sound(sound_file)

    def start_spinner(self):
        self.spinner_idx = 0
        self.spinner_label.setVisible(True)
        self.spinner_timer.start()

    def stop_spinner(self):
        self.spinner_timer.stop()
        self.spinner_label.setVisible(False)

    def tick_spinner(self):
        self.spinner_label.setText(self.spinner_chars[self.spinner_idx % len(self.spinner_chars)])
        self.spinner_idx += 1

    def open_extensions(self):
        """Open the Extension Manager dialog (non-modal — Declas stays interactive)."""
        # Re-use an already-open dialog instead of stacking multiple instances.
        if hasattr(self, "ext_dlg") and self.ext_dlg.isVisible():
            self.ext_dlg.raise_()
            self.ext_dlg.activateWindow()
            return
        self.ext_dlg = ExtensionManagerDialog(parent=self)
        self.ext_dlg.extension_changed.connect(self.reload_extensions)
        self.ext_dlg.show()
        self.ext_dlg.raise_()
        self.ext_dlg.activateWindow()

    def open_publish_guidelines(self):
        """Open the Publish guidelines dialog."""
        if hasattr(self, "pub_dlg") and self.pub_dlg.isVisible():
            self.pub_dlg.raise_()
            self.pub_dlg.activateWindow()
            return
        self.pub_dlg = PublishGuidelinesDialog(parent=self)
        self.pub_dlg.show()
        self.pub_dlg.raise_()
        self.pub_dlg.activateWindow()

    def reload_extensions(self):
        """Refresh the global extension registry after install / remove."""
        global INSTALLED_EXTENSIONS
        try:
            INSTALLED_EXTENSIONS = scan_extensions()
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
                # Pre-compute binary expansion columns from tag definitions
                tag_defs = load_tag_definitions()
                binary_cols = {
                    t["title"]: t["values"]
                    for t in tag_defs
                    if t.get("values") and t.get("type") in ("text", "")
                }

                for jsf in json_files:
                    jsf_folder = Path(jsf).parent

                    # Tags are saved in the detections/ subfolder
                    tags_path = jsf_folder / "detections" / "custom_tags.json"
                    custom_tags_data = {}
                    if tags_path.exists():
                        try:
                            with open(tags_path) as tf:
                                custom_tags_data = json.load(tf)
                        except Exception:
                            pass

                    # Video frames are tagged in frames/detections/ — merge those in
                    frame_tags_path = jsf_folder / "frames" / "detections" / "custom_tags.json"
                    if frame_tags_path.exists():
                        try:
                            with open(frame_tags_path) as tf:
                                custom_tags_data.update(json.load(tf))
                        except Exception:
                            pass

                    with open(jsf, "r") as f:
                        content = json.load(f)
                    if content == {}:
                        continue

                    for each_c in content:
                        row = content[each_c]
                        if not isinstance(row, dict):
                            continue

                        # Expand individual detections into separate rows (one per animal).
                        # If distance estimation ran, each bbox has its own distance_m.
                        # Without distance data, keep the original single row.
                        det_list = row.pop("detections", None)
                        has_distances = det_list and any(
                            isinstance(d.get("distance_m"), (int, float)) for d in det_list
                        )
                        if has_distances:
                            base_rows = []
                            for det in det_list:
                                r = dict(row)
                                r["count"] = 1
                                r["distance_m"] = det.get("distance_m")
                                base_rows.append(r)
                        else:
                            row.setdefault("distance_m", None)
                            base_rows = [row]

                        # Look up tag entries once per JSON key (shared across all base rows)
                        media_name = Path(row.get("media_path", "")).name
                        tag_entries = custom_tags_data.get(media_name, [])
                        if isinstance(tag_entries, dict):
                            tag_entries = [tag_entries] if tag_entries else []

                        if not tag_entries and row.get("media_type") == "video":
                            frame_stem = Path(row.get("media_path", "")).stem
                            video_stem = frame_stem.rsplit("_f", 1)[0]
                            found = next(
                                (v for k, v in custom_tags_data.items()
                                 if Path(k).stem == video_stem),
                                []
                            )
                            if isinstance(found, dict):
                                found = [found] if found else []
                            tag_entries = found

                        for base_row in base_rows:
                            # Restore species/station columns from folder structure
                            if not run_on_main_dir:
                                base_row["species"] = jsf_folder.name
                                base_row["station"] = jsf_folder.parent.name

                            if tag_entries:
                                for entry in tag_entries:
                                    row_copy = dict(base_row)
                                    for title, allowed in binary_cols.items():
                                        val = entry.get(title, "")
                                        for v in allowed:
                                            row_copy[f"{title}_{v}"] = 1 if val == v else 0
                                    for title, val in entry.items():
                                        if title not in binary_cols:
                                            row_copy[title] = val
                                    content_data.append(row_copy)
                            else:
                                for t in tag_defs:
                                    title = t["title"]
                                    if title in binary_cols:
                                        for v in binary_cols[title]:
                                            base_row[f"{title}_{v}"] = None
                                    else:
                                        base_row[title] = None
                                content_data.append(base_row)

                success_table_build(f"Table built successfully and saved at {folder}")

                pd.DataFrame(content_data).to_csv(str(Path(folder, "detections.csv")), index=False)

            except Exception as e:
                unsuccess_table_build(f"{e}")

        else:
            missed_path()


## QThread
def collect_leaf_dirs(root: Path, skip: frozenset) -> list:
    """Return all dirs under root (excluding skip names) that directly contain media."""
    result = []
    try:
        children = [c for c in root.iterdir() if c.is_dir() and c.name not in skip]
    except PermissionError:
        return result
    for child in children:
        try:
            has_media = any(
                f.suffix in IMAGE_EXT or f.suffix in VIDEO_EXT
                for f in child.iterdir() if f.is_file()
            )
        except PermissionError:
            continue
        if has_media:
            result.append(child)
        else:
            result.extend(collect_leaf_dirs(child, skip))
    return result


def process_directory(dp, log_queue):
    parameters  = load_json()
    model_weight = load_weight()

    mt = parameters.get("model_type", "")
    is_ext = mt in INSTALLED_EXTENSIONS
    if not is_ext:
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
            if is_ext:
                ext_info = INSTALLED_EXTENSIONS[mt]
                if ext_info["status"] != "ready":
                    if log_queue:
                        log_queue.put(f"Extension '{mt}' weights not downloaded.")
                    return "Extension weights missing."
                adapter = load_adapter(ext_info, device=parameters["device"])
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
                adapter = load_adapter(ext_info, device=parameters["device"])

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


class DistanceWorker(QThread):
    """Run depth estimation on detection results after detection completes."""

    distance_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    progress = pyqtSignal(str) # short status message for the status bar

    def __init__(self, folder_path, inference_param, parent=None):
        super().__init__(parent)
        self.folder_path = Path(folder_path)
        self.depth_model_name = inference_param.get("depth_model", "")
        self.fov_table = inference_param.get("fov_table", {})
        self.run_on_main_dir = inference_param.get("run_on_main_dir", False)
        self.device = inference_param.get("device", "cpu")

    def run(self):
        try:
            import numpy as np
            from model_extensions.loader import scan_extensions, load_adapter

            exts = scan_extensions()
            if self.depth_model_name not in exts:
                self.error_occurred.emit(
                    f"Depth model '{self.depth_model_name}' not found. "
                    "Install it via the Extension Manager.")
                return

            manifest = exts[self.depth_model_name].get("manifest", {})
            hf_id = manifest.get("model_hf_id", "")
            if hf_id:
                self.download_model_with_progress(hf_id)
            else:
                self.progress.emit("Distance estimation: loading depth model…")
            adapter = load_adapter(exts[self.depth_model_name], device=self.device)
            self.progress.emit("Distance estimation: running…")

            root = self.folder_path
            if self.run_on_main_dir:
                json_files = [root / "detections.json"]
            else:
                json_files = [p for p in root.rglob("detections.json")
                              if "frames" not in p.parts]
            json_files = [p for p in json_files if p.exists()]

            if not json_files:
                self.error_occurred.emit("No detections.json found for distance estimation.")
                return

            BATCH = 4
            for jsf in json_files:
                jsf_folder = jsf.parent

                with open(jsf, encoding="utf-8") as f:
                    detections = json.load(f)

                # Case-insensitive stem → path index for every image in the folder.
                # Covers JPG, PNG, TIF, BMP so no extension is silently skipped.
                IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
                folder_index: dict = {
                    p.stem.lower(): p
                    for p in jsf_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                }

                def resolve_image(entry: dict, key: str):
                    """Return the image Path for this detection entry, or None."""
                    # 1. media_path points directly to the file
                    mp = entry.get("media_path", "")
                    if mp:
                        mp_path = Path(mp)
                        if mp_path.exists():
                            return mp_path
                        # Path moved but filename unchanged → look in the same folder
                        local = jsf_folder / mp_path.name
                        if local.exists():
                            return local

                    # 2. Strip known category suffix from the JSON key
                    stem = key
                    for suffix in ("_animal", "_person", "_vehicle", "_Unknown"):
                        if stem.endswith(suffix):
                            stem = stem[: -len(suffix)]
                            break
                    else:
                        # Unknown species suffix: strip everything after the last '_'
                        if "_" in stem:
                            stem = stem.rsplit("_", 1)[0]

                    return folder_index.get(stem.lower())

                # Collect (orig_path, entry, det_list); deduplicate images
                work = []
                seen: dict = {}
                unique_paths = []
                for key, entry in detections.items():
                    if not isinstance(entry, dict):
                        continue
                    det_list = entry.get("detections")
                    if not det_list:
                        continue
                    orig = resolve_image(entry, key)
                    if orig is None:
                        continue
                    work.append((orig, entry, det_list))
                    if orig not in seen:
                        seen[orig] = None
                        unique_paths.append(orig)

                if not work:
                    continue

                n = len(unique_paths)
                station = jsf_folder.name

                import math
                if self.fov_table:
                    sl = station.lower()
                    fov_deg = next(
                        (v for k, v in self.fov_table.items() if k.lower() == sl),
                        None,
                    )
                else:
                    fov_deg = None

                depth_maps: dict = {}
                done = 0
                for i in range(0, n, BATCH):
                    batch_paths = unique_paths[i:i + BATCH]
                    maps = adapter.predict_depth_batch(
                        [str(p) for p in batch_paths], fov_deg=fov_deg)
                    for p, dm in zip(batch_paths, maps):
                        depth_maps[p] = dm
                    done += len(batch_paths)
                    pct = int(done / n * 100)
                    self.progress.emit(f"Distance estimation: {pct}% for {station}")

                # Assign distances from cached depth maps
                changed = False
                for orig, entry, det_list in work:
                    depth_map = depth_maps.get(orig)
                    if depth_map is None:
                        continue
                    h, w = depth_map.shape[:2]
                    for det in det_list:
                        bbox = det.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                        cx_px = int((x1 + x2) / 2)
                        cy_px = int((y1 + y2) / 2)
                        raw_depth = float(depth_map[
                            min(h - 1, cy_px), min(w - 1, cx_px)])
                        if fov_deg:
                            # Pinhole model: map pixel position to angle from
                            # optical axis using the camera's half-FOV.
                            half_fov_rad = math.radians(fov_deg / 2.0)
                            # Normalised position: -1 = left edge, +1 = right edge
                            pos = (cx_px - w / 2.0) / (w / 2.0)
                            angle_rad = math.atan(
                                pos * math.tan(half_fov_rad))
                            cos_a = math.cos(angle_rad)
                            distance = raw_depth / cos_a if cos_a > 0 else raw_depth
                        else:
                            distance = raw_depth
                        det["distance_m"] = round(distance, 2)
                        changed = True

                if changed:
                    with open(jsf, "w", encoding="utf-8") as f:
                        json.dump(detections, f, indent=4)

            self.distance_done.emit("Distance estimation complete ✅")

        except Exception as e:
            import traceback
            self.error_occurred.emit(
                f"Distance estimation error: {e}\n{traceback.format_exc()}")

    def download_model_with_progress(self, repo_id: str) -> None:
        try:
            from huggingface_hub import try_to_load_from_cache, snapshot_download
            import tqdm as tqdm_mod

            cached = try_to_load_from_cache(repo_id, "config.json")
            if cached is not None:
                self.progress.emit("Distance estimation: loading depth model…")
                return

            emit = self.progress.emit

            class ProgressBar(tqdm_mod.tqdm):
                def update(self, n=1):
                    super().update(n)
                    if self.total and self.total > 0:
                        mb_done = self.n / 1_048_576
                        mb_total = self.total / 1_048_576
                        pct = int(100 * self.n / self.total)
                        emit(f"Downloading model: {mb_done:.0f}/{mb_total:.0f} MB  {pct}%")

            emit("Downloading depth model (first time only)…")
            snapshot_download(repo_id=repo_id, tqdm_class=ProgressBar)
            emit("Distance estimation: loading depth model…")

        except Exception:
            self.progress.emit("Distance estimation: loading depth model…")


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

        SKIP = {"detections", "frames"}

        def is_bad(r):
            if r is None:
                return True
            rl = r.lower()
            return "error" in rl or "no media" in rl or "missing" in rl

        try:
            dir_path = Path(self.folder_path)

            if self.main_subdir:
                self.log_queue.put(f"\U0001F504 Processing: {dir_path.name}\n")
                result = process_directory(dir_path, self.log_queue)
                if is_bad(result):
                    self.error_occurred.emit(result or "No media found or error occurred")
                else:
                    self.detection_done.emit(result)

            else:
                SKIP_FS = frozenset(SKIP)
                station_dirs = [
                    d for d in dir_path.iterdir()
                    if d.is_dir() and d.name not in SKIP_FS
                ]
                if not station_dirs:
                    self.error_occurred.emit(
                        "\u274C No valid subdirectories found. Select the parent folder "
                        "that contains station sub-folders."
                    )
                    return

                all_errors = []
                any_success = False

                for station in station_dirs:
                    leaf_dirs = collect_leaf_dirs(station, SKIP_FS)
                    if not leaf_dirs:
                        self.log_queue.put(
                            f"\u26A0\uFE0F  Skipping '{station.name}': no media found\n"
                        )
                        continue
                    self.log_queue.put(
                        f"\U0001F504 Station: {station.name}"
                    )
                    try:
                        with ThreadPoolExecutor() as executor:
                            result_list = list(
                                executor.map(process_directory_wrapper,
                                             [(d, self.log_queue) for d in leaf_dirs])
                            )
                        for d, r in zip(leaf_dirs, result_list):
                            if is_bad(r):
                                all_errors.append(r or "Unknown error")
                            else:
                                any_success = True
                                self.log_queue.put(f"\u2705 {d.name}\n")
                    except Exception as exc:
                        all_errors.append(str(exc))
                        self.log_queue.put(f"\u274C Error in {station.name}: {exc}\n")

                if any_success:
                    emoji = ['\U0001F38A', '\U0001F389', '\u2705', '\U0001F917']
                    self.detection_done.emit(f"Completed successfully {choice(emoji)}")
                elif all_errors:
                    self.error_occurred.emit(all_errors[0])
                else:
                    self.error_occurred.emit(
                        "\u274C No processable folders found. Check your directory structure."
                    )

        except Exception as exc:
            self.error_occurred.emit(f"Unexpected error: {exc}")
        finally:
            self.log_emitter.stop()
            self.log_emitter.wait()