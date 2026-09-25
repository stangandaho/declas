from pathlib import Path

from PyQt5.QtCore import Qt, QSize, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPolygonF
from PyQt5.QtWidgets import (QApplication, QListWidget, QListWidgetItem, QListView,
                             QAbstractItemView, QFrame, QStyle)

THUMB_SIZE = QSize(320, 240)   # decoded size; icons never upscale past it
CELL_PADDING = 14
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def load_thumbnail(path: str, size: QSize):
    """Decode a small RGB preview of an image, or of a video's first frame."""
    from PIL import Image, ImageOps
    try:
        if Path(path).suffix.lower() in VIDEO_SUFFIXES:
            import cv2
            capture = cv2.VideoCapture(path)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                return None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(path)
            img.draft("RGB", (size.width() * 2, size.height() * 2))  # fast JPEG downscale
            img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size.width(), size.height()))
        data = img.tobytes("raw", "RGB")
        return QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888).copy()
    except Exception:
        return None


def add_play_badge(pixmap: QPixmap) -> QPixmap:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    cx, cy = pixmap.width() / 2, pixmap.height() / 2
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(0, 0, 0, 140))
    painter.drawEllipse(QPointF(cx, cy), 18, 18)
    painter.setBrush(QColor(255, 255, 255, 230))
    painter.drawPolygon(QPolygonF([QPointF(cx - 6, cy - 10), QPointF(cx - 6, cy + 10),
                                   QPointF(cx + 11, cy)]))
    painter.end()
    return pixmap


class ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(str, QImage)

    def __init__(self, paths, size):
        super().__init__()
        self.paths = paths
        self.size = size
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        for path in self.paths:
            if self.stopped:
                return
            image = load_thumbnail(path, self.size)
            if image is not None and not self.stopped:
                self.thumbnail_ready.emit(path, image)


class ThumbnailList(QListWidget):
    """Grid of media thumbnails for one folder; the open media has a grey border."""

    media_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "QListWidget { outline: 0; background: transparent; }"
            "QListWidget::item { border: 3px solid transparent; border-radius: 4px; }"
            "QListWidget::item:hover { background: rgba(128, 128, 128, 40); }"
            "QListWidget::item:selected { border: 3px solid #808080; background: transparent; }"
        )

        placeholder = QPixmap(THUMB_SIZE)
        placeholder.fill(QColor(128, 128, 128, 50))
        self.placeholder = QIcon(placeholder)
        self.items = {}      # normalised path -> item
        self.workers = []

        self.currentItemChanged.connect(self.on_current_changed)
        QApplication.instance().aboutToQuit.connect(self.stop_workers)
        self.fit_columns()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_columns()

    def fit_columns(self):
        """One column of thumbnails as wide as the sidebar."""
        # Reserve room for the scrollbar, so it appearing doesn't reflow the grid
        scrollbar = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        cell = max(CELL_PADDING + 40, self.width() - 2 * self.frameWidth() - scrollbar - 2)
        icon_w = min(cell - CELL_PADDING, THUMB_SIZE.width())
        icon_h = icon_w * THUMB_SIZE.height() // THUMB_SIZE.width()
        grid = QSize(cell, icon_h + CELL_PADDING)
        if grid != self.gridSize():
            self.setIconSize(QSize(icon_w, icon_h))
            self.setGridSize(grid)

    def show_folder(self, paths, current=None):
        keys = [str(Path(p)) for p in paths]
        if keys != list(self.items):
            self.rebuild(keys)
        self.highlight(current)

    def rebuild(self, paths):
        for worker in self.workers:
            worker.stop()
        self.blockSignals(True)
        self.clear()
        self.items = {}
        for path in paths:
            item = QListWidgetItem(self.placeholder, "")
            item.setData(Qt.UserRole, path)
            item.setToolTip(Path(path).name)
            self.addItem(item)
            self.items[path] = item
        self.blockSignals(False)

        if paths:
            worker = ThumbnailWorker(paths, THUMB_SIZE)
            worker.thumbnail_ready.connect(self.set_thumbnail)
            worker.finished.connect(lambda w=worker: self.workers.remove(w) if w in self.workers else None)
            self.workers.append(worker)
            worker.start()

    def set_thumbnail(self, path, image):
        item = self.items.get(path)
        if item is None:   # belongs to a folder that is no longer shown
            return
        pixmap = QPixmap.fromImage(image)
        if Path(path).suffix.lower() in VIDEO_SUFFIXES:
            pixmap = add_play_badge(pixmap)
        icon = QIcon()
        # Same pixmap when selected, so Qt does not tint the thumbnail
        icon.addPixmap(pixmap, QIcon.Normal)
        icon.addPixmap(pixmap, QIcon.Selected)
        item.setIcon(icon)

    def highlight(self, path):
        item = self.items.get(str(Path(path))) if path else None
        self.blockSignals(True)
        if item is not None:
            self.setCurrentItem(item)
            self.scrollToItem(item)
        else:
            self.clearSelection()
        self.blockSignals(False)

    def on_current_changed(self, current, previous):
        if current is not None:
            self.media_selected.emit(current.data(Qt.UserRole))

    def stop_workers(self):
        for worker in list(self.workers):
            worker.stop()
            worker.wait(2000)
