from PyQt5.QtCore import Qt, QEvent, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget

DRAW_COLOR = QColor(0, 200, 50)
DELETE_COLOR = QColor(220, 30, 30)


class BoxEditOverlay(QWidget):
    """Transparent layer over the image label to draw or delete bounding boxes.

    Only visible while a mode is active, so the label behaves normally otherwise.
    Coordinates emitted and received are in original-image pixels.
    """

    box_drawn = pyqtSignal(list)                  # [x1, y1, x2, y2]
    box_delete_requested = pyqtSignal(str, int)   # (json_key, detection index)

    MIN_SIZE = 5   # ignore accidental clicks smaller than this (screen pixels)

    def __init__(self, label, magnifier=None, box_provider=None):
        super().__init__(label)
        self.label = label
        self.magnifier = magnifier
        self.box_provider = box_provider   # callable returning [(json_key, index, bbox)]
        self.mode = None                   # "draw", "delete" or None
        self.image_w = 0
        self.image_h = 0
        self.boxes = []
        self.drag_start = None
        self.drag_end = None
        self.hovered = None
        self.setMouseTracking(True)
        label.installEventFilter(self)
        self.hide()

    # API

    def set_mode(self, mode) -> None:
        self.mode = mode
        self.drag_start = self.drag_end = None
        self.hovered = None
        if mode:
            self.refresh_boxes()
            self.setGeometry(self.label.rect())
            self.setCursor(Qt.CrossCursor if mode == "draw" else Qt.PointingHandCursor)
            self.show()
            self.raise_()
        else:
            self.hide()
        self.update()

    def set_image_size(self, width: int, height: int) -> None:
        self.image_w, self.image_h = width, height
        self.update()

    def refresh_boxes(self) -> None:
        self.boxes = self.box_provider() if self.box_provider else []
        self.hovered = None
        self.update()

    # geometry

    def image_frame(self):
        """Return (x_offset, y_offset, scale) of the pixmap shown in the label."""
        pix = self.label.pixmap()
        if pix is None or pix.isNull() or not self.image_w:
            return None
        scale = pix.width() / self.image_w
        x_off = (self.label.width() - pix.width()) / 2
        y_off = (self.label.height() - pix.height()) / 2
        return x_off, y_off, scale

    def to_image(self, point, frame):
        x_off, y_off, scale = frame
        x = min(max((point.x() - x_off) / scale, 0), self.image_w)
        y = min(max((point.y() - y_off) / scale, 0), self.image_h)
        return x, y

    def to_widget(self, bbox, frame):
        x_off, y_off, scale = frame
        x1, y1, x2, y2 = bbox
        return QRectF(x_off + x1 * scale, y_off + y1 * scale,
                      (x2 - x1) * scale, (y2 - y1) * scale)

    def box_at(self, point):
        """Index of the smallest box under *point* (so nested boxes stay reachable)."""
        frame = self.image_frame()
        if frame is None:
            return None
        hits = [(self.to_widget(bbox, frame), i) for i, (k, n, bbox) in enumerate(self.boxes)]
        hits = [(r.width() * r.height(), i) for r, i in hits if r.contains(point)]
        return min(hits)[1] if hits else None

    # events

    def eventFilter(self, obj, event) -> bool:
        if obj is self.label and event.type() == QEvent.Resize:
            self.setGeometry(self.label.rect())
        return False

    def enterEvent(self, event):
        # The shown media may have changed while the cursor was elsewhere.
        self.refresh_boxes()

    def leaveEvent(self, event):
        if self.magnifier:
            self.magnifier.overlay.hide()
        self.hovered = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.mode == "draw" and self.image_frame():
            self.drag_start = self.drag_end = event.pos()
        elif self.mode == "delete":
            hit = self.box_at(event.pos())
            if hit is not None:
                key, index, bbox = self.boxes[hit]
                self.hovered = None
                self.box_delete_requested.emit(key, index)

    def mouseMoveEvent(self, event):
        if self.magnifier and self.magnifier.active:
            self.magnifier.update(event.pos())
        if self.drag_start is not None:
            self.drag_end = event.pos()
        elif self.mode == "delete":
            self.hovered = self.box_at(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None or event.button() != Qt.LeftButton:
            return
        rect = QRect(self.drag_start, event.pos()).normalized()
        self.drag_start = self.drag_end = None
        self.update()
        frame = self.image_frame()
        if frame is None or rect.width() < self.MIN_SIZE or rect.height() < self.MIN_SIZE:
            return
        x1, y1 = self.to_image(rect.topLeft(), frame)
        x2, y2 = self.to_image(rect.bottomRight(), frame)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            self.box_drawn.emit([x1, y1, x2, y2])

    def paintEvent(self, event):
        painter = QPainter(self)
        frame = self.image_frame()

        if self.mode == "delete" and frame:
            for i, (key, index, bbox) in enumerate(self.boxes):
                rect = self.to_widget(bbox, frame)
                if i == self.hovered:
                    painter.fillRect(rect, QColor(DELETE_COLOR.red(), DELETE_COLOR.green(),
                                                  DELETE_COLOR.blue(), 90))
                    painter.setPen(QPen(DELETE_COLOR, 2))
                else:
                    painter.setPen(QPen(DELETE_COLOR, 1, Qt.DashLine))
                painter.drawRect(rect)

        if self.drag_start is not None:
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.fillRect(rect, QColor(DRAW_COLOR.red(), DRAW_COLOR.green(),
                                          DRAW_COLOR.blue(), 40))
            painter.setPen(QPen(DRAW_COLOR, 2, Qt.DashLine))
            painter.drawRect(rect)
        painter.end()
