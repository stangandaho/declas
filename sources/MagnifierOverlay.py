from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QEvent, QObject, QRectF
from PyQt5.QtGui import QPainter, QPainterPath, QPen, QColor, QPixmap, QIcon


class MagnifierOverlay(QWidget):
    """Frameless rounded-rectangle lens showing a zoomed image patch."""

    SIZE = 220 # lens side in pixels
    ZOOM = 2 # magnification factor
    RADIUS = 14 # corner radius
    BORDER = 1 # border thickness

    def __init__(self):
        super().__init__(None,
                         Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._patch = None

    def set_patch(self, pixmap: QPixmap) -> None:
        self._patch = pixmap
        self.update()

    def paintEvent(self, _event):
        if self._patch is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Clip everything to a rounded rectangle
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(0.5, 0.5, self.SIZE - 1, self.SIZE - 1),
            self.RADIUS, self.RADIUS,
        )
        painter.setClipPath(path)
        painter.drawPixmap(self.rect(), self._patch)

        # Subtle border ring
        painter.setClipping(False)
        painter.setPen(QPen(QColor(50, 50, 50, 210), self.BORDER))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(
            QRectF(self.BORDER * 0.5, self.BORDER * 0.5,
                   self.SIZE - self.BORDER, self.SIZE - self.BORDER),
            self.RADIUS, self.RADIUS,
        )
        painter.end()


class MagnifierFilter(QObject):
    """Event filter installed on a QLabel that drives the MagnifierOverlay."""

    def __init__(self, label: QWidget, overlay: MagnifierOverlay):
        super().__init__()
        self._label   = label
        self._overlay = overlay
        self._pixmap  = None   # original full-resolution QPixmap
        self._active  = False

    #  API

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._label.setCursor(Qt.CrossCursor)
        else:
            self._overlay.hide()
            self._label.unsetCursor()

    def set_pixmap(self, pixmap) -> None:
        """Call this with the original (full-res) QPixmap whenever a new image loads."""
        self._pixmap = pixmap

    #  event filter

    def eventFilter(self, obj, event) -> bool:
        if not self._active or obj is not self._label:
            return False
        t = event.type()
        if t == QEvent.MouseMove:
            self._update(event.pos())
        elif t in (QEvent.Leave, QEvent.Hide):
            self._overlay.hide()
        return False   # never consume events

    # impl

    def _update(self, cursor_pos) -> None:
        pix = self._pixmap
        if pix is None or pix.isNull():
            return

        lw, lh = self._label.width(), self._label.height()
        pw, ph = pix.width(), pix.height()

        # KeepAspectRatio scale factor and letterbox offsets (label is centred)
        scale  = min(lw / pw, lh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        xo, yo = (lw - sw) // 2, (lh - sh) // 2

        rel_x = cursor_pos.x() - xo
        rel_y = cursor_pos.y() - yo

        # Hide if cursor is in the letterbox area
        if rel_x < 0 or rel_y < 0 or rel_x >= sw or rel_y >= sh:
            self._overlay.hide()
            return

        # Map to original-pixmap coordinates
        orig_x = int(rel_x / scale)
        orig_y = int(rel_y / scale)

        # Source patch: SIZE/ZOOM pixels from the original, centred on cursor
        src_side = MagnifierOverlay.SIZE // MagnifierOverlay.ZOOM
        src_x = max(0, min(orig_x - src_side // 2, pw - src_side))
        src_y = max(0, min(orig_y - src_side // 2, ph - src_side))
        src_w = min(src_side, pw)
        src_h = min(src_side, ph)

        patch = pix.copy(src_x, src_y, src_w, src_h)
        patch = patch.scaled(
            MagnifierOverlay.SIZE, MagnifierOverlay.SIZE,
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
        )
        self._overlay.set_patch(patch)

        # Position the overlay to the bottom-right of cursor; flip near edges
        gpos   = self._label.mapToGlobal(cursor_pos)
        screen = QApplication.desktop().screenGeometry(self._label)
        ox = gpos.x() + 24
        oy = gpos.y() + 24
        if ox + MagnifierOverlay.SIZE > screen.right():
            ox = gpos.x() - MagnifierOverlay.SIZE - 24
        if oy + MagnifierOverlay.SIZE > screen.bottom():
            oy = gpos.y() - MagnifierOverlay.SIZE - 24

        self._overlay.move(ox, oy)
        if not self._overlay.isVisible():
            self._overlay.show()