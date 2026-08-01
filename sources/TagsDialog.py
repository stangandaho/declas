import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QHeaderView,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

TAG_TYPES = ["float", "integer", "text", "date", "boolean"]
TAGS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "custom_tags.json"


def load_tag_definitions():
    if TAGS_CONFIG.exists():
        try:
            with open(TAGS_CONFIG) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_tag_definitions(tags):
    TAGS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGS_CONFIG, "w") as f:
        json.dump(tags, f, indent=2)


def load_media_tags(folder, filename):
    """Return the saved custom-tag values for *filename* inside *folder*, or {}."""
    path = Path(folder) / "custom_tags.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get(str(filename), {})
    except Exception:
        return {}


def save_media_tags(folder, filename, values):
    """Persist *values* (dict) for *filename* in folder/custom_tags.json."""
    path = Path(folder) / "custom_tags.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data[str(filename)] = values
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class TagsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Tags")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(480, 320)
        self.setMaximumSize(700, 600)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Tag Title", "Data Type"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Tag")
        add_btn.clicked.connect(lambda: self.add_row())
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_row)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_and_close)
        save_btn.setDefault(True)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        for tag in load_tag_definitions():
            self.add_row(tag.get("title", ""), tag.get("type", "float"))

    def add_row(self, title="", type_="float"):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(title))
        combo = QComboBox()
        combo.addItems(TAG_TYPES)
        combo.setCurrentText(type_ if type_ in TAG_TYPES else "float")
        self.table.setCellWidget(row, 1, combo)
        # Only enter edit mode when user clicks Add Tag (empty title), not when loading saved tags
        if not title and self.isVisible():
            self.table.editItem(self.table.item(row, 0))

    def remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def save_and_close(self):
        tags = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            combo = self.table.cellWidget(row, 1)
            title = item.text().strip() if item else ""
            if title:
                tags.append({"title": title, "type": combo.currentText()})
        save_tag_definitions(tags)
        self.accept()
