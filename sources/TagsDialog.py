import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QHeaderView, QLineEdit,
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
    """Return saved tag entries for *filename* inside *folder* as a list of dicts."""
    path = Path(folder) / "custom_tags.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        val = data.get(str(filename), [])
        if isinstance(val, dict):
            return [val] if val else []
        return val if isinstance(val, list) else []
    except Exception:
        return []


def save_media_tags(folder, filename, entries):
    """Persist *entries* (list of dicts) for *filename* in folder/custom_tags.json."""
    path = Path(folder) / "custom_tags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data[str(filename)] = entries
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class TagsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Tags")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(600, 320)
        self.setMaximumSize(900, 600)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Tag Title", "Data Type", "Predefined Values (comma-separated)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
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
            self.add_row(
                tag.get("title", ""),
                tag.get("type", "float"),
                ", ".join(tag.get("values", [])),
            )

    def add_row(self, title="", type_="float", values=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(title))
        combo = QComboBox()
        combo.addItems(TAG_TYPES)
        combo.setCurrentText(type_ if type_ in TAG_TYPES else "float")
        self.table.setCellWidget(row, 1, combo)
        self.table.setItem(row, 2, QTableWidgetItem(values))
        if not title and self.isVisible():
            self.table.editItem(self.table.item(row, 0))

    def remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def save_and_close(self):
        tags = []
        for row in range(self.table.rowCount()):
            title_item = self.table.item(row, 0)
            combo = self.table.cellWidget(row, 1)
            vals_item = self.table.item(row, 2)
            title = title_item.text().strip() if title_item else ""
            if not title:
                continue
            raw_vals = vals_item.text() if vals_item else ""
            values = [v.strip() for v in raw_vals.split(",") if v.strip()]
            tags.append({"title": title, "type": combo.currentText(), "values": values})
        save_tag_definitions(tags)
        self.accept()
