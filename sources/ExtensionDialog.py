"""Extension Manager dialog — browse the online registry and manage installed extensions."""

import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMetaObject
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QSizePolicy, QTabWidget,
    QTextEdit, QListWidget, QMessageBox, QTextBrowser,
)

_DECLAS_ROOT = Path(__file__).resolve().parent.parent
if str(_DECLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECLAS_ROOT))

from model_extensions._registry import fetch_registry, download_extension, remove_extension
from model_extensions._loader   import scan_extensions


# Background workers

class _FetchWorker(QThread):
    done = pyqtSignal(dict)

    def run(self):
        self.done.emit(fetch_registry())


class _DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, manifest: dict):
        super().__init__()
        self.manifest = manifest

    def run(self):
        ok = download_extension(self.manifest,
                                progress_callback=self.progress.emit)
        self.finished.emit(ok)


# Main dialog

class ExtensionManagerDialog(QDialog):
    """Shows the online registry (Available tab) and installed extensions (Installed tab)."""

    extension_changed = pyqtSignal()  # emitted after install or remove

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Extensions")
        self.setMinimumSize(820, 560)
        # Qt.Window makes it a proper standalone window (not modal child).
        # Explicitly list every hint we want so the close button is never lost.
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowMinimizeButtonHint
        )

        icon_path = str(_DECLAS_ROOT / "icons" / "logo.png")
        if Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        self._registry: dict = {}
        self._installed: dict = {}
        self._workers: list = []

        root = QVBoxLayout(self)
        root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._build_installed_tab()
        self._build_available_tab()

        # Log / progress area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        self._log.setPlaceholderText("Activity log …")
        root.addWidget(self._log)

        self._refresh_installed()
        self._start_fetch()

    # Available tab

    def _build_available_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        self._status_lbl = QLabel("Fetching registry …")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setAlignment(Qt.AlignTop)
        self._cards_layout.setSpacing(8)
        scroll.setWidget(self._cards_widget)
        layout.addWidget(scroll)

        self._tabs.addTab(container, "Available")

    def _start_fetch(self):
        self._fetch_worker = _FetchWorker()
        self._fetch_worker.done.connect(self._on_registry_fetched)
        self._fetch_worker.start()

    def _on_registry_fetched(self, registry: dict):
        self._registry = registry
        error = registry.get("error")
        models = registry.get("models", [])

        # Clear old cards
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if error:
            self._status_lbl.setText(f"⚠  Could not reach registry: {error}")
        elif not models:
            self._status_lbl.setText("No models found in registry.")
        else:
            self._status_lbl.setText(f"{len(models)} model(s) available online.")
            for m in models:
                self._cards_layout.addWidget(self._make_card(m))

    def _make_card(self, m: dict) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(frame)
        layout.setSpacing(4)

        # Title row
        top = QHBoxLayout()
        bold = QFont(); bold.setBold(True)
        title = QLabel(f"{m.get('display_name', m.get('name'))}  v{m.get('version', '?')}")
        title.setFont(bold)
        top.addWidget(title)
        top.addStretch()
        # Task + arch badges
        task = m.get("task", "")
        arch = m.get("model_arch", m.get("input_type", ""))
        badge = QLabel(f"[{task}]  [{arch}]")
        badge.setStyleSheet("color: gray; font-size: 10px;")
        top.addWidget(badge)
        layout.addLayout(top)

        # Meta row (developer · region · size · environment)
        meta_parts = []
        if m.get("developer"): meta_parts.append(m["developer"])
        if m.get("region"): meta_parts.append(m["region"])
        if m.get("size_mb"):   meta_parts.append(f"{m['size_mb']} MB")
        if m.get("environment"): meta_parts.append(m["environment"])
        if meta_parts:
            meta = QLabel("  |  ".join(meta_parts))
            meta.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(meta)

        # Description
        desc = QLabel(m.get("description", ""))
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # License + links
        links = []
        lic = m.get("license", "")
        lic_url = m.get("license_url", "")
        cit_url = m.get("citation_url", "")
        inf_url = m.get("info_url", "")
        if lic:
            lic_txt = (f'<a href="{lic_url}">{lic}</a>' if lic_url else lic)
            links.append(f"License: {lic_txt}")
        if cit_url:
            links.append(f'<a href="{cit_url}">Citation</a>')
        if inf_url:
            links.append(f'<a href="{inf_url}">Info page</a>')
        if links:
            link_lbl = QLabel("  |  ".join(links))
            link_lbl.setOpenExternalLinks(True)
            link_lbl.setStyleSheet("font-size: 10px;")
            layout.addWidget(link_lbl)

        # Species preview
        classes = m.get("classes", [])
        if classes:
            preview = ", ".join(classes[:12]) + (f"  … (+{len(classes)-12} more)" if len(classes) > 12 else "")
            cls_lbl = QLabel(f"Classes: {preview}")
            cls_lbl.setWordWrap(True)
            cls_lbl.setStyleSheet("font-size: 10px;")
            layout.addWidget(cls_lbl)

        # Download button
        bottom = QHBoxLayout()
        bottom.addStretch()
        is_installed = m.get("name") in self._installed
        btn = QPushButton("Installed ✓" if is_installed else "Download")
        btn.setEnabled(not is_installed)
        btn.setFixedWidth(120)
        btn.clicked.connect(
            lambda _, manifest=m, b=btn: self._start_download(manifest, b)
        )
        bottom.addWidget(btn)
        layout.addLayout(bottom)

        return frame

    def _start_download(self, manifest: dict, btn: QPushButton):
        btn.setEnabled(False)
        btn.setText("Downloading …")
        worker = _DownloadWorker(manifest)
        worker.progress.connect(self._log.append)
        worker.finished.connect(
            lambda ok, b=btn, m=manifest: self._on_download_done(ok, b, m)
        )
        self._workers.append(worker)
        worker.start()

    def _on_download_done(self, ok: bool, btn: QPushButton, manifest: dict):
        if ok:
            btn.setText("Installed ✓")
            self._refresh_installed()
            self.extension_changed.emit()
        else:
            btn.setEnabled(True)
            btn.setText("Download")

    # Installed tab

    def _build_installed_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        self._inst_list = QListWidget()
        layout.addWidget(self._inst_list)

        row = QHBoxLayout()
        row.addStretch()
        self._del_btn = QPushButton("Delete selected")
        self._del_btn.clicked.connect(self._delete_selected)
        row.addWidget(self._del_btn)
        layout.addLayout(row)

        self._tabs.addTab(container, "Installed")

    def _refresh_installed(self):
        self._installed = scan_extensions()
        self._inst_list.clear()
        for name, info in self._installed.items():
            m = info.get("manifest", {})
            status = "Active" if info["status"] == "ready" else "⚠ Weights missing"
            text = (
                f"{m.get('display_name', name)}  "
                f"v{m.get('version', '?')}  by  "
                f"{m.get('author', '')}  and  {status}"
            )
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            self._inst_list.addItem(item)

        self._tabs.setTabText(0, f"Installed ({len(self._installed)})")

    def _delete_selected(self):
        item = self._inst_list.currentItem()
        if not item:
            return
        name = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete extension",
            f"Remove '{name}' and all its files from disk?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            ok = remove_extension(name)
            if ok:
                self._log.append(f"Removed {name}.")
                self._refresh_installed()
                self.extension_changed.emit()
            else:
                self._log.append(f"Could not remove {name} — directory not found.")


# Publish guidelines

_PUBLISH_HTML = """
<html><body style="font-family:sans-serif; font-size:15px; margin:16px;">

<h2>Publishing a Model Extension</h2>
<p>Anyone can contribute a detection or classification model to Declas.
Each extension is a small directory with two files: a <code>manifest.json</code>
and an <code>adapter.py</code>.</p>

<hr/>
<h3>1. Directory layout</h3>
<pre style="background:#f4f4f4;padding:8px;border-radius:4px;font-size:15px;">
model_extensions/
  my_model_v1/
    manifest.json ← metadata (required)
    adapter.py ← inference code (required)
    README.md ← optional but appreciated
</pre>

<h3>2. manifest.json fields</h3>
<table border="0" cellspacing="4">
  <tr><td><b>name</b></td><td>Unique snake_case identifier, e.g. <code>my_model_v1</code></td></tr>
  <tr><td><b>display_name</b></td><td>Human-readable name shown in the UI</td></tr>
  <tr><td><b>version</b></td><td>Semantic version string, e.g. <code>1.0</code></td></tr>
  <tr><td><b>author</b></td><td>Your name or organisation</td></tr>
  <tr><td><b>email</b></td><td>Contact email (optional)</td></tr>
  <tr><td><b>description</b></td><td>One-sentence description of what the model does</td></tr>
  <tr><td><b>license</b></td><td>SPDX license identifier, e.g. <code>MIT</code>, <code>CC-BY-4.0</code></td></tr>
  <tr><td><b>license_url</b></td><td>URL to the full licence text</td></tr>
  <tr><td><b>citation_url</b></td><td>DOI or paper URL (optional)</td></tr>
  <tr><td><b>info_url</b></td><td>Project homepage or GitHub repository</td></tr>
  <tr><td><b>region</b></td><td>Geographic scope, e.g. <code>Global</code>, <code>Amazon</code></td></tr>
  <tr><td><b>task</b></td><td><code>detection</code> or <code>classification</code></td></tr>
  <tr><td><b>framework</b></td><td>e.g. <code>ultralytics</code>, <code>pywildlife</code>, <code>timm</code></td></tr>
  <tr><td><b>model_arch</b></td><td>e.g. <code>yolov9</code>, <code>resnet</code>, <code>vit</code></td></tr>
  <tr><td><b>environment</b></td><td>Runtime requirement, e.g. <code>PyTorch</code></td></tr>
  <tr><td><b>input_type</b></td><td><code>detect</code> or <code>classify</code></td></tr>
  <tr><td><b>model_file</b></td><td>Filename of the weights file, e.g. <code>my_model.pt</code></td></tr>
  <tr><td><b>download_url</b></td><td>Direct URL to download the weights (Zenodo, HF, GitHub Releases…)</td></tr>
  <tr><td><b>size_mb</b></td><td>Approximate size in MB</td></tr>
  <tr><td><b>zip_url</b></td><td>Optional ZIP bundling adapter + auxiliaries (leave blank otherwise)</td></tr>
  <tr><td><b>adapter</b></td><td>Filename of the adapter script, default <code>adapter.py</code></td></tr>
  <tr><td><b>classes</b></td><td>JSON array of class/species names the model outputs</td></tr>
</table>

<h3>3. adapter.py</h3>
<p>Your adapter must inherit from <code>ModelAdapter</code> and implement two methods:</p>
<pre style="background:#f4f4f4;padding:8px;border-radius:4px;font-size:15px">
from model_extensions._base import ModelAdapter

class MyModelAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        \"\"\"Load the model weights.  Called once before inference.\"\"\"
        # model_path is the full path to the downloaded weights file.
        ...

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        \"\"\"Run inference on one image.
        Return a list of dicts, each with:
          'species'    (str)   — predicted class / species name
          'confidence' (float) — score in [0, 1]
          'bbox'       (list)  — [x1, y1, x2, y2] pixel coords, or []
        \"\"\"
        ...
        return [{'species': 'lion', 'confidence': 0.92, 'bbox': [10, 20, 300, 400]}]
</pre>

<h3>4. Host your weights</h3>
<ul>
  <li><b>Zenodo</b>: free, DOI-minted, permanent.  Ideal for academic models.</li>
  <li><b>Hugging Face Hub</b>:easy versioning, large-file support.</li>
  <li><b>GitHub Releases</b>: fine for files under 2 GB.</li>
</ul>
<p>The <code>download_url</code> in your manifest must be a direct download link
(no redirect pages).</p>

<h3>5. Submit</h3>
<ol>
  <li>Fork the <a href="https://github.com/stangandaho/declas">Declas repository</a>.</li>
  <li>Add your extension directory under <code>model_extensions/</code>.</li>
  <li>Add an entry to <code>model_extensions/registry.json</code> following the
      same schema as the existing entries.</li>
  <li>Open a Pull Request with a short description of the model</li>
</ol>
<p>The maintainers will review the adapter code, test the weights URL,
and merge when ready.  Thank you for contributing!</p>

</body></html>
"""

class PublishGuidelinesDialog(QDialog):
    """Shows the guidelines for publishing a model extension."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publish a Model Extension")
        self.setMinimumSize(700, 580)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMaximizeButtonHint
        )

        icon_path = str(_DECLAS_ROOT / "icons" / "logo.png")
        if Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(_PUBLISH_HTML)
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)
