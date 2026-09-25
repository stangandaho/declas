import subprocess
import sys
import os

# Suppress console windows that subprocess/os.system calls open on Windows in frozen builds
if sys.platform == "win32" and getattr(sys, "frozen", False):
    _orig_Popen = subprocess.Popen
    def _silent_Popen(*args, **kwargs):
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        return _orig_Popen(*args, **kwargs)
    subprocess.Popen = _silent_Popen

    # os.system() bypasses Popen — redirect it through our patched subprocess
    def _silent_os_system(cmd):
        return subprocess.call(cmd, shell=True)
    os.system = _silent_os_system

# PyQt5 ships an old msvcp140.dll that onnxruntime cannot initialise with.
# Loading onnxruntime first makes Windows keep the newer runtime for the whole process.
try:
    import onnxruntime
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

# Redirect stdout and stderr to a log file
import tempfile
temp_dir = tempfile.gettempdir()
log_file_path = os.path.join(temp_dir, "declas_running_verbose.log")
log_file = open(log_file_path, "w")
sys.stdout = log_file
sys.stderr = log_file

# Add the directory containing 'sources' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))
from sources.Declas import Declas


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  

    app = QApplication(sys.argv)

    # Qt's own texts (OK/Cancel/Yes/No buttons, file dialogs) in the chosen language
    from PyQt5.QtCore import QTranslator, QLibraryInfo
    from i18n import LANGUAGE
    qt_translator = QTranslator()
    if LANGUAGE == "fr" and qt_translator.load(
            "qtbase_fr", QLibraryInfo.location(QLibraryInfo.TranslationsPath)):
        app.installTranslator(qt_translator)

    MainWindow = Declas()
    font = QFont("Montserrat", 11)
    app.setFont(font)
    MainWindow.show()

    sys.exit(app.exec_())