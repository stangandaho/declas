from PyQt5.QtWidgets import QApplication
import sys, os
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

    MainWindow = Declas()
    font = QFont("Montserrat", 11)
    app.setFont(font)
    MainWindow.show()

    sys.exit(app.exec_())