from PyQt5.QtWidgets import QApplication
import sys, os
from PyQt5.QtGui import QFont


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