from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
import os

#icon_file
# Set dialog icon
icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.jpg') )
icon_file = icon_file.replace("sources", "")

def general_error(error):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText(str(error))
    msgBox.setWindowTitle("Error")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def missed_path():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText("Choose an image")
    msgBox.setWindowTitle("Missed image")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def missed_folder():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText("Choose a directory that contains images")
    msgBox.setWindowTitle("Missed directory")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def missed_path():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText("Choose an image, directory that contains images, or import a detection/classification file.")
    msgBox.setWindowTitle("Missed path")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def success_table_build(message):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText(message)
    msgBox.setWindowTitle("Successful")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def unsuccess_table_build(message):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText(message)
    msgBox.setWindowTitle("Error")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def missed_gps():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Warning)
    msgBox.setText("No GPS info found")
    msgBox.setWindowTitle("Empty metadata")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def no_weight():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Warning)
    msgBox.setText("They are no model. Add one !")
    msgBox.setWindowTitle("No model")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()

def invalid_edit():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText("The edit is invalid. Check and try again")
    msgBox.setWindowTitle("Invalid edit")
    msgBox.setWindowIcon(QIcon(icon_file))
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()