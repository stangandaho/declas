import json
from PyQt5.QtWidgets import QWidget, QTableWidget, QTableWidgetItem, QPushButton, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont, QIcon
from pathlib import Path
import sys, os

DECLAS_ROOT = Path(__file__).resolve().parent.parent

class WeightTable(QWidget):
   def __init__(self, weight_data:dict) -> None:
      super(WeightTable, self).__init__()
      self.weight_data = weight_data
      loadUi(f"{DECLAS_ROOT}/ui/WeightTable.ui", self)

      font = QFont()
      font.setPointSize(12)  # Set font size to 12 points
      self.setFont(font)
      self.icon_file = os.path.normpath( os.path.join(os.path.dirname(__file__), 'icons', 'logo.jpg') )
      icon_file = self.icon_file.replace("sources", "")
      self.setWindowIcon(QIcon(icon_file))

      self.setWindowTitle("Weight Table Widget")
      self.resize(600, 400)

      self.table = self.findChild(QTableWidget, "tableWidget")
      self.table.setColumnCount(6)
      self.table.setHorizontalHeaderLabels(['Id', 'Weight', 'Path', 'Size', 'Add Date', 'Action'])  # Last column for delete button
            

   def add_row(self, id, weight, path, size, add_date):
        # Get the current row count
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # Insert data into the row
        self.table.setItem(row_position, 0, QTableWidgetItem(id))
        self.table.setItem(row_position, 1, QTableWidgetItem(weight))
        self.table.setItem(row_position, 2, QTableWidgetItem(path))
        self.table.setItem(row_position, 3, QTableWidgetItem(size))
        self.table.setItem(row_position, 4, QTableWidgetItem(add_date))
        
        # Add delete button
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.remove_row(row_position))
        self.table.setCellWidget(row_position, 5, delete_button)

   def remove_row(self, row):
        weight_data = self.weight_data
        weight_path = str(Path(f"{DECLAS_ROOT}/config/model_/mdls/mdl.json"))
        # Get the value from the desired column (e.g., column 1 for 'Weight')
        weight_item = self.table.item(row, 1)  # Column 1: 'Weight'
        if weight_item:
            weight_value = weight_item.text()
            del weight_data[weight_value]
            with open(weight_path, "w") as updated:
                json.dump(weight_data, updated)
        
        # Now remove the row
        self.table.removeRow(row)

# def main():
#     app = QApplication(sys.argv)
#     window = WeightTable()

#     window.show()

#     sys.exit(app.exec_())

# if __name__ == '__main__':
#       main()
      
    
      
    