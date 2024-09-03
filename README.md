# declas
Species detection and classification software designed to handle efficiently camera trap data. 

## Declas - Species Detection and Recognition Software

### Overview

Declas is an advanced software tool designed for detecting and recognizing species in camera trap data. Whether you are a wildlife researcher, a conservationist, or an enthusiast, Declas provides an intuitive interface and powerful algorithms to help you analyze and interpret the data captured by camera traps. The software leverages computer vision techniques to identify different species automatically, saving you time and improving the accuracy of your data analysis.

![Declas Interface](https://raw.githubusercontent.com/stangandaho/declas/main/declas.jpg?raw=true)

### Features

- 🦌**Species Detection**: Automatically detect and identify species in camera trap images.
- 👨🏾‍💻**User-Friendly Interface**: Intuitive graphical interface for easy operation.
- 🗃️**Mass detection or classification**: Analyse single image or folder or multipleimage with high optimization.
- ⚙︎**Customizable Settings**: Adjust settings and parameters for species recognition based on specific requirements.
- 📄**Comprehensive Reports**: Generate detailed reports with recognized species and relevant metadata in json file.
- 🤖**Extensibility**: Easily extendable to include more species or custom recognition models.

### Installation (On Windows OS)

#### Option 1: Download the Pre-packaged Executable

1. **Download the Executable**: 
   - Visit the [releases page](#) (link to be added) and download the latest version of the `Declas.exe` file.

2. **Run the Software** (`recommended`):
   - Unzipe the downloaded file and go into `Declas` folder. Double-click `Declas.exe` file to launch the software.

#### Option 2: Clone the Repository and Run from Source

1. **Clone the Repository**:
   - Open a terminal and run the following command:
     ```bash
     git clone https://github.com/yourusername/declas.git
     cd declas
     ```

2. **Install Python and Dependencies**:
   - Ensure you have Python 3.7+ installed on your system. You can check this by running:
     ```bash
     python --version
     ```
   - Install the required libraries from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Software**:
   - Execute the main file to launch Declas:
     ```bash
     python main.py
     ```


### Usage

1. **Launching Declas**:
   - Upon running the software, the main window will appear. 
   - You can select the folder containing your camera trap images.

2. **Settings**:
   - Customize species detection settings under the "Settings" tab for more tailored results.

3. **Models**
    - Add a weight (for detection or classification) you relevant to your purpose.

3. **Processing Images**:
   - Click on the "Analyze" button to start the species detection process.
   - The results will be displayed within the interface, and you can export reports as needed.


### Contributing

We welcome contributions from the community. If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request detailing the changes you've made.

### License

Declas is licensed under the MIT License. See `LICENSE` for more information.

### Contact

For any inquiries or support, please contact us at stangandaho@gmail.com or visit our [GitHub Issues](https://github.com/stangandaho/declas/issues) page.
