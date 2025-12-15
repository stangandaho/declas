[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10495310.svg)](https://doi.org/10.5281/zenodo.14166440)

## Automated Wildlife Detection and Species Classification
Declas is an open source software designed for animal detection and species classification on camera trap data. It leverages computer vision models, saving the time for data processing and cleaning, and improving the accuracy of data analysis.

![Declas Interface](https://raw.githubusercontent.com/stangandaho/declas/main/app_interface.jpg?raw=true)

## Features

- 🦌**Animal detection and Species identification**
- 👨🏾‍💻**User-Friendly Interface**
- 🗃️**Mass detection or classification** - Analyse single image or folder(s) with high optimization.
- ⚙︎**Customizable Settings** - Adjust settings and parameters for animal detection and species recognition based on specific requirements.
- 📄**Comprehensive Reports** - Generate detailed reports with detected animals or recognized species and relevant metadata in JSON and CSV.
- 🌫️**Filter images** - Separate empty images from images with at least one animal
- 🤖**Extensibility** - Easily extendable to include more species or custom recognition models.

## Installation
This guide will provide step-by-step instructions to install and run the Declas software from source on Linux, macOS, and Windows systems.

## Table of Contents

1. [Windows Installation](#windows-installation-)
2. [Linux Installation](#linux-installation-)
2. [macOS Installation](#macos-installation-)
4. [Troubleshooting](#troubleshooting)

---
## Windows Installation 🪟
---

### Option 1: Run Executable
1. [Download](https://zenodo.org/records/15811203/files/Declas_050725.zip?download=1) the zip file.

2. Unzip the downloaded file and go into `Declas` folder. Double-click `Declas.exe` file to launch the software.
3. Create a shortcut on Desktop (optional) pressing `Shift` + `Right click` on `Declas.exe`. Got to `Send to` -> `Desktop (create shortcut)`. On the Desktop, the shortcut can be renamed as `Declas`.

### Option 2: Install from source
### 1. Install Prerequisites
#### a. Python Installation
1. Download and install Python 3.7+ from [python.org](https://www.python.org/downloads/).

2. **Important**: During installation, ensure that the option to **"Add Python to PATH"** is selected.

3. Verify the installation by opening Command Prompt and running:

   ```bash
   python --version
   ```

#### b. Git Installation
Download and install Git from [git-scm.com](https://git-scm.com/). During the installation, use the default settings. Verify the installation by opening Command Prompt and running:

```bash
git --version
```

#### c. Optional: Virtual Environment (Recommended)
To create a virtual environment, run the following commands in Command Prompt or PowerShell:

```bash
python -m venv venv
venv\Scripts\activate
```
You will need to activate the virtual environment by running `venv\Scripts\activate`.

### 2. Clone the Repository
Clone the repository by running the following commands in Command Prompt:

```bash
git clone https://github.com/yourusername/declas.git
cd declas
```

### 3. Install Dependencies
Inside the project directory, install the dependencies using:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated if you're using one.

### 4. Run the Software
To run the Declas software, execute the following in Command Prompt or PowerShell:

```bash
python main.py
```

---
## Linux Installation 🐧
---
### 1. Install Prerequisites
#### a. Python Installation

1. Open a terminal and install Python 3.7+ using the following commands:

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. Verify the installation:

   ```bash
   python3 --version
   ```

#### b. Git Installation
To clone the repository, Git must be installed. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

#### c. Optional: Virtual Environment (Recommended)
It's a good idea to use a virtual environment to isolate dependencies. Set up a virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll need to activate the virtual environment doing: `source .venv/bin/activate`.

### 2. Clone the Repository
Once Git is installed, clone the repository using the following commands:

```bash
git clone https://github.com/stangandaho/declas.git
cd declas
```

### 3. Install Dependencies
Inside the project directory (`declas`), install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated before running the above command if you're using one.

### 4. Run the Software
Once all dependencies are installed, you can run the Declas software using:

```bash
python3 main.py
```

---
## macOS Installation 🍏
---
### 1. Install Prerequisites
#### a. Python Installation
1. Open the terminal and install Python 3.7+ using [Homebrew](https://brew.sh/). If you don’t have Homebrew installed, first install it by running:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Once Homebrew is installed, use it to install Python:

   ```bash
   brew install python
   ```

3. Verify the installation:

   ```bash
   python3 --version
   ```

#### b. Git Installation
To clone the repository, Git must be installed. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Verify the installation:

```bash
git --version
```

#### c. Optional: Virtual Environment (Recommended)
To create a virtual environment, run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```
You'll need to activate the virtual environment by running `source venv/bin/activate`.

### 2. Clone the Repository
Once Git is installed, clone the repository using:

```bash
git clone https://github.com/stangandaho/declas.git
cd declas
```

### 3. Install Dependencies
Inside the project directory, install the required dependencies:

```bash
pip install -r requirements.txt
```
Make sure the virtual environment is activated before running this command if you're using one.

### 4. Run the Software
To run Declas, execute:

```bash
python3 main.py
```

## Troubleshooting
### Common Issues
1. **Pip not found or outdated**:
   - If you encounter an error related to `pip`, ensure that `pip` is installed and up to date:
   
     ```bash
     python -m ensurepip --upgrade
     pip install --upgrade pip
     ```

2. **Permission Denied (Linux/macOS)**:
   - If you encounter permission issues during installation, try using `pip` with `sudo` (Linux/macOS):

     ```bash
     sudo pip install -r requirements.txt
     ```

3. **Virtual Environment Not Activated**:
   - Ensure that your virtual environment is activated before running `pip install` or `python main.py`.

4. **Python Not Recognized on Windows**:
   - If you encounter errors like "python not recognized", ensure Python was added to the system's PATH. You can modify the PATH manually through **System Properties > Environment Variables** or reinstall Python with the option to add to PATH selected.

### Additional Help
If you encounter any issues, feel free to check the [GitHub Issues](https://github.com/yourusername/declas/issues) page for possible solutions or create a new issue if the problem persists.


## Contributing
I welcome contributions from the community. If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request detailing the changes you've made.

## License
Declas is licensed under the MIT License. See `LICENSE` for more information.

## Contact
For any inquiries or support, please contact me at stangandaho@gmail.com.