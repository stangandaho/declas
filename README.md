[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20951592.svg)](https://doi.org/10.5281/zenodo.20951592)

## Automated Wildlife Detection and Species Classification

Declas is a free, open-source desktop application for AI-powered detection and species classification of wildlife camera trap images and videos. It runs on local machine with no internet required (except to download models), and supports pluggable model extensions so regional or taxon-specific classifiers can be added.

![Declas Interface](https://raw.githubusercontent.com/stangandaho/declas/main/app_interface.jpg?raw=true)

## Features

- Process images and videos on one or multiples directories
- Built-in support for MegaDetector v6 and YOLOv8-based classifiers
- Install community-contributed, region- or taxon-specific classifiers as plug-in extensions from within the application
- Attach typed fields (numeric, text, date, boolean) to individual media files and export them alongside detection results
- GPS tagged images are plotted on an interactive map
- Merge detection results and custom tags into a single analysis-ready CSV
- Separate images with detections from blank triggers

## Installation

Declas provides installers for Windows, macOS, and Linux. Download the file for your system from the [latest release](https://github.com/stangandaho/declas/releases/latest).

### Windows

1. Download `Declas_Setup_1.2.0.exe` from the [latest release](https://github.com/stangandaho/declas/releases/latest).
2. Double-click the downloaded file and follow the installation wizard.
3. Once installed, launch Declas from the Start Menu or the Desktop shortcut.

To open Declas without install it, [download this zip](https://zenodo.org/records/21755286/files/Declas_020826.zip?download=1) file, unzip and double-click on the executable file (.exe)


> **Note:** Because the installer is not code-signed, Windows SmartScreen may show an "unknown publisher" warning the first time you run it. Click **More info**, then **Run anyway** to continue.

### macOS

1. Download `Declas-macOS.dmg` from the [latest release](https://github.com/stangandaho/declas/releases/latest).
2. Open the `.dmg` file and drag **Declas** into your **Applications** folder.
3. Launch Declas from Applications.

> **Note:** Because the app is not notarized, macOS Gatekeeper may block the first launch. Right-click (or Control-click) the app and choose **Open**, then confirm. You only need to do this once. If macOS reports the app as "damaged", open Terminal and run `xattr -cr /Applications/Declas.app`, then launch it again.

### Linux

1. Download `Declas-Linux.tar.gz` from the [latest release](https://github.com/stangandaho/declas/releases/latest).
2. Extract the archive:

   ```
   tar -xzf Declas-Linux.tar.gz
   ```

3. Enter the extracted folder and run the executable:

   ```
   cd Declas
   ./Declas
   ```

   If needed, make it executable first with `chmod +x Declas`.


## Troubleshooting

**The application will not start**
Make sure you downloaded the correct file for your operating system and that the download completed fully. On Windows and macOS, see the notes above about SmartScreen and Gatekeeper.

On Linux, ensure the file is executable: `chmod +x Declas`. If you see a message about a missing system library such as `libGL.so.1`, install it with your package manager, for example `sudo apt-get install libgl1` on Debian and Ubuntu.

For any other issue, please check the [GitHub Issues](https://github.com/stangandaho/declas/issues) page or open a new issue.

## Contributing

Contributions from the community are welcome. If you would like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request describing the changes you have made.

## License

Declas is licensed under the MIT License. See `LICENSE` for more information.

## Contact

For any inquiries or support, please contact <stangandaho@gmail.com>.