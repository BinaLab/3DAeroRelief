# 3DAeroRelief: The First 3D Benchmark UAV Dataset for Post-Disaster Assessment
3DAeroRelief is a high-resolution 3D point cloud dataset designed to benchmark semantic segmentation algorithms in post-disaster scenarios.

![3DAeroRelief Dataset Examples](images/1.png)

## Download
The 3D point cloud data (Areas 1-8) is available for download via Dropbox:
- **[Download 3DAeroRelief Dataset](https://www.dropbox.com/scl/fo/591kalyu9x3hea0vxqota/AC6iCldjaaDgy0m4gLALqFs?rlkey=we9vlfigv720laws6a04fybfz&st=2t6xwj98&dl=0)**

# Dataset Structure
The project is organized into two parts: the **Code Repository** (hosted here on GitHub) and the **3D Data** (hosted externally).

### 1. Repository Contents (GitHub)
Files included in this repository:
- `ColmapConfig/`: Contains the specific COLMAP configuration files used to generate the reconstructions.
- `code/`: Python utility scripts for data handling.
- `labels.txt`: List of semantic class names and IDs.

### 2. External Dataset Contents (Dropbox)
Files included in the download link:
- `Area_<n>/`: Contains the 3D data for each location.
    - `pp<n>.ply`: The raw RGB 3D point cloud (Geometry + Texture).
    - `segmentpp<n>.ply`: The associated label.

```
(GitHub Repository)              (External Dropbox)
3DAeroRelief/                    3DAeroRelief_Dataset/
├── ColmapConfig/                ├── Area_1/
│   ├── extractor.ini            │   ├── pp1.ply
│   ├── mapper.ini               │   └── segmentpp1.ply
│   └── matcher.ini              ├── Area_2/ 
├── code/                        │   ├── pp1.ply
│   └── add_labels_and_viz.py    │   └── segmentpp1.ply
└── labels.txt                   └── Area_3/ ... Area_8/
```

# Usage and Data Processing
In the raw dataset, geometry (coordinates/colors) and semantic labels are stored in separate files (`pp<n>.ply` and `segmentpp<n>.ply`). We provide a Python script to merge these into a single, training-ready PLY file containing a `label` field.

## Prerequisites
Install the required dependencies:
```
pip install numpy scipy open3d 
```
## Merging Labels and Geometry
Use the `code/add_labels_and_viz.py` script to align labels to the geometry using Nearest Neighbor matching and save the result.

### Basic Command:
```
python code/add_labels_and_viz.py --input 3DAeroRelief --output processed_data
```
**Options:**

- `--viz`: Generates additional visualization files (original RGB and false-color labels).

- `--areas <Area_Name>`: Process specific areas (e.g., `--areas Area_1 Area_2`).

- `--workers <int>`: Specify the number of parallel CPU processes (default: uses all available cores).

### Output Format
The script generates a binary PLY file with the following properties, suitable for loading into standard 3D learning frameworks:

- `x`, `y`, `z`: 3D Coordinates.
- `red`, `green`, `blue`: RGB Texture.
- `label`: Integer Class ID (0-4).

# Classes
The dataset classifies points into distinct semantic categories. The merging script maps the raw label colors to the following integer IDs:

| ID | Class Name | Color (RGB) |
| :--- | :--- | :--- |
| 0 | Background | (0, 0, 0) |
| 1 | Building-Damage | (230, 25, 75) |
| 2 | Building-No-Damage | (70, 240, 240) |
| 3 | Road | (255, 255, 25) |
| 4 | Tree | (0, 128, 0) |

Refer to `labels.txt` for the source mapping configuration.

# License
- **Dataset**: Licensed under CC0. You are free to share and adapt the data, provided you give appropriate credit.
- **Code**: The provided utility scripts are licensed under the MIT License.

# Citation
If you use this dataset in your research, please cite our paper.
```
@misc{le20253daerorelief3dbenchmarkuav,
      title={3DAeroRelief: The first 3D Benchmark UAV Dataset for Post-Disaster Assessment}, 
      author={Nhut Le and Ehsan Karimi and Maryam Rahnemoonfar},
      year={2025},
      eprint={2509.11097},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11097}, 
}
```
# Contact
For questions regarding the dataset or the paper, please open an issue in this repository or contact the corresponding author at: maryam@lehigh.edu