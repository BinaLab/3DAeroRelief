# 3DAeroRelief: The First 3D Benchmark UAV Dataset for Post-Disaster Assessment
3DAeroRelief is a high-resolution 3D point cloud dataset designed to benchmark semantic segmentation algorithms in post-disaster scenarios.

# Dataset Structure
The dataset is organized into 8 distinct areas, along with configuration files for reproducibility and helper code for preprocessing the data.
- `Area_<n>/`: Contains the 3D data for each location.
    - `pp<n>.ply`: The raw RGB 3D point cloud (Geometry + Texture).
    - `segmentpp<n>.ply`: The associated label.
- `ColmapConfig/`: Contains the specific COLMAP configuration files used to generate the reconstructions.
- `code/`: Python utility scripts for data handling.
- `labels.txt`: List of semantic class names and IDs.

```
3DAeroRelief/
├── Area_1/
│   ├── pp1.ply          # Raw Point Cloud
│   └── segmentpp1.ply   # Semantic Labels
|   └── ...  
├── Area_2/              # (Designated Test Set)
│   ├── pp1.ply
│   └── segmentpp1.ply
|   └── ... 
├── Area_3/ ... Area_8/
├── ColmapConfig/
│   ├── extractor.ini    # Feature extraction settings
│   ├── mapper.ini       # Sparse reconstruction settings
│   └── matcher.ini      # Feature matching settings
├── code/
│   └── add_labels_and_viz.py
└── labels.txt
```
## Classes
The dataset classifies points into distinct semantic categories (e.g., Undamaged Building, Damaged Building, Background). Refer to labels.txt for the complete list of class mappings.

## License
- **Dataset**: Licensed under CC0. You are free to share and adapt the data, provided you give appropriate credit.
- **Code**: The provided utility scripts are licensed under the MIT License.

## Citation
If you use this dataset in your research, please cite our paper.

## Contact
For questions regarding the dataset or the paper, please open an issue in this repository or contact the corresponding author at: maryam@lehigh.edu