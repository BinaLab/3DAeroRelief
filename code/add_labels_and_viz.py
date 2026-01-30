import numpy as np
import argparse
import sys
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import open3d as o3d
from scipy.spatial import cKDTree

# ---- Helper -----
def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines

def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


# --- Configuration ---

COLOR_MAP = {
    (0, 0, 0): 0,          # Background
    (230, 25, 75): 1,      # Building-Damage
    (70, 240, 240): 2,     # Building-No-Damage
    (255, 255, 25): 3,     # Road
    (0, 128, 0): 4         # Tree
}

LABEL2NAMES = {
    0: 'Background',
    1: 'Building-Damage',
    2: 'Building-No-Damage',
    3: 'Road',
    4: 'Tree'
}

# --- Processing Functions ---

def read_ply_with_open3d(path):
    """Reads PLY using Open3D and returns numpy arrays."""
    # o3d.io.read_point_cloud handles ASCII/Binary automatically
    pcd = o3d.io.read_point_cloud(str(path))
    
    if not pcd.has_points():
        raise ValueError(f"Empty point cloud: {path}")

    # Extract XYZ (float32)
    xyz = np.asarray(pcd.points).astype(np.float32)
    
    # Extract RGB (Open3D loads as float 0-1, convert to uint8 0-255)
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors)
        # Check if actually 0-1 (standard for o3d) or 0-255 (rare but possible)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    else:
        # Default white
        rgb = np.full_like(xyz, 255, dtype=np.uint8)
        
    return xyz, rgb

def map_colors_to_labels(colors):
    """Robust Nearest Neighbor Color Matching."""
    target_colors = np.array(list(COLOR_MAP.keys()), dtype=np.float32)
    target_labels = np.array(list(COLOR_MAP.values()), dtype=np.int32)
    
    colors_f = colors.astype(np.float32)
    
    # Broadcasting distance calculation
    # colors_f: [N, 3], target: [5, 3] -> [N, 5]
    dists = np.linalg.norm(colors_f[:, None, :] - target_colors[None, :, :], axis=2)
    
    min_dist_idx = np.argmin(dists, axis=1)
    labels = target_labels[min_dist_idx]
    
    return labels.astype(np.int32)

def save_labeled_ply(xyz, rgb, labels, output_path):
    """Saves XYZ, RGB, and Label using helper_ply.write_ply."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for custom writer
    field_list = [xyz, rgb, labels.astype(np.int32)]
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'label']
    
    success = write_ply(str(output_path), field_list, field_names)
    if not success:
        print(f"Error saving PLY to {output_path}.")

def save_labeled_ply_visualization(xyz, rgb_orig, labels, output_base_name):
    """Saves visualization PLYs (split RGB and False-Color Labels)."""
    
    # 1. Save Original RGB View
    path_rgb = f"{output_base_name}_vis_rgb.ply"
    write_ply(path_rgb, [xyz, rgb_orig], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # 2. Save Label Color View
    LABEL_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}
    label_colors = np.zeros_like(rgb_orig)
    
    for lbl, color in LABEL_TO_COLOR.items():
        mask = (labels == lbl)
        label_colors[mask] = color
        
    path_lbl = f"{output_base_name}_vis_labels.ply"
    write_ply(path_lbl, [xyz, label_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

# --- Worker Function ---

def process_file_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing."""
    pp_path, input_root, output_root, viz = args
    pp_path = Path(pp_path)
    
    label_stem = "segment" + pp_path.stem
    label_path = pp_path.parent / (label_stem + ".ply")
    
    if not label_path.exists():
        return f"Skipped (No Label File): {pp_path.name}"

    try:
        # 1. Read Geometry using Open3D
        xyz_geo, rgb_geo = read_ply_with_open3d(pp_path)
        
        # 2. Read Labels using Open3D
        xyz_lbl, rgb_lbl = read_ply_with_open3d(label_path)
        
    except Exception as e:
        return f"Error reading {pp_path.name}: {e}"

    # 3. Map colors to label indices (on Label Cloud)
    labels_src = map_colors_to_labels(rgb_lbl)

    # 4. Spatial Matching (KDTree) to transfer labels to Geometry Cloud
    # Even if point counts differ, we find the closest labeled point for every geometry point.
    try:
        tree = cKDTree(xyz_lbl)
        # Query nearest neighbor (k=1)
        _, indices = tree.query(xyz_geo, k=1, workers=1)
        
        labels_aligned = labels_src[indices]
        
    except Exception as e:
        return f"Error matching labels for {pp_path.name}: {e}"
    
    # 5. Save
    rel_path = pp_path.relative_to(input_root)
    output_path = Path(output_root) / rel_path
    
    save_labeled_ply(xyz_geo, rgb_geo, labels_aligned, output_path)
    
    if viz:
        base_name = os.path.splitext(output_path)[0]
        # Pass data directly to avoid re-reading
        save_labeled_ply_visualization(xyz_geo, rgb_geo, labels_aligned, base_name)
        
    return f"Processed: {pp_path.name} -> {labels_aligned.shape[0]} pts"

# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add labels to AeroRelief3D PLY and visualize (Parallel + Open3D).")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing Area_*/pp*.ply")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save processed files")
    parser.add_argument("--areas", type=str, nargs='*', help="Optional: List of Areas to process (e.g. Area_1).")
    parser.add_argument("--viz", action="store_true", help="Save visualization PLYs")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist.")
        sys.exit(1)

    target_areas = set(args.areas) if args.areas else None
    tasks = []

    print(f"Scanning {input_root} for files...")
    
    if input_root.is_file():
         if input_root.name.startswith("pp") and input_root.suffix == ".ply":
             area_name = input_root.parent.name
             if not target_areas or area_name in target_areas:
                 tasks.append((str(input_root), str(input_root.parent), args.output, args.viz))
    else:
        for root, dirs, files in os.walk(input_root):
            current_folder_name = Path(root).name
            if target_areas and current_folder_name not in target_areas:
                if current_folder_name.startswith("Area_"):
                    continue

            for file in files:
                if file.startswith("pp") and file.endswith(".ply"):
                    pp_path = Path(root) / file
                    tasks.append((str(pp_path), str(input_root), args.output, args.viz))

    if not tasks:
        print("No matching files found.")
        sys.exit(0)

    num_workers = args.workers if args.workers else cpu_count()
    print(f"Found {len(tasks)} files. Processing with {num_workers} workers...")
    
    start_time = time.time()
    
    # Use 'spawn' context for Open3D compatibility in multiprocessing on Linux/MacOS
    # Open3D + Forking can sometimes deadlock
    ctx =  sys.modules['multiprocessing'].get_context('spawn')
    
    with ctx.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(process_file_wrapper, tasks):
            print(res)
            
    print(f"Done in {time.time() - start_time:.2f}s.")