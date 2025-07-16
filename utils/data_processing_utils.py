import matplotlib.pyplot as plt
import pandas as pd
import os
import ezc3d
import opensim as osim
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
from pycpd import DeformableRegistration
from typing import List, Dict
import random
import re
import numpy as np
from scipy.stats import ttest_rel, wilcoxon



def convert_c3d_to_trc(c3d_file, output_trc=None, start_time=0.0, end_time=None, include_markers=None, markers_to_clean=None):
    """
    Convert a C3D file to a TRC file compatible with OpenSim.

    This function extracts 3D marker data from a `.c3d` file, filters and optionally renames markers,
    applies coordinate system adjustments, and exports the result to a `.trc` file format.

    Parameters:
        c3d_file (str): Path to the input `.c3d` file.
        output_trc (str, optional): Path for the output `.trc` file. If None, replaces extension of `c3d_file`.
        start_time (float, optional): Start time in seconds to begin extracting frames (default: 0.0).
        end_time (float, optional): End time in seconds to stop extracting frames (default: until end of file).
        include_markers (List[str], optional): List of marker names to keep. If None, keeps all markers.
        markers_to_clean (List[str], optional): List of marker name substrings to normalize. 
            Any marker name containing a substring will be renamed using that base name.

    Returns:
        int: Total number of frames in the original C3D file.
    """

    if output_trc is None:
        output_trc = os.path.splitext(c3d_file)[0] + '.trc'

    c3d = ezc3d.c3d(c3d_file)
    points = c3d['data']['points']  # shape: (4, nb_markers, nb_frames)
    labels = c3d['parameters']['POINT']['LABELS']['value']
    n_frames = points.shape[2]
    print(n_frames)
    rate = c3d['parameters']['POINT']['RATE']['value'][0]

    # Filtrer les marqueurs si nécessaire
    if include_markers is not None:
        keep_indices = [i for i, label in enumerate(labels) if label in include_markers]
        labels = [labels[i] for i in keep_indices]
        points = points[:, keep_indices, :]
    else:
        keep_indices = list(range(len(labels)))

    # Nettoyage des noms de marqueurs si demandé
    if markers_to_clean is not None:
        cleaned_labels = []
        for label in labels:
            new_label = label
            for base in markers_to_clean:
                if base in label:
                    new_label = base
                    break  # On prend le premier match
            cleaned_labels.append(new_label)
        labels = cleaned_labels

    n_markers = len(labels)
    frame_time = 1.0 / rate

    start_frame = int(start_time * rate)
    end_frame = int(end_time * rate) if end_time is not None else n_frames
    print(start_frame, end_frame)

    data = []
    for frame_idx in range(start_frame, end_frame):
        frame_data = []
        time_sec = frame_idx * frame_time
        frame_data.append(time_sec)
        for marker_idx in range(n_markers):
            x = points[0, marker_idx, frame_idx]
            y = points[1, marker_idx, frame_idx]
            z = points[2, marker_idx, frame_idx]
            x, y, z = x, z, -y
            frame_data.extend([x, y, z])
        data.append(frame_data)

    columns = ['Time']
    for label in labels:
        columns.extend([f'{label}_X', f'{label}_Y', f'{label}_Z'])

    df = pd.DataFrame(data, columns=columns)

    with open(output_trc, 'w') as f:
        f.write(f'PathFileType\t4\t(X/Y/Z)\t{os.path.basename(output_trc)}\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write(f'{rate}\t{rate}\t{end_frame-start_frame}\t{n_markers}\tmm\t{rate}\t{start_frame+1}\t{end_frame-start_frame}\n')

        header = ['Frame#', 'Time']
        for label in labels:
            header.extend([label, '', ''])
        f.write('\t'.join(header) + '\n')

        subheader = ['', '']
        for idx in range(1, n_markers + 1):
            subheader.extend([f'X{idx}', f'Y{idx}', f'Z{idx}'])
        f.write('\t'.join(subheader) + '\n')

        for idx, row in df.iterrows():
            row_data = [str(idx + 1)] + [f'{v:.5f}' for v in row.values]
            f.write('\t'.join(row_data) + '\n')

    print(f'✅ Conversion terminée : {output_trc}')
    return n_frames





def convert_trc_to_c3d(trc_path, source_c3d_path=None):
    """
    Convert a TRC file to a C3D file, optionally including analog data from a source C3D.

    This function:
    - Parses marker positions from a TRC file and applies a coordinate transformation 
      to match the C3D coordinate system (x = -z, y = -x, z = y).
    - Constructs a new C3D file with marker data and proper metadata (frame rate, labels, etc.).
    - Optionally imports and synchronizes analog (e.g. force plate) data from another C3D file.
    - Automatically writes the resulting C3D file with the same base name as the TRC input.

    Parameters:
        trc_path (str): Path to the input `.trc` file.
        source_c3d_path (str, optional): Path to an existing `.c3d` file containing analog data to be copied.

    Returns:
        None. Writes a `.c3d` file to disk with the same base name as the input TRC.

    Example:
        convert_trc_to_c3d("trial.trc", source_c3d_path="forces.c3d")
    """
    with open(trc_path, 'r') as f:
        lines = f.readlines()
        marker_names = lines[3].strip().split('\t')[2::3]

    trc_data = np.genfromtxt(trc_path, skip_header=5, delimiter='\t')[:, 1:]
    times = trc_data[:, 0]
    frame_rate = round((len(times) - 1) / (times[-1] - times[0]))
    nb_frames = trc_data.shape[0]
    nb_markers = len(marker_names)

    coords = trc_data[:, 1:].reshape(nb_frames, nb_markers, 3) * 1000
    transformed_coords = np.zeros_like(coords)
    transformed_coords[:, :, 0] = coords[:, :, 0]
    transformed_coords[:, :, 1] = -coords[:, :, 2]
    transformed_coords[:, :, 2] = coords[:, :, 1]

    new_c3d = ezc3d.c3d()
    new_c3d['data']['points'] = np.zeros((4, nb_markers, nb_frames))
    new_c3d['data']['points'][:3, :, :] = transformed_coords.transpose(2, 1, 0)
    new_c3d['parameters']['POINT']['LABELS']['value'] = marker_names
    new_c3d['header']['points']['frame_rate'] = frame_rate

    new_c3d['parameters']['POINT']['RATE'] = {
        'description': 'Frame rate for point data',
        'dimension': [1],
        'value': np.array([float(frame_rate)], dtype=np.float64),
        'type': 4,
        'is_locked': False
    }

    if source_c3d_path:
        print(f"🔄 Ajout des données de force depuis : {source_c3d_path}")
        source_c3d = ezc3d.c3d(source_c3d_path)

        analog_data = source_c3d['data']['analogs']  # (1, nChannels, nAnalogFrames)
        analog_labels = source_c3d['parameters']['ANALOG']['LABELS']['value']
        analog_rate = source_c3d['header']['analogs']['frame_rate']

        expected_analog_frames = int(nb_frames * analog_rate / frame_rate)
        actual_analog_frames = analog_data.shape[2]

        if actual_analog_frames > expected_analog_frames:
            print(f"✂️ Trimming analog data from {actual_analog_frames} to {expected_analog_frames} frames")
            analog_data = analog_data[:, :, :expected_analog_frames]
        elif actual_analog_frames < expected_analog_frames:
            pad_width = expected_analog_frames - actual_analog_frames
            print(f"⚠️ Padding analog data with {pad_width} frames")
            analog_data = np.pad(analog_data, ((0, 0), (0, 0), (0, pad_width)))

        new_c3d['data']['analogs'] = analog_data
        new_c3d['parameters']['ANALOG']['LABELS']['value'] = analog_labels
        new_c3d['header']['analogs']['frame_rate'] = analog_rate

        new_c3d['parameters']['ANALOG']['RATE'] = {
            'description': 'Analog samples per second',
            'dimension': [1],
            'value': np.array([float(analog_rate)], dtype=np.float64),
            'type': 4,
            'is_locked': False
        }

    output_path = trc_path.replace('.trc', '.c3d')
    print(f"💾 Écriture du fichier C3D : {output_path}")
    new_c3d.write(output_path)
    print(f"✅ Fichier C3D sauvegardé : {output_path}")




def convert_markers(marker_list):
    """
    Convert a list of marker strings in the form 'NAMEANGLE' into a dictionary mapping marker names to 3D vectors.

    The direction vector is selected based on the marker group (e.g., pelvis vs. limb markers),
    and depends on the numerical angle suffix. This function assumes fixed 2D rotations (0, 90, 180, 270 degrees)
    around a dominant axis (X or Z) depending on the marker's anatomical role.

    Parameters:
        marker_list (List[str]): List of strings like 'RASIS90', 'RLEK180', etc.

    Returns:
        dict: Mapping from marker name (str) to direction vector (tuple of floats).

    Raises:
        ValueError: If an unknown marker name or unsupported angle is encountered.
    """

    group_z = {"RASIS", "LASIS", "LPSIS", "RPSIS"}
    group_x = {"RLEK", "LLEK", "RMEK", "LMEK", "RLM", "LLM", "RMM", "LMM"}

    angle_to_vector_z = {
        0: (0, 0, 0.01),
        90: (0, 0.01, 0),
        180: (0, 0, -0.01),
        270: (0, -0.01, 0)
    }

    angle_to_vector_x = {
        0: (0.01, 0, 0),
        90: (0, 0.01, 0),
        180: (-0.01, 0, 0),
        270: (0, -0.01, 0)
    }

    marker_dict = {}
    for item in marker_list:
        for i, char in enumerate(item):
            if char.isdigit():
                name = item[:i]
                angle = int(item[i:])
                # Choisir la table de correspondance appropriée
                if name in group_z:
                    angle_to_vector = angle_to_vector_z
                elif name in group_x:
                    angle_to_vector = angle_to_vector_x
                else:
                    raise ValueError(f"Nom de marqueur inconnu : {name}")
                vector = angle_to_vector.get(angle)
                if vector is not None:
                    marker_dict[name] = vector
                else:
                    raise ValueError(f"Angle inconnu : {angle}")
                break
    return marker_dict


def convert_markers_(marker_list):
    """
    Convert a list of marker strings in the form 'NAMEANGLE' into a dictionary of 3D vectors.

    Unlike `convert_markers`, this version applies a fixed mapping for all markers, ignoring marker-specific axis logic.
    Only the angle suffix determines the direction vector.

    Parameters:
        marker_list (List[str]): List of strings like 'LLTHI90', 'C7180', etc.

    Returns:
        dict: Mapping from marker name (str) to a corresponding 3D direction vector (tuple of floats).

    Raises:
        ValueError: If an unsupported angle is encountered.
    """
    angle_to_vector = {
        0: (0, 0, 0.01),
        90: (0, 0.01, 0),
        180: (0, 0, -0.01),
        270: (0, -0.01, 0)
    }

    marker_dict = {}
    for item in marker_list:
        for i, char in enumerate(item):
            if char.isdigit():
                name = item[:i]
                angle = int(item[i:])
                vector = angle_to_vector.get(angle)
                if vector is not None:
                    marker_dict[name] = vector
                else:
                    raise ValueError(f"Angle inconnu : {angle}")
                break
    return marker_dict


def clean_marker_names(marker_line):
    """
    Clean a list of marker names by removing angle suffixes (0, 90, 180, 270) and empty entries.

    Keeps specific markers like 'T10' unchanged.  
    Used to normalize marker names for comparison or matching.

    Parameters:
        marker_line (List[str]): List of raw marker names (possibly with angle suffixes).

    Returns:
        List[str]: Cleaned list of marker names with directional suffixes removed.
    """
    cleaned = []
    for name in marker_line:
        if name:
            if name == "T10":
                cleaned_name = name 
            else:
                cleaned_name = re.sub(r'(0|90|180|270)$', '', name)
            cleaned.append(cleaned_name)
    return cleaned


def generate_random_marker_combinations(markers, n):
    """
    Generate `n` unique random combinations of orientation-labeled marker names.

    Each marker can be left unchanged (interpreted as "correct") or be suffixed with a random
    directional label from [0, 90, 180, 270]. Ensures all combinations are unique.

    Parameters:
        markers (List[str]): List of base marker names.
        n (int): Number of unique combinations to generate.

    Returns:
        List[List[str]]: A list containing `n` marker name combinations with random orientations.
    """
    orientations = ["correct", "0", "90", "180", "270"]
    unique_combinations = set()

    while len(unique_combinations) < n:
        combo = []
        for marker in markers:
            orientation = random.choice(orientations)
            if orientation == "correct":
                combo.append(marker)
            else:
                combo.append(f"{marker}{orientation}")
        unique_combinations.add(tuple(combo))

    return [list(c) for c in unique_combinations]


def extract_static_pose(df: pd.DataFrame, marker_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract the 3D coordinates of a static pose for specified markers from a TRC-style DataFrame.

    Assumes the coordinates for each marker are located in columns named Xn, Yn, Zn,
    where n corresponds to the marker index (1-based). Only the first frame (row) is used.

    Parameters:
        df (pd.DataFrame): DataFrame containing TRC marker data.
        marker_names (List[str]): List of marker names to extract.

    Returns:
        Dict[str, np.ndarray]: Mapping of marker names to their 3D coordinates as numpy arrays.
    """
    pose = {}
    for i, marker in enumerate(marker_names):
        cols = [f'X{i+1}', f'Y{i+1}', f'Z{i+1}']
        if all(col in df.columns for col in cols):
            coords = df[cols].iloc[0].values.astype(float)
            pose[marker] = coords
    return pose



def compute_segment_center(pose: Dict[str, np.ndarray], markers: List[str]) -> np.ndarray:
    """
    Compute the geometric center (centroid) of a body segment from selected markers.

    Parameters:
        pose (Dict[str, np.ndarray]): Dictionary mapping marker names to their 3D coordinates.
        markers (List[str]): List of marker names defining the segment.

    Returns:
        np.ndarray: 3D coordinates of the segment's center (mean position of provided markers).
    """
    return np.mean([pose[m] for m in markers if m in pose], axis=0)

def apply_translation(pose: Dict[str, np.ndarray], markers: List[str], translation: np.ndarray) -> None:
    """
    Apply a translation vector to a subset of markers in a static pose.

    Subtracts the given translation from the coordinates of each specified marker, in-place.

    Parameters:
        pose (Dict[str, np.ndarray]): Dictionary of marker names to their 3D coordinates.
        markers (List[str]): List of marker names to which the translation will be applied.
        translation (np.ndarray): 3D translation vector to subtract.
    """
    for m in markers:
        if m in pose:
            pose[m] = pose[m] - translation

def non_rigid_align(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Perform non-rigid alignment of source points to target points using CPD registration.

    Uses Coherent Point Drift (CPD) to deform the source set to match the target set.

    Parameters:
        source_points (np.ndarray): Source point cloud of shape (N, 3).
        target_points (np.ndarray): Target point cloud of shape (N, 3).

    Returns:
        np.ndarray: Transformed source points after alignment (same shape as input).
    """
    reg = DeformableRegistration(X=target_points, Y=source_points)
    TY, _ = reg.register()
    return TY

def prepare_aligned_calibrations(file_paths: List[str],
                                  output_dir: str = "aligned_calibs",
                                  non_rigid: bool = True,
                                  recenter_strategy: str = "pelvis") -> List[np.ndarray]:
    """
    Load and align multiple calibration files (static marker poses) into a common reference space.

    This function:
    - Loads marker data from TRC files.
    - Optionally performs non-rigid alignment using Coherent Point Drift (CPD).
    - Recenters each pose using a chosen anatomical strategy: "pelvis", "segments", or "none".
    - Saves the aligned poses as CSV files and returns them as a list of arrays.

    Parameters:
        file_paths (List[str]): List of file paths to TRC calibration files.
        output_dir (str): Directory where aligned CSV files will be saved. Defaults to "aligned_calibs".
        non_rigid (bool): Whether to apply non-rigid CPD alignment. If False, only recentering is applied.
        recenter_strategy (str): Strategy for recentering poses. Options are:
            - "pelvis": Recenters all markers around the pelvis segment center.
            - "segments": Recenters each anatomical segment independently.
            - "none": No recentering is applied.

    Returns:
        List[np.ndarray]: List of aligned marker arrays, one per file, in millimeters.
    """
    os.makedirs(output_dir, exist_ok=True)

    SEGMENT_GROUPS = {
        "pelvis": ["LASIS", "RASIS", "LPSIS", "RPSIS"],
        "left_knee": ["LLEK", "LMEK"],
        "right_knee": ["RLEK", "RMEK"],
        "left_ankle": ["LLM", "LMM"],
        "right_ankle": ["RLM", "RMM"]
    }

    DEPENDENCY_MAP = {
        "pelvis": ["LLTHI", "RLTHI", "C7", "T10", "XIPH", "JN"],
        "left_knee": ["LLSHA"],
        "right_knee": ["RLSHA"],
        "left_ankle": ["LHEE", "LMT2", "LMT5"],
        "right_ankle": ["RHEE", "RMT2", "RMT5"]
    }

    point_sets = []
    marker_order = None
    poses_raw = []

    for path in file_paths:
        df, markers = load_trc_file_stimuloop(path)
        print(markers)
        pose = extract_static_pose(df, markers)
        if marker_order is None:
            marker_order = markers
        poses_raw.append(pose)
        points = np.array([pose[m] for m in marker_order]) / 1000.0  # convert mm to meters
        point_sets.append(points)

    if non_rigid:
        template = np.mean(np.stack(point_sets), axis=0)

    aligned_sets = []
    for i, pose in enumerate(poses_raw):
        # Step 1: base point cloud
        points = np.array([pose[m] for m in marker_order]) / 1000.0

        if non_rigid:
            TY = non_rigid_align(points, template)
        else:
            TY = points.copy()

        pose_aligned = {m: TY[j] for j, m in enumerate(marker_order)}

        if recenter_strategy == "pelvis":
            pelvis_center = compute_segment_center(pose_aligned, SEGMENT_GROUPS["pelvis"])
            for m in marker_order:
                pose_aligned[m] -= pelvis_center

        elif recenter_strategy == "segments":
            for segment, seg_markers in SEGMENT_GROUPS.items():
                center = compute_segment_center(pose_aligned, seg_markers)
                to_translate = seg_markers + DEPENDENCY_MAP.get(segment, [])
                apply_translation(pose_aligned, to_translate, center)

        # else: "none" → do nothing

        aligned_pose_array = np.array([pose_aligned[m] for m in marker_order]) * 1000.0  # back to mm
        aligned_sets.append(aligned_pose_array)

        df_out = pd.DataFrame(aligned_pose_array, columns=['X', 'Y', 'Z'])
        df_out['Marker'] = marker_order
        df_out.to_csv(os.path.join(output_dir, f"aligned_day{i}.csv"), index=False)

    return aligned_sets



def extract_gait_cycles(source_path):
    """
    Extract 'Foot Off' event times for the left foot from one or more CSV files.

    If a directory is provided, all CSV files inside are processed and merged by gait cycle index.
    Only rows labeled as 'Foot Off' and 'Left' in the respective columns are considered.

    Parameters:
        source_path (str): Path to a single CSV file or a directory containing multiple CSV files.

    Returns:
        pd.DataFrame: A DataFrame where each column contains the 'Foot Off' times per file,
                      indexed by gait cycle number.
    """
    all_gait_cycles = pd.DataFrame()

    if os.path.isfile(source_path) and source_path.endswith(".csv"):
        files = [source_path]
    elif os.path.isdir(source_path):
        files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith(".csv")]
    else:
        raise ValueError("Le chemin spécifié n'est ni un fichier .csv valide ni un dossier.")

    for filepath in files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, skiprows=2)
        df.columns = ["Subject", "Context", "Name", "Time (s)", "Description"]

        foot_off_df = df[(df["Name"].str.strip() == "Foot Off") & (df["Context"].str.strip() == "Left")]
        foot_off_df = foot_off_df.reset_index(drop=True)
        foot_off_df["Gait Cycle"] = foot_off_df.index + 1

        gait_cycles = foot_off_df[["Gait Cycle", "Time (s)"]]
        gait_cycles.rename(columns={"Time (s)": f"Time"}, inplace=True)

        if all_gait_cycles.empty:
            all_gait_cycles = gait_cycles
        else:
            all_gait_cycles = pd.merge(all_gait_cycles, gait_cycles, on="Gait Cycle", how="inner")

    return all_gait_cycles