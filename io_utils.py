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


from data_processing_utils import clean_marker_names

def find_latest_calibration_file(folder_path):
    """
    Finds the latest calibration C3D file in a folder based on the highest numeric index in the filename.

    Only considers files starting with "calibration" and ending with ".c3d", excluding any that contain 
    a dot (.) in the base filename (to avoid intermediate or hidden versions).

    Parameters:
        folder_path (str): Path to the directory to search in.

    Returns:
        str or None: Full path to the calibration file with the highest index, or None if no match is found.
    """
    calibration_files = []
    for f in os.listdir(folder_path):
        if f.lower().startswith("calibration") and f.lower().endswith(".c3d"):
            name_without_ext = f[:-4]  # sans .c3d
            if '.' not in name_without_ext:  # rejette si "." dans le nom
                calibration_files.append(f)

    if not calibration_files:
        return None

    def extract_numeric_index(filename):
        name = filename[:-4]  # enlever .c3d
        matches = re.findall(r'\d+', name)
        return int(matches[-1]) if matches else -1

    best_file = max(calibration_files, key=extract_numeric_index)
    return os.path.join(folder_path, best_file)




def collect_calibration_files(source_folder, final_folder):
    """
    Collects the latest calibration C3D file from each session of each patient and copies them into a central folder.

    For each patient directory in the source folder, the function searches for subdirectories matching 
    the pattern *_T{index} (e.g., _T0 to _T15), finds the latest calibration file using `find_latest_calibration_file`, 
    and copies it to a standardized output folder structure.

    Parameters:
        source_folder (str): Path to the root directory containing patient subfolders.
        final_folder (str): Destination directory where organized calibration files will be stored.

    Side Effects:
        Creates directories and copies files to the destination path.
        Prints the progress and warnings for missing calibration files.
    """
    os.makedirs(final_folder, exist_ok=True)

    for patient_name in os.listdir(source_folder):
        patient_source_path = os.path.join(source_folder, patient_name)
        if not os.path.isdir(patient_source_path):
            continue

        print(f"Processing patient: {patient_name}")
        patient_final_path = os.path.join(final_folder, patient_name)
        os.makedirs(patient_final_path, exist_ok=True)

        for i in range(16):
            suffix = f"_T{i}"
            matching_dirs = [
                d for d in os.listdir(patient_source_path)
                if d.endswith(suffix) and os.path.isdir(os.path.join(patient_source_path, d))
            ]

            for session_dir in matching_dirs:
                session_path = os.path.join(patient_source_path, session_dir)
                calibration_file_path = find_latest_calibration_file(session_path)

                if calibration_file_path:
                    dest_filename = f"CalibrationT{i}.c3d"
                    dest_path = os.path.join(patient_final_path, dest_filename)

                    shutil.copy2(calibration_file_path, dest_path)
                    print(f"✅ Copied: {calibration_file_path} -> {dest_path}")
                else:
                    print(f"⚠️ Warning: No valid calibration*.c3d found in {session_path}")



    
def load_trc_file(filepath):
    """
    Loads a TRC file and extracts its data and marker names.

    Parses the file header to retrieve marker labels and loads the associated time-series
    data into a pandas DataFrame. Assumes standard TRC format with metadata in the first 5 lines.

    Parameters:
        filepath (str): Path to the .trc file.

    Returns:
        tuple:
            - DataFrame: Marker coordinate data with time and frame information.
            - List[str]: Names of the markers.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    marker_line = lines[3].strip().split('\t')[2:]
    marker_names = [marker_line[i] for i in range(0, len(marker_line), 3)]

    xyz_headers = lines[4].strip().split('\t')
    headers = ['Frame#', 'Time'] + xyz_headers

    df = pd.read_csv(filepath, sep='\t', skiprows=5, header=None)
    df.columns = headers

    return df, marker_names




def load_trc_file_stimuloop(filepath: str) -> (pd.DataFrame, List[str]):
    """
    Loads a TRC file in Stimuloop format and extracts both marker data and marker names.

    Similar to `load_trc_file`, but includes internal print statements for debugging and assumes 
    consistent formatting from Stimuloop TRC exports.

    Parameters:
        filepath (str): Path to the TRC file.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame containing frame, time, and 3D marker coordinates.
            - List[str]: List of marker names extracted from the header.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        print(lines)

    marker_line = lines[3].strip().split('\t')[2:]
    print(marker_line)
    marker_names = [marker_line[i] for i in range(0, len(marker_line), 3)]
    print(marker_names)
    xyz_headers = lines[4].strip().split('\t')
    headers = ['Frame#', 'Time'] + xyz_headers
    print(headers)
    df = pd.read_csv(filepath, sep='\t', skiprows=5, header=None)
    df.columns = headers

    return df, marker_names




def load_mot_file(filepath):
    """
    Loads a .mot (motion) file typically used in OpenSim and extracts its data and kinematic variable names.

    The function skips the header until it finds the 'endheader' keyword, then reads the column names and data.

    Parameters:
        filepath (str): Path to the .mot file.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame containing time and kinematic variables.
            - List[str]: List of kinematic variable names (excluding the 'time' column).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_end_idx = next(i for i, line in enumerate(lines) if 'endheader' in line.lower())

    column_headers = lines[header_end_idx + 1].strip().split('\t')

    df = pd.read_csv(filepath, sep='\t', skiprows=header_end_idx + 2, header=None)
    df.columns = column_headers

    return df, column_headers[1:]  


def save_aligned_trc(aligned_pose: np.ndarray,
                     reference_trc_path: str,
                     output_trc_path: str,
                     marker_names: List[str],
                     start_time: float = 0.0,
                     end_time: float = 1.0,
                     frequency: float = 100.0):
    """
    Saves an aligned static pose into a new TRC file, using the header from a reference TRC file.

    This function reconstructs a valid OpenSim-compatible TRC file by injecting aligned marker coordinates 
    and rebuilding the header with proper marker labels and metadata.

    Parameters:
        aligned_pose (np.ndarray): 2D array of shape (n_markers, 3) with XYZ coordinates in mm.
        reference_trc_path (str): Path to an existing TRC file to reuse its header structure.
        output_trc_path (str): Output path where the new TRC will be saved.
        marker_names (List[str]): List of marker names in order corresponding to the pose.
        start_time (float): Starting timestamp for the output data.
        end_time (float): Ending timestamp for the output data.
        frequency (float): Sampling frequency in Hz (frames per second).

    Returns:
        None: Writes a new TRC file to the specified output path.
    """
    with open(reference_trc_path, 'r') as f:
        lines = f.readlines()

    header_end_index = None
    for i, line in enumerate(lines):
        if line.startswith('Frame#'):
            header_end_index = i + 2
            break
    static_header = lines[:header_end_index - 2]

    marker_line = ['Frame#', 'Time'] + marker_names
    suffix_line = ['', '']
    for i in range(1, len(marker_names) + 1):
        suffix_line.extend([f'X{i}', f'Y{i}', f'Z{i}'])

    marker_line_str = '\t'.join(marker_line) + '\n'
    suffix_line_str = '\t'.join(suffix_line) + '\n'
    header_lines = static_header + [marker_line_str, suffix_line_str]

    duration = end_time - start_time
    n_frames = int(duration * frequency) + 1
    time_values = np.linspace(start_time, end_time, n_frames)
    frame_numbers = np.arange(1, n_frames + 1)

    flat_coords = aligned_pose.flatten()
    data_rows = []
    for frame, t in zip(frame_numbers, time_values):
        row = [frame, f"{t:.5f}"] + [f"{c:.5f}" for c in flat_coords]
        data_rows.append(row)

    with open(output_trc_path, 'w') as f:
        f.writelines(header_lines)
        for row in data_rows:
            f.write('\t'.join(map(str, row)) + '\n')

    print(f"✅ Fichier TRC aligné sauvegardé : {output_trc_path}")





def load_average_and_save_trc_bis(folder_path, reference_trc_path, output_path, reference_marker_order, startswith='static_Julie_combo', opensim=False):
    """
    Loads multiple TRC files, averages their marker trajectories, and saves the result to a new TRC file.

    The function:
    - Filters TRC files based on a given prefix.
    - Aligns columns to a reference marker order.
    - Computes the average 3D coordinates across files.
    - Builds a consistent TRC header from a reference file.
    - Saves the averaged data to disk in OpenSim-compatible format.

    Parameters:
        folder_path (str): Path to the folder containing TRC files to average.
        reference_trc_path (str): Path to a reference TRC file used to build the output header.
        output_path (str): Path where the averaged TRC file will be saved.
        reference_marker_order (List[str]): Ordered list of marker names expected in the output file.
        startswith (str): Prefix to filter which TRC files in the folder are processed.
        opensim (bool): If True, scales the averaged coordinates to millimeters for OpenSim compatibility.

    Side Effects:
        Writes an averaged TRC file to `output_path`.
    """
    # Liste les fichiers .trc correspondant au préfixe
    trc_files = sorted([f for f in os.listdir(folder_path) if f.startswith(startswith)])
    dfs = []

    for filename in trc_files:
        print(filename)
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('Frame#'):
                marker_names_line = lines[i].strip().split('\t')
                coord_suffixes_line = lines[i + 1].strip().split('\t')
                header_line_idx = i
                break

        raw_marker_names = marker_names_line[2:]
        cleaned_markers = clean_marker_names(raw_marker_names)

        # Construit les noms complets des colonnes
        current_full_columns = ['Frame#', 'Time']
        for marker in cleaned_markers:
            current_full_columns.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

        df = pd.read_csv(file_path, sep='\t', skiprows=header_line_idx + 1)
        if opensim:
            df = df.iloc[:, :-1]  # retire la dernière colonne vide éventuelle

        df.columns = ['Frame#', 'Time'] + coord_suffixes_line
        df.columns = current_full_columns

        # Réordonne les colonnes selon la référence
        full_column_names = ['Frame#', 'Time']
        for marker in reference_marker_order:
            full_column_names.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

        # Vérifie la présence de toutes les colonnes nécessaires
        missing = [col for col in full_column_names if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing columns in {filename}: {missing}")

        df = df[full_column_names]
        dfs.append(df)

    # Tronque tous les DataFrames à la même longueur
    min_len = min(len(df) for df in dfs)
    dfs = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]

    # Moyenne des coordonnées
    frame_time = dfs[0][['Frame#', 'Time']]
    coord_columns = [col for col in full_column_names if col not in ['Frame#', 'Time']]
    data_stack = np.stack([df[coord_columns].to_numpy() for df in dfs])
    avg_data = data_stack.mean(axis=0)

    if opensim:
        avg_data *= 1000  # Convertit en mm si nécessaire

    averaged_coords = pd.DataFrame(avg_data, columns=coord_columns)
    averaged_df = pd.concat([frame_time, averaged_coords], axis=1)

    # Lit l'en-tête du fichier de référence
    with open(reference_trc_path, 'r') as f:
        reference_lines = f.readlines()

    # Repère jusqu’où garder l’en-tête original
    header_end_index = None
    for i, line in enumerate(reference_lines):
        if line.startswith('Frame#'):
            header_end_index = i + 2
            break
    static_header = reference_lines[:header_end_index - 2]  # garde les lignes avant les noms de marqueurs

    # Reconstruit dynamiquement l’en-tête correct
    marker_line = ['Frame#', 'Time'] + reference_marker_order
    suffix_line = ['', '']
    for i in range(1, len(reference_marker_order) + 1):
        suffix_line.extend([f'X{i}', f'Y{i}', f'Z{i}'])
    marker_line_str = '\t'.join(marker_line) + '\n'
    suffix_line_str = '\t'.join(suffix_line) + '\n'
    header_lines = static_header + [marker_line_str, suffix_line_str]

    # Écrit le fichier de sortie
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        averaged_df.to_csv(f, sep='\t', index=False, header=False, float_format='%.5f')

    print(f"✅ Fichier .trc moyen sauvegardé : {output_path}")


