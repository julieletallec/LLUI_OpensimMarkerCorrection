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

from io_utils import load_trc_file, load_mot_file

def compute_rmse(gt, est):
    """
    Compute the Root Mean Square Error (RMSE) between two arrays.

    Parameters:
        gt (array-like): Ground truth values.
        est (array-like): Estimated or predicted values to compare against the ground truth.

    Returns:
        float: The RMSE value, representing the average magnitude of the error.
    """
    return np.sqrt(np.mean((gt - est) ** 2))



def significance_label(p_value, alpha=0.05):
    """
    Return a qualitative label based on statistical significance.

    Parameters:
        p_value (float): The p-value from a statistical test.
        alpha (float, optional): Significance threshold. Default is 0.05.

    Returns:
        str: "Significant" if p_value < alpha, otherwise "Not significant".
    """
    return "Significant" if p_value < alpha else "Not significant"




def analyze_correction_effect_per_day_with_marker_details__(
    ground_truth_path: str,
    files_before: List[str],
    files_after: List[str],
    markers_to_compare: List[str],
    t_min: float = None,
    t_max: float = None,
    combo_list: List[str] = None
):
    """
    Analyze and visualize the per-marker and global effect of correction methods on TRC data 
    across multiple sessions or conditions, using Root Mean Square Error (RMSE) against a ground truth.

    This function:
    - Computes RMSE per marker and globally (mean of all markers) for each session before and after correction.
    - Performs paired statistical tests (t-test and Wilcoxon) to evaluate improvement significance.
    - Generates line plots and bar charts to visually compare RMSE before and after correction.

    Parameters:
        ground_truth_path (str): Path to the ground truth TRC file.
        files_before (List[str]): List of TRC file paths before correction.
        files_after (List[str]): List of TRC file paths after correction.
        markers_to_compare (List[str]): List of marker names to include in the analysis.
        t_min (float, optional): Start time (in seconds) for evaluation window. Defaults to entire time range.
        t_max (float, optional): End time (in seconds) for evaluation window. Defaults to entire time range.
        combo_list (List[str], optional): Labels for each session/condition for plotting. Optional.

    Returns:
        Tuple:
            - per_marker_errors_before (Dict[str, List[float]]): RMSE per marker before correction for each session.
            - per_marker_errors_after (Dict[str, List[float]]): RMSE per marker after correction for each session.
            - global_errors_before (List[float]): Mean RMSE over all markers before correction.
            - global_errors_after (List[float]): Mean RMSE over all markers after correction.
    """
    assert len(files_before) == len(files_after), "Listes avant/après doivent avoir la même longueur"

    df_gt, marker_names_gt = load_trc_file(ground_truth_path)
    if t_min is not None:
        df_gt = df_gt[df_gt["Time"] >= t_min]
    if t_max is not None:
        df_gt = df_gt[df_gt["Time"] <= t_max]

    marker_indices_gt = {}
    for marker in markers_to_compare:
        if marker in marker_names_gt:
            idx = marker_names_gt.index(marker)
            marker_indices_gt[marker] = [f'X{idx+1}', f'Y{idx+1}', f'Z{idx+1}']
        else:
            print(f"[!] Marqueur '{marker}' absent du ground truth. Ignoré.")

    per_marker_errors_before = {marker: [] for marker in marker_indices_gt}
    per_marker_errors_after = {marker: [] for marker in marker_indices_gt}
    global_errors_before = []
    global_errors_after = []

    for path_before, path_after in zip(files_before, files_after):
        df_before, marker_names_before = load_trc_file(path_before)
        df_after, marker_names_after = load_trc_file(path_after)

        if t_min is not None:
            df_before = df_before[df_before["Time"] >= t_min]
            df_after = df_after[df_after["Time"] >= t_min]
        if t_max is not None:
            df_before = df_before[df_before["Time"] <= t_max]
            df_after = df_after[df_after["Time"] <= t_max]

        rmse_day_before = []
        rmse_day_after = []

        for marker, gt_cols in marker_indices_gt.items():
            if marker not in marker_names_before or marker not in marker_names_after:
                print(f"[!] Marqueur '{marker}' manquant dans {path_before} ou {path_after}. Ignoré.")
                continue

            idx_b = marker_names_before.index(marker)
            idx_a = marker_names_after.index(marker)

            before_cols = [f'X{idx_b+1}', f'Y{idx_b+1}', f'Z{idx_b+1}']
            after_cols = [f'X{idx_a+1}', f'Y{idx_a+1}', f'Z{idx_a+1}']

            gt_data = df_gt[gt_cols].reset_index(drop=True)
            before_data = df_before[before_cols].reset_index(drop=True)
            after_data = df_after[after_cols].reset_index(drop=True)

            min_len = min(len(gt_data), len(before_data), len(after_data))
            gt_trim = gt_data.iloc[:min_len]
            before_trim = before_data.iloc[:min_len]
            after_trim = after_data.iloc[:min_len]

            rmse_b = np.mean([
                compute_rmse(gt_trim.iloc[i], before_trim.iloc[i]) for i in range(min_len)
            ])
            rmse_a = np.mean([
                compute_rmse(gt_trim.iloc[i], after_trim.iloc[i]) for i in range(min_len)
            ])

            per_marker_errors_before[marker].append(rmse_b)
            per_marker_errors_after[marker].append(rmse_a)
            rmse_day_before.append(rmse_b)
            rmse_day_after.append(rmse_a)

        if rmse_day_before and rmse_day_after:
            global_errors_before.append(np.mean(rmse_day_before))
            global_errors_after.append(np.mean(rmse_day_after))

    print("\n=== Global analysis (mean on all markers / day) ===")
    print(f"Mean Error BEFORE correction : {np.mean(global_errors_before):.4f}")
    print(f"Mean Error AFTER correction : {np.mean(global_errors_after):.4f}")
    t_stat, p_t = ttest_rel(global_errors_before, global_errors_after)
    w_stat, p_w = wilcoxon(global_errors_before, global_errors_after)
    print(f"t test : t = {t_stat:.3f}, p = {p_t:.4f} --> {significance_label(p_t)}")
    print(f"Wilcoxon : W = {w_stat:.3f}, p = {p_w:.4f} --> {significance_label(p_w)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(global_errors_before, label='Before Correction (GroundTruth vs Displaced)', marker='o')
    ax.plot(global_errors_after, label='After Correction (GroundTruth vs Corrected)', marker='o')
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.2,
               label='Camera accuracy threshold (0.01 m)')
    ax.set_xlabel("Condition" if combo_list else "Index", fontsize=14)
    ax.set_ylabel("Mean RMSE (m) – all markers", fontsize=14)
    ax.set_title("Mean RMSE per condition – All markers", fontsize=16)

    if combo_list is not None:
        ax.set_xticks(range(len(combo_list)))
        ax.set_xticklabels(combo_list, rotation=45, ha='right')

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n=== Detailed analysis per marker ===")
    for marker in per_marker_errors_before:
        eb = per_marker_errors_before[marker]
        ea = per_marker_errors_after[marker]
        if len(eb) < 2:
            print(f"[!] Trop peu de données pour {marker}")
            continue

        if all(np.isclose(eb[i], ea[i]) for i in range(len(eb))):
            continue  # marker didn't change over any day

        t_stat, p_t = ttest_rel(eb, ea)
        w_stat, p_w = wilcoxon(eb, ea)

        print(f"\n🟢 {marker}")
        print(f"  Mean BEFORE : {np.mean(eb):.4f}")
        print(f"  Mean AFTER  : {np.mean(ea):.4f}")
        print(f"  t-test    : t = {t_stat:.3f}, p = {p_t:.4f} → {significance_label(p_t)}")
        print(f"  Wilcoxon  : W = {w_stat:.3f}, p = {p_w:.4f} → {significance_label(p_w)}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eb, label='Before Correction (GroundTruth vs Displaced)', marker='o')
        ax.plot(ea, label='After Correction (GroundTruth vs Corrected)', marker='o')
        ax.axhline(0.01, color='red', linestyle='--', linewidth=1.2,
                   label='Camera accuracy threshold (0.01 m)')
        ax.set_title(f"RMSE per condition – {marker}", fontsize=16)
        ax.set_xlabel("Condition" if combo_list else "Index", fontsize=14)
        ax.set_ylabel("RMSE (m)", fontsize=14)

        if combo_list is not None:
            ax.set_xticks(range(len(eb)))
            ax.set_xticklabels(combo_list, rotation=45, ha='right')

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    # === Final Bar Chart Summary with StdDev and Significance ===
    print("\n=== Bar Chart: Mean RMSE Before vs After per Marker ===")
    markers = list(per_marker_errors_before.keys())
    means_before = [np.mean(per_marker_errors_before[m]) for m in markers]
    means_after = [np.mean(per_marker_errors_after[m]) for m in markers]
    std_before = [np.std(per_marker_errors_before[m]) for m in markers]
    std_after = [np.std(per_marker_errors_after[m]) for m in markers]
    p_values = [ttest_rel(per_marker_errors_before[m], per_marker_errors_after[m]).pvalue for m in markers]

    x = np.arange(len(markers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, means_before, width, yerr=std_before, capsize=5, label='Before Correction')
    bars2 = ax.bar(x + width/2, means_after, width, yerr=std_after, capsize=5, label='After Correction')

    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.2, label='Camera threshold (0.01 m)')
    ax.set_ylabel('Mean RMSE (m)', fontsize=14)
    ax.set_title('Mean RMSE per Marker – Before vs After Correction', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(markers, rotation=90)
    ax.legend()
    ax.grid(True)

    # Add significance asterisks
    for i, p in enumerate(p_values):
        if p < 0.05:
            y_max = max(means_before[i] + std_before[i], means_after[i] + std_after[i])
            ax.text(x[i], y_max + 0.001, '*', ha='center', va='bottom', fontsize=16, color='black')

    plt.tight_layout()
    plt.show()

    return per_marker_errors_before, per_marker_errors_after, global_errors_before, global_errors_after



def analyze_correction_effect_per_day_kinematics_(
    ground_truth_path: str,
    files_before: List[str],
    files_after: List[str],
    kinematics_to_compare: List[str],
    t_min: float = None,
    t_max: float = None,
    combo_list: List[str] = None
):
    """
    Analyze and visualize the effect of correction strategies on kinematic outputs over multiple sessions.

    This function:
    - Compares pre- and post-correction .mot files to a ground truth by computing RMSE values
      for each selected kinematic variable.
    - Performs paired t-tests and Wilcoxon tests to assess statistical significance.
    - Visualizes global and per-kinematic errors using line plots and bar charts.

    Parameters:
        ground_truth_path (str): Path to the ground truth .mot file.
        files_before (List[str]): List of .mot file paths before correction.
        files_after (List[str]): List of .mot file paths after correction.
        kinematics_to_compare (List[str]): Names of kinematic variables to evaluate (e.g. joint angles).
        t_min (float, optional): Start time (in seconds) for evaluation. Defaults to full time range.
        t_max (float, optional): End time (in seconds) for evaluation. Defaults to full time range.
        combo_list (List[str], optional): Labels for the different test sessions (used for plotting x-axis).

    Returns:
        Tuple:
            - per_kin_errors_before (Dict[str, List[float]]): RMSE per kinematic variable before correction.
            - per_kin_errors_after (Dict[str, List[float]]): RMSE per kinematic variable after correction.
            - global_errors_before (List[float]): Mean RMSE across all kinematics before correction per session.
            - global_errors_after (List[float]): Mean RMSE across all kinematics after correction per session.
    """

    assert len(files_before) == len(files_after), "files_before and files_after must have same length"
    df_gt, kin_names_gt = load_mot_file(ground_truth_path)
    if t_min is not None:
        df_gt = df_gt[df_gt["time"] >= t_min]
    if t_max is not None:
        df_gt = df_gt[df_gt["time"] <= t_max]

    per_kin_errors_before = {kin: [] for kin in kinematics_to_compare}
    per_kin_errors_after = {kin: [] for kin in kinematics_to_compare}
    global_errors_before = []
    global_errors_after = []
    for path_before, path_after in zip(files_before, files_after):
        df_before, kin_names_before = load_mot_file(path_before)
        df_after, kin_names_after = load_mot_file(path_after)

        if t_min is not None:
            df_before = df_before[df_before["time"] >= t_min]
            df_after = df_after[df_after["time"] >= t_min]
        if t_max is not None:
            df_before = df_before[df_before["time"] <= t_max]
            df_after = df_after[df_after["time"] <= t_max]

        rmse_day_before = []
        rmse_day_after = []

        for kin in kinematics_to_compare:
            if kin not in df_before.columns or kin not in df_after.columns or kin not in df_gt.columns:
                print(f"[!] '{kin}' missing in one or more datasets. Skipped.")
                continue

            gt_data = df_gt[kin].reset_index(drop=True)
            before_data = df_before[kin].reset_index(drop=True)
            after_data = df_after[kin].reset_index(drop=True)

            min_len = min(len(gt_data), len(before_data), len(after_data))
            if min_len == 0:
                print(f"[!] No overlapping data for {kin} – skipping.")
                continue

            gt_trim = gt_data.iloc[:min_len]
            before_trim = before_data.iloc[:min_len]
            after_trim = after_data.iloc[:min_len]

            rmse_b = compute_rmse(gt_trim.values, before_trim.values)
            rmse_a = compute_rmse(gt_trim.values, after_trim.values)

            per_kin_errors_before[kin].append(rmse_b)
            per_kin_errors_after[kin].append(rmse_a)
            rmse_day_before.append(rmse_b)
            rmse_day_after.append(rmse_a)

        if rmse_day_before and rmse_day_after:
            global_errors_before.append(np.mean(rmse_day_before))
            global_errors_after.append(np.mean(rmse_day_after))

    if not global_errors_before or not global_errors_after:
        print("\n[!] No valid data points found for global RMSE analysis.")
        return per_kin_errors_before, per_kin_errors_after, global_errors_before, global_errors_after

    print("\n=== Global Kinematic Analysis ===")
    print(f"Mean Error BEFORE correction : {np.mean(global_errors_before):.4f}")
    print(f"Mean Error AFTER correction : {np.mean(global_errors_after):.4f}")
    t_stat, p_t = ttest_rel(global_errors_before, global_errors_after)
    w_stat, p_w = wilcoxon(global_errors_before, global_errors_after)
    print(f"t-test     : t = {t_stat:.3f}, p = {p_t:.4f} → {significance_label(p_t)}")
    print(f"Wilcoxon   : W = {w_stat:.3f}, p = {p_w:.4f} → {significance_label(p_w)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(global_errors_before, label='Before Correction (GroundTruth vs Displaced)', marker='o')
    ax.plot(global_errors_after, label='After correction (GroundTruth vs Corrected)', marker='o')
    ax.axhline(5, color='red', linestyle='--', linewidth=1.2, label='Minimal Detectable Change (5°)')
    ax.set_title("Global RMSE per Condition – All Kinematics", fontsize=16)
    ax.set_xlabel("Condition", fontsize=14)
    ax.set_ylabel("RMSE (degrees)", fontsize=14)

    if combo_list:
        filtered_labels = combo_list
        ax.set_xticks(range(len(global_errors_before)))
        ax.set_xticklabels(filtered_labels[:len(global_errors_before)], rotation=45, ha='right')

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n=== Per-Kinematic Analysis ===")
    for kin in per_kin_errors_before:
        eb = per_kin_errors_before[kin]
        ea = per_kin_errors_after[kin]
        if len(eb) < 2:
            print(f"[!] Not enough data for {kin}")
            continue

        if all(np.isclose(eb[i], ea[i]) for i in range(len(eb))):
            continue  # no change across conditions

        t_stat, p_t = ttest_rel(eb, ea)
        w_stat, p_w = wilcoxon(eb, ea)

        print(f"\n🟢 {kin}")
        print(f"  Mean BEFORE : {np.mean(eb):.4f}")
        print(f"  Mean AFTER  : {np.mean(ea):.4f}")
        print(f"  t-test    : t = {t_stat:.3f}, p = {p_t:.4f} → {significance_label(p_t)}")
        print(f"  Wilcoxon  : W = {w_stat:.3f}, p = {p_w:.4f} → {significance_label(p_w)}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eb, label='Before Correction (GroundTruth vs Displaced)', marker='o')
        ax.plot(ea, label='After Correction (GroundTruth vs Corrected)', marker='o')
        ax.axhline(5, color='red', linestyle='--', linewidth=1.2,
                   label='Minimal Detectable Change (5°)')
        ax.set_title(f"RMSE per Condition – {kin}", fontsize=16)
        ax.set_xlabel("Condition", fontsize=14)
        ax.set_ylabel("RMSE (degrees)", fontsize=14)

        if combo_list:
            filtered_labels = combo_list
            ax.set_xticks(range(len(eb)))
            ax.set_xticklabels(filtered_labels[:len(eb)], rotation=45, ha='right')

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    # === Final Summary Bar Chart with Std and Significance ===
    print("\n=== Summary Bar Chart: Mean RMSE Before vs After per Kinematic ===")
    kins = list(per_kin_errors_before.keys())
    means_before = [np.mean(per_kin_errors_before[k]) for k in kins]
    means_after = [np.mean(per_kin_errors_after[k]) for k in kins]
    std_before = [np.std(per_kin_errors_before[k]) for k in kins]
    std_after = [np.std(per_kin_errors_after[k]) for k in kins]
    p_values = [ttest_rel(per_kin_errors_before[k], per_kin_errors_after[k]).pvalue for k in kins]

    x = np.arange(len(kins))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, means_before, width, yerr=std_before, capsize=5, label='Before Correction')
    bars2 = ax.bar(x + width/2, means_after, width, yerr=std_after, capsize=5, label='After Correction')

    ax.axhline(5, color='red', linestyle='--', linewidth=1.2, label='Minimal Detectable Change (5°)')
    ax.set_ylabel('Mean RMSE (degrees)', fontsize=14)
    ax.set_title('Mean RMSE per Kinematic – Before vs After Correction', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(kins, rotation=90)
    ax.legend()
    ax.grid(True)

    # Add significance stars
    for i, p in enumerate(p_values):
        if p < 0.05:
            y_max = max(means_before[i] + std_before[i], means_after[i] + std_after[i])
            ax.text(x[i], y_max + 0.1, '*', ha='center', va='bottom', fontsize=16, color='black')

    plt.tight_layout()
    plt.show()

    return per_kin_errors_before, per_kin_errors_after, global_errors_before, global_errors_after




def compare_trc_files(file_entries: List[dict], markers_to_plot: List[str]):
    """
    Plot and compare the trajectories of specified markers across multiple TRC files.

    For each marker, this function generates a 3-panel subplot (X, Y, Z) showing the marker’s 
    position over time for all provided TRC files. Optional scaling and time window cropping 
    can be applied to each file.

    Parameters:
        file_entries (List[dict]): List of dictionaries, each representing a TRC file with optional metadata.
            Each dictionary may contain:
                - "path" (str): Path to the TRC file. (required)
                - "name" (str): Display name for legend. Defaults to file path.
                - "scale" (float): Scaling factor for coordinates. Defaults to 1.0.
                - "t_min" (float): Minimum time to include (in seconds).
                - "t_max" (float): Maximum time to include (in seconds).
                - "color" (str): Line color for the plot.
                - "width" (float): Line width. Defaults to 1.5.

        markers_to_plot (List[str]): Names of markers to compare.

    Returns:
        None. Displays comparison plots using matplotlib.
    """
    dataframes = []
    marker_maps = []
    plot_styles = []

    for entry in file_entries:
        path = entry["path"]
        scale = entry.get("scale", 1.0)
        t_min = entry.get("t_min", None)
        t_max = entry.get("t_max", None)
        name = entry.get("name", path)
        color = entry.get("color", None)
        width = entry.get("width", 1.5)

        df, marker_names = load_trc_file(path)

        if t_min is not None:
            df = df[df['Time'] >= t_min]
        if t_max is not None:
            df = df[df['Time'] <= t_max]

        coord_cols = [col for col in df.columns if col.startswith(('X', 'Y', 'Z'))]
        df[coord_cols] *= scale

        marker_map = {name: (f'X{i+1}', f'Y{i+1}', f'Z{i+1}') for i, name in enumerate(marker_names)}

        dataframes.append(df)
        marker_maps.append(marker_map)
        plot_styles.append({"name": name, "color": color, "width": width})

    for marker in markers_to_plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        fig.suptitle(f"Comparaison du marqueur : {marker}")

        for i, df in enumerate(dataframes):
            marker_map = marker_maps[i]
            style = plot_styles[i]
            if marker not in marker_map:
                print(f"Marqueur '{marker}' non trouvé dans {file_entries[i]['path']}")
                continue

            x_col, y_col, z_col = marker_map[marker]
            time = df['Time']

            axs[0].plot(time, df[x_col], label=style["name"], color=style["color"], linewidth=style["width"])
            axs[1].plot(time, df[y_col], label=style["name"], color=style["color"], linewidth=style["width"])
            axs[2].plot(time, df[z_col], label=style["name"], color=style["color"], linewidth=style["width"])

        axs[0].set_ylabel('X')
        axs[1].set_ylabel('Y')
        axs[2].set_ylabel('Z')
        axs[2].set_xlabel('Temps (s)')

        # Légende globale unique
        lines, labels = [], []
        for ax in axs:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
            ax.legend().remove()

        #fig.legend(lines, labels, loc='right', ncol=len(dataframes), bbox_to_anchor=(0.5, 1.05))

        plt.tight_layout()
        plt.show()






def compare_mot_files(file_entries: List[dict], kinematics_to_plot: List[str]):
    """
    Plot and compare the time-series of specified kinematic variables from multiple .mot files.

    This function overlays each selected kinematic variable across the provided .mot files 
    to visually assess differences. Optional scaling and time filtering can be applied.

    Parameters:
        file_entries (List[dict]): List of dictionaries, each describing a .mot file and its plot style.
            Each dictionary may include:
                - "path" (str): Path to the .mot file. (required)
                - "name" (str): Label used in the plot legend. Defaults to the file path.
                - "scale" (float): Optional scaling factor for kinematic values. Defaults to 1.0.
                - "t_min" (float): Optional minimum time to include (in seconds).
                - "t_max" (float): Optional maximum time to include (in seconds).
                - "color" (str): Line color for the plot.
                - "width" (float): Line width. Defaults to 1.5.

        kinematics_to_plot (List[str]): Names of kinematic variables to compare.

    Returns:
        None. Displays comparison plots using matplotlib.
    """

    dataframes = []
    plot_styles = []

    for entry in file_entries:
        path = entry["path"]
        scale = entry.get("scale", 1.0)
        t_min = entry.get("t_min", None)
        t_max = entry.get("t_max", None)
        name = entry.get("name", path)
        color = entry.get("color", None)
        width = entry.get("width", 1.5)

        df, kinematic_names = load_mot_file(path)

        if t_min is not None:
            df = df[df['time'] >= t_min]
        if t_max is not None:
            df = df[df['time'] <= t_max]

        # Apply scaling to all kinematic columns (excluding time)
        kin_cols = [col for col in df.columns if col != 'time']
        df[kin_cols] *= scale

        dataframes.append(df)
        plot_styles.append({"name": name, "color": color, "width": width})

    for kin in kinematics_to_plot:
        plt.figure(figsize=(10, 5))
        plt.title(f"Kinematic Comparison: {kin}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")

        for i, df in enumerate(dataframes):
            style = plot_styles[i]
            if kin not in df.columns:
                print(f"Kinematic '{kin}' not found in {file_entries[i]['path']}")
                continue

            plt.plot(df['time'], df[kin], label=style["name"], color=style["color"], linewidth=style["width"])

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()







def compare_markers_to_average(csv_folder: str, subject: str, output_plot_prefix: str = None):
    """
    Compare aligned marker positions across multiple sessions against their average configuration.

    This function:
    - Computes per-marker Euclidean and axis-specific errors relative to the mean marker position.
    - Visualizes the variation per marker across sessions using boxplots and stripplots.
    - Optionally saves the plots to disk.

    Parameters:
        csv_folder (str): Path to the folder containing aligned CSV files (one per session/day).
        subject (str): Name of the subject (used for plot titles).
        output_plot_prefix (str, optional): If provided, saves each plot with this prefix. Otherwise, plots are shown.

    Returns:
        None. Generates and displays/saves plots of positional errors per marker across sessions.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    aligned_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    all_poses = []
    marker_names = None

    for file in aligned_files:
        df = pd.read_csv(os.path.join(csv_folder, file))
        df = df.sort_values("Marker").reset_index(drop=True)
        if marker_names is None:
            marker_names = df["Marker"].tolist()
        coords = df[["X", "Y", "Z"]].values
        all_poses.append(coords)

    all_poses = np.stack(all_poses)  # shape (N_days, N_markers, 3)
    mean_model = np.mean(all_poses, axis=0)  # shape (N_markers, 3)

    # Build full dataframe
    plot_data = []
    for day_index, pose in enumerate(all_poses):
        for i, marker in enumerate(marker_names):
            error_vector = pose[i] - mean_model[i]
            dist = np.linalg.norm(error_vector)
            plot_data.append({
                "Marker": marker,
                "Day": f"Day {day_index}",
                "Error (m)": dist,
                "Error X": error_vector[0],
                "Error Y": error_vector[1],
                "Error Z": error_vector[2],
            })

    df_plot = pd.DataFrame(plot_data)

    # Common settings
    palette = sns.color_palette("husl", n_colors=len(aligned_files))
    legend_args = dict(title="Day", bbox_to_anchor=(1.01, 1), loc='upper left')

    def plot_error_component(y_col, title_suffix):
        plt.figure(figsize=(16, 6))
        sns.boxplot(data=df_plot, x="Marker", y=y_col, color='white', linewidth=1.2)
        sns.stripplot(data=df_plot, x="Marker", y=y_col, hue="Day",
                      palette=palette, size=5, jitter=0.2, dodge=False, alpha=0.8)

        plt.axhline(10, color='red', linestyle='--', label='Camera threshold (0.01 m)')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{title_suffix} per Marker for {subject}", fontsize=14)
        plt.ylabel(f"{title_suffix} (mm)")
        plt.grid(True)
        plt.legend(**legend_args)
        plt.tight_layout()
        if output_plot_prefix:
            filename = f"{output_plot_prefix}_{y_col.replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300)
            print(f"📊 Saved: {filename}")
            plt.close()
        else:
            plt.show()

    # 1 – Euclidean
    plot_error_component("Error (m)", "Total Euclidean Error")

    # 2 – X component
    plot_error_component("Error X", "X-Axis Error")

    # 3 – Y component
    plot_error_component("Error Y", "Y-Axis Error")

    # 4 – Z component
    plot_error_component("Error Z", "Z-Axis Error")






def compare_markers_by_day(csv_folder: str, subject: str, output_plot_prefix: str = None):
    """
    Analyze and visualize per-day variability of marker positions grouped by anatomical regions.

    This function:
    - Computes Euclidean and directional (X, Y, Z) errors of each marker relative to the multi-day average position.
    - Groups markers into anatomical regions (e.g., pelvis, thigh, foot) for color-coded visualization.
    - Generates per-day boxplots and scatter overlays to identify variability trends.
    - Optionally saves each plot with a specified filename prefix.

    Parameters:
        csv_folder (str): Path to the folder containing aligned marker CSV files (one per day/session).
        subject (str): Subject name used in plot titles.
        output_plot_prefix (str, optional): Prefix to use for saving output figures. If not provided, plots are shown.

    Returns:
        None. Displays or saves per-day error plots for each error dimension.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    # ---- 1. Définir les groupes de marqueurs par zone ----
    torso_markers = ["C7", "T10", "XIPH", "JN"]
    pelvis_markers = ["LASIS", "RASIS", "LPSIS", "RPSIS"]
    right_knee = ["RMEK", "RLEK"]
    left_knee = ["LMEK", "LLEK"]
    right_foot = ["RMM", "RLM", "RMT2", "RMT5", "RHEE"]
    left_foot = ["LMM", "LLM", "LMT2", "LMT5", "LHEE"]
    thighs = ["LLSHA", "RLSHA", "LLTHI", "RLTHI"]

    color_map = {
        "torso": "#1f77b4",      # bleu
        "pelvis": "#2ca02c",     # vert
        "right_knee": "#ff7f0e", # orange
        "left_knee": "#d62728",  # rouge
        "right_foot": "#9467bd", # violet
        "left_foot": "#8c564b",  # brun
        "thighs": "#17becf"      # cyan
    }

    def get_marker_group(marker):
        if marker in torso_markers: return "torso"
        if marker in pelvis_markers: return "pelvis"
        if marker in right_knee: return "right_knee"
        if marker in left_knee: return "left_knee"
        if marker in right_foot: return "right_foot"
        if marker in left_foot: return "left_foot"
        if marker in thighs: return "thighs"
        return "other"

    # ---- 2. Charger les fichiers CSV et structurer les données ----
    aligned_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    print(aligned_files)
    all_poses = []
    marker_names = None

    for file in aligned_files:
        df = pd.read_csv(os.path.join(csv_folder, file))
        df = df.sort_values("Marker").reset_index(drop=True)
        if marker_names is None:
            marker_names = df["Marker"].tolist()
        coords = df[["X", "Y", "Z"]].values
        all_poses.append(coords)

    all_poses = np.stack(all_poses)  # shape (N_days, N_markers, 3)
    mean_model = np.mean(all_poses, axis=0)  # shape (N_markers, 3)

    # ---- 3. Préparation des données pour le graphique ----
    plot_data = []
    for day_index, pose in enumerate(all_poses):
        for i, marker in enumerate(marker_names):
            group = get_marker_group(marker)
            error_vector = pose[i] - mean_model[i]
            dist = np.linalg.norm(error_vector)
            plot_data.append({
                "Day": f"Day {day_index}",
                "Marker": marker,
                "Group": group,
                "Error (m)": dist,
                "Error X": error_vector[0],
                "Error Y": error_vector[1],
                "Error Z": error_vector[2],
            })

    df_plot = pd.DataFrame(plot_data)

    # ---- 4. Palette par groupe ----
    group_palette = {group: color_map[group] for group in df_plot["Group"].unique()}

    def plot_error_by_day(y_col, title_suffix):
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_plot, x="Day", y=y_col, color='white', linewidth=1.2)
        sns.stripplot(
            data=df_plot,
            x="Day",
            y=y_col,
            hue="Group",
            palette=group_palette,
            size=5,
            jitter=0.25,
            dodge=False,
            alpha=0.8
        )

        plt.axhline(10, color='red', linestyle='--', label='Camera threshold (0.01 m)')
        plt.title(f"{title_suffix} per Day for {subject}", fontsize=14)
        plt.ylabel(f"{title_suffix} (mm)")
        plt.grid(True)
        plt.legend(title="Body Region", bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()

        if output_plot_prefix:
            filename = f"{output_plot_prefix}_{y_col.replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300)
            print(f"📊 Saved: {filename}")
            plt.close()
        else:
            plt.show()

    # ---- 5. Générer les graphiques ----
    plot_error_by_day("Error (m)", "Total Euclidean Error")
    plot_error_by_day("Error X", "X-Axis Error")
    plot_error_by_day("Error Y", "Y-Axis Error")
    plot_error_by_day("Error Z", "Z-Axis Error")
