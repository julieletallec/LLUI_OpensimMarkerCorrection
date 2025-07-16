# `analysis_utils.py`

This module provides a suite of analysis and visualization tools for evaluating the **effectiveness of motion data correction**, focusing on **marker trajectories (TRC)** and **kinematic signals (MOT)**.  
It enables statistical comparison, RMSE evaluation, and visual inspection across sessions or correction stages.

---

## Contents

| Function | Description |
|----------|-------------|
| **`compute_rmse(gt, est)`** | Compute the root-mean-square error between ground truth and estimated vectors. |
| **`significance_label(p_value, alpha=0.05)`** | Return a significance label ("Significant" / "Not significant") based on a p-value and threshold. |
| **`analyze_correction_effect_per_day_with_marker_details__`** | Analyze and compare marker-wise RMSE before/after correction across multiple TRC sessions. Includes statistical tests and detailed plots per marker. |
| **`analyze_correction_effect_per_day_kinematics_`** | Analyze joint angle (MOT) data to evaluate kinematic correction effects across conditions. Computes RMSE and plots per-variable statistics. |
| **`compare_trc_files(file_entries, markers_to_plot)`** | Overlay marker trajectories (X, Y, Z) from multiple TRC files. Useful for visual inspection of correction results. |
| **`compare_mot_files(file_entries, kinematics_to_plot)`** | Plot selected kinematic variables over time from multiple MOT files for comparison. |
| **`compare_markers_to_average(csv_folder, subject, output_plot_prefix=None)`** | Compare aligned marker positions from multiple sessions to the group average. Plots total and axis-specific errors per marker. |
| **`compare_markers_by_day(csv_folder, subject, output_plot_prefix=None)`** | Analyze marker error distributions per day, grouped by body region. Highlights segment-wise drift or inconsistency. |

---




# `data_processing_utils.py`

This utility module provides core functionality for processing and preparing motion capture datasets. It includes tools for converting file formats (TRC ↔ C3D), transforming marker orientations, aligning static poses across sessions, and extracting key gait events. These functions are particularly useful for workflows involving OpenSim, motion analysis, and marker-based calibration procedures.

---

## Contents

| Function | Description |
|----------|-------------|
| `convert_c3d_to_trc` | Converts a `.c3d` file into a `.trc` file, optionally filtering time ranges and renaming markers for OpenSim compatibility. |
| `convert_trc_to_c3d` | Converts a `.trc` file into a `.c3d` file, optionally merging analog (e.g., force) data from a source `.c3d`. |
| `convert_markers` | Maps formatted marker names (e.g., `RASIS90`) to 3D orientation vectors based on anatomical context. |
| `convert_markers_` | Simplified marker-to-vector converter using fixed angle mappings, without context-specific logic. |
| `clean_marker_names` | Cleans a list of marker names by removing directional suffixes (`0`, `90`, etc.) while preserving key labels. |
| `generate_random_marker_combinations` | Generates `n` randomized combinations of marker orientations for perturbation or robustness testing. |
| `extract_static_pose` | Extracts 3D coordinates of specified markers from the first frame of a motion capture trial. |
| `compute_segment_center` | Computes the geometric center of a group of markers representing a body segment. |
| `apply_translation` | Applies a translation vector to selected markers in a pose (modifies the data in place). |
| `non_rigid_align` | Performs non-rigid alignment (Coherent Point Drift) between two point clouds. |
| `prepare_aligned_calibrations` | Loads and aligns multiple static calibration TRC files using non-rigid registration and optional recentring strategies. |
| `extract_gait_cycles` | Extracts left foot 'Foot Off' event times from one or more CSV gait event files and aligns them by cycle index. |

---
