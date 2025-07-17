
# DETAILED FILE DESCRIPTION
---
# 📁 FINAL_USABLE_PIPELINES
## 📄 `01_translate_markerset.ipynb`

## 📄 `02_add_marker.ipynb`

## 📄 `03_correct_marker_displacement_15_days.ipynb`
---
# 📁 TEST_PIPELINES
---
# 📁 example_c3d_files
---
# 📁 example_gaitevents_files
---
# 📁 example_marker_files
---
# 📁 example_model_files
---
# 📁 example_osim_setup_files
---
# 📁 example_model_files
---
# 📁 utils
## `analysis_utils.py`

This module provides a suite of analysis and visualization tools for evaluating the **effectiveness of motion data correction**, focusing on **marker trajectories (TRC)** and **kinematic signals (MOT)**.  
It enables statistical comparison, RMSE evaluation, and visual inspection across sessions or correction stages.

---

| Function | Description |
|----------|-------------|
| `compute_rmse` | Compute the root-mean-square error between ground truth and estimated vectors. |
| `significance_label` | Return a significance label ("Significant" / "Not significant") based on a p-value and threshold. |
| `analyze_correction_effect_per_day_with_marker_details__` | Analyze and compare marker-wise RMSE before/after correction across multiple TRC sessions. Includes statistical tests and detailed plots per marker. |
| `analyze_correction_effect_per_day_kinematics_` | Analyze joint angle (MOT) data to evaluate kinematic correction effects across conditions. Computes RMSE and plots per-variable statistics. |
| `compare_trc_files` | Overlay marker trajectories (X, Y, Z) from multiple TRC files. Useful for visual inspection of correction results. |
| `compare_mot_files` | Plot selected kinematic variables over time from multiple MOT files for comparison. |
| `compare_markers_to_average` | Compare aligned marker positions from multiple sessions to the group average. Plots total and axis-specific errors per marker. |
| `compare_markers_by_day` | Analyze marker error distributions per day, grouped by body region. Highlights segment-wise drift or inconsistency. |

---




## `data_processing_utils.py`

This utility module provides core functionality for processing and preparing motion capture datasets. It includes tools for converting file formats (TRC ↔ C3D), transforming marker orientations, aligning static poses across sessions, and extracting key gait events. These functions are particularly useful for workflows involving OpenSim, motion analysis, and marker-based calibration procedures.

---

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


## `io_utils.py`

This module provides utility functions for managing input/output operations with TRC and MOT files, as well as organizing calibration data. It supports OpenSim-compatible formats and structured processing workflows.

---

| Function | Description |
|----------|-------------|
| `find_latest_calibration_file` | Finds the latest `calibration*.c3d` file in the given folder, based on the highest numeric suffix and ignoring files with dots. |
| `collect_calibration_files` | Iterates through a patient/session directory structure and copies the latest calibration files into a common output directory. |
| `load_trc_file` | Loads a TRC file and returns both the DataFrame of marker trajectories and the list of marker names. |
| `load_trc_file_stimuloop` | Loads a TRC file with debug print statements, intended for use with Stimuloop-style marker formats. |
| `load_mot_file` | Loads an OpenSim `.mot` file by skipping the header and extracting kinematic data and labels. |
| `save_aligned_trc` | Saves a 3D aligned marker pose array to a new TRC file using the structure from a reference TRC file. |
| `load_average_and_save_trc_bis` | Loads multiple TRC files, averages the marker data across all files, and saves the result to a new TRC file with a consistent header and marker order. |

---


## `opensim_kinematics_utils.py`

This module provides utility functions for creating, modifying, and executing OpenSim XML configurations for scaling, inverse kinematics, and point kinematics workflows. It also includes tools for marker processing and TRC reconstruction from analysis results.

---

| Function | Description |
|----------|-------------|
| `update_scaling_xml` | Updates a scaling setup XML file by replacing parameters like mass, height, marker files, and model output paths. |
| `generate_ik_setup_xml` | Creates an Inverse Kinematics XML setup file with marker and coordinate tasks, using model and marker input files. |
| `generate_pointkin_xml_from_marker_file` | Builds an AnalyzeTool XML file with `PointKinematics` analyses for each marker defined in a marker XML file. |
| `generate_trc_from_stos` | Reads multiple `.sto` files from a `PointKinematics` analysis and compiles them into a single `.trc` file. |
| `add_marker_to_opensim_file` | Adds one or more markers to an existing OpenSim marker set XML, including geometry and visualization settings. |
| `deplacer_markers` | Applies translations to specified markers in a marker XML file by adjusting their coordinate values. |
| `extract_markers_from_xml` | Parses an OpenSim marker XML file and returns a dictionary of marker names and their coordinates. |
| `scaling` | Runs the full OpenSim scaling workflow: converts static C3D to TRC, updates XML config, runs ScaleTool, and manages output. |
| `inverse_kinematics` | Executes the Inverse Kinematics tool based on motion and calibration data, producing a `.mot` file. |
| `point_kinematics` | Runs a PointKinematics analysis for specified markers and converts the result into a usable `.trc` file. |

---


## `pipeline_utils.py`

This module provides high-level orchestration utilities to simulate, displace, correct, and analyze marker sets using OpenSim pipelines. It automates batch processing of marker perturbation experiments for both real and simulated data.

---

| Function | Description |
|----------|-------------|
| `run_displacement_pipeline_opensim` | Runs a pipeline to simulate marker displacement on OpenSim-generated data (ground truth), generate noisy marker sets, perform scaling and IK, and assess results through point kinematics. |
| `run_displacement_correction_pipeline` | Averages multiple displaced static TRC files into a corrected marker file, rescales a model, and reprocesses dynamic trials using the corrected calibration for evaluation. |
| `run_displacement_pipeline` | Executes a full marker displacement pipeline using real C3D data: generates displaced marker combinations, performs scaling and IK on both real and optionally simulated data, and computes point kinematics for evaluation. |

---
