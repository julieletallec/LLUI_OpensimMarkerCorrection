# analysis_utils.py

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
