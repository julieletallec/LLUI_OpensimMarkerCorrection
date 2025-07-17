
# DETAILED FILE DESCRIPTION
---
## 📁 FINAL_USABLE_PIPELINES
### 📄 `01_translate_markerset.ipynb`
#### ✅ What This Notebook Does

This notebook simulates the use of a **different marker set** by:

1. **Scaling** an OpenSim model using a C3D calibration file and the original markers.
2. Running **Inverse Kinematics** (IK) on both calibration and dynamic trials with the original marker set.
3. Running **Point Kinematics** using a **new marker set** on the IK results.
4. Saving the resulting `.trc` files and converting them to `.c3d` files with embedded force/analog data.

> 🔄 This simulates marker placement variation or alternative setups, useful for validation, robustness testing, or sensor configuration evaluation.

#### 🎯 Goal & Outcome

- **Goal:** Generate new `.trc` and `.c3d` files representing the motion as if it had been captured using a different marker configuration.
- **Outcome:** You get transformed files with updated marker trajectories aligned to the new marker set.

#### 📂 Requirements to Run

- A C3D file for static calibration
- A C3D file for dynamic motion capture
- A model marker set XML (original configuration)
- A new marker set XML (target configuration)
- An OpenSim musculoskeletal model
- A scaling setup XML template

#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL` | Path to the OpenSim model file | `"gait2354_simbody.osim"` |
| `ORIGINAL_MARKERSET_FILE` | Path to original marker set XML | `"example_markerset_files/HMB2_MODEL_markers.xml"` |
| `NEW_MARKERSET_FILE` | Path to new marker set XML | `"example_markerset_files/HMB2_MODEL_LUCILLE_markers.xml"` |
| `C3D_CALIBRATION_FILE` | Path to static trial | `"example_c3d_files/Calibration_Mathieu.c3d"` |
| `C3D_DYNAMIC_FILE` | Path to dynamic walking trial | `"example_c3d_files/Walk_Mathieu.c3d"` |
| `TEMPLATE_SCALING_SETUP_FILE` | XML config template for scaling | `"template_scaling_setup.xml"` |
| `RESULTS_FOLDER` | Folder where outputs are saved | `"results_marker_translation"` |
| `PREFIX` | Prefix used in output filenames | `"translation"` |

### 📄 `02_add_marker.ipynb`

#### ✅ What This Notebook Does

This notebook is used to **manually add a missing marker** to a motion capture trial, using its estimated coordinates and associated body part in the OpenSim model. The steps simulate how the marker would have moved if it had been captured during the session.

1. **Scales** an OpenSim model using a calibration C3D and the marker set (without the missing marker).
2. Runs **Inverse Kinematics** to extract model motion from the incomplete C3D.
3. **Adds the missing marker** to the marker set XML with known body-relative coordinates.
4. Runs **Point Kinematics** to reconstruct the trajectory of the added marker.
5. **Generates a new `.trc` and `.c3d` file** including the synthetic marker trajectory.

> ➕ This workflow is useful when retroactively adding virtual markers to trials where some were not physically captured.

#### 🎯 Goal & Outcome

- **Goal:** Insert a missing marker into a motion trial based on model kinematics and known body-relative position.
- **Outcome:** You obtain `.trc` and `.c3d` files that include the reconstructed trajectory of the newly added marker.

#### 📂 Requirements to Run

- A C3D file for static calibration
- A C3D file missing the marker to be added
- A model marker set XML (excluding the missing marker)
- Coordinates of the missing marker in the frame of the associated body part
- Name of the OpenSim body part to which the marker belongs
- An OpenSim musculoskeletal model
- A scaling setup XML template

#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL` | Path to the OpenSim model file | `"gait2354_simbody.osim"` |
| `MARKERSET_FILE` | Path to marker set XML **excluding** the missing marker | `"example_markerset_files/HMB2_MODEL_LUCILLE_markers.xml"` |
| `MARKERS` | List of marker names (used for scaling & IK) | `[...marker names...]` |
| `C3D_CALIBRATION_FILE` | C3D for scaling | `"example_c3d_files/GCAP_S_03_TPose.c3d"` |
| `C3D_FILE_TO_ADD_MARKER` | C3D file that lacks the marker | `"example_c3d_files/GCAP_S_03_TPose.c3d"` |
| `TEMPLATE_SCALING_SETUP_FILE` | XML config template for scaling | `"template_scaling_setup.xml"` |
| `NAME` | Name of the marker to add | `"RStatic8"` |
| `COORDINATES` | 3D position of the marker in local frame | `(0.0412685, -0.0218856, 0.0594629)` |
| `BODY_PART` | OpenSim body to attach the marker to | `"toes_r"` |
| `UPDATED_MARKERSET` | Path to save the new XML with added marker | `"results_add_marker/HBM2_MODEL_markers_Lucille_updated.xml"` |
| `RESULTS_FOLDER` | Folder where outputs are saved | `"results_add_marker"` |
| `PREFIX` | Prefix used in output filenames | `"add_marker"` |


### 📄 `03_correct_marker_displacement_15_days.ipynb`
#### ✅ What This Notebook Does

This notebook performs **marker displacement correction** across 15 days of repeated calibration trials by:

1. Converting each day’s calibration `.c3d` file into a `.trc` file.
2. Averaging all 15 `.trc` files to produce a single, stable calibration marker configuration.
3. Scaling an OpenSim model using the averaged calibration file.
4. Using this corrected model to re-run **Inverse Kinematics** and **Point Kinematics** on each dynamic trial.
5. Producing corrected `.trc` and `.c3d` files for all trials.

> 🛠️ This workflow reduces the effect of day-to-day marker placement variability and improves consistency across sessions.


#### 🎯 Goal & Outcome

- **Goal:** Generate more stable and consistent kinematics by averaging multiple calibrations and applying the result to correct all dynamic trials.
- **Outcome:** You obtain 15 corrected `.trc` and `.c3d` files with more reliable marker positions.



#### 📂 Requirements to Run

- 15 daily calibration `.c3d` files (e.g., `CalibrationT1.c3d` to `CalibrationT15.c3d`)
- 15 dynamic `.c3d` files (e.g., `BaselineT1.c3d` to `BaselineT15.c3d`)
- A model marker set XML
- An OpenSim musculoskeletal model
- A scaling setup XML template



#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_FOLDER` | Folder containing `.c3d` files | `"S001"` |
| `MARKERSET_FILE` | Path to model marker set XML | `"example_markerset_files/HMB2_MODEL_markers.xml"` |
| `MARKERS` | List of markers used for IK and averaging | `[...marker names...]` |
| `TEMPLATE_SCALING_SETUP_FILE` | XML config template for scaling | `"template_scaling_setup.xml"` |
| `N_DAYS` | Number of repeated sessions (calibration + dynamic) | `15` |
| `FILES_TO_CORRECT` | Base name of dynamic trials to correct | `"Baseline"` |
| `RESULTS_FOLDER` | Folder where outputs are saved | `"results_marker_displacement_correction"` |
| `PREFIX` | Prefix used for output model and motion filenames | `"averaged"` |

---
## 📁 TEST_PIPELINES
### 📄 `01_test_caren_simulated_displacements.ipynb`

#### ✅ What This Notebook Does

This notebook **simulates and evaluates marker displacement scenarios** using motion capture data recorded with the CAREN system. It performs a **systematic validation** of the correction pipeline by:

1. **Generating Ground Truth Data**
   - Uses unaltered markers to scale a model, compute joint angles (`.mot`) and marker trajectories (`.trc`) using OpenSim.
   - Ground truth `.trc` and `.mot` files undergo the exact same processing as displaced and corrected files to ensure fair comparison.

2. **Simulating Marker Displacement**
   - Simulated displacements are **not synthetically applied** but are instead **recorded using additional physical markers** placed near the true anatomical landmarks.
   - These displaced markers are present in the original `.c3d` files and follow a naming convention like:
     - `RASIS0`, `RASIS90`, `RASIS180`, `RASIS270`
     - where the suffix indicates the **directional offset** (e.g., 90° = anterior)
   - 15 unique displacement combinations are defined and tested by selecting different versions of these displaced markers.

3. **Correction via Averaged Calibration**
   - A model is built using the average of all 15 displaced calibration trials.
   - It is then used to correct the dynamic trials and generate **corrected `.trc` and `.mot` files**.
   - These are compared against both the displaced and ground truth versions using RMSE and statistical testing.

> 🎯 This notebook is used to **quantify how displacement affects kinematic analysis**, and how effective the correction process is in recovering ground truth.


#### 🎯 Goal & Outcome

- **Goal:** Assess and visualize the effects of marker displacements on gait analysis, and evaluate how correction improves results.
- **Outcome:** 
  - 15 displaced and 15 corrected marker/joint angle files (`.trc`, `.mot`)
  - Ground truth reference files
  - RMSE & statistical analysis (marker-level and joint-level)


#### 📂 Requirements to Run

- A calibration `.c3d` file containing **both ground truth and displaced markers**
- A dynamic walking `.c3d` trial with the same marker layout
- A `.csv` gait events file aligned to the dynamic trial
- OpenSim musculoskeletal model and marker set (`.osim`, `.xml`)
- A scaling setup template (`.xml`)

#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `SUBJECT` | Participant identifier | `"Mathieu"` |
| `N_CYCLES` | Number of gait cycles to include | `10` |
| `N_COMBOS` | Number of marker displacement configurations to evaluate | `15` |
| `MARKERS_WITH_DISPLACEMENT` | Markers with multiple placed versions (displaced) | `["RASIS", "LASIS", "RPSIS", "LPSIS"]` |
| `MARKERS_WITHOUT_DISPLACEMENT` | Markers placed only once (ground truth only) | `["LLTHI", "LHEE", ..., "T10"]` |
| `COMBOS` | List of marker combinations to use for each displacement trial | `[['RASIS90', 'LASIS0', ...], ...]` |
| `KINEMATICS` | Joint angles to track and compare | `["hip_flexion_r", "pelvis_tilt", ...]` |
| `RESULTS_FOLDER` | Folder to store all outputs | `"test_caren_displacements_Mathieu"` |


#### 🔁 Adapt for New Experiments

To use this notebook with a **different experimental setup**, you only need to update a few key parameters:

- **Change which markers are considered displaced** by editing `MARKERS_WITH_DISPLACEMENT`.
- **Adjust the marker combination sets** via the `COMBOS` list.
- **Replace `.c3d` and marker set files** if working with a new subject or setup.
- Optionally, change `KINEMATICS` if evaluating different joint angles.

This makes the pipeline **fully reusable** for testing other marker placements, participant configurations, or perturbation strategies.


### 📄 `02_test_opensim_displacements.ipynb`
#### ✅ What This Notebook Does

This notebook **predicts and evaluates the impact of marker displacements** using **fully virtual simulations in OpenSim**. Unlike the CAREN-based study (see `01_test_caren_simulated_displacements.ipynb`), this pipeline simulates displacements by modifying marker definitions and using **OpenSim's PointKinematics** tool—no additional physical markers or modified `.c3d` files are required.

> 💡 This is ideal for **designing or pre-evaluating experimental setups** *before* conducting real trials. You can test how displacing specific markers affects kinematics, and estimate correction outcomes in advance.

The pipeline is organized in 4 main parts:

1. **Ground Truth Generation**
   - Derives clean `.trc` and `.mot` data from calibration and walking `.c3d` trials.
   - Uses **double PointKinematics** steps to ensure consistency and comparability with displaced/virtual data.

2. **Simulated Marker Displacements**
   - Marker displacements are simulated by modifying marker definitions in the marker set XML.
   - No physical recording of displaced markers is needed—their positions are **simulated from joint angles** using OpenSim.

3. **Correction with Averaged Calibration**
   - All displaced calibration trials are averaged to create a "neutral" model.
   - This model is then used to correct the walking trials, producing "corrected" `.trc` and `.mot` outputs.

4. **Evaluation**
   - Displaced and corrected results are compared against ground truth in terms of **marker trajectories** and **joint angles**, using RMSE.


#### 🎯 Goal & Outcome

- **Goal:** Predict how virtual displacements of anatomical markers affect motion analysis and assess the effectiveness of model-based correction.
- **Outcome:** 
  - 15 displaced and 15 corrected `.trc` and `.mot` files
  - Ground truth versions generated under comparable conditions
  - Per-marker and per-kinematic RMSE evaluations before and after correction


#### 🤖 How Displacements Are Simulated

- Markers like `"RASIS"` or `"LLEK"` are redefined with **virtual positional offsets** in the OpenSim marker set XML.
- These offsets simulate displacements of **10–20 mm** in anterior, posterior, medial, or lateral directions.
- The simulation uses:
  - **`deplacer_markers()`** to update marker definitions
  - **OpenSim's `PointKinematics`** to compute virtual marker trajectories from `.mot` files


#### 📂 Requirements to Run

- Original `.c3d` calibration and walking files containing **only ground truth markers**
- A gait event file aligned with the dynamic trial
- OpenSim musculoskeletal model and marker set (`.osim`, `.xml`)
- A scaling setup template (`.xml`)


#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `SUBJECT` | Participant identifier | `"Mathieu"` |
| `N_CYCLES` | Number of gait cycles to include | `10` |
| `N_COMBOS` | Number of marker displacement configurations to simulate | `15` |
| `MARKERS_WITH_DISPLACEMENT` | Markers to simulate displacement on | `["RASIS", "LLEK", "LMM", ...]` |
| `KINEMATICS` | Joint angles to track and compare | `["hip_flexion_r", "pelvis_tilt", ...]` |
| `RESULTS_FOLDER` | Folder to store all outputs | `"test_opensim_displacements_Mathieu"` |


#### 🔁 Adapt for New Experiments

To use this notebook for other experimental contexts or hypotheses:

- **Change the displaced markers** by updating `MARKERS_WITH_DISPLACEMENT`.
- **Modify displacement logic** inside `deplacer_markers()` to simulate different magnitudes or directions.
- **Adjust the number of combos or gait cycles** to match your validation needs.

> 🛠 This pipeline is fully modular and allows easy adaptation for future pilot testing or protocol development.


### 📄 `03_test_quantification_of_model_displacements.ipynb`

#### ✅ What This Notebook Does

This notebook evaluates **multi-day calibration consistency** across several StimuLOOP patients by comparing the marker positions obtained from daily static trials with those from a **single averaged (corrected) model**.

The calibration `.c3d` files from 16 training days are converted to `.trc`, aligned, and then averaged to generate a representative, corrected calibration. The **marker displacement error** between the daily calibrations and this averaged model is then analyzed.

> 📊 This analysis is useful to understand how much calibration variation occurs in real-world, longitudinal data collection, and whether a single averaged model might be a more robust reference.


#### 🎯 Goal & Outcome

- **Goal:** Quantify marker position variability across 16 sessions and evaluate the feasibility of replacing per-day scaling with a single averaged calibration.
- **Outcome:** For each patient:
  - One `.trc` file representing the averaged static calibration
  - 16 individually scaled calibration files
  - Marker-wise and day-wise displacement plots against the averaged model


#### 🔬 What’s Being Analyzed

- For each patient:
  - 16 daily `.c3d` calibration files (T0–T15)
  - Converted to `.trc`, then **aligned segment-wise** (based on each anatomical segment’s local marker CoM)
  - Averaged into a **single `.trc` file**
  - Used to scale a "corrected" OpenSim model
  - Marker distances between each individual day and the averaged model are computed and plotted


#### 📂 Requirements to Run

- Folder structure like: `stimuloop patient data/S001/CalibrationT0.c3d` ... `CalibrationT15.c3d`
- 16 static calibration trials per patient
- OpenSim model and marker set
- A scaling setup XML template


#### 🔧 Changeable Parameters

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_FOLDER` | Location of StimuLOOP patient `.c3d` data | `"stimuloop patient data"` |
| `RESULTS_FOLDER` | Where all outputs and plots are saved | `"stimuloop_displacement_error_results"` |
| `N_DAYS` | Number of calibration days to process | `16` |
| `MARKERS` | Marker list used for analysis and alignment | `[“LASIS”, “RASIS”, ...]` |
| `PATIENT_ID` | ID of patient for single-patient evaluation | `"S001"` |


#### 👨‍⚕️ Two Modes of Operation

1. **Batch Mode (All Patients):**
   - Automatically processes every patient folder in `DATA_FOLDER`
   - Skips any with missing `.c3d` files

2. **Single Patient Mode:**
   - Manually specify `PATIENT_ID`
   - Useful for debugging or visualizing a single case


#### 📈 What You Get

- 16 aligned `.trc` calibration files per patient
- 1 averaged `.trc` calibration file per patient
- 1 scaled model based on the average
- 📊 Two diagnostic plots:
  - **Per-marker RMSE across days** (boxplot per marker)
  - **Per-day RMSE across markers** (boxplot per day)


#### 🔁 Adapt for New Studies

You can easily adapt this pipeline for:

- A different number of calibration sessions (e.g., `N_DAYS = 5`)
- Different marker sets (`MARKERS`)
- Custom alignment strategies (`non_rigid=True`, `recenter_strategy="pelvis"`)

> 🧪 This pipeline is ideal for analyzing inter-session consistency or designing protocols where per-day recalibration is to be minimized.


---
## 📁 example_c3d_files
---
## 📁 example_gaitevents_files
---
## 📁 example_marker_files

This folder contains XML files that define marker sets for use in OpenSim simulations.

#### 🧩 What is a marker set?
A marker set in OpenSim specifies:
- The **name** of each marker.
- The **body segment** it is attached to (e.g., `pelvis`, `femur_r`, etc.).
- The **location** (3D coordinates in meters) of the marker relative to its segment.
- Additional visual and simulation parameters (e.g., visibility, fixed status).

Each `<Marker>` in the XML file looks like this:

```xml
<Marker name="LASIS">
    <body>pelvis</body>
    <location>0.02 0.03 -0.128</location>
    <fixed>true</fixed>
    ...
</Marker>
```



---
## 📁 example_model_files
---
## 📁 example_osim_setup_files
---
## 📁 example_model_files
---
## 📁 utils
### 📄 `analysis_utils.py`

This module provides a suite of analysis and visualization tools for evaluating the **effectiveness of motion data correction**, focusing on **marker trajectories (TRC)** and **kinematic signals (MOT)**.  
It enables statistical comparison, RMSE evaluation, and visual inspection across sessions or correction stages.


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





### 📄 `data_processing_utils.py`

This utility module provides core functionality for processing and preparing motion capture datasets. It includes tools for converting file formats (TRC ↔ C3D), transforming marker orientations, aligning static poses across sessions, and extracting key gait events. These functions are particularly useful for workflows involving OpenSim, motion analysis, and marker-based calibration procedures.


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



### 📄 `io_utils.py`

This module provides utility functions for managing input/output operations with TRC and MOT files, as well as organizing calibration data. It supports OpenSim-compatible formats and structured processing workflows.


| Function | Description |
|----------|-------------|
| `find_latest_calibration_file` | Finds the latest `calibration*.c3d` file in the given folder, based on the highest numeric suffix and ignoring files with dots. |
| `collect_calibration_files` | Iterates through a patient/session directory structure and copies the latest calibration files into a common output directory. |
| `load_trc_file` | Loads a TRC file and returns both the DataFrame of marker trajectories and the list of marker names. |
| `load_trc_file_stimuloop` | Loads a TRC file with debug print statements, intended for use with Stimuloop-style marker formats. |
| `load_mot_file` | Loads an OpenSim `.mot` file by skipping the header and extracting kinematic data and labels. |
| `save_aligned_trc` | Saves a 3D aligned marker pose array to a new TRC file using the structure from a reference TRC file. |
| `load_average_and_save_trc_bis` | Loads multiple TRC files, averages the marker data across all files, and saves the result to a new TRC file with a consistent header and marker order. |



### 📄 `opensim_kinematics_utils.py`

This module provides utility functions for creating, modifying, and executing OpenSim XML configurations for scaling, inverse kinematics, and point kinematics workflows. It also includes tools for marker processing and TRC reconstruction from analysis results.


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



### 📄 `pipeline_utils.py`

This module provides high-level orchestration utilities to simulate, displace, correct, and analyze marker sets using OpenSim pipelines. It automates batch processing of marker perturbation experiments for both real and simulated data.


| Function | Description |
|----------|-------------|
| `run_displacement_pipeline_opensim` | Runs a pipeline to simulate marker displacement on OpenSim-generated data (ground truth), generate noisy marker sets, perform scaling and IK, and assess results through point kinematics. |
| `run_displacement_correction_pipeline` | Averages multiple displaced static TRC files into a corrected marker file, rescales a model, and reprocesses dynamic trials using the corrected calibration for evaluation. |
| `run_displacement_pipeline` | Executes a full marker displacement pipeline using real C3D data: generates displaced marker combinations, performs scaling and IK on both real and optionally simulated data, and computes point kinematics for evaluation. |

---
