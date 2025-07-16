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

from opensim_kinematics_utils import scaling, inverse_kinematics, point_kinematics, deplacer_markers
from io_utils import extract_gait_cycles, load_average_and_save_trc_bis
from data_processing_utils import generate_random_marker_combinations, convert_markers, convert_c3d_to_trc






def run_displacement_pipeline_opensim (default_markers_list: List[str], markers_to_displace: List[str], base_markerset_xml_file, template_scaling_setup_xml_file, n_combos, mot_groundtruth_calib_file, mot_groundtruth_walking_file, results_folder, gait_event_file=None, n_cycles=None, start_time = None, end_time = None, combos=None):
    """
    Runs a full OpenSim displacement pipeline simulating marker misplacement.

    For each marker combination, the pipeline:
    - Displaces specified markers according to defined or randomly generated orientations.
    - Computes static and dynamic point kinematics using the ground-truth model.
    - Performs scaling with displaced marker sets.
    - Runs inverse kinematics on both original and displaced marker sets.
    - Computes point kinematics post-IK to evaluate the impact of displacement.

    Args:
        default_markers_list (List[str]): List of fixed, non-displaced markers.
        markers_to_displace (List[str]): List of markers to simulate misplacement for.
        base_markerset_xml_file (str): Path to the original marker set XML file.
        template_scaling_setup_xml_file (str): Path to the XML template for scaling setup.
        n_combos (int): Number of random marker combinations to generate if `combos` is not provided.
        mot_groundtruth_calib_file (str): Path to the ground-truth static `.mot` file.
        mot_groundtruth_walking_file (str): Path to the ground-truth dynamic `.mot` file.
        results_folder (str): Directory to save intermediate and final results.
        gait_event_file (str, optional): Path to gait cycle event CSV file.
        n_cycles (int, optional): Number of gait cycles to use for time bounds.
        start_time (float, optional): Start time for dynamic trials (overridden by `gait_event_file` if provided).
        end_time (float, optional): End time for dynamic trials (overridden by `gait_event_file` if provided).
        combos (List[List[str]], optional): Predefined marker orientation combinations.

    Returns:
        List[List[str]]: The list of marker combinations used during the pipeline.
    """
    all_markers = default_markers_list + markers_to_displace

    gait_events_df = extract_gait_cycles(gait_event_file)
    gait_events_df = gait_events_df[:n_cycles]
    if (start_time, end_time) == (None, None):
        start_time = gait_events_df["Time"].iloc[0]
        end_time = gait_events_df["Time"].iloc[-1]


    if combos == None:
        combos = generate_random_marker_combinations(markers_to_displace, n_combos)

    print(combos)
    
    for i, combo in enumerate (combos):

        combo_marker_set_xml_file = results_folder+f"\combo{i}_markers.xml"
        deplacer_markers(base_markerset_xml_file, combo_marker_set_xml_file, convert_markers(combo))


        point_kinematics (f"combo{i}_calib",
                      "static",
                      combo_marker_set_xml_file,
                      results_folder,
                      os.path.join(results_folder, "groundtruth_scaled_model_markers.osim"),
                      mot_groundtruth_calib_file,
                      0,
                      1)
        point_kinematics (f"combo{i}_walking",
                      "dynamic",
                      combo_marker_set_xml_file,
                      results_folder,
                      os.path.join(results_folder, "groundtruth_scaled_model_markers.osim"),
                      mot_groundtruth_walking_file,
                      gait_event_file=gait_event_file,
                      n_cycles=n_cycles)
        
        scaling (f"combo{i}",
                    results_folder,
                    all_markers,
                    template_scaling_setup_xml_file,
                    base_markerset_xml_file,
                    trc_calib_file = os.path.join(results_folder, f"combo{i}_calib_static.trc"))
        
        inverse_kinematics(f"combo{i}_walking",
                    "dynamic",
                   all_markers,
                   combo_marker_set_xml_file,
                   os.path.join(results_folder, f"combo{i}_scaled_model_markers.osim"),
                   results_folder,
                   n_cycles=n_cycles,
                   gait_event_file = gait_event_file,
                   trc_dynamic_file=os.path.join(results_folder, f"combo{i}_walking_dynamic.trc"))
        
        point_kinematics(f"point_kin_combo{i}_walking",
                      "dynamic",
                      base_markerset_xml_file,
                      results_folder,
                      os.path.join(results_folder, f"combo{i}_scaled_model_markers.osim"),
                      os.path.join(results_folder, f"combo{i}_walking_dynamic_motion.mot"),
                      gait_event_file=gait_event_file,
                      n_cycles=n_cycles)
        
        inverse_kinematics(f"point_kin_combo{i}_walking",
                    "dynamic",
                   all_markers,
                   combo_marker_set_xml_file,
                   os.path.join(results_folder, f"combo{i}_scaled_model_markers.osim"),
                   results_folder,
                   n_cycles=n_cycles,
                   gait_event_file = gait_event_file,
                   trc_dynamic_file=os.path.join(results_folder, f"point_kin_combo{i}_walking_dynamic.trc"))
    
    return combos






def run_displacement_correction_pipeline (base_markerset_xml_file, gait_event_file, n_cycles, n_combos, results_folder, all_markers: List[str], gt_calib_trc_file, template_scaling_setup_xml_file):
    """
    Runs a correction pipeline to mitigate marker displacement effects by averaging across multiple displaced calibrations.

    This pipeline:
    - Aggregates multiple displaced static calibrations to compute an averaged marker configuration.
    - Performs scaling using the averaged static pose.
    - Runs inverse kinematics and point kinematics on each displacement case using the corrected (averaged) model.
    - Re-evaluates the corrected pipeline output for dynamic trials.

    Args:
        base_markerset_xml_file (str): Path to the original (non-displaced) marker set XML file.
        gait_event_file (str): Path to the CSV file containing gait cycle event timestamps.
        n_cycles (int): Number of gait cycles to extract from the gait event file.
        n_combos (int): Number of marker displacement variations to process.
        results_folder (str): Directory where all intermediate and final output files are stored.
        all_markers (List[str]): List of all marker names (fixed + displaced).
        gt_calib_trc_file (str): Path to the reference static TRC file (used for header and marker ordering).
        template_scaling_setup_xml_file (str): Path to the XML scaling setup template file.

    Returns:
        None
    """
    gait_events_df = extract_gait_cycles(gait_event_file)
    gait_events_df = gait_events_df[:n_cycles]
    start_time = gait_events_df["Time"].iloc[0]
    end_time = gait_events_df["Time"].iloc[-1]

    prefixes = tuple(f"combo{i}_calib_static" for i in range(n_combos))
    corrected_static_trc_file = os.path.join(results_folder, "average_calib_static.trc")

    load_average_and_save_trc_bis(
            folder_path=results_folder,
            reference_trc_path=gt_calib_trc_file,
            output_path=corrected_static_trc_file,
            reference_marker_order=all_markers,
            startswith=prefixes,
            opensim=True)
            
    scaling("average",
            results_folder,
            all_markers,
            template_scaling_setup_xml_file,
            base_markerset_xml_file,
            trc_calib_file = corrected_static_trc_file)

    for i in range (n_combos):

        inverse_kinematics(f"average_combo{i}_walking",
                   "dynamic",
                   all_markers,
                   base_markerset_xml_file,
                   os.path.join(results_folder, "average_scaled_model_markers.osim"),
                   results_folder,
                   trc_dynamic_file=os.path.join(results_folder, f"combo{i}_walking_dynamic.trc"),
                   dyn_initial_time = start_time,
                   dyn_final_time = end_time)

        point_kinematics (f"point_kin_average_combo{i}_walking",
                      "dynamic",
                      base_markerset_xml_file,
                      results_folder,
                      os.path.join(results_folder, "average_scaled_model_markers.osim"),
                      os.path.join(results_folder, f"average_combo{i}_walking_dynamic_motion.mot"),
                      gait_event_file=gait_event_file,
                      n_cycles=n_cycles)

        inverse_kinematics(f"point_kin_average_combo{i}_walking",
                    "dynamic",
                    all_markers,
                    base_markerset_xml_file,
                    os.path.join(results_folder, "average_scaled_model_markers.osim"),
                    results_folder,
                    trc_dynamic_file=os.path.join(results_folder, f"point_kin_average_combo{i}_walking_dynamic.trc"),
                    dyn_initial_time = start_time,
                    dyn_final_time = end_time)
        





def run_displacement_pipeline (default_markers_list: List[str], markers_to_displace: List[str], base_markerset_xml_file, template_scaling_setup_xml_file, n_combos, c3d_groundtruth_calib_file, c3d_groundtruth_walking_file, results_folder, gait_event_file=None, n_cycles=None, start_time = None, end_time = None, combos=None, opensim_simulations=False):
    """
    Executes a marker displacement analysis pipeline using real C3D data to simulate marker noise and assess its impact.

    This function:
    - Applies random or predefined marker displacements to simulate noise.
    - Converts corresponding static and dynamic C3D trials to TRC format using displaced markers.
    - Updates marker sets accordingly.
    - Runs scaling, inverse kinematics, and point kinematics on the displaced marker sets.
    - Optionally compares with OpenSim-simulated marker sets for consistency and validation.

    Args:
        default_markers_list (List[str]): List of original markers not subject to displacement.
        markers_to_displace (List[str]): Subset of markers that will be randomly displaced.
        base_markerset_xml_file (str): Path to the original marker set XML.
        template_scaling_setup_xml_file (str): XML setup file used as a base for scaling operations.
        n_combos (int): Number of random displacement combinations to generate.
        c3d_groundtruth_calib_file (str): Path to the ground truth static C3D file.
        c3d_groundtruth_walking_file (str): Path to the ground truth dynamic C3D file.
        results_folder (str): Directory where all output files will be saved.
        gait_event_file (str, optional): CSV file used to extract gait cycle timings.
        n_cycles (int, optional): Number of gait cycles to extract from event file.
        start_time (float, optional): Optional fixed start time for dynamic trial.
        end_time (float, optional): Optional fixed end time for dynamic trial.
        combos (List[List[str]], optional): If provided, overrides random displacement with these marker combinations.
        opensim_simulations (bool, optional): If True, reprocesses additional OpenSim marker simulations using the displaced configurations.

    Returns:
        List[List[str]]: List of marker displacement combinations used during the pipeline.
    """
    all_markers = default_markers_list + markers_to_displace

    gait_events_df = extract_gait_cycles(gait_event_file)
    gait_events_df = gait_events_df[:n_cycles]
    if (start_time, end_time) == (None, None):
        start_time = gait_events_df["Time"].iloc[0]
        end_time = gait_events_df["Time"].iloc[-1]


    if combos == None:
        combos = generate_random_marker_combinations(markers_to_displace, n_combos)


    for i, combo in enumerate (combos):

        all_markers_with_displacements = default_markers_list + combo
        static_trc = os.path.join(results_folder, f"real_combo{i}_calib_static.trc")
        dynamic_trc = os.path.join(results_folder, f"real_combo{i}_walking_dynamic.trc")
        convert_c3d_to_trc(c3d_groundtruth_calib_file, output_trc=static_trc, start_time=0.2, end_time=1, include_markers=all_markers_with_displacements, markers_to_clean=markers_to_displace)
        convert_c3d_to_trc(c3d_groundtruth_walking_file, output_trc=dynamic_trc, start_time=start_time, end_time=end_time, include_markers=all_markers_with_displacements, markers_to_clean=markers_to_displace)

        combo_marker_set_xml_file = results_folder+f"\combo{i}_markers.xml"
        deplacer_markers(base_markerset_xml_file, combo_marker_set_xml_file, convert_markers(combo))

        scaling(f"real_combo{i}",
            results_folder,
            all_markers,
            template_scaling_setup_xml_file,
            base_markerset_xml_file,
            trc_calib_file = os.path.join(results_folder, f"real_combo{i}_calib_static.trc")
            )
        
        inverse_kinematics(f"real_combo{i}_walking",
                        "dynamic",
                    all_markers,
                    base_markerset_xml_file,
                    os.path.join(results_folder, f"real_combo{i}_scaled_model_markers.osim"),
                    results_folder,
                    n_cycles=n_cycles,
                    gait_event_file = gait_event_file,
                    trc_dynamic_file=os.path.join(results_folder, f"real_combo{i}_walking_dynamic.trc"))

        point_kinematics (f"point_kin_real_combo{i}_walking",
                        "dynamic",
                        base_markerset_xml_file,
                        results_folder,
                        os.path.join(results_folder, f"real_combo{i}_scaled_model_markers.osim"),
                        os.path.join(results_folder, f"real_combo{i}_walking_dynamic_motion.mot"),
                        gait_event_file=gait_event_file,
                        n_cycles=n_cycles)
        
        inverse_kinematics(f"point_kin_combo{i}_walking",
                    "dynamic",
                    all_markers,
                    base_markerset_xml_file,
                    os.path.join(results_folder, f"real_combo{i}_scaled_model_markers.osim"),
                    results_folder,
                    trc_dynamic_file=os.path.join(results_folder, f"point_kin_real_combo{i}_walking_dynamic.trc"),
                    dyn_initial_time = start_time,
                    dyn_final_time = end_time)



        if opensim_simulations == True:

            combo_marker_set_xml_file = results_folder+f"\combo{i}_markers.xml"
            deplacer_markers(base_markerset_xml_file, combo_marker_set_xml_file, convert_markers(combo))

            point_kinematics (f"combo{i}_calib",
                      "static",
                      combo_marker_set_xml_file,
                      results_folder,
                      os.path.join(results_folder, "groundtruth_scaled_model_markers.osim"),
                      os.path.join(results_folder, "groundtruth_calib_static_motion.mot"),
                      0,
                      1)
            point_kinematics (f"combo{i}_walking",
                      "dynamic",
                      combo_marker_set_xml_file,
                      results_folder,
                      os.path.join(results_folder, "groundtruth_scaled_model_markers.osim"),
                      os.path.join(results_folder, "groundtruth_walking_dynamic_motion.mot"),
                      gait_event_file=gait_event_file,
                      n_cycles=n_cycles)
            
            scaling(f"combo{i}",
                    results_folder,
                    all_markers,
                    template_scaling_setup_xml_file,
                    base_markerset_xml_file,
                    trc_calib_file = os.path.join(results_folder, f"combo{i}_calib_static.trc"))


            inverse_kinematics(f"combo{i}_walking",
                    "dynamic",
                   all_markers,
                   combo_marker_set_xml_file,
                   os.path.join(results_folder, f"combo{i}_scaled_model_markers.osim"),
                   results_folder,
                   n_cycles=n_cycles,
                   gait_event_file = gait_event_file,
                   trc_dynamic_file=os.path.join(results_folder, f"combo{i}_walking_dynamic.trc"))
            
    return combos
