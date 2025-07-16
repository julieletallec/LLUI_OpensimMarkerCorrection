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





def update_scaling_xml(
    input_path,
    output_path,
    marker_set_file=None,
    marker_file=None,
    model_scaler_output_model_file=None,
    marker_placer_output_model_file=None,
    output_scale_file=None,
    mass=None,
    height=None
):
    """
    Update specific fields in an OpenSim scaling XML configuration file.

    This function modifies an existing XML file used for scaling a musculoskeletal model in OpenSim.
    It updates fields such as mass, height, marker set paths, and output model or scale files,
    and saves the updated XML to a new location.

    Args:
        input_path (str): Path to the input XML scaling file.
        output_path (str): Path to save the modified XML file.
        marker_set_file (str, optional): Path to a marker set XML file.
        marker_file (str, optional): Path to the TRC file containing motion capture markers.
        model_scaler_output_model_file (str, optional): Path for the output model from ModelScaler.
        marker_placer_output_model_file (str, optional): Path for the output model from MarkerPlacer.
        output_scale_file (str, optional): Path for the output scaling file.
        mass (float, optional): Subject mass (in kg) to update in the file.
        height (float, optional): Subject height (in meters) to update in the file.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    ns = {"": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    
    def find_and_set(tag_name, new_value, parent=root):
        for elem in parent.iter(tag_name):
            elem.text = new_value
    if mass is not None:
        find_and_set("mass", str(mass))
    if height is not None:
        find_and_set("height", str(height))
    if marker_set_file:
        find_and_set("marker_set_file", marker_set_file)
    if marker_file:
        find_and_set("marker_file", marker_file)
    if model_scaler_output_model_file:
        model_scaler = root.find(".//ModelScaler", ns)
        find_and_set("output_model_file", model_scaler_output_model_file, model_scaler)
    if output_scale_file:
        find_and_set("output_scale_file", output_scale_file)
    if marker_placer_output_model_file:
        marker_placer = root.find(".//MarkerPlacer", ns)
        find_and_set("output_model_file", marker_placer_output_model_file, marker_placer)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)



def generate_ik_setup_xml(
    model_file,
    marker_file,
    output_motion_file,
    markerset_file,
    coordinate_file="",
    time_range=(0.4, 1.6),
    constraint_weight=20,
    accuracy=1e-5,
    default_marker_weight=1.0,
    coordinate_tasks=None,
    output_path="Setup_IK.xml"
):
    """
    Generates an OpenSim Inverse Kinematics (IK) setup XML file.

    This function creates a complete IK XML configuration file, including marker and coordinate tracking tasks,
    for use with the OpenSim InverseKinematicsTool.

    Args:
        model_file (str): Path to the OpenSim model file (.osim).
        marker_file (str): Path to the input TRC marker file.
        output_motion_file (str): Path where the resulting .mot motion file will be saved.
        markerset_file (str): Path to the marker set XML to retrieve marker names and build IKMarkerTasks.
        coordinate_file (str, optional): Path to a coordinate file for tracking (default is empty).
        time_range (tuple): Tuple of start and end time (in seconds) for the IK analysis.
        constraint_weight (float): Weight for satisfying model constraints.
        accuracy (float): Numerical accuracy threshold for IK solver convergence.
        default_marker_weight (float): Default weight applied to all marker tasks.
        coordinate_tasks (list of dict, optional): List of coordinate tracking tasks with keys: name, apply, weight, value_type, value.
        output_path (str): Output path where the XML file will be saved.

    Side Effects:
        Writes a fully formatted IK setup XML file to the specified `output_path`.
    """
    coordinate_tasks = coordinate_tasks or []

    def elem(tag, text=None):
        e = ET.Element(tag)
        e.text = str(text).lower() if isinstance(text, bool) else str(text)
        return e

    # Créer le document de base
    root = ET.Element("OpenSimDocument", Version="30000")
    tool = ET.SubElement(root, "InverseKinematicsTool", name="subject01")

    tool.append(elem("model_file", model_file))
    tool.append(elem("constraint_weight", constraint_weight))
    tool.append(elem("accuracy", accuracy))

    # Début du bloc IKTaskSet
    ik_task_set = ET.SubElement(tool, "IKTaskSet", name="ik_tasks")
    objects = ET.SubElement(ik_task_set, "objects")

    # Parse du fichier MarkerSet
    marker_tree = ET.parse(markerset_file)
    marker_root = marker_tree.getroot()
    for marker in marker_root.findall(".//Marker"):
        name = marker.attrib.get("name")
        if name:
            task_elem = ET.SubElement(objects, "IKMarkerTask", name=name)
            task_elem.append(elem("apply", True))
            task_elem.append(elem("weight", default_marker_weight))

    # Ajout des coordinate tasks (facultatif)
    for coord in coordinate_tasks:
        print(coord)
        task_elem = ET.SubElement(objects, "IKCoordinateTask", name=coord["name"])
        task_elem.append(elem("apply", coord.get("apply", True)))
        task_elem.append(elem("weight", coord.get("weight", 1)))
        task_elem.append(elem("value_type", coord.get("value_type", "default_value")))
        task_elem.append(elem("value", coord.get("value", 0)))

    ET.SubElement(ik_task_set, "groups")

    # Fichiers et paramètres
    tool.append(elem("marker_file", marker_file))
    tool.append(elem("coordinate_file", coordinate_file))
    tool.append(elem("time_range", f"{time_range[0]} {time_range[1]}"))
    tool.append(elem("output_motion_file", output_motion_file))

    # Mise en forme propre avec minidom
    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)








def generate_pointkin_xml_from_marker_file(marker_file_path, output_path,
                                           model_file="gait2354_simbody_scaled.osim",
                                           results_directory="PointKinematics_results",
                                           initial_time=0, final_time=199.5):
    """
    Generates an OpenSim AnalyzeTool XML configuration for point kinematics based on a marker set file.

    This function creates an XML file that configures the PointKinematics analysis for each marker defined in
    the provided marker XML file. Each marker is associated with a body and location and will be tracked over time.

    Args:
        marker_file_path (str): Path to the OpenSim marker set (.xml) file.
        output_path (str): Path where the output XML configuration will be saved.
        model_file (str): Path to the OpenSim model file (.osim). Default is "gait2354_simbody_scaled.osim".
        results_directory (str): Directory where OpenSim should store analysis results.
        initial_time (float): Start time of the analysis.
        final_time (float): End time of the analysis.

    Side Effects:
        Writes a fully formatted XML AnalyzeTool file for point kinematics to `output_path`.
    """
    def create_element_with_text(tag, text):
        el = ET.Element(tag)
        el.text = str(text)
        return el

    # Parse marker XML
    tree = ET.parse(marker_file_path)
    root = tree.getroot()
    markers = root.findall(".//Marker")

    # Create XML structure
    opensim_root = ET.Element("OpenSimDocument", Version="40500")
    tool = ET.SubElement(opensim_root, "AnalyzeTool", name="3DGaitModel2354-scaled")

    # Header
    tool.append(create_element_with_text("model_file", model_file))
    tool.append(create_element_with_text("replace_force_set", "false"))
    tool.append(create_element_with_text("force_set_files", ""))
    tool.append(create_element_with_text("results_directory", results_directory))
    tool.append(create_element_with_text("output_precision", "8"))
    tool.append(create_element_with_text("initial_time", str(initial_time)))
    tool.append(create_element_with_text("final_time", str(final_time)))
    tool.append(create_element_with_text("solve_for_equilibrium_for_auxiliary_states", "false"))
    tool.append(create_element_with_text("maximum_number_of_integrator_steps", "20000"))
    tool.append(create_element_with_text("maximum_integrator_step_size", "1"))
    tool.append(create_element_with_text("minimum_integrator_step_size", "1e-08"))
    tool.append(create_element_with_text("integrator_error_tolerance", "1e-05"))

    # AnalysisSet
    analysis_set = ET.SubElement(tool, "AnalysisSet", name="Analyses")
    objects = ET.SubElement(analysis_set, "objects")

    for m in markers:
        name = m.attrib["name"]
        body = m.find("body").text
        location = m.find("location").text.strip()

        k = ET.SubElement(objects, "PointKinematics", name="PointKinematics")
        k.append(create_element_with_text("on", "true"))
        k.append(create_element_with_text("start_time", str(initial_time)))
        k.append(create_element_with_text("end_time", str(final_time)))
        k.append(create_element_with_text("step_interval", "1"))
        k.append(create_element_with_text("in_degrees", "true"))
        k.append(create_element_with_text("body_name", body))
        k.append(create_element_with_text("relative_to_body_name", "none"))
        k.append(create_element_with_text("point_name", name))
        k.append(create_element_with_text("point", location))

    ET.SubElement(analysis_set, "groups")

    # ControllerSet and other files
    controller_set = ET.SubElement(tool, "ControllerSet", name="Controllers")
    ET.SubElement(controller_set, "objects")
    ET.SubElement(controller_set, "groups")
    tool.append(create_element_with_text("external_loads_file", ""))
    tool.append(create_element_with_text("states_file", ""))
    tool.append(create_element_with_text("coordinates_file", ""))
    tool.append(create_element_with_text("speeds_file", ""))
    tool.append(create_element_with_text("lowpass_cutoff_frequency_for_coordinates", "-1"))

    # Beautify & save
    rough_string = ET.tostring(opensim_root, 'utf-8')
    pretty_xml = minidom.parseString(rough_string).toprettyxml(indent="	")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)







def generate_trc_from_stos(marker_dict, sto_folder, output_trc_path, data_rate=100, units="m"):
    """
    Generates a TRC file from a set of OpenSim PointKinematics .sto files.

    This function reads multiple .sto files (each representing 3D point trajectories for a single marker)
    and compiles them into a single .trc file compatible with OpenSim, preserving time information.

    Args:
        marker_dict (dict): Dictionary of marker names to any values (the values are unused).
                            Only keys (marker names) are used to locate .sto files.
        sto_folder (str): Path to the folder containing the .sto files.
        output_trc_path (str): Destination path for the generated .trc file.
        data_rate (float, optional): Sampling frequency in Hz. Defaults to 100.
        units (str, optional): Unit of measurement for marker positions. Defaults to "m".

    Side Effects:
        Writes a formatted .trc file to `output_trc_path`.

    Notes:
        Expects .sto files to follow the naming convention:
        `3DGaitModel2354-scaled_PointKinematics_<marker>_pos.sto`.
    """
    sample_marker = next(iter(marker_dict))
    sample_path = os.path.join(sto_folder, f"3DGaitModel2354-scaled_PointKinematics_{sample_marker}_pos.sto")
    df_sample = pd.read_csv(sample_path, sep='\t', skiprows=7)
    n_frames = len(df_sample)
    time = df_sample['time']
    df_out = pd.DataFrame()
    df_out['Frame#'] = range(1, n_frames + 1)
    df_out['Time'] = time
    marker_names = []
    for marker in marker_dict:
        sto_file = os.path.join(sto_folder, f"3DGaitModel2354-scaled_PointKinematics_{marker}_pos.sto")
        if not os.path.exists(sto_file):
            print(f"Skipping missing marker file: {marker}")
            continue
        df = pd.read_csv(sto_file, sep='\t', skiprows=7)
        df_out[f"{marker}_X"] = df['state_0']
        df_out[f"{marker}_Y"] = df['state_1']
        df_out[f"{marker}_Z"] = df['state_2']
        marker_names.append(marker)
    # Write TRC file
    with open(output_trc_path, 'w') as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tgenerated_output.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate:.2f}\t{data_rate:.2f}\t{n_frames}\t{len(marker_names)}\t{units}\t{data_rate:.2f}\t1\t{n_frames}\n")
        f.write("Frame#\tTime\t")
        for marker in marker_names:
            f.write(f"{marker}\t\t\t")
        f.write("\n")

        f.write("\t\t")
        for i in range(1, len(marker_names)+1):
            f.write(f"X{i}\tY{i}\tZ{i}\t")
        f.write("\n")
        for i in range(n_frames):
            row = f"{df_out.at[i, 'Frame#']}\t{df_out.at[i, 'Time']:.5f}\t"
            for marker in marker_names:
                row += (
                    f"{df_out.at[i, f'{marker}_X']:.5f}\t"
                    f"{df_out.at[i, f'{marker}_Y']:.5f}\t"
                    f"{df_out.at[i, f'{marker}_Z']:.5f}\t"
                )
            f.write(row.strip() + "\n")
    print(f"TRC file saved: {output_trc_path}")


def add_marker_to_opensim_file(file_path, marker_name, coordinates, body_part, output_file=None):
    """
    Adds one or multiple markers to an OpenSim marker set (.xml) file.

    This function appends new marker definitions (with associated coordinates and body segments)
    to an existing OpenSim marker set XML file, and writes the updated file with proper formatting.

    Args:
        file_path (str): Path to the original marker .xml file.
        marker_name (str or list of str): Name(s) of the marker(s) to add.
        coordinates (tuple or list of tuples): (x, y, z) coordinate(s) for each marker.
        body_part (str or list of str): Name(s) of the body segment each marker is attached to.
        output_file (str, optional): Path to save the updated file. If None, the original file is overwritten.

    Raises:
        ValueError: If input lengths are mismatched or required XML sections are missing.

    Side Effects:
        Writes the updated XML with new markers to `output_file` or overwrites `file_path`.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    marker_set = root.find(".//MarkerSet/objects")
    if marker_set is None:
        raise ValueError("No MarkerSet/objects section found in XML.")

    # Normalize input to lists
    if isinstance(marker_name, str):
        marker_name = [marker_name]
    if isinstance(coordinates, (tuple, list)) and not isinstance(coordinates[0], (tuple, list)):
        coordinates = [coordinates]
    if isinstance(body_part, str):
        body_part = [body_part]

    if not (len(marker_name) == len(coordinates) == len(body_part)):
        raise ValueError("marker_name, coordinates, and body_part must have the same length.")

    for name, coord, part in zip(marker_name, coordinates, body_part):
        # Create new marker element
        marker = ET.Element('Marker', name=name)

        ET.SubElement(marker, 'body').text = part
        ET.SubElement(marker, 'location').text = f"{coord[0]} {coord[1]} {coord[2]}"
        ET.SubElement(marker, 'fixed').text = 'true'

        visible_object = ET.SubElement(marker, 'VisibleObject', name="")
        geometry_set = ET.SubElement(visible_object, 'GeometrySet', name="")
        ET.SubElement(geometry_set, 'objects')
        ET.SubElement(geometry_set, 'groups')
        ET.SubElement(visible_object, 'scale_factors').text = '1.00000000 1.00000000 1.00000000'
        ET.SubElement(visible_object, 'transform').text = '0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000'
        ET.SubElement(visible_object, 'show_axes').text = 'false'
        ET.SubElement(visible_object, 'display_preference').text = '4'

        # Append new marker
        marker_set.append(marker)

    # Generate a pretty XML string
    xml_string = ET.tostring(root, encoding='utf-8')
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="    ")

    # Remove empty lines (common issue with toprettyxml)
    pretty_xml = "\n".join([line for line in pretty_xml.split('\n') if line.strip()])

    # Save the result
    save_path = output_file if output_file else file_path
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"{len(marker_name)} marker(s) added and saved to '{save_path}' with proper formatting.")





def deplacer_markers(input_marker_file, output_marker_file, deplacements):
    """
    Moves specified markers in an OpenSim XML marker file by updating their coordinates.

    Args:
        input_marker_file (str): Path to the input XML file containing markers.
        output_marker_file (str): Path to the output XML file with updated markers.
        deplacements (dict): Dictionary of displacements to apply,
                             formatted as {marker_name: (dx, dy, dz)}.

    Side effects:
        Writes a new XML file with updated marker coordinates.
    """
    tree = ET.parse(input_marker_file)
    root = tree.getroot()

    # Parcourir tous les markers dans le fichier
    for marker in root.findall(".//Marker"):
        nom = marker.attrib.get("name")
        if nom in deplacements:
            dx, dy, dz = deplacements[nom]
            loc_elem = marker.find("location")
            if loc_elem is not None:
                coords = list(map(float, loc_elem.text.split()))
                new_coords = [coords[0] + dx, coords[1] + dy, coords[2] + dz]
                loc_elem.text = "       {:.8f}       {:.8f}       {:.8f}".format(*new_coords)

    # Sauvegarder le fichier modifié
    tree.write(output_marker_file, encoding="UTF-8", xml_declaration=True)




def extract_markers_from_xml(xml_file_path):
    """
    Extracts a dictionary of marker names and their 3D coordinates from an OpenSim .xml file.

    Args:
        xml_file_path (str): Path to the OpenSim marker XML file.

    Returns:
        dict: A dictionary where keys are marker names and values are [x, y, z] coordinate lists.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    ns = {"osim": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    markers_dict = {}

    for marker in root.findall(".//Marker", ns):
        name = marker.get("name")
        location_tag = marker.find("location", ns)
        if location_tag is not None:
            coords = list(map(float, location_tag.text.strip().split()))
            markers_dict[name] = coords

    return markers_dict




def scaling(filename_prefix, output_folder, markers, template_scaling_setup, base_marker_set, c3d_calib_file = None, trc_calib_file = None):
    """
    Runs the OpenSim scaling tool to generate a subject-specific scaled model based on a static trial.

    Args:
        filename_prefix (str): Prefix used for naming output files.
        output_folder (str): Directory where output files will be saved.
        markers (list): List of marker names to include in the scaling process.
        template_scaling_setup (str): Path to the XML template for the OpenSim ScaleTool.
        base_marker_set (str): Path to the marker set file to use.
        c3d_calib_file (str, optional): Path to a C3D static calibration file (if used).
        trc_calib_file (str, optional): Path to a TRC file (used if no C3D file is provided).

    Returns:
        None. Outputs are written to the output folder, and temporary XML setup is deleted after execution.
    """
    if c3d_calib_file is not None:
        static_trc = os.path.join(output_folder, f"{filename_prefix}_calib_static.trc")
        convert_c3d_to_trc(c3d_calib_file, output_trc=static_trc, start_time=0, end_time=1, include_markers=markers)
    else:
        static_trc = trc_calib_file


    # define paths 
    scaled_model = os.path.join(output_folder, f"{filename_prefix}_scaled_model.osim")
    model_with_markers = os.path.join(output_folder, f"{filename_prefix}_scaled_model_markers.osim")
    scale_output_file = os.path.join(output_folder, f"{filename_prefix}_scaleSet.xml")
    scale_setup = f"{filename_prefix}_scaling_setup.xml"

    update_scaling_xml(
        input_path=template_scaling_setup,
        output_path=scale_setup,
        marker_set_file=base_marker_set,
        marker_file=static_trc,
        model_scaler_output_model_file=scaled_model,
        marker_placer_output_model_file=model_with_markers,
        output_scale_file=scale_output_file
    )

    osim.ScaleTool(scale_setup).run()
    try: os.remove(scale_setup)
    except FileNotFoundError: pass



def inverse_kinematics(filename_prefix, motion_type, markers, base_marker_set, model_with_markers, output_folder, c3d_dynamic_file=None, trc_dynamic_file=None, n_cycles=None, dyn_initial_time = None, dyn_final_time = None, gait_event_file = None):
    """
    Runs the OpenSim Inverse Kinematics (IK) tool using a dynamic motion trial and a scaled model.

    Args:
        filename_prefix (str): Prefix used for naming output files.
        motion_type (str): Name describing the type of motion (e.g., 'walking', 'squat').
        markers (list): List of marker names to include in the IK process.
        base_marker_set (str): Path to the marker set XML file used during IK.
        model_with_markers (str): Path to the scaled OpenSim model with placed markers.
        output_folder (str): Directory where output files will be saved.
        c3d_dynamic_file (str, optional): Path to a C3D file for the dynamic trial (if used).
        trc_dynamic_file (str, optional): Path to a TRC file for the dynamic trial (used if no C3D file is provided).
        n_cycles (int, optional): Number of gait cycles to include (used if gait_event_file is provided).
        dyn_initial_time (float, optional): Start time of the motion window (in seconds).
        dyn_final_time (float, optional): End time of the motion window (in seconds).
        gait_event_file (str, optional): Path to CSV file with gait event data for determining time window.

    Returns:
        None. Outputs the motion file (.mot) and runs IK using OpenSim.
    """    
    dynamic_trc = os.path.join(output_folder, f"{filename_prefix}_{motion_type}.trc")

    if (dyn_initial_time, dyn_final_time) == (None, None): 
        gait_events_df = extract_gait_cycles(gait_event_file)
        gait_events_df = gait_events_df[:n_cycles]
        dyn_initial_time = gait_events_df["Time"].iloc[0]
        dyn_final_time = gait_events_df["Time"].iloc[-1]

    if c3d_dynamic_file is not None:
        dynamic_trc = os.path.join(output_folder, f"{filename_prefix}_{motion_type}.trc")
        convert_c3d_to_trc(c3d_dynamic_file, output_trc=dynamic_trc, start_time=dyn_initial_time, end_time=dyn_final_time, include_markers=markers)
    else:
        dynamic_trc = trc_dynamic_file

    #define paths 
    ik_output_mot = os.path.join(output_folder, f"{filename_prefix}_{motion_type}_motion.mot")
    ik_setup = f"{filename_prefix}_ik_setup.xml"

    generate_ik_setup_xml(
        model_file=model_with_markers,
        marker_file=dynamic_trc,
        output_motion_file=ik_output_mot,
        markerset_file=base_marker_set,
        output_path=ik_setup,
        time_range=(dyn_initial_time, dyn_final_time)
    )
    
    osim.InverseKinematicsTool(ik_setup).run()
    try: os.remove(ik_setup)
    except FileNotFoundError: pass




def point_kinematics (filename_prefix, motion_type, markers_to_monitor, output_folder, model_file, motion_file, start_time = None, end_time = None, gait_event_file = None, n_cycles = None):
    """
    Runs a PointKinematics analysis in OpenSim using virtual markers and generates a TRC file from the resulting data.

    Args:
        filename_prefix (str): Prefix used to name generated files.
        motion_type (str): Type of motion (e.g., 'walk', 'run') to label the output.
        markers_to_monitor (str): Path to the marker XML file containing virtual markers to track.
        output_folder (str): Folder where output files (XML, TRC) will be saved.
        model_file (str): Path to the OpenSim model file.
        motion_file (str): Path to the motion (.mot) file from inverse kinematics.
        start_time (float, optional): Start time for the analysis. Ignored if gait_event_file is used.
        end_time (float, optional): End time for the analysis. Ignored if gait_event_file is used.
        gait_event_file (str, optional): Path to CSV file containing gait events to infer time window.
        n_cycles (int, optional): Number of gait cycles to analyze if gait_event_file is provided.

    Returns:
        None. Generates a TRC file with tracked virtual marker positions and cleans up intermediate results.
    """
    if (start_time, end_time) == (None, None): 
        gait_events_df = extract_gait_cycles(gait_event_file)
        gait_events_df = gait_events_df[:n_cycles]
        start_time = gait_events_df["Time"].iloc[0]
        end_time = gait_events_df["Time"].iloc[-1]

    pointkin_setup = f"{filename_prefix}_pointkin_setup.xml"

    generate_pointkin_xml_from_marker_file(markers_to_monitor, pointkin_setup)

    pointkin_results_dir = os.path.join(output_folder, f"{filename_prefix}_pointkinematics")


    tree = ET.parse(pointkin_setup)
    root = tree.getroot()

    for elem in root.iter():
        if elem.tag == "model_file":
            elem.text = model_file  
        elif elem.tag == "coordinates_file":
            elem.text = motion_file
        elif elem.tag == "initial_time":
            elem.text = f"{start_time:.3f}"
        elif elem.tag == "final_time":
            elem.text = f"{end_time:.3f}"
        elif elem.tag == "results_directory":
            elem.text = pointkin_results_dir

    tree.write(pointkin_setup)
    
    try:
        pig_tool = osim.AnalyzeTool(pointkin_setup)
        pig_tool.run()
    except Exception as e:
        print("❌ Erreur pendant le PointKinematics AnalyzeTool :", e)
    
    try: os.remove(pointkin_setup)
    except FileNotFoundError: pass


    virtual_marker_dict = extract_markers_from_xml(markers_to_monitor)

    generate_trc_from_stos(virtual_marker_dict, sto_folder=pointkin_results_dir, output_trc_path=output_folder+f"\{filename_prefix}_{motion_type}.trc")
    
    try:
        shutil.rmtree(pointkin_results_dir)
    except Exception as e:
        print(f"⚠️ Impossible de supprimer le dossier {pointkin_results_dir} :", e)