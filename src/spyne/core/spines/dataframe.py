""" Created on Mon Sep 15 11:49:07 2025
    @author: dcupolillo """

from __future__ import annotations
import pandas as pd
import numpy as np
from spyne.core.spines.analysis.spatial_distances import (
    distance_along_neurite, get_path_to_root, path_distance,
    prepare_neurite_distance_tools)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spyne.core.spines.spinedataset import SpineDataset


def to_dataframe(
    spine_dataset: SpineDataset,
    calcium_events_binary_BLA: np.ndarray,
    calcium_events_binary_CA3: np.ndarray,
    n_events_threshold: int = 1,
) -> pd.DataFrame:
    """
    Convert spine data from a SpineDataset into a pandas DataFrame.
    Parameters
    ----------
    spine_dataset : SpineDataset
        The SpineDataset instance containing spine data and predictions.
    calcium_events_binary_BLA : np.ndarray
        Binary array indicating predicted calcium events for BLA spines.
    calcium_events_binary_CA3 : np.ndarray
        Binary array indicating predicted calcium events for CA3 spines.
    n_events_threshold : int, optional
        Minimum number of events to consider a spine as active (default is 1).

    Returns
    -------
    pd.DataFrame
        DataFrame containing processed spine data with additional
        computed columns.
    """
    
    data_list = spine_dataset.spines_data
    neuron_id = spine_dataset._dataset.name
    nodes_list = spine_dataset._dataset._morph.neuron

    df = spines_to_dataframe(
        data_list,
        neuron_id,
        calcium_events_binary_BLA,
        calcium_events_binary_CA3,
        n_events_threshold,
        nodes_list
    )

    return df


def spines_to_dataframe(
        data_list: list,
        neuron_id: str,
        prediction_binary_BLA: np.ndarray,
        prediction_binary_CA3: np.ndarray,
        n_events_threshold: int,
        nodes_list: list
) -> pd.DataFrame:
    """
    This function converts a list of spine dictionaries
    into a pandas DataFrame, adds a boolean 'spine_active' column
    based on calcium event predictions,
    and computes distances to nearest and consecutive neighboring spines.

    Parameters
    ----------
    data_list : list
        List of dictionaries, each containing spine information
        with required keys.
    neuron_id : str
        Identifier for the neuron to which the spines belong.
    prediction_binary_BLA : np.ndarray
        Binary array indicating predicted calcium events for each spine.
    prediction_binary_CA3 : np.ndarray
        Binary array indicating predicted calcium events for each spine.
    n_events_threshold : int
        Minimum number of events to consider a spine as active.
    nodes_list : list
        List of nodes representing the neurite structure
        for distance calculations.
    """

    df = spine_dicts_to_dataframe(data_list, neuron_id)

    df = add_spine_active_column(
        df, prediction_binary_BLA, "BLA", n_events_threshold)
    df = add_spine_active_column(
        df, prediction_binary_CA3, "CA3", n_events_threshold)

    df = add_nearest_neighbor_distance_column(df, "BLA", nodes_list)
    df = add_consecutive_neighbor_distance_column(df, "BLA", nodes_list)

    df = add_nearest_neighbor_distance_column(df, "CA3", nodes_list)
    df = add_consecutive_neighbor_distance_column(df, "CA3", nodes_list)

    return df


def spine_dicts_to_dataframe(
        data_list: list,
        neuron_id: str
) -> pd.DataFrame:
    """
    Convert a list of spine dictionaries into a pandas DataFrame.

    Parameters:
    ----------
    data_list : list
        List of dictionaries, each containing spine information
        with required keys.

    Returns:
    -------
    pd.DataFrame
        DataFrame with 'spine_index' as the index and all other
        keys as columns, excluding the 'mask' column.

    Raises:
    ------
    ValueError
        If data_list is not a list of dictionaries or
        if required keys are missing.
    """
    if not isinstance(data_list, list):
        raise ValueError("data_list must be a list of dictionaries.")

    if not all(isinstance(d, dict) for d in data_list):
        raise ValueError("All items in data_list must be dictionaries.")

    # Check for required keys in each dictionary
    required_keys = {
        'roi_n', 'roi_z', 'centroid_pix', 'mask', 'compartment',
        'branch_id', 'branch_degree', 'spine_area_pix', 'spine_area_um',
        'centroid_fov', 'centroid_fov_um', 'spine_index', 'distance_from_root',
        'root_node_id', 'closest_node_id', 'distance_from_shaft'}

    for n, d in enumerate(data_list):
        if not required_keys.issubset(d.keys()):
            raise ValueError(
                f"Dictionary at index {n} is missing required keys "
                f"{required_keys - d.keys()}.")

    df = pd.DataFrame(data_list, index=None)

    # Remove the `mask` column if it exists
    if "mask" in df.columns:
        df = df.drop(columns=["mask"])

    if "spine_index" not in df.columns:
        raise ValueError("Each dictionary must contain 'spine_index'.")

    df['neuron_id'] = neuron_id

    assert df.shape[0] == len(data_list), (
        "DataFrame row count does not match input list length.")

    return df.set_index("spine_index")


def add_spine_active_column(
        dataframe: pd.DataFrame,
        prediction_binary: np.ndarray,
        input_identity: str,
        n_events_threshold: int = 1

) -> pd.DataFrame:
    """
    Add a boolean 'spine_active' column to the DataFrame based
    on calcium event predictions. A spine is considered active if
    the number of predicted events exceeds the threshold.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spine information.
    prediction_binary : np.ndarray
        Binary array indicating predicted calcium events for each spine.
    input_identity : str
        Identifier for the type of input ('BLA' or 'CA3').
    n_events_threshold : int, optional
        Minimum number of events to consider a spine as active (default is 1).

    Returns:
    -------
    pd.DataFrame
        DataFrame with an added 'spine_active' column indicating active spines.

    Raises:
    ------
    ValueError
        If input parameters are of incorrect types or values.

    Notes:
    -----
    The 'input_identity' parameter must be either 'BLA' or 'CA3'.
    The 'n_events_threshold' must be a positive integer.

    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame.")

    if not isinstance(prediction_binary, np.ndarray):
        raise ValueError("prediction_binary must be a numpy ndarray.")

    if input_identity not in ["BLA", "CA3"]:
        raise ValueError("input_identity must be either 'BLA' or 'CA3'.")

    if not isinstance(n_events_threshold, int) or n_events_threshold < 1:
        raise ValueError("n_events_threshold must be a positive integer.")

    event_counts = prediction_binary.sum(axis=1)
    is_active = event_counts > n_events_threshold
    dataframe[f'is_{input_identity}'] = is_active

    return dataframe


def randomize_identity_column(
        dataframe: pd.DataFrame,
        input_identity: str,
        seed: int = 42
) -> pd.DataFrame:
    """
    Randomly shuffle the boolean values in the 'is_{input_identity}' column
    within each branch, preserving the overall distribution of active spines.
    Parameters:
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spine information with 'branch_id' column.
    input_identity : str
        Identifier for the type of input ('BLA' or 'CA3').
    seed : int, optional
        Random seed for reproducibility (default is 42).
    Returns:
    -------
    pd.DataFrame
        DataFrame with the 'is_{input_identity}' column values shuffled
        within each branch.
    Raises:
    ------
    ValueError
        If input parameters are of incorrect types or values.
    Notes:
    -----   
    The 'input_identity' parameter must be either 'BLA' or 'CA3'.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame.")

    if input_identity not in ["BLA", "CA3"]:
        raise ValueError("input_identity must be either 'BLA' or 'CA3'.")
    
    if f'is_{input_identity}' not in dataframe.columns:
        raise ValueError(
            f"DataFrame must contain 'is_{input_identity}' column.")
    
    np.random.seed(seed)
    
    grouped = dataframe.groupby('branch_id')

    for _, group in grouped:
        
        identities = group[f'is_{input_identity}'].values
        np.random.shuffle(identities)
        dataframe.loc[group.index, f'is_{input_identity}'] = identities

    return dataframe


def add_nearest_neighbor_distance_column(
        dataframe: pd.DataFrame,
        input_identity: str,
        nodes_list: list
) -> pd.DataFrame:
    """
    Add a 'nearest_neighbor_distance' column to the DataFrame,
    calculating the distance to the nearest spine.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spine information with 'centroid' column.
    input_identity : str
        Identifier for the type of input ('BLA' or 'CA3').
    nodes_list : list
        List of nodes representing the neurite structure
        for distance calculations.

    Returns:
    -------
    pd.DataFrame
        DataFrame with an added 'nearest_neighbor_distance' column.
    Raises:
    ------
    ValueError
        If input parameters are of incorrect types or values.
    Notes:
    -----
    The 'input_identity' parameter must be either 'BLA' or 'CA3'.
    Uses `distance_along_neurite()` function to compute distances.
    """

    node_map, paths_to_root = prepare_neurite_distance_tools(nodes_list)

    # Initialize the column with NaN values
    dataframe[f"nearest_neighbor_distance_{input_identity}"] = np.nan

    dataframe_by_input = dataframe[dataframe[f'is_{input_identity}'] == True]

    # Group spines by branch_id
    grouped = dataframe_by_input.groupby('branch_id')

    for _, branch_df in grouped:
        spine_indices = branch_df.index.to_list()
        node_ids = branch_df['closest_node_id'].to_list()
        n = len(spine_indices)

        # Only one spine on branch
        if n < 2:
            dataframe.loc[
                spine_indices[0],
                f'nearest_neighbor_distance_{input_identity}'
            ] = np.nan
            continue

        # For each spine, compute minimum distance to any other spine
        # (excluding self and zero)
        for i, idx in enumerate(spine_indices):
            
            min_dist = np.inf

            for j in range(len(spine_indices)):
                if i == j:
                    continue
                
                d = distance_along_neurite(node_map, paths_to_root, node_ids[i], node_ids[j])
                
                if d > 0 and d < min_dist:
                    min_dist = d
            
            dataframe.loc[
                idx,
                f'nearest_neighbor_distance_{input_identity}'
            ] = min_dist if min_dist != np.inf else np.nan

    return dataframe


def add_nearest_neighbor_distance_column_fast(
        dataframe: pd.DataFrame,
        input_identity: str,
        node_map: dict,
        paths_to_root: dict
) -> pd.DataFrame:
    """
    Add a 'nearest_neighbor_distance' column using precomputed node_map and paths_to_root.
    """

    dataframe = dataframe.copy()
    dataframe[f"nearest_neighbor_distance_{input_identity}"] = np.nan
    dataframe_by_input = dataframe[dataframe[f'is_{input_identity}'] == True]
    grouped = dataframe_by_input.groupby('branch_id')
    
    for _, branch_df in grouped:
    
        spine_indices = branch_df.index.to_list()
        node_ids = branch_df['closest_node_id'].to_list()
        n = len(spine_indices)
    
        if n < 2:
            dataframe.loc[
                spine_indices[0],
                f'nearest_neighbor_distance_{input_identity}'
            ] = np.nan
            continue
    
        for i, idx in enumerate(spine_indices):
            min_dist = np.inf
            for j in range(len(spine_indices)):
                if i == j:
                    continue
                d = distance_along_neurite(node_map, paths_to_root, node_ids[i], node_ids[j])
                if d > 0 and d < min_dist:
                    min_dist = d
    
            dataframe.loc[
                idx,
                f'nearest_neighbor_distance_{input_identity}'
            ] = min_dist if min_dist != np.inf else np.nan
    
    return dataframe


def add_consecutive_neighbor_distance_column(
        dataframe: pd.DataFrame,
        input_identity: str,
        nodes_list: list
) -> pd.DataFrame:
    """
    Add a 'consecutive_neighbor_distance' column to the DataFrame,
    calculating the distance to the next spine along the neurite.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spine information with 'centroid' column.
    input_identity : str
        Identifier for the type of input ('BLA' or 'CA3').
    nodes_list : list
        List of nodes representing the neurite structure
        for distance calculations.

    Returns:
    -------
    pd.DataFrame
        DataFrame with an added 'consecutive_neighbor_distance' column.

    Raises:
    ------
    ValueError
        If input parameters are of incorrect types or values.

    Notes:
    -----
    The 'input_identity' parameter must be either 'BLA' or 'CA3'.
    Uses `distance_along_neurite()` function to compute distances.
    """

    node_map, paths_to_root = prepare_neurite_distance_tools(nodes_list)

    # Initialize the column with NaN values
    dataframe[f"consecutive_neighbor_distance_{input_identity}"] = np.nan

    dataframe_by_input = dataframe[dataframe[f'is_{input_identity}'] == True]

    # Group spines by branch_id
    grouped = dataframe_by_input.groupby('branch_id')

    for _, branch_df in grouped:
        branch_df = branch_df.copy()
        branch_df['distance_from_soma'] = branch_df['closest_node_id'].apply(
            lambda nid: path_distance(
                node_map,
                (path := get_path_to_root(node_map, nid)),
                stop_node_id=path[-1]
            )
        )
        branch_df = branch_df.sort_values('closest_node_id')
        spine_indices = branch_df.index.to_list()
        node_ids = branch_df['closest_node_id'].to_list()
        n = len(spine_indices)

        if n < 2:
            dataframe.loc[
                spine_indices[0],
                f'consecutive_neighbor_distance_{input_identity}'
            ] = np.nan
            continue

        # For each spine, compute minimum distance to any spine
        # further along the neurite (sorted order)
        for i, idx in enumerate(spine_indices):
            
            min_dist = np.inf
            
            for j in range(i + 1, n): # Only spines after i in the sorted list
                d = distance_along_neurite(node_map, paths_to_root, node_ids[i], node_ids[j])
                if d > 0 and d < min_dist:
                    min_dist = d
            
            dataframe.loc[
                idx,
                f'consecutive_neighbor_distance_{input_identity}'
            ] = min_dist if min_dist != np.inf else np.nan

    return dataframe
