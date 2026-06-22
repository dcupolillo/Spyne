import numpy as np


def euclidean_distance(
        point1: tuple,
        point2: tuple
) -> float:
    """
    Compute the Euclidean distance between two points.

    Parameters
    ----------
    point1 : tuple
        The (x, y, z) coordinates of the first point.
    point2 : tuple
        The (x, y, z) coordinates of the second point.

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    return np.round(np.linalg.norm(np.array(point1) - np.array(point2)), 2)


def find_closest_node(
        spine: dict,
        nodes_list: list) -> int:
    """
    Find the ID of the closest node in the morphology
    to the given spine position.

    Parameters
    ----------
    spine_position : tuple
        The (x, y, z) coordinates of the spine position.
    nodes_list : list
        List of nodes in the morphology.

    Returns
    -------
    int
        The id of the closest node to the spine position.
    """
    closest_node = None
    min_distance = float("inf")

    spine_position = (
        spine["centroid_fov_um"][0],
        spine["centroid_fov_um"][1],
        spine["roi_z"])

    # branch_id is be set when building the spine dict,
    # during segmentation post-processing
    branch_id = spine["branch_id"]

    # Only consider nodes from the same branch, excluding soma nodes
    branch_nodes = [
        node for node in nodes_list
        if hasattr(node, "branch_id") and
        node.branch_id == branch_id and
        getattr(node, 'type', None) != 'soma']

    if not branch_nodes:
        raise ValueError(f"No non-soma nodes found for branch_id {branch_id}.")

    for node in branch_nodes:
        node_position = (node.x, node.y, node.z)
        distance = euclidean_distance(node_position, spine_position)

        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node.id


def get_path_to_root(
        node_map: dict,
        node_id: int
) -> list:
    """
    Return the path from a node to the soma (node with parent_id == 0)
    as a list of node ids (including the root).

    Parameters
    ----------
    node_map : dict
        A mapping from node id to node object with .parent_id attribute.
    node_id : int
        The starting node id.
    """

    path = []
    
    while node_id in node_map:
        if isinstance(node_map[node_id], dict):
            parent_id = node_map[node_id]["_parent_id"]
        else:
            parent_id = node_map[node_id].parent_id
        
        path.append(node_id)
        
        if parent_id == 0:
            break
        
        node_id = parent_id
    
    return path


def path_distance(
        node_map: dict,
        path: list,
        stop_node_id: int
) -> float:
    """
    Sum the distance along a path of node ids up to (but not including) stop_node_id.

    Parameters
    ----------
    node_map : dict
        A mapping from node id to node object with .x, .y, .z, .parent_id attributes.
    path : list
        List of node ids representing the path.
    stop_node_id : int
        The node id at which to stop summing distances (not included in sum).

    Returns
    -------
    float
        The total distance along the path up to stop_node_id.
    """
    if not isinstance(path, list):
        raise ValueError("Path must be a list of node ids.")
    
    if not path or path[-1] == stop_node_id:
        return 0.0
    
    if stop_node_id not in path:
        raise ValueError("stop_node_id must be in the path.")
    
    dist = 0.0
    for i in range(path.index(stop_node_id)):
        
        n1 = node_map[path[i]]
        
        if isinstance(n1, dict):
            parent_id = n1["_parent_id"]
            x1, y1, z1 = n1["x"], n1["y"], n1["z"]
            n2 = node_map[parent_id]
            x2, y2, z2 = n2["x"], n2["y"], n2["z"]
        else:
            parent_id = n1.parent_id
            x1, y1, z1 = n1.x, n1.y, n1.z
            n2 = node_map[parent_id]
            x2, y2, z2 = n2.x, n2.y, n2.z

        # If parent_id is -1 or not in node_map, stop
        if parent_id == -1 or parent_id not in node_map:
            break
        dist += euclidean_distance((x1, y1, z1), (x2, y2, z2))
    
    return dist


def prepare_neurite_distance_tools(nodes_list: list) -> tuple:
    """
    Precompute node_map and paths to root for all nodes
    for fast repeated distance calculations.
    Returns:
        node_map: dict of node_id -> node
        paths_to_root: dict of node_id -> path to root (list of node_ids)
    """
    if all(isinstance(node, dict) for node in nodes_list):
        node_map = {node["_id"]: node for node in nodes_list}
        get_parent = lambda n: node_map[n]["_parent_id"]
    elif all(hasattr(node, "id") for node in nodes_list):
        node_map = {node.id: node for node in nodes_list}
        get_parent = lambda n: node_map[n].parent_id

    paths_to_root = {}
    for node_id in node_map:
        path = []
        nid = node_id
        while nid in node_map and get_parent(nid) != 0:
            path.append(nid)
            nid = get_parent(nid)
        path.append(nid)
        paths_to_root[node_id] = path
    
    return node_map, paths_to_root


def distance_along_neurite(
        node_map: dict,
        paths_to_root: dict,
        node_id1: int,
        node_id2: int
) -> float:
    """
    Compute the distance along the neurite between two nodes using
    precomputed paths to root.
    
    Parameters
    ----------
    node_map : dict
        A mapping from node id to node object with .x, .y, .z, .parent_id attributes.
    paths_to_root : dict
        A mapping from node id to its path to root (list of node ids).
    node_id1 : int
        The id of the first node.
    node_id2 : int
        The id of the second node.
    
    Returns
    -------
    float
        The distance along the neurite between the two nodes.
    """

    path1 = paths_to_root[node_id1]
    path2 = paths_to_root[node_id2]

    # Find lowest common ancestor (LCA)
    set2 = set(path2)
    lca = next((nid for nid in path1 if nid in set2), None)
    if lca is None:
        # No connection found
        return np.nan

    # Path from node1 to LCA
    dist1 = path_distance(node_map, path1, lca)
    # Path from node2 to LCA
    dist2 = path_distance(node_map, path2, lca)
    
    return np.round(dist1 + dist2, 2)


def find_root(
        nodes_list: list,
        node_id: int
) -> int:
    """
    Find the ID of the closest root node (soma or branching point)
    to the given node in the morphology.

    Assumes that the nodes are ordered by their position along the neurite
    with IDs incrementing with distance from origin.

    Parameters
    ----------
    nodes_list : list
        List of nodes in the morphology.
    node_id : int
        ID of the node to find the closest origin for.

    Returns
    -------
    int
        The closest root node index to the given node.
    """
    if node_id > nodes_list[-1].id:
        raise IndexError("Node index out of range")

    node_index = node_id - 1 # -1 because 0-based numeration

    for node in nodes_list[:node_index][::-1]:
        parent_id = node.parent_id
            
        # Check if parent is a fork or soma
        if nodes_list[parent_id - 1].is_fork: 
            return node.id - 1  # Return current node (first node of branch segment)

        if nodes_list[parent_id - 1].type == "soma":
            return node.id - 1  # Return current node (first dendritic node from soma)
    
    # Fallback soma
    return 0


def interspine_distance(
        nodes_list: list,
        spine1: dict,
        spine2: dict
) -> float:
    """
    Compute the distance between two spines in 3D space.
    Distance is calculated between the points where the spine necks
    connects to the dendritic shaft.

    Parameters
    ----------
    spine1 : dict
        The first spine's (x, y, z) coordinates.
    spine2 : dict
        The second spine's (x, y, z) coordinates.

    Returns
    -------
    float
        The Euclidean distance between the two spines.
    """

    node_map, paths_to_root = prepare_neurite_distance_tools(nodes_list)
    
    closest_node_1 = nodes_list[spine1['closest_node_id'] - 1]
    closest_node_2 = nodes_list[spine2['closest_node_id'] - 1]

    # Distance along neurite between nodes
    internode_distance = distance_along_neurite(
        node_map,
        paths_to_root,
        closest_node_1.id,
        closest_node_2.id
    )

    return np.round(internode_distance, 2)
