from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from joblib import Parallel, delayed

from Helping_Code.CustomHyperrectangle import minimize_hyperrectangle_distance_dual

@dataclass
class StateModel:
    # Identifiers
    state_identifier: int

    # Regression model info: y = Mx + m0, with error h
    M: Optional[np.ndarray] = None
    m0: Optional[np.ndarray] = None
    h: Optional[float] = None

    # Data / bookkeeping
    n_samples: int = 0
    indices: List[int] = field(default_factory=list)

    # Geometry
    bounds: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    vertices: Optional[np.ndarray] = None
    image_vertices: Optional[np.ndarray] = None

    transition_to: List[int] = field(default_factory=list)

   
def print_states(states: List[StateModel], fields_to_print: List[str]) -> None:
    for st in states:
        output = [f"State {st.state_identifier}"]
        for field in fields_to_print:
            if hasattr(st, field):
                value = getattr(st, field)
                output.append(f"{field}={value}")
            else:
                output.append(f"{field}=<not found>")
        print(", ".join(output))

# Usage
#print_states(states, ["n_samples", "indices"])

def collect_state_field(states: List[StateModel], field_name: str) -> List[Any]:
    """
    Return a list with the value of `field_name` from each StateModel in `states`.
    Raises AttributeError if the field doesn't exist on StateModel.
    """
    if not states:
        return []
    # Fail fast if the attribute doesn't exist
    if not hasattr(states[0], field_name):
        raise AttributeError(f"`StateModel` has no attribute '{field_name}'")
    return [getattr(st, field_name) for st in states]

# usage
# vals_h = collect_state_field(states, "h")



def compute_row_of_TransitionMatrix(i:int, numLeaves:int, elevated_vertices:np.ndarray, vertices_of_hyperrectangle:List[np.ndarray], h:float):
    local_row = np.zeros(numLeaves)
    vertices_P1 = elevated_vertices

    for j in range(numLeaves):
        vertices_P2 = vertices_of_hyperrectangle[j]
        _, _, min_distance = minimize_hyperrectangle_distance_dual(vertices_P1, vertices_P2)
        # print(f"min_distance = {min_distance} | h = {h}")
        if min_distance < h:
            local_row[j] = 1
    return i, local_row


def getStateTransitionMatrix( vertices_of_hyperrectangle:List[np.ndarray], elevated_vertices:List[np.ndarray], h_values:List[int], n_jobs=-1)->np.ndarray:
    numLeaves = len(h_values)
    state_graph_transition_matrix = np.zeros((numLeaves, numLeaves))

    # Run rows in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_row_of_TransitionMatrix)(i, numLeaves, elevated_vertices[i], vertices_of_hyperrectangle, h_values[i])
        for i in range(numLeaves)
    )

    # Collect results
    for i, local_row in results:
        state_graph_transition_matrix[i, :] = local_row

    return state_graph_transition_matrix

def calculate_state_transition(states: List[StateModel], n_jobs=-1) -> np.ndarray:
    vertices_of_hyperrectangles = collect_state_field(states, "vertices")
    elevated_vertices = collect_state_field(states, "image_vertices")
    h_values = collect_state_field(states, "h")
    numLeaves = len(states)
    state_graph_transition_matrix = np.zeros((numLeaves, numLeaves))

    # Run rows in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_row_of_TransitionMatrix)(i, numLeaves, elevated_vertices[i], vertices_of_hyperrectangles, h_values[i])
        for i in range(numLeaves)
    )

    # Collect results
    for i, local_row in results:
        idx = np.where(local_row == 1)[0].tolist()
        states[i].transition_to = idx
        state_graph_transition_matrix[i, :] = local_row
        
    return state_graph_transition_matrix