import cvxpy as cp
from collections import defaultdict
import numpy as np

from joblib import Parallel, delayed
from typing import List, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from CustomAbstraction.CustomAbstractionHelper import StateModel

from Helping_Code.HelpingFunctions import get_leaf_samples
#from Helping_Code.HelpingFunctions import normalized_root_mean_square_error

from Helping_Code.CustomHyperrectangle import vertices_from_bounds_dict


def cvxpy_minimax(X_leaf, y_leaf):
    """
    Multi-output minimax regression using CVXPY.
    Returns M (m x d), m0 (m,), h (float)
    """
    n, d = X_leaf.shape
    if y_leaf.ndim == 1:
        y_leaf = y_leaf.reshape(-1, 1)
    m = y_leaf.shape[1]

    # Decision variables
    M = cp.Variable((m, d))
    m0 = cp.Variable(m)
    h = cp.Variable(nonneg=True)

    constraints = []
    for i in range(n):
        x_i = X_leaf[i, :]
        y_i = y_leaf[i, :]
        pred = M @ x_i + m0
        residual = pred - y_i
        constraints.append(cp.sum_squares(residual) <= h)

    # Objective: minimize worst-case squared error h
    problem = cp.Problem(cp.Minimize(h), constraints)
    problem.solve(solver=cp.SCS)  # or ECOS, OSQP, GUROBI if licensed

    return M.value, m0.value, h.value
'''
# previouse Working version
def gurobi_minimax(X_leaf, y_leaf):
    """
    Multi-output minimax regression using Gurobi.
    Returns M (m x d), m0 (m,), h (float)
    """
    try:
        n, d = X_leaf.shape
        if y_leaf.ndim == 1:
            y_leaf = y_leaf.reshape(-1, 1)
        m = y_leaf.shape[1]

        model = gp.Model("minimax_regression")
        model.setParam("OutputFlag", 0)  # silent

        # Decision variables
        M = model.addVars(m, d, lb=-GRB.INFINITY, name="M")
        m0 = model.addVars(m, lb=-GRB.INFINITY, name="m0")
        h = model.addVar(lb=0, name="h")

        # Constraints: for each sample, squared norm <= h
        for i in range(n):
            diff_sq_terms = []
            for k in range(m):
                expr = m0[k]
                for j in range(d):
                    expr += M[k, j] * X_leaf[i, j]
                diff_sq_terms.append((expr - y_leaf[i, k]) * (expr - y_leaf[i, k]))
            model.addConstr(gp.quicksum(diff_sq_terms) <= h)

        model.setObjective(h, GRB.MINIMIZE)
        model.optimize()

        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or model.SolCount > 0:
            M_val = np.array([[M[k, j].X for j in range(d)] for k in range(m)])
            m0_val = np.array([m0[k].X for k in range(m)])
            h_val = h.X
        else:
            M_val, m0_val, h_val = None, None, np.inf

        return M_val, m0_val, h_val
    except gp.GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            return cvxpy_minimax(X_leaf, y_leaf)   # your fallback path
        raise
'''
# Updated Working version
def gurobi_minimax(X_leaf, y_leaf):
    """
    Minimax regression (multi-output) with squared L2 error.
    Minimize h s.t. for every sample i: sum_k (m0_k + M_k·x_i - y_{ik})^2 <= h
    Returns: M (m x d), m0 (m,), h (float)
    """
    # Shapes
    n, d = X_leaf.shape
    if y_leaf.ndim == 1:
        y_leaf = y_leaf.reshape(-1, 1)
    m = y_leaf.shape[1]

    try:
        model = gp.Model("minimax_regression")
        model.Params.OutputFlag = 0  # silent

        # Decision variables
        M = model.addVars(m, d, lb=-GRB.INFINITY, name="M")
        m0 = model.addVars(m, lb=-GRB.INFINITY, name="m0")
        h  = model.addVar(lb=0.0, name="h")

        # Quadratic constraints: for each sample i, sum_k (affine)^2 <= h
        for i in range(n):
            quad_terms = []
            for k in range(m):
                # expr = m0[k] + sum_j M[k,j] * X[i,j] - y[i,k]
                expr = m0[k]
                for j in range(d):
                    expr += M[k, j] * float(X_leaf[i, j])  # X is data (constant)
                expr -= float(y_leaf[i, k])
                quad_terms.append(expr * expr)  # (affine)^2 -> QuadExpr

            # Move h to LHS so all quadratic stays on LHS
            model.addQConstr(gp.quicksum(quad_terms) - h <= 0.0)

        # Objective: minimize h
        model.setObjective(h, GRB.MINIMIZE)
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or model.SolCount > 0:
            M_val  = np.array([[M[k, j].X for j in range(d)] for k in range(m)])
            m0_val = np.array([m0[k].X for k in range(m)])
            h_val  = float(h.X)
            return M_val, m0_val, h_val
        else:
            return None, None, np.inf

    except gp.GurobiError as e:
        # License-size fallback or other Gurobi errors
        if "Model too large for size-limited license" in str(e):
            # Define this in your codebase
            return cvxpy_minimax(X_leaf, y_leaf)
        raise
    

def process_leaf(
    leaf_id: int,
    indices: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: str
) -> Optional[StateModel]:
    X_leaf = X_train[indices]
    y_leaf = y_train[indices]

    if X_leaf.shape[0] == 0:
        return None  # nothing reached this leaf

    # Choose optimizer to get M, m0, h
    if optimizer.lower() == "gurobi":
        M, m0, h = gurobi_minimax(X_leaf, y_leaf)
    else:
        M, m0, h = cvxpy_minimax(X_leaf, y_leaf, squared=True)

    # Build bounds by feature index (safe & compact)
    bounds_by_idx: Dict[int, Tuple[float, float]] = {
        i: (float(X_leaf[:, i].min()), float(X_leaf[:, i].max()))
        for i in range(X_leaf.shape[1])
    }

    # Compute vertices from bounds
    vertices = vertices_from_bounds_dict(bounds_by_idx)  # shape: (2^d, d)

    # Compute image vertices if model is available
    image_vertices = vertices @ M.T + m0  # (2^d, p)

    state = StateModel(
        state_identifier=leaf_id,
        M=M,
        m0=m0,
        h=float(h),
        n_samples=len(indices),
        indices=list(indices),
        bounds=bounds_by_idx,
        vertices=vertices,
        image_vertices=image_vertices,
    )

    return state


def train_States_on_leaves_parallel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tree,
    optimizer: str = "gurobi",
    n_jobs: int = -1,
) -> List[StateModel]:
    """
    Train constrained optimization models on tree leaves in parallel,
    returning a list of StateModel objects (one per non-empty leaf).
    """
    leaf_samples: Dict[int, List[int]] = get_leaf_samples(tree, X_train)


    # Run leaf computations in parallel -> returns List[Optional[StateModel]]
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_leaf)(leaf_id, indices, X_train, y_train, optimizer)
        for leaf_id, indices in leaf_samples
    )

    # Keep only valid states
    states: List[StateModel] = [st for st in results if st is not None]
    return states


def predict_with_tree(new_X: np.ndarray, tree, states: List[StateModel]) -> np.ndarray:
    """
    Predict y for new_X by routing through the sklearn DecisionTree to get leaf IDs,
    then applying y = Mx + m0 for the matching StateModel.

    Args:
        new_X: (N, m) input matrix
        tree: fitted sklearn DecisionTreeRegressor (or tree_.apply-capable estimator)
        states: list of StateModel with state_identifier == leaf_id, and M, m0 set

    Returns:
        y_pred: (N, p) predictions
    """
    if new_X.ndim != 2:
        raise ValueError("new_X must be 2D (N, m).")

    # map leaf_id -> StateModel
    id2state: Dict[int, StateModel] = {s.state_identifier: s for s in states}

    # figure out output dim p from any state that has a model
    st0 = next((s for s in states if s.M is not None and s.m0 is not None), None)
    if st0 is None:
        raise ValueError("No StateModel has M and m0 set.")
    p = st0.m0.shape[0]
    m = st0.M.shape[1]

    if new_X.shape[1] != m:
        raise ValueError(f"new_X has {new_X.shape[1]} features but M expects {m}.")

    # route samples to leaves
    leaf_ids = tree.apply(new_X)  # shape (N,)
    y_pred = np.empty((new_X.shape[0], p), dtype=float)

    # vectorized by leaf
    unique_leafs = np.unique(leaf_ids)
    for lid in unique_leafs:
        mask = (leaf_ids == lid)
        st = id2state.get(int(lid))
        if st is None:
            # No model for this leaf — fill NaNs (or raise if you prefer)
            y_pred[mask, :] = np.nan
            continue
        if st.M is None or st.m0 is None:
            y_pred[mask, :] = np.nan
            continue

        X_block = new_X[mask]                   # (k, m)
        # y = X_block @ M.T + m0
        y_block = X_block @ st.M.T + st.m0      # (k, p)
        y_pred[mask, :] = y_block

    return y_pred

def _contains(x: np.ndarray, bounds: Dict[int, Tuple[float, float]]) -> bool:
    # True if x is inside all bounded dims
    for d, (mn, mx) in bounds.items():
        v = x[d]
        if not (mn <= v <= mx):
            return False
    return True

def predict_by_bounds(new_X: np.ndarray, states: List[StateModel]) -> np.ndarray:
    """
    Predict y by selecting the first StateModel whose bounds contain each sample.
    If none matches or model missing, returns NaNs for that row.
    """
    if new_X.ndim != 2:
        raise ValueError("new_X must be 2D (N, m).")

    # infer p from any state with model
    st0 = next((s for s in states if s.M is not None and s.m0 is not None), None)
    if st0 is None:
        raise ValueError("No StateModel has M and m0 set.")
    p = st0.m0.shape[0]
    m = st0.M.shape[1]
    if new_X.shape[1] != m:
        raise ValueError(f"new_X has {new_X.shape[1]} features but M expects {m}.")

    y_pred = np.full((new_X.shape[0], p), np.nan, dtype=float)

    for i, x in enumerate(new_X):
        chosen = None
        for st in states:
            if st.M is None or st.m0 is None:
                continue
            if _contains(x, st.bounds):
                chosen = st
                break
        if chosen is not None:
            y_pred[i, :] = x @ chosen.M.T + chosen.m0  # (p,)

    return y_pred

# def get_leaf_samples(tree_model, X):
#     leaf_indices = tree_model.apply(X)
#     leaf_samples = defaultdict(list)
#     for i, leaf in enumerate(leaf_indices):
#         leaf_samples[leaf].append(i)
#     return leaf_samples


# def train_COF_on_leaves(X_train, y_train, tree,feature_names=None, optimizer = "gurobi"):
#     # optimizer can be { "gurobi" or "CVXPY + SCS"}
#     leaf_samples = get_leaf_samples(tree, X_train)
#     tree_extracted_info = []
#     if feature_names is None:
#         feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

#     for leaf_id, indices in leaf_samples.items():
#         X_leaf = X_train[indices]
#         y_leaf = y_train[indices]
#         if optimizer == "gurobi":
#             M, m0, h = constrained_optimization_gurobi(X_leaf, y_leaf)
#         else:
#             M, m0, h = constrained_optimization(X_leaf, y_leaf)
#         model = {} # models = {leaf_Index:{h, M, m0}}
#         model["leaf_id"] = leaf_id
#         model["CO_Model"] = {'M': M, 'm0': m0, 'h': h }
#         model["no_samples"] = len(indices)
#         model["indices"] = indices
        
#         bounds = {
#                 feature_names[i]: (X_leaf[:, i].min(), X_leaf[:, i].max())
#                 for i in range(X_leaf.shape[1])
#             }
#         model["bounds"] = bounds
#         tree_extracted_info.append(model)
#     return tree_extracted_info










# def predict_from_COF(COF_model_tree, X_new, tree):
#     """
#     Predicts outputs for new samples X_new based on the COF model.

#     Parameters:
#     -----------
#     COF_model_tree : list of dict
#         Output from train_COF_on_leaves.
#         Each dict contains 'leaf_id', 'CO_Model': {'M','m0','h'}, etc.
#     X_new : np.ndarray
#         New input samples, shape (n_samples, n_features)
#     leaf_indices : np.ndarray or None
#         Optional precomputed leaf indices for each sample. 
#         If None, function assumes you will map samples to leaves externally.

#     Returns:
#     --------
#     y_pred : np.ndarray
#         Predicted outputs, shape (n_samples, n_outputs)
#     """
#     n_samples = X_new.shape[0]
#     n_outputs = next(iter(COF_model_tree))['CO_Model']['M'].shape[0]

#     y_pred = np.zeros((n_samples, n_outputs))

#     leaf_samples = get_leaf_samples(tree, X_new)

#     for leaf_id, indices in leaf_samples.items():
#         # X_leaf = X_new[indices]
#         leaf_model = leaf_model = next(item for item in COF_model_tree if item['leaf_id'] == leaf_id) 
#         M = leaf_model['CO_Model']['M']
#         m0 = leaf_model['CO_Model']['m0']
#         y_pred[indices] = X_new[indices] @ M.T + m0
    
    

#     return y_pred


# def get_elevated_vertices(COF_model_tree, vertices):
#     elevated_vertices = []
#     for idx, item in enumerate(COF_model_tree):
#         leaf_model = item['CO_Model']
#         M = leaf_model['M']
#         m0 = leaf_model['m0']
#         elevated_vertices.append(vertices[idx] @ M.T + m0)
    
#     return elevated_vertices
            
def counts_of_daviation_in_testing(new_X: np.ndarray, new_y: np.ndarray, states: List[StateModel], error_bound: float = 0.001) -> np.ndarray:
    """
    Predict y for new_X using the list of StateModel objects.
    Each StateModel must have M and m0 and h defined.

    Args:
        new_X: (N, m) testing input matrix
        new_X: (N, m) testing output matrix
        states: list of StateModel with M, m0 set
    Returns:
        n_daviation: (counts) predictions which daviate more than h
    """

    if new_X.ndim != 2:
        raise ValueError("new_X must be 2D (N, m).")
    
    # figure out output dim p from any state that has a model
    st0 = next((s for s in states if s.M is not None and s.m0 is not None and s.h is not None), None)
    if st0 is None:
        raise ValueError("No StateModel has M and m0 set.")
    p = st0.m0.shape[0]
    m = st0.M.shape[1]
    if new_X.shape[1] != m:
        raise ValueError(f"new_X has {new_X.shape[1]} features but M expects {m}.")
    daviation_counts = 0
    for i, x in enumerate(new_X):
        chosen = None
        for st in states:
            if st.M is None or st.m0 is None:
                continue
            if _contains(x, st.bounds):
                chosen = st
                break
        if chosen is not None:
            if np.linalg.norm(x @ chosen.M.T + chosen.m0 - new_y[i]) > error_bound:
                daviation_counts += 1
    return daviation_counts 
