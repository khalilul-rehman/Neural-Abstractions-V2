# na_trajectories_from_states.py
# Build trajectories from initial-box vertices using StateModel objects (no XML).
# Plots invariant polytopes (no image polytopes) + trajectories.

from CustomAbstraction.CustomAbstractionHelper import StateModel
from typing import Optional, List, Dict, Tuple
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt




# -------------------- Helpers --------------------
def in_bounds(x: np.ndarray, bounds: Dict[int, Tuple[float, float]], tol: float = 1e-12) -> bool:
    for i, (lo, hi) in bounds.items():
        if x[i] < lo - tol or x[i] > hi + tol:
            return False
    return True

def euler_step(M: np.ndarray, m0: np.ndarray, u: np.ndarray, x: np.ndarray, dt: float) -> np.ndarray:
    return x + dt * (M @ x + m0 + u)

def u_from_h(h: Optional[float], n: int, mode: str = "plus") -> np.ndarray:
    """
    Piecewise-constant input per step derived from scalar h.
    modes:
      - "zero":  u = 0
      - "plus":  u = +sqrt(h) * 1_n
      - "minus": u = -sqrt(h) * 1_n
      - "mid":   u = 0  (same as zero; kept for readability)
    """
    if h is None or h < 0:
        return np.zeros(n)
    r = math.sqrt(h)
    if mode == "plus":
        return np.full(n, r)
    if mode == "minus":
        return np.full(n, -r)
    # zero / mid
    return np.zeros(n)

def pick_state_containing(x: np.ndarray, states: List[StateModel]) -> Optional[StateModel]:
    for s in states:
        if in_bounds(x, s.bounds):
            return s
    return None

def next_state_by_edges(cur: StateModel, x_next: np.ndarray, states_by_id: Dict[int, StateModel]) -> Optional[StateModel]:
    # Prefer declared transitions
    for tid in cur.transition_to:
        st = states_by_id.get(tid)
        if st is not None and in_bounds(x_next, st.bounds):
            return st
    # Fallback: any state whose bounds contain x_next
    for st in states_by_id.values():
        if in_bounds(x_next, st.bounds):
            return st
    return None

def box_vertices(initial_box: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    initial_box: [(lo0, hi0), (lo1, hi1), ...]
    Returns all 2^n corner vertices as np.ndarrays.
    """
    corners = []
    for vals in itertools.product(*[[lo, hi] for (lo, hi) in initial_box]):
        corners.append(np.array(vals, dtype=float))
    return corners


# -------------------- Simulation --------------------
def simulate_from_vertex(states: List[StateModel],
                         x0: np.ndarray,
                         dt: float,
                         T: float,
                         u_mode: str = "plus") -> Tuple[np.ndarray, List[int]]:
    """
    Simulate a single trajectory starting at vertex x0.
    - Routes via state's transition_to; falls back to any containing state if needed.
    - Uses u from h via u_mode (default 'plus' => +sqrt(h) per component).
    Returns (traj_points array, visited_state_ids list).
    """
    assert len(states) > 0, "states must be non-empty"
    n = states[0].M.shape[0]
    x = x0.copy()
    # Ensure dimension matches model
    if x.shape[0] != n:
        raise ValueError(f"Initial point dimension {x.shape[0]} != model dimension {n}")

    # Preindex
    states_by_id = {s.state_identifier: s for s in states}

    # Pick starting state that contains x0
    cur = pick_state_containing(x, states)
    if cur is None:
        # If no state contains the point, choose the one whose box center is closest (robust fallback)
        def center(b): return np.array([(lo + hi) * 0.5 for (_, (lo, hi)) in sorted(b.items())])
        cur = min(states, key=lambda s: np.linalg.norm(x - center(s.bounds)))
    visited = [cur.state_identifier]
    traj = [x.copy()]

    steps = max(1, int(math.ceil(T / dt)))
    for _ in range(steps):
        # Input derived from current state's h
        u = u_from_h(cur.h, n, mode=u_mode)
        x_next = euler_step(cur.M, cur.m0, u, x, dt)

        # Stay if still inside
        if in_bounds(x_next, cur.bounds):
            x = x_next
        else:
            # Try edges first, then any state
            nxt = next_state_by_edges(cur, x_next, states_by_id)
            if nxt is None:
                # cannot move anywhere; stop trajectory
                break
            cur = nxt
            x = x_next

        traj.append(x.copy())
        visited.append(cur.state_identifier)

    return np.vstack(traj), visited


# -------------------- Plotting --------------------
def plot_polytopes_and_trajectories(states: List[StateModel],
                                    trajectories: List[np.ndarray],
                                    visited_ids: List[List[int]],
                                    initial_box: Optional[List[Tuple[float, float]]] = None,
                                    forbidden_box: Optional[List[Tuple[float, float]]] = None,
                                    dims: Tuple[int, int] = (0, 1),
                                    save_path: Optional[str] = None,
                                    title: Optional[str] = None):
    """
    Plot only invariant polytopes (rectangles from bounds) in (x_i, x_j) plane,
    plus multiple trajectories (one per initial vertex).
    """
    i, j = dims
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"x{j}")
    ax.set_title("Invariant polytopes and trajectories")

    # Colors per state
    palette = list(plt.rcParams['axes.prop_cycle'].by_key().get('color', [])) or [f"C{k}" for k in range(10)]
    color_by_id: Dict[int, str] = {}
    for idx, s in enumerate(states):
        color_by_id[s.state_identifier] = palette[idx % len(palette)]

    # Draw polytopes (base/invariant), only if bounded on (i,j)
    for s in states:
        if i not in s.bounds or j not in s.bounds:
            continue
        (xi_lo, xi_hi) = s.bounds[i]
        (xj_lo, xj_hi) = s.bounds[j]
        if xi_lo is None or xi_hi is None or xj_lo is None or xj_hi is None:
            continue
        rect = np.array([[xi_lo, xj_lo], [xi_hi, xj_lo], [xi_hi, xj_hi], [xi_lo, xj_hi]])
        c = color_by_id[s.state_identifier]
        ax.fill(rect[:, 0], rect[:, 1], facecolor=c, alpha=0.18) #, label=f"state_{s.state_identifier} polytope"
        ax.plot(rect[:, 0], rect[:, 1], color=c, linewidth=1.2)

    # Initial box (if given)
    if initial_box is not None and len(initial_box) > max(i, j):
        (i_lo, i_hi) = initial_box[i]
        (j_lo, j_hi) = initial_box[j]
        if all(v is not None for v in [i_lo, i_hi, j_lo, j_hi]):
            R = np.array([[i_lo, j_lo], [i_hi, j_lo], [i_hi, j_hi], [i_lo, j_hi]])
            ax.fill(R[:, 0], R[:, 1], facecolor="black", alpha=0.12, label="initial set")
            ax.plot(R[:, 0], R[:, 1], color="black", linewidth=1.5)

    # Forbidden box (if given)
    if forbidden_box is not None and len(forbidden_box) > max(i, j):
        (i_lo, i_hi) = forbidden_box[i]
        (j_lo, j_hi) = forbidden_box[j]
        if all(v is not None for v in [i_lo, i_hi, j_lo, j_hi]):
            R = np.array([[i_lo, j_lo], [i_hi, j_lo], [i_hi, j_hi], [i_lo, j_hi]])
            ax.fill(R[:, 0], R[:, 1], facecolor="red", alpha=0.15, label="forbidden")
            ax.plot(R[:, 0], R[:, 1], color="red", linewidth=1.5)

    # Plot trajectories
    for k, traj in enumerate(trajectories):
        if traj.shape[0] == 0:
            continue
        # color segments by the state they were in (if provided)
        ids = visited_ids[k] if k < len(visited_ids) else None
        if ids is None or len(ids) != traj.shape[0]:
            # single color fallback
            ax.plot(traj[:, i], traj[:, j], color="k", linewidth=2.0)
        else:
            for t in range(traj.shape[0] - 1):
                cid = ids[t]
                c = color_by_id.get(cid, "k")
                p = traj[t, [i, j]]
                q = traj[t + 1, [i, j]]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c, linewidth=2.0)
            # markers
            ax.plot(traj[0, i], traj[0, j], "ko", markersize=5)
            ax.plot(traj[-1, i], traj[-1, j], "kx", markersize=6)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); H = []; L = []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    if H:
        ax.legend(H, L, loc="best")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


# -------------------- Driver (example usage) --------------------
def run_all_trajectories(states: List[StateModel],
                         initial_box: List[Tuple[float, float]],
                         dt: float = 0.01,
                         T: float = 1.5,
                         u_mode: str = "plus",    # "+sqrt(h)" by default as you requested
                         dims: Tuple[int, int] = (0, 1),
                         forbidden_box: Optional[List[Tuple[float, float]]] = None,
                         save_path: Optional[str] = None,
                         title: Optional[str] = None):
    """
    Generates one trajectory per initial-box vertex and plots them with polytopes.
    """
    # Build vertices of the initial box
    verts = box_vertices(initial_box)

    trajectories: List[np.ndarray] = []
    visited_ids: List[List[int]] = []

    for v in verts:
        traj, visited = simulate_from_vertex(states, v, dt=dt, T=T, u_mode=u_mode)
        trajectories.append(traj)
        visited_ids.append(visited)

    plot_polytopes_and_trajectories(
        states=states,
        trajectories=trajectories,
        visited_ids=visited_ids,
        initial_box=initial_box,
        forbidden_box=forbidden_box,
        dims=dims,
        save_path=save_path,
        title=title
    )


# -------------------- If you want to test quickly --------------------
if __name__ == "__main__":
    # Minimal illustrative toy (replace with your real 'states' + initial_box)
    # 2D example with two states that cover adjacent x0 intervals.
    s0 = StateModel(
        state_identifier=0,
        M=np.array([[0.5, -1.0],
                    [3.0, -1.0]]),
        m0=np.array([-0.1, 0.0]),
        h=0.01,  # u magnitude = sqrt(0.01) = 0.1 per component
        bounds={0: (0.0, 0.6), 1: (-1.1, 1.1)},
        transition_to=[1]
    )
    s1 = StateModel(
        state_identifier=1,
        M=np.array([[-1.3, -1.0],
                    [3.0, -1.0]]),
        m0=np.array([0.08, 0.0]),
        h=0.01,
        bounds={0: (0.6, 1.1), 1: (-1.1, 1.1)},
        transition_to=[]
    )
    states = [s0, s1]

    initial_box = [(0.45, 0.5), (-0.6, -0.55)]   # x0,x1
    forbidden_box = [(0.3, 0.35), (0.5, 0.6)]

    run_all_trajectories(states, initial_box, dt=0.05, T=1.0,
                         u_mode="plus", dims=(0, 1), forbidden_box=forbidden_box)

'''
u as √h: by default u_mode="plus" sets 
𝑢
𝑖
=
+
ℎ
u
i
	​

=+
h
	​

 for all i in the current location. Switch to "minus" or "zero" if you want the opposite extreme or no input. You can also change it step-by-step based on your policy.

Transitions: the simulator looks at transition_to first; if none matches, it falls back to “any state whose bounds contain the point.” That keeps things going even if a specific edge wasn’t listed.

Multiple trajectories: one per initial vertex (all 
2
𝑛
2
n
 corners of the initial box). If you prefer only the center, just call simulate_from_vertex once with the midpoint.

If you want me to export the trajectories to CSV as well (per vertex, with state IDs and timestamps), I can add a save_csv=True option in a few lines.
'''