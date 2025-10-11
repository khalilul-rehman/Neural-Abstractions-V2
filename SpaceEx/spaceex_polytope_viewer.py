# spaceex_polytope_viewer.py

import re
import math
import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------- numeric + text utils -------------------------

_num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def _num_sanitize(s: str) -> float:
    s = s.replace(" ", "")
    # normalize adjacent signs
    s = s.replace("+-", "-").replace("-+", "-").replace("++", "+")
    return float(s)


# ------------------------- invariant parsing -------------------------

def _parse_bounds_from_invariant(inv_text: str,
                                 var_prefix: str) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """
    Parse constraints of the form:
      lo <= var  and/or  var <= hi
    where var looks like 'x0', 'x1', ... or 'u0', 'u1', ...
    Returns dict: idx -> (lo, hi) where either can be None if missing.
    """
    s = (inv_text or "").replace("&amp;", "&")
    parts = [p.strip() for p in s.split("&") if p.strip()]

    lows: Dict[int, float] = {}
    highs: Dict[int, float] = {}

    le_var_pat = re.compile(rf"^\s*({_num})\s*<=\s*{re.escape(var_prefix)}(\d+)\s*$")
    var_le_pat = re.compile(rf"^\s*{re.escape(var_prefix)}(\d+)\s*<=\s*({_num})\s*$")

    for p in parts:
        m = le_var_pat.match(p)
        if m:
            lo = _num_sanitize(m.group(1))
            idx = int(m.group(2))
            lows[idx] = lo
            continue
        m = var_le_pat.match(p)
        if m:
            idx = int(m.group(1))
            hi = _num_sanitize(m.group(2))
            highs[idx] = hi
            continue

    idxs = set(lows.keys()) | set(highs.keys())
    out: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for i in idxs:
        out[i] = (lows.get(i, None), highs.get(i, None))
    return out


# ------------------------- flow parsing -------------------------

def _infer_n_from_flow(flow_text: str) -> int:
    """
    Infer state dimension from flow equations by scanning LHS 'xk' and RHS 'xj' usages.
    Returns max index + 1, or 0 if nothing found.
    """
    s = (flow_text or "").replace("&amp;", "&")
    idxs = []

    # LHS: xk' ==
    for m in re.finditer(r"\bx(\d+)'\s*==", s):
        idxs.append(int(m.group(1)))

    # RHS: references to xj
    for m in re.finditer(r"\bx(\d+)\b", s):
        idxs.append(int(m.group(1)))

    return (max(idxs) + 1) if idxs else 0


def _parse_flow(flow_text: str, n: int):
    """
    Parse flow lines like:
      x0'==a00*x0 + a01*x1 + ... + b0 [+ u0]
    Returns (M, m0, uses_u: bool)
    If flow_text is empty/missing, returns zero dynamics of size n (or (1) if n<1).
    """
    uses_u = False
    n = max(1, n)
    M = np.zeros((n, n), dtype=float)
    m0 = np.zeros((n,), dtype=float)

    s_all = (flow_text or "").replace("&amp;", "&").strip()
    if not s_all:
        return M, m0, uses_u

    for eq in s_all.split("&"):
        eq = eq.strip()
        if not eq:
            continue
        m = re.match(r"^\s*x(\d+)'\s*==\s*(.*)$", eq)
        if not m:
            # ignore lines like "t'==1"
            continue
        i = int(m.group(1))
        rhs = m.group(2).strip()

        # if LHS index exceeds current n, grow arrays
        if i >= n:
            newn = i + 1
            M = np.pad(M, ((0, newn - n), (0, newn - n)), mode="constant")
            m0 = np.pad(m0, (0, newn - n), mode="constant")
            n = newn

        # normalize adjacent signs to avoid "+-c"
        rhs = re.sub(r"\+\s*-\s*", "-", rhs)
        rhs = re.sub(r"-\s*\+\s*", "-", rhs)
        rhs = re.sub(r"\+\s*\+\s*", "+", rhs)

        # detect and drop "+ u_i" (if present) but remember it's used
        if re.search(rf"\bu{i}\b", rhs):
            uses_u = True
            rhs = re.sub(rf"\+?\s*u{i}\b", "", rhs)

        # sum coefficients for xj terms; if j >= n, grow arrays
        for t in re.finditer(rf"([+-]?\s*{_num})\s*\*\s*x(\d+)\b", rhs):
            a = _num_sanitize(t.group(1))
            j = int(t.group(2))
            if j >= n:
                newn = j + 1
                M = np.pad(M, ((0, newn - n), (0, newn - n)), mode="constant")
                m0 = np.pad(m0, (0, newn - n), mode="constant")
                n = newn
            M[i, j] += a

        # remove all "a*xj" parts, remaining numbers are constants to sum
        tmp = re.sub(rf"[+-]?\s*{_num}\s*\*\s*x\d+\b", "", rhs)
        consts = re.findall(rf"([+-]?\s*{_num})", tmp)
        m0[i] = sum(_num_sanitize(c) for c in consts) if consts else 0.0

    return M, m0, uses_u


# ------------------------- geometry helpers -------------------------

def _hyperrectangle_vertices(bounds: Dict[int, Tuple[Optional[float], Optional[float]]], dims: List[int]):
    """
    Given bounds dict (lo,hi) and a list of dims to include, returns list of vertices (len=dims).
    If any chosen dim has None lo/hi, returns empty list (cannot enumerate unbounded).
    """
    intervals = []
    for d in dims:
        lo, hi = bounds.get(d, (None, None))
        if lo is None or hi is None:
            return []  # unbounded -> cannot enumerate finite vertices
        intervals.append([lo, hi])
    verts = []
    for corners in itertools.product(*intervals):
        verts.append(np.array(corners, dtype=float))
    return verts


def _affine_one_step_vertices(M, m0, verts_x, u_box, dt):
    """
    Compute images: x+ = x + dt*(M x + m_0) + dt*u, with u in u_box (hyperrectangle).
    If u_box is None, u = 0.
    Returns a list of image points (not convex-hulled).
    """
    pts = []
    n = len(m0)

    # corners for u
    if u_box is None:
        u_corners = [np.zeros(n)]
    else:
        # if any dimension unbounded -> just use zero (conservative plotting)
        if any((u_box.get(i, (0.0, 0.0))[0] is None or u_box.get(i, (0.0, 0.0))[1] is None) for i in range(n)):
            u_corners = [np.zeros(n)]
        else:
            u_corners = _hyperrectangle_vertices(u_box, list(range(n)))
            if not u_corners:
                u_corners = [np.zeros(n)]

    for x_full in verts_x:
        x_full = np.asarray(x_full, dtype=float)
        # ensure length n
        if len(x_full) != n:
            continue
        for u in u_corners:
            xdot = M @ x_full + m0 + u
            pts.append(x_full + dt * xdot)
    return pts


def _polygon_from_points_xy(points_xy):
    """
    Return 2D polygon (x0,x1) as Nx2 array in convex hull order, or None if too few points.
    """
    if len(points_xy) < 3:
        return None
    P = np.unique(np.asarray(points_xy), axis=0)
    if len(P) < 3:
        return None
    # sort by x then y
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


# ------------------------- parsing: XML + CFG -------------------------

@dataclass
class LocationData:
    name: str
    inv_text: str
    flow_text: str
    x_bounds: Dict[int, Tuple[Optional[float], Optional[float]]]
    u_bounds: Dict[int, Tuple[Optional[float], Optional[float]]]
    M: np.ndarray
    m0: np.ndarray
    uses_u: bool


def parse_spaceex_xml(xml_path: str) -> List[LocationData]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"sx": "http://www-verimag.imag.fr/xml-namespaces/sspaceex"}

    locs = []
    for loc in root.findall(".//sx:location", ns):
        name = loc.get("name", "")

        inv_node = loc.find("sx:invariant", ns)
        flow_node = loc.find("sx:flow", ns)

        inv_text = inv_node.text if (inv_node is not None and inv_node.text) else ""
        flow_text = flow_node.text if (flow_node is not None and flow_node.text) else ""

        # parse bounds
        xb = _parse_bounds_from_invariant(inv_text, "x")
        ub = _parse_bounds_from_invariant(inv_text, "u")

        # infer n from BOTH invariant and flow (take the max)
        n_inv = (max(xb.keys()) + 1) if xb else 0
        n_flow = _infer_n_from_flow(flow_text)
        n = max(n_inv, n_flow, 1)

        # parse flow with the correct n
        M, m0, uses_u = _parse_flow(flow_text, n)

        # if there is no x-flow at all (e.g., only t'==1), skip plotting this loc
        if M.size == 0 or m0.size == 0:
            continue

        locs.append(LocationData(
            name=name, inv_text=inv_text, flow_text=flow_text,
            x_bounds=xb, u_bounds=ub, M=M, m0=m0, uses_u=uses_u
        ))
    return locs


@dataclass
class CfgData:
    sampling_time: float
    time_horizon: float
    initially: str
    forbidden: str


def parse_cfg(cfg_path: str) -> CfgData:
    with open(cfg_path, "r") as f:
        text = f.read()

    def get_float(key, default):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, flags=re.MULTILINE)
        if not m:
            return default
        try:
            return float(m.group(1).strip().strip("'\""))
        except:
            return default

    def get_str_in_quotes(key, default=""):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*\"([^\"]*)\"", text, flags=re.MULTILINE)
        return m.group(1) if m else default

    sampling = get_float("sampling-time", 0.01)
    horizon = get_float("time-horizon", 1.0)
    initially = get_str_in_quotes("initially", "")
    forbidden = get_str_in_quotes("forbidden", "")
    return CfgData(sampling, horizon, initially, forbidden)


def parse_box_from_constraint_str(constr: str,
                                  var_prefix: str) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Reuse invariant parser on the quoted strings from the cfg (same syntax)."""
    return _parse_bounds_from_invariant(constr, var_prefix)


# ------------------------- main visualization logic -------------------------

def visualize(xml_path: str, cfg_path: str, save_path: Optional[str] = None):
    locs = parse_spaceex_xml(xml_path)
    cfg = parse_cfg(cfg_path)

    # Initial & forbidden boxes (x only)
    init_box = parse_box_from_constraint_str(cfg.initially, "x")
    forb_box = parse_box_from_constraint_str(cfg.forbidden, "x")

    dt = cfg.sampling_time

    # Prepare plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("SpaceEx polytopes and one-step images (x0–x1)")

    # consistent colors per location
    tableau = list(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    if not tableau:
        tableau = [f"C{i}" for i in range(10)]

    # Plot per location
    for li, loc in enumerate(locs):
        xb = loc.x_bounds
        if not xb:
            continue

        # determine n from flow matrix
        n = loc.M.shape[0]
        dims_all = list(range(n))

        # if any x-dim is unbounded -> cannot enumerate vertices; skip plotting for this loc
        unbounded = False
        for d in dims_all:
            lo, hi = xb.get(d, (None, None))
            if lo is None or hi is None:
                if d < 2:
                    print(f"[warn] location '{loc.name}' has x{d} unbounded; skipping its polygon.")
                unbounded = True
                break
        if unbounded:
            continue

        # Full-dim vertices for x
        x_full_verts = _hyperrectangle_vertices(xb, dims_all)
        if not x_full_verts:
            print(f"[info] location '{loc.name}': cannot enumerate vertices (unbounded); skipping.")
            continue

        # u-box (n dims) for image computation; if +u_i appears, use u-bounds from invariant; else u=0
        if loc.uses_u:
            u_box: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
            for k in range(n):
                if k in loc.u_bounds:
                    u_box[k] = loc.u_bounds[k]
                else:
                    u_box[k] = (0.0, 0.0)
        else:
            u_box = {k: (0.0, 0.0) for k in range(n)}

        # images for all corner combinations of x and u
        x_plus_pts = _affine_one_step_vertices(loc.M, loc.m0, x_full_verts, u_box, dt)

        # Build 2D polygons (x0,x1)
        poly_xy = _polygon_from_points_xy([(v[0], v[1]) for v in x_full_verts])
        img_xy = _polygon_from_points_xy([(p[0], p[1]) for p in x_plus_pts])

        # pick a consistent color for this location
        color = tableau[li % len(tableau)]

        # Plot state polytope (same hue)
        if poly_xy is not None:
            ax.fill(poly_xy[:, 0], poly_xy[:, 1], facecolor=color, edgecolor=color, alpha=0.25,
                    label=f"{loc.name} polytope")
            ax.plot(poly_xy[:, 0], poly_xy[:, 1], color=color, linewidth=1.2)

        # Plot image polytope (same hue, lighter + dashed edge)
        if img_xy is not None:
            ax.fill(img_xy[:, 0], img_xy[:, 1], facecolor=color, edgecolor=color, alpha=0.12)
            ax.plot(img_xy[:, 0], img_xy[:, 1], color=color, linestyle="--", linewidth=1.2,
                    label=f"{loc.name} image")

    # Plot initial rectangle (if finite x0,x1)
    if init_box and 0 in init_box and 1 in init_box:
        x0b, x1b = init_box[0], init_box[1]
        if all(b is not None for b in x0b) and all(b is not None for b in x1b):
            x0_lo, x0_hi = x0b
            x1_lo, x1_hi = x1b
            init_rect = np.array([[x0_lo, x1_lo], [x0_hi, x1_lo], [x0_hi, x1_hi], [x0_lo, x1_hi]])
            ax.fill(init_rect[:, 0], init_rect[:, 1], facecolor="black", alpha=0.12, label="initial")
            ax.plot(init_rect[:, 0], init_rect[:, 1], color="black", linewidth=1.2)

    # Plot forbidden rectangle (if finite x0,x1)
    if forb_box and 0 in forb_box and 1 in forb_box:
        x0b, x1b = forb_box[0], forb_box[1]
        if all(b is not None for b in x0b) and all(b is not None for b in x1b):
            x0_lo, x0_hi = x0b
            x1_lo, x1_hi = x1b
            forb_rect = np.array([[x0_lo, x1_lo], [x0_hi, x1_lo], [x0_hi, x1_hi], [x0_lo, x1_hi]])
            ax.fill(forb_rect[:, 0], forb_rect[:, 1], facecolor="red", alpha=0.15, label="forbidden")
            ax.plot(forb_rect[:, 0], forb_rect[:, 1], color="red", linewidth=1.2)

    # Avoid duplicate legend entries by using unique labels
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h); uniq_l.append(l); seen.add(l)
    ax.legend(uniq_h, uniq_l, loc="best")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Visualize SpaceEx location polytopes and one-step images (x0–x1)")
    ap.add_argument("--xml", required=True, help="SpaceEx model XML")
    ap.add_argument("--cfg", required=True, help="SpaceEx CFG")
    ap.add_argument("--out", default=None, help="Optional path to save the PNG")
    args = ap.parse_args()
    visualize(args.xml, args.cfg, args.out)
