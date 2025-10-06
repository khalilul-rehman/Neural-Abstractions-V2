from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
import math
import os
from xml.sax.saxutils import escape

from CustomAbstraction.CustomAbstractionHelper import StateModel

# ----------------- Small helpers -----------------
def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return ("{:.12g}".format(float(v))).replace("nan", "0")

def _xml(s: str) -> str:
    # IMPORTANT: escape &, <, >
    # This ensures " & " becomes "&amp;" in XML text nodes.
    return escape(s)  # default escapes &, <, >

def _ordered_dims(bounds: Dict[int, Tuple[float, float]]) -> List[int]:
    return sorted(bounds.keys())

def _bounds_to_box_constr(bounds: Dict[int, Tuple[float, float]], var_names: List[str], sep=" & "):
    parts = []
    for d in _ordered_dims(bounds):
        lo, hi = bounds[d]
        v = var_names[d]
        parts.append(f"{_fmt(lo)} <= {v}")
        parts.append(f"{v} <= {_fmt(hi)}")
    return sep.join(parts)

def _matrix_to_flow(M: np.ndarray, m0: np.ndarray, xvars: List[str],
                    uvars: Optional[List[str]] = None, sep=" & "):
    M = np.asarray(M, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    n = len(xvars)
    flows = []
    for i in range(n):
        terms = []
        for j in range(n):
            a = M[i, j]
            if abs(a) > 1e-15:
                terms.append(f"{_fmt(a)}*{xvars[j]}")
        rhs = " + ".join(terms) if terms else "0"
        rhs += f" + {_fmt(m0[i])}"
        if uvars is not None:
            rhs += f" + {uvars[i]}"
        flows.append(f"{xvars[i]}'=={rhs}")
    return sep.join(flows)

def _u_bounds_sqrt_h(h_scalar: Optional[float], uvars: List[str], sep=" & "):
    if h_scalar is None or h_scalar < 0:
        return ""
    hbar = math.sqrt(h_scalar)
    parts = []
    for ui in uvars:
        parts.append(f"-{_fmt(hbar)} <= {ui}")
        parts.append(f"{ui} <= {_fmt(hbar)}")
    return sep.join(parts)

def _boxes_intersect(boxA: List[Tuple[float, float]], boxB: List[Tuple[float, float]]) -> bool:
    # axis-aligned box intersection
    for (a_lo, a_hi), (b_lo, b_hi) in zip(boxA, boxB):
        if a_hi < b_lo or b_hi < a_lo:
            return False
    return True

# ----------------- Builders (flow, invariant, guard, location, transition) -----------------
def build_flow_text(state: StateModel, xvars: List[str], uvars: Optional[List[str]], bounded_time: bool):
    txt = _matrix_to_flow(state.M, state.m0, xvars, uvars)
    if bounded_time:
        txt = txt + " & t'==1"
    return txt

def build_invariant_text(state: StateModel, xvars: List[str], uvars: Optional[List[str]], bounded_time: bool, T: float):
    # x-box always present
    box = _bounds_to_box_constr(state.bounds, xvars)
    parts = [box]
    # u-bounds with sqrt(h)
    if uvars:
        ub = _u_bounds_sqrt_h(state.h, uvars)
        if ub:
            parts.append(ub)
    if bounded_time:
        parts.append(f"t <= {_fmt(T)}")
    return " & ".join(parts)

def build_guard_text(src: StateModel, dst: StateModel, xvars: List[str], bounded_time: bool, T: float):
    # target's x-only invariant + time cap
    g = _bounds_to_box_constr(dst.bounds, xvars)
    if bounded_time:
        g = g + f" & t <= {_fmt(T)}"
    return g

def build_location_xml(loc_id: str, name: str, flow_text: str, inv_text: str,
                       x: int, y: int, w: int = 150, h: int = 60) -> str:
    return (
        f'  <location id="{loc_id}" name="{escape(name)}" x="{x}" y="{y}" width="{w}" height="{h}">\n'
        f"    <invariant>{_xml(inv_text)}</invariant>\n"
        f"    <flow>{_xml(flow_text)}</flow>\n"
        f"  </location>\n"
    )

def build_transition_xml(src_id: str, dst_id: str, guard_text: Optional[str] = None,
                         label: Optional[str] = None) -> str:
    lab = f"    <label>{escape(label)}</label>\n" if label else ""
    grd = f"    <guard>{_xml(guard_text)}</guard>\n" if guard_text else ""
    return (
        f'  <transition source="{src_id}" target="{dst_id}">\n'
        f"{lab}{grd}"
        f"  </transition>\n"
    )

# ----------------- Writers: XML + CFG -----------------
def write_spaceex_xml(
    states: List[StateModel],
    out_xml_path: str,
    system_name: str = "NA_like_system",
    bounded_time: bool = True,              # you asked for exact NA pattern
    T: float = 1.5,
    initial_box: Optional[List[Tuple[float, float]]] = None,   # [(lo,hi)]*n
    unsafe_box: Optional[List[Tuple[float, float]]] = None,
    add_label_param: bool = False,
    init_mode: str = "init_location"        # "init_location" (NA-like) | "global_initially"
):
    if not states:
        raise ValueError("No states provided")

    # infer dimension and var names
    n = states[0].M.shape[0]
    xvars = [f"x{i}" for i in range(n)]
    use_u = any(s.h is not None for s in states)
    uvars = [f"u{i}" for i in range(n)] if use_u else None

    # header + params
    header = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">',
        f'  <component id="{escape(system_name)}">'
    ]
    for xv in xvars:
        header.append(f'    <param name="{xv}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>')
    if use_u:
        for uv in uvars:
            header.append(f'    <param name="{uv}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="false"/>')
    if bounded_time:
        header.append('    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>')
    if add_label_param:
        header.append('    <param name="tau" type="label" local="true"/>')
    header = "\n".join(header) + "\n"

    # locations
    body = []
    L = len(states)
    max_cols = min(10, L)
    x_spacing, y_spacing = 220, 150
    w, h = 160, 64
    id_map = {}  # state_identifier -> "loc{k}"

    for k, st in enumerate(states):
        loc_id = f"loc{k}"
        id_map[st.state_identifier] = loc_id
        col, row = k % max_cols, k // max_cols
        x, y = col * x_spacing, row * y_spacing
        flow = build_flow_text(st, xvars, uvars, bounded_time)
        inv = build_invariant_text(st, xvars, uvars, bounded_time, T)
        body.append(build_location_xml(loc_id, f"state_{st.state_identifier}", flow, inv, x, y, w, h))

    # End location (bounded time)
    end_id = None
    if bounded_time:
        end_id = f"loc{L}"
        # flow: x'==0 & t'==0 ; invariant: t >= T
        end_flow = " & ".join([f"{xv}'==0" for xv in xvars] + ["t'==0"])
        end_inv  = f"t >= {_fmt(T)}"
        body.append(
            f'  <location id="{end_id}" name="End" x="{(L%max_cols)*x_spacing}" y="{(L//max_cols)*y_spacing}" width="{w}" height="{h}">\n'
            f"    <invariant>{_xml(end_inv)}</invariant>\n"
            f"    <flow>{_xml(end_flow)}</flow>\n"
            f"  </location>\n"
        )

    # Init location (NA-like)
    init_xml = ""
    init_loc_id = None
    if initial_box is not None and init_mode == "init_location":
        init_loc_id = f"loc{L + (1 if bounded_time else 0)}"
        init_flow = " & ".join([f"{xv}'==0" for xv in xvars] + (["t'==1"] if bounded_time else []))
        # (NA often left Init inv empty; we keep it empty to mirror)
        body.append(
            f'  <location id="{init_loc_id}" name="Init" x="0" y="{(L//max_cols+1)*y_spacing}" width="{w}" height="{h}">\n'
            f"    <flow>{_xml(init_flow)}</flow>\n"
            f"  </location>\n"
        )
        # <initially> referencing Init with the initial box
        init_box_dict = {i: tuple(initial_box[i]) for i in range(len(initial_box))}
        init_constr = _bounds_to_box_constr(init_box_dict, xvars)
        init_xml = (
            "  <initially>\n"
            f"    <constr>{_xml(init_constr)}</constr>\n"
            f"    <loc>{init_loc_id}</loc>\n"
            "  </initially>\n"
        )

    # transitions (between normal states)
    trans = []
    for st in states:
        src = id_map[st.state_identifier]
        for tgt_id in st.transition_to:
            if tgt_id not in id_map:
                continue
            dst = id_map[tgt_id]
            # if the souce and target are the same, skip self-loop
            if src == dst:
                continue
            guard = build_guard_text(st, next(s for s in states if s.state_identifier == tgt_id),
                                     xvars, bounded_time, T)
            trans.append(build_transition_xml(src, dst, guard_text=guard, label=("tau" if add_label_param else None)))
        if bounded_time and end_id:
            trans.append(build_transition_xml(src, end_id, guard_text=f"t >= {_fmt(T)}"))

    # transitions from Init -> states whose x-box intersects initial_box
    if initial_box is not None and init_mode == "init_location" and init_loc_id is not None:
        ibox = list(initial_box)  # [(lo,hi)]*n
        for st in states:
            sb = [st.bounds[i] for i in range(len(ibox))]
            if _boxes_intersect(sb, ibox):
                trans.append(build_transition_xml(init_loc_id, id_map[st.state_identifier], guard_text=None))

    # Optional: global <initially> without Init location
    if initial_box is not None and init_mode == "global_initially":
        init_box_dict = {i: tuple(initial_box[i]) for i in range(len(initial_box))}
        init_constr = _bounds_to_box_constr(init_box_dict, xvars)
        # start in the first state's location by default
        init_xml = (
            "  <initially>\n"
            f"    <constr>{_xml(init_constr)}</constr>\n"
            f"    <loc>{id_map[states[0].state_identifier]}</loc>\n"
            "  </initially>\n"
        )

    # forbidden (unsafe) region
    forb_xml = ""
    if unsafe_box is not None:
        unsafe_dict = {i: tuple(unsafe_box[i]) for i in range(len(unsafe_box))}
        forb_constr = _bounds_to_box_constr(unsafe_dict, xvars)
        forb_xml = (
            "  <forbidden>\n"
            f"    <constr>{_xml(forb_constr)}</constr>\n"
            "  </forbidden>\n"
        )

    footer = "  </component>\n</sspaceex>\n"
    xml_text = header + "".join(body) + "".join(trans) + init_xml + forb_xml + footer

    with open(out_xml_path, "w") as f:
        f.write(xml_text)

    return out_xml_path

def write_spaceex_cfg(
    cfg_path: str,
    system_name: str,
    state_var_names: List[str],
    initial_box: Optional[List[Tuple[float, float]]],
    time_horizon: float,
    scenario: str = "phaver",
    directions: str = "oct",
    sampling_time: float = 0.01,
    iter_max: int = 100,
    output_format: str = "GEN"
):
    # NOTE: CFG is not XML, so keep plain "&" (SpaceEx expects that here).
    if initial_box is None:
        initially = ""
    else:
        init_dict = {i: tuple(initial_box[i]) for i in range(len(initial_box))}
        initially = _bounds_to_box_constr(init_dict, [f"x{i}" for i in range(len(initial_box))], sep=" & ")
    cfg = f"""system = {system_name}
initially = "{initially}"
scenario = "{scenario}"
directions = "{directions}"
sampling-time = {sampling_time}
time-horizon = {time_horizon}
iter-max = {iter_max}
output-variables = "{', '.join(state_var_names)}"
output-format = "{output_format}"
"""
    with open(cfg_path, "w") as f:
        f.write(cfg)
    return cfg_path

# --------------- Convenience wrapper ---------------
def export_spaceex(
    states: List[StateModel],
    out_xml_path: str,
    system_name: str = "NA_like_system",
    T: float = 1.5,
    initial_box: Optional[List[Tuple[float, float]]] = None,
    unsafe_box: Optional[List[Tuple[float, float]]] = None,
    init_mode: str = "init_location"  # "init_location" (NA-like) or "global_initially"
):
    xml_path = write_spaceex_xml(
        states=states,
        out_xml_path=out_xml_path,
        system_name=system_name,
        bounded_time=True,
        T=T,
        initial_box=initial_box,
        unsafe_box=unsafe_box,
        add_label_param=False,
        init_mode=init_mode
    )
    n = states[0].M.shape[0]
    cfg_path = os.path.splitext(out_xml_path)[0] + ".cfg"
    write_spaceex_cfg(
        cfg_path=cfg_path,
        system_name=system_name,
        state_var_names=[f"x{i}" for i in range(n)],
        initial_box=initial_box,          # cfg-level initial set
        time_horizon=T
    )
    return xml_path, cfg_path
