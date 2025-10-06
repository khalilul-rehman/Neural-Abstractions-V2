# cof_to_spaceex_v098f_safe.py
import numpy as np
from xml.sax.saxutils import escape
import os
import math


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return ("{:.12g}".format(float(v))).replace("nan", "0")


def xml_escape(s: str):
    return escape(s, {"<": "&lt;", ">": "&gt;"})


def poly_box_to_constr(bounds, var_names):
    cons = []
    for (mn, mx), var in zip(bounds, var_names):
        cons.append(f"{fmt(mn)} <= {var}")
        cons.append(f"{var} <= {fmt(mx)}")
    return " & ".join(cons)


def matrix_to_flow(M, m0, var_names):
    flows = []
    for i, var in enumerate(var_names):
        terms = []
        for j, var2 in enumerate(var_names):
            a = float(M[i][j])
            if abs(a) > 1e-15:
                terms.append(f"{fmt(a)}*{var2}")
        # build RHS correctly whether or not there are linear terms
        if terms:
            rhs = " + ".join(terms) + f" + {fmt(m0[i])}"
        else:
            rhs = f"{fmt(m0[i])}"
        flows.append(f"{var}' == {rhs}")
    return " & ".join(flows)


def write_spaceex_cfg(cfg_path, system_name, state_var_names, initial_box, tau_max):
    init_constr = poly_box_to_constr(initial_box, state_var_names)
    cfg_text = f"""system = {system_name}
initially = "{init_constr}"
scenario = "phaver"
directions = "oct"
sampling-time = 0.01
time-horizon = {tau_max}
iter-max = 100
output-variables = "{', '.join(state_var_names)}"
output-format = "GEN"
"""
    with open(cfg_path, "w") as f:
        f.write(cfg_text)
    print(f"Wrote SpaceEx CFG to {cfg_path}")


def build_spaceex_xml_from_cof(
    cof_list,
    transitions,
    outpath="pwa_spaceex.xml",
    state_var_names=None,
    tau_max=1.0,
    initial_box=None,
    unsafe_box=None,
    system_name="System_from_COF"
):
    if len(cof_list) == 0:
        raise ValueError("Empty COF list")

    first_bounds = cof_list[0]['bounds']
    feature_key_order = sorted(
        list(first_bounds.keys()),
        key=lambda s: int(''.join(filter(str.isdigit, s)) or -1)
    )

    regions = []
    for item in cof_list:
        bdict = item['bounds']
        bounds = [[float(bdict[k][0]), float(bdict[k][1])] for k in feature_key_order]
        M = np.array(item['CO_Model']['M'], dtype=float).tolist()
        m0 = np.array(item['CO_Model']['m0'], dtype=float).tolist()
        regions.append({
            'name': f"leaf_{int(item['leaf_id'])}",
            'bounds': bounds,
            'M': M,
            'm0': m0
        })

    dim = len(regions[0]['bounds'])
    if state_var_names is None:
        state_var_names = [f"x{i}" for i in range(dim)]

    # Header
    header = f"""<?xml version="1.0" encoding="UTF-8"?>
<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">
  <component id="{escape(system_name)}">
"""
    for v in state_var_names:
        header += f'    <param name="{escape(v)}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'
    # Label for transitions
    header += '    <param name="tau" type="label" local="true"/>\n'

    # Locations with compact automatic coordinates
    body = ""
    num_regions = len(regions)
    max_cols = min(10, num_regions)  # compact grid
    x_spacing = 200
    y_spacing = 150
    width = 150
    height = 60

    for idx, reg in enumerate(regions):
        lname = escape(reg['name'])
        col = idx % max_cols
        row = idx // max_cols
        x_coord = col * x_spacing
        y_coord = row * y_spacing
        body += f'    <location id="loc{idx}" name="{lname}" x="{x_coord}" y="{y_coord}" width="{width}" height="{height}">\n'
        inv_box = poly_box_to_constr(reg['bounds'], state_var_names)
        body += f"      <invariant>{xml_escape(inv_box)}</invariant>\n"
        flow_terms = matrix_to_flow(reg['M'], reg['m0'], state_var_names)
        body += f"      <flow>{xml_escape(flow_terms)}</flow>\n"
        body += "    </location>\n"

    # Transitions
    trans_arr = np.array(transitions, dtype=int)
    L = len(regions)
    for i in range(L):
        for j in range(L):
            if trans_arr[i, j] == 0:
                continue
            body += f"""    <transition source="loc{i}" target="loc{j}">
      <label>tau</label>
    </transition>
"""

    # Initial
    init_box = initial_box if initial_box is not None else regions[0]['bounds']
    init_constr = poly_box_to_constr(init_box, state_var_names)
    init_xml = f"""    <initially>
      <constr>{xml_escape(init_constr)}</constr>
      <loc>loc0</loc>
    </initially>
"""

    # Forbidden
    forbidden_xml = ""
    if unsafe_box is not None:
        unsafe_poly = poly_box_to_constr(unsafe_box, state_var_names)
        forbidden_xml = f"""    <forbidden>
      <constr>{xml_escape(unsafe_poly)}</constr>
    </forbidden>
"""

    footer = """  </component>
</sspaceex>
"""

    xml = header + body + init_xml + forbidden_xml + footer
    with open(outpath, "w") as f:
        f.write(xml)
    print(f"Wrote SpaceEx XML to {outpath} (L={len(regions)}, dim={dim})")

    # CFG
    cfg_path = os.path.splitext(outpath)[0] + ".cfg"
    write_spaceex_cfg(cfg_path, system_name, state_var_names, init_box, tau_max)
    return outpath, cfg_path


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    cof_list = [
        {
            'leaf_id': 0,
            'bounds': {'x0': [0, 1], 'x1': [0, 1]},
            'CO_Model': {'M': [[-1, 0], [0, -1]], 'm0': [0, 0]}
        },
        {
            'leaf_id': 1,
            'bounds': {'x0': [1, 2], 'x1': [0, 1]},
            'CO_Model': {'M': [[-0.5, 0], [0, -0.5]], 'm0': [0, 0]}
        }
    ]

    transition_matrix = [[0, 1], [0, 0]]
    initial_box = [[0.2, 0.3], [0.2, 0.3]]
    unsafe_box = [[1.5, 1.8], [0.2, 0.4]]
    tau_max = 2.0

    build_spaceex_xml_from_cof(
        cof_list=cof_list,
        transitions=transition_matrix,
        outpath="toy_system_v098f_safe.xml",
        state_var_names=["x", "y"],
        tau_max=tau_max,
        initial_box=initial_box,
        unsafe_box=unsafe_box,
        system_name="ToySystem_v098f_safe"
    )
