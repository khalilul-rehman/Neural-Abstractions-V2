
# import subprocess

# # Define the SpaceEx command as a list of arguments
# cmd = [
#     sapceex_path + "spaceex",
#     "--model-file", model_xml_path,
#     "--rel-err", "1.0E-12",
#     "--abs-err", "1.0E-15",
#     "--output-error", "0",
#     "--scenario", "stc",
#     "--directions", "oct",
#     "--set-aggregation", "none",
#     "--verbosity", "m",
#     "--time-horizon", "1.5",
#     "--sampling-time", "1",
#     "--simu-init-sampling-points", "0",
#     "--flowpipe-tolerance", "0.01",
#     "--flowpipe-tolerance-rel", "0",
#     "--iter-max", "-1",
#     "--initially", "x0>=0.45 & x0<=0.5 & x1>=-0.6 & x1<=-0.55 & u0==0 & u1==0 & t==0",
#     "--forbidden", "x0>=0.3 & x0<=0.35 & x1>=0.5 & x1<=0.6",
#     "--system", "NA",
#     "--output-format", "TXT",
#     "--output-variables", "x0,x1",
#     "--output-file", "spaceex.gen"
# ]

# # Run the command and capture stdout and stderr
# result = subprocess.run(cmd, capture_output=True, text=True)

# # stdout and stderr are now available as strings
# output_text = result.stdout
# error_text = result.stderr

# # Print the output if needed
# print("SpaceEx Output:\n", output_text)
# print("SpaceEx Errors:\n", error_text)

# # Example: check if forbidden states were reachable
# if "Forbidden states are reachable" in output_text:
#     forbidden_reached = True
# else:
#     forbidden_reached = False

# print("Forbidden reached?", forbidden_reached)


import subprocess
import os


def run_spaceex(
    model_file,
    system,
    initially,
    forbidden,
    output_file="spaceex.gen",
    output_vars="x0,x1",
    output_format="TXT",
    time_horizon=1.5,
    scenario="stc",
    directions="oct",
    rel_err="1.0E-12",
    abs_err="1.0E-15",
    flowpipe_tol="0.01",
    iter_max="-1",
    folder_model_xml_path = r"/Users/khalilulrehman/Academic/Phd Italy 2023_26/University of LAquila/Research Papers tasks/Benchmarks_Systems/Robust Control for Dynamical Systems With Non-Gaussian Noise via Formal Abstractions/Neural Abstractions V2/SpaceEx/experiments/"  # Path to your SpaceEx model file
):
    """
    Runs SpaceEx from Python and returns:
    1. Terminal output as a string
    2. Output file contents as a list of polygons (list of lists of [x, y])
    """

    sapceex_path = r"/Users/khalilulrehman/ExtraToolsInstalledHere/spaceex_exe_osx2/"  # Update this path to your SpaceEx installation if needed
    


    if not os.path.exists(folder_model_xml_path + model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Construct the SpaceEx command
    cmd = [
        sapceex_path + "spaceex",
        "--model-file", folder_model_xml_path + model_file,
        "--rel-err", rel_err,
        "--abs-err", abs_err,
        "--output-error", "0",
        "--scenario", scenario,
        "--directions", directions,
        "--set-aggregation", "none",
        "--verbosity", "m",
        "--time-horizon", str(time_horizon),
        "--sampling-time", "1",
        "--simu-init-sampling-points", "0",
        "--flowpipe-tolerance", flowpipe_tol,
        "--flowpipe-tolerance-rel", "0",
        "--iter-max", iter_max,
        "--initially", initially,
        "--forbidden", forbidden,
        "--system", system,
        "--output-format", output_format,
        "--output-variables", output_vars,
        "--output-file", folder_model_xml_path + output_file
    ]

    # Run SpaceEx and capture stdout/stderr
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Terminal output
    output_text = result.stdout
    error_text = result.stderr

    # Read output file if it exists
    polygons = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            current_poly = []
            for line in f:
                line = line.strip()
                if not line:
                    if current_poly:
                        polygons.append(current_poly)
                        current_poly = []
                    continue
                # Each line is expected to be "x y"
                try:
                    coords = list(map(float, line.split()))
                    current_poly.append(coords)
                except ValueError:
                    continue
            if current_poly:
                polygons.append(current_poly)

    # Check if forbidden states were reachable
    forbidden_reached = True if "Forbidden states are reachable" in output_text else False
    
    
    return forbidden_reached, output_text, error_text, polygons


if __name__ == "__main__":
    # Example usage
    model_file = "jet_spaceex.xml"  # Update with your model file
    system = "Jet_NA"
    initially = "x0>=0.45 & x0<=0.5 & x1>=-0.6 & x1<=-0.55 & u0==0 & u1==0 & t==0"
    forbidden = "x0>=0.3 & x0<=0.35 & x1>=0.5 & x1<=0.6"

    forbidden_reached, output_text, error_text, polygons = run_spaceex(
        model_file=model_file,
        system=system,
        initially=initially,
        forbidden=forbidden
    )

    print("SpaceEx Output:\n", output_text)
    print("SpaceEx Errors:\n", error_text)
    print(f"Number of polygons in output: {len(polygons)}")
    for i, poly in enumerate(polygons):
        print(f"Polygon {i+1}: {poly}")

    print("Forbidden state values reached?", "Yes" if forbidden_reached else "No")