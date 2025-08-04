import json

def convert_json_to_traj(json_path, output_path="traj.txt"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    lines = []
    for frame in data["frames"]:
        matrix = frame["transform_matrix"]
        flat = [f"{v:.18e}" for row in matrix for v in row]
        line = " ".join(flat)
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {len(lines)} poses to '{output_path}'.")

# 示例用法
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert transform matrices JSON to traj.txt")
    parser.add_argument("json_path", help="Path to JSON file (e.g. transforms.json)")
    parser.add_argument("--output", default="traj.txt", help="Output file name")
    args = parser.parse_args()

    convert_json_to_traj(args.json_path, args.output)
