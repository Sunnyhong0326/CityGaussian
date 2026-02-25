# -*- coding: utf-8 -*-
# Export LoBEGS-format block_bounds.json for CityGaussian uniform partitions.

import argparse
import json
import os

import numpy as np
import yaml


def _load_block_dim_and_source(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_params = cfg.get("model_params", {})
    block_dim = model_params.get("block_dim", None)
    source_path = model_params.get("source_path", None)
    if block_dim is None:
        raise ValueError("Missing model_params.block_dim in config.")
    if source_path is None:
        raise ValueError("Missing model_params.source_path in config.")
    return block_dim, source_path


def _compute_uniform_segments(block_dim, plane_indices):
    v_ax, h_ax, _ = plane_indices
    dims = [block_dim[0], block_dim[1], block_dim[2]]
    v_dim = dims[v_ax]
    h_dim = dims[h_ax]

    segments = []
    if v_dim > 1:
        for i in range(1, v_dim):
            v = float(i) / v_dim
            segments.append([v, 0.0, v, 1.0])
    if h_dim > 1:
        for j in range(1, h_dim):
            h = float(j) / h_dim
            segments.append([0.0, h, 1.0, h])
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Export LoBEGS-format block_bounds.json for CityGaussian uniform partitions."
    )
    parser.add_argument("--config", required=True, help="CityGaussian config yaml.")
    parser.add_argument("--output_dir", default="", help="Output directory (defaults to source_path/data_partitions).")
    parser.add_argument(
        "--block_dim",
        default="",
        help="Override block_dim as 'x,y,z' (e.g., '4,1,5').",
    )
    args = parser.parse_args()

    block_dim, source_path = _load_block_dim_and_source(args.config)
    if args.block_dim:
        block_dim = [int(x) for x in args.block_dim.split(",")]
    if len(block_dim) != 3:
        raise ValueError("block_dim must have 3 integers [x, y, z].")

    output_dir = args.output_dir or os.path.join(source_path, "data_partitions")
    os.makedirs(output_dir, exist_ok=True)

    bx, by, bz = block_dim
    blocks = []
    block_id = 0
    for z in range(bz):
        for y in range(by):
            for x in range(bx):
                min_x = float(x) / bx
                max_x = float(x + 1) / bx
                min_y = float(y) / by
                max_y = float(y + 1) / by
                min_z = float(z) / bz
                max_z = float(z + 1) / bz
                open_edges = [ False, False, False, False]
                blocks.append(
                    {
                        "block_id": block_id,
                        "aabb": [min_x, max_x, min_y, max_y, min_z, max_z],
                        "open_edges": open_edges,
                    }
                )
                block_id += 1

    plane_indices = [0, 2, 1] if by == 1 else [0, 1, 2]
    segments = _compute_uniform_segments(block_dim, plane_indices)
    payload = {
        "partition_type": "grid",
        "contracted_space": True,
        "aabb_format": "[minx, maxx, miny, maxy, minz, maxz]",
        "plane_indices": plane_indices,
        "segments": segments,
        "blocks": blocks,
    }

    out_path = os.path.join(output_dir, "block_bounds.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
