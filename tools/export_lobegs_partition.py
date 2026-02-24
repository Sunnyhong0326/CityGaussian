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


def _compute_segments(blocks, plane_indices):
    v_ax, h_ax, _ = plane_indices
    v_vals = []
    h_vals = []
    for blk in blocks:
        aabb = np.asarray(blk["aabb"], dtype=np.float64)
        v_vals.extend([aabb[2 * v_ax], aabb[2 * v_ax + 1]])
        h_vals.extend([aabb[2 * h_ax], aabb[2 * h_ax + 1]])

    v_min, v_max = min(v_vals), max(v_vals)
    h_min, h_max = min(h_vals), max(h_vals)
    v_span = max(v_max - v_min, 1e-6)
    h_span = max(h_max - h_min, 1e-6)
    v_eps = 1e-6 * v_span
    h_eps = 1e-6 * h_span

    def _snap(x0, y0, x1, y1):
        sx0 = round(x0 / v_eps) * v_eps
        sy0 = round(y0 / h_eps) * h_eps
        sx1 = round(x1 / v_eps) * v_eps
        sy1 = round(y1 / h_eps) * h_eps
        return sx0, sy0, sx1, sy1

    edge_set = set()
    segments = []
    for blk in blocks:
        aabb = np.asarray(blk["aabb"], dtype=np.float64)
        v0 = float(aabb[2 * v_ax])
        v1 = float(aabb[2 * v_ax + 1])
        h0 = float(aabb[2 * h_ax])
        h1 = float(aabb[2 * h_ax + 1])
        edges = [
            (v0, h0, v1, h0),
            (v1, h0, v1, h1),
            (v1, h1, v0, h1),
            (v0, h1, v0, h0),
        ]
        for x0, y0, x1, y1 in edges:
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            x0, y0, x1, y1 = _snap(x0, y0, x1, y1)
            key = (x0, y0, x1, y1)
            if key in edge_set:
                continue
            edge_set.add(key)
            segments.append([float(x0), float(y0), float(x1), float(y1)])
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
                open_edges = [
                    x == 0,
                    x == bx - 1,
                    y == 0,
                    y == by - 1,
                    z == 0,
                    z == bz - 1,
                ]
                blocks.append(
                    {
                        "block_id": block_id,
                        "aabb": [min_x, max_x, min_y, max_y, min_z, max_z],
                        "open_edges": open_edges,
                    }
                )
                block_id += 1

    segments = _compute_segments(blocks, [0, 1, 2])
    payload = {
        "partition_type": "grid",
        "contracted_space": True,
        "aabb_format": "[minx, maxx, miny, maxy, minz, maxz]",
        "plane_indices": [0, 1, 2],
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
