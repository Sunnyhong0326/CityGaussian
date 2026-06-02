#!/usr/bin/env python3
import os
import subprocess

BASH = "./scripts/run_citygs.sh"  # <-- your bash script path

EXPERIMENTS = [
    
    # COLMAP
    # dict(
    #     COARSE_CONFIG="colmap/rubble_coarse",
    #     CONFIG="colmap/rubble_c9_r4",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/rubble_split/test",
    #     out_name="test",
    #     max_block_id="8",
    # ),
    # dict(
    #     COARSE_CONFIG="colmap/sciart_coarse",
    #     CONFIG="colmap/sciart_c9_r4",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/sci-art_split/test",
    #     out_name="test",
    #     max_block_id="8",
    # ),
    # dict(
    #     COARSE_CONFIG="colmap/canteen_coarse",
    #     CONFIG="colmap/canteen_c9",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/Occlude3D/Canteen/test",
    #     out_name="test",
    #     max_block_id="8",
    # ),
    # dict(
    #     COARSE_CONFIG="colmap/building_coarse",
    #     CONFIG="colmap/building_c20_r4",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/building_split/test",
    #     out_name="test",
    #     max_block_id="19",
    # ),
    # dict(
    #     COARSE_CONFIG="colmap/nthu_campus_coarse",
    #     CONFIG="colmap/nthu_campus_c8_r4",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/NTHU_ABCDE/test",
    #     out_name="test",
    #     max_block_id="7",
    # ),
    # dict(
    #     COARSE_CONFIG="colmap/residence_coarse",
    #     CONFIG="colmap/residence_c20_r4",
    #     TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/residence_split/test",
    #     out_name="test",
    #     max_block_id="19",
    # ),
    dict(
        COARSE_CONFIG="colmap/mc_aerial_coarse",
        CONFIG="colmap/mc_aerial_c36",
        TEST_PATH="/mnt/nfs/sunnyhong/data/COLMAP/matrix_city_aerial/test",
        out_name="test",
        max_block_id="35",
    ),
]

for exp in EXPERIMENTS:
    env = os.environ.copy()
    env.update(exp)
    env.setdefault("SAVE_EVERY", "200")

    print(f"\n=== Running: CONFIG={exp['CONFIG']} ===")
    subprocess.run(["bash", BASH], env=env, check=True)

print("\nAll experiments finished.")
