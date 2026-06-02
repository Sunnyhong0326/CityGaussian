import csv
import os
from utils.general_utils import format_seconds


def write_optimization_profile(dataset, max_memory_allocated, elapsed_time):
    profile_path = getattr(dataset, "profile_path", None) or dataset.model_path
    csv_file_path = os.path.join(profile_path, "optimization_time.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(
                ["Block_id", "Max Memory Allocated (GB)", "Elapsed Time (Seconds)", "Elapsed Time (HH:MM:SS)"]
            )
        elapsed_time_seconds = elapsed_time
        elapsed_time_formatted = format_seconds(elapsed_time)
        csv_writer.writerow([getattr(dataset, "block_id", None), max_memory_allocated, elapsed_time_seconds, elapsed_time_formatted])


def write_ply_save_time_profile(dataset, iteration, elapsed_time):
    profile_path = getattr(dataset, "profile_path", None) or dataset.model_path
    csv_file_path = os.path.join(profile_path, "ply_save_time.csv")
    file_exists = os.path.isfile(csv_file_path)
    ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Block_id", "Iteration", "Elapsed Time (Seconds)", "Elapsed Time (HH:MM:SS)", "PLY Path"])
        csv_writer.writerow([getattr(dataset, "block_id", getattr(dataset, "partition_id", None)), iteration, elapsed_time, format_seconds(elapsed_time), ply_path])
