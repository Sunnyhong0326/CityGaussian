#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = image_path.split("images/")[1].split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    field_names = vertices.data.dtype.names or ()
    if 'nx' in field_names and 'ny' in field_names and 'nz' in field_names:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        # Fall back to zero normals for PLYs that don't carry them (e.g.
        # MatrixCity BigCity's all_ds32_*.ply). Matches vanilla
        # gaussian-splatting/scene/dataset_readers.py.
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=None, partition=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    if partition is not None:
        filtered_cam_infos = []
        for i in range(partition.shape[0]):
            if partition[i]:
                filtered_cam_infos.append(cam_infos[i])
        cam_infos = filtered_cam_infos if len(filtered_cam_infos) >= 50 else []
        print(f"Filtered Cameras: {len(filtered_cam_infos)}. ")
    
    if eval:
        import math
        if llffhold is None:
            eval_image_num = max(math.ceil(0.05 * len(cam_infos)), 1)
            llffhold = max(len(cam_infos) // eval_image_num, 8)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # image_path = os.path.join(path, cam_name)
            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def _matrixcity_frame_image_name(frame, extension=".png"):
    image_name = frame.get("file_name", frame.get("file_path", ""))
    if not image_name:
        raise KeyError("MatrixCity frame is missing both 'file_name' and 'file_path'.")
    if Path(image_name).suffix:
        return image_name
    return f"{image_name}{extension}"

def _matrixcity_image_path(path, mode, image_name):
    root = Path(path)
    candidates = [
        root.parent.parent / mode / image_name,
        root / mode / image_name,
        root.parent / mode / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])

def isMatrixCityScene(path):
    root = Path(path)
    transforms_path = root / "transforms_train.json"
    if not transforms_path.exists() or not any(root.glob("*.ply")):
        return False

    train_root_candidates = [root.parent.parent / "train", root / "train", root.parent / "train"]
    return any(candidate.is_dir() for candidate in train_root_candidates)

def readMatrixCityCameras(path, transformsfile, mode, extension=".png"):
    transforms_path = Path(path) / transformsfile
    with transforms_path.open("r", encoding="utf-8") as json_file:
        contents = json.load(json_file)

    fovx = contents.get("camera_angle_x")
    fl_x = contents.get("fl_x")
    fl_y = contents.get("fl_y")
    cam_infos = []

    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        if idx == 0 or (idx + 1) % 1000 == 0 or idx + 1 == len(frames):
            sys.stdout.write('\r')
            sys.stdout.write("Reading MatrixCity {} camera {}/{}".format(mode, idx + 1, len(frames)))
            sys.stdout.flush()

        image_name = _matrixcity_frame_image_name(frame, extension)
        image_path = _matrixcity_image_path(path, mode, image_name)
        if not os.path.exists(image_path):
            print(f"\nFile {image_path} not found, skipping...")
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        # OpenGL/Blender axes (Y up, Z back) -> COLMAP axes (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # PIL's Image.open is lazy; keep the handle alive on CameraInfo.image
        # to match readColmapCameras / readCamerasFromTransforms.
        image = Image.open(image_path)
        width, height = image.size

        frame_fl_x = frame.get("fl_x", fl_x)
        frame_fl_y = frame.get("fl_y", fl_y)
        if fovx is not None:
            FovX = fovx
            FovY = focal2fov(fov2focal(fovx, width), height)
        elif frame_fl_x is not None and frame_fl_y is not None:
            FovX = focal2fov(frame_fl_x, width)
            FovY = focal2fov(frame_fl_y, height)
        else:
            raise KeyError(f"MatrixCity camera intrinsics missing in {transforms_path}")

        cam_infos.append(CameraInfo(
            uid=len(cam_infos),
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=Path(image_name).stem,
            width=width,
            height=height,
        ))

    sys.stdout.write('\n')
    return cam_infos

def readMatrixCityInfo(path, eval, partition=None, extension=".png", load_train=True):
    print(f"Reading MatrixCity BigCity dataset from {path}")
    if load_train:
        train_cam_infos = readMatrixCityCameras(path, "transforms_train.json", "train", extension)

        # Deterministic order so partition indices are stable across runs;
        # data_partition.py builds its mask against this same ordering.
        train_cam_infos = sorted(train_cam_infos, key=lambda c: c.image_path)
        train_cam_infos = [c._replace(uid=i) for i, c in enumerate(train_cam_infos)]

        # Mirror readColmapSceneInfo:149-155 partition filter, including the
        # >=50 fallback that keeps tiny blocks from becoming empty.
        if partition is not None:
            filtered_cam_infos = [train_cam_infos[i] for i in range(partition.shape[0]) if partition[i]]
            train_cam_infos = filtered_cam_infos if len(filtered_cam_infos) >= 50 else train_cam_infos
            print(f"Filtered Cameras: {len(filtered_cam_infos)}. ")
    else:
        # Test-only rendering (e.g. render_large.py --eval --skip_train):
        # skip parsing the (much larger) train split entirely.
        train_cam_infos = []

    test_json_path = Path(path) / "transforms_test.json"
    if eval and test_json_path.exists():
        test_cam_infos = readMatrixCityCameras(path, "transforms_test.json", "test", extension)
    else:
        test_cam_infos = []

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    # cameras_extent is unused when loading a trained model, but getNerfppNorm
    # needs a non-empty camera list; fall back to test cams in test-only mode.
    nerf_normalization = getNerfppNorm(train_cam_infos if train_cam_infos else test_cam_infos)

    ply_candidates = sorted(Path(path).glob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"No MatrixCity initial point cloud .ply found in {path}")
    ply_path = str(ply_candidates[0])
    print(f"[INFO] MatrixCity initial point cloud: {ply_path}")
    pcd = fetchPly(ply_path)

    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "MatrixCity": readMatrixCityInfo,
}