'''
Main file for building Semantic-TSDF.
'''

import time

import cv2
import numpy as np

from tsdf import tsdf
from tsdf import utils as tsdfutils

def main(n_imgs, voxel_size, use_raw_depth):

    depth_name = 'rawdepth' if use_raw_depth else 'depth'

    # Compute 3D bounds (in world coordinates) around convex hull of all camera view frustums in dataset.
    print("Estimating voxel volume bounds...")
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):

        # Read depth image and camera pose.
        depth_im = cv2.imread("data/frame-{:06d}.{}.png".format(i, depth_name), -1).astype(float) / 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt("data/frame-{:06d}.pose.txt".format(i))  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull.
        view_frust_pts = tsdfutils.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # Initialize voxel volume.
    print("Initializing voxel volume...")
    tsdf_vol = tsdf.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together.
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame {:d}/{:d}".format(i + 1, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("data/frame-{:06d}.color.jpg".format(i)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("data/frame-{:06d}.{}.png".format(i, depth_name), -1).astype(float) / 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt("data/frame-{:06d}.pose.txt".format(i))  # 4x4 rigid transformation matrix

        # Integrate observation into voxel volume (assume color aligned with depth).
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:02f}".format(fps))

    # Get mesh from voxel volume and save to disk.
    print("Saving to mesh.ply...")
    verts,faces,norms,colors = tsdf_vol.get_mesh()
    tsdfutils.meshwrite("mesh_{}_{}{}.ply".format(n_imgs, voxel_size, '_raw' if use_raw_depth else ''), verts, faces, norms, colors)


if __name__ == "__main__":
    n_imgs = 227  # default = 154 for KITTI sequence 0005, 227 for KITTI sequence 0106
    main(n_imgs, 0.125, True)
    main(n_imgs, 0.25, True)
    main(n_imgs, 0.5, True)
    main(n_imgs, 1.0, True)
    main(n_imgs, 0.125, False)
    main(n_imgs, 0.25, False)
    main(n_imgs, 0.5, False)
    main(n_imgs, 1.0, False)
