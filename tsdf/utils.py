'''
TSDF utility functions.
'''

import numpy as np


# Get corners of 3D camera view frustum of depth image.
def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0, 0],
                               (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1, 1],
                                np.array([0, max_depth, max_depth, max_depth, max_depth])])
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(cam_pose[:3, 3].reshape(3, 1), (1, view_frust_pts.shape[1]))  # from camera to world coordinates
    return view_frust_pts


# Save 3D mesh to a polygon .ply file.
def meshwrite(filename, verts, faces, norms, colors):

    # Write header.
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex {:d}\n".format(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face {:d}\n".format(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list.
    for i in range(verts.shape[0]):
        ply_file.write("{:f} {:f} {:f} {:f} {:f} {:f} {:d} {:d} {:d}\n".format(
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2]))
    
    # Write face list.
    for i in range(faces.shape[0]):
        ply_file.write("3 {:d} {:d} {:d}\n".format(faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()