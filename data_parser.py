'''
Convert raw KITTI data to RGB-D images + pose + camera intrinsics.
'''
import os
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import png
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from PIL import Image


from network.model import RFMobileNetV2Plus
from network.modules import ABN, decode_segmap


warnings.filterwarnings("ignore")


def main(basedir, date, drive, image_h, image_w, image_ratio):

    # Load the data.
    dataset = pykitti.raw(basedir, date, drive)
    num_of_frames = len(dataset)  # default: len(dataset)
    if num_of_frames == 0:
        return

    # Set compress ratio.
    image_h_new = int(image_h * image_ratio / 8) * 8
    image_w_new = int(image_w * image_ratio / 8) * 8
    image_h_seg = int(image_h * max(0.5, image_ratio) / 8) * 8
    image_w_seg = int(image_w * max(0.5, image_ratio) / 8) * 8

    # We use right RGB camera (cam2) only.
    intrinsic_cam2 = dataset.calib.K_cam2
    intrinsic_cam2[0, 0] *= image_ratio
    intrinsic_cam2[1, 1] *= image_ratio
    intrinsic_cam2[0, 2] *= image_ratio
    intrinsic_cam2[1, 2] *= image_ratio
    
    if not os.path.exists('data'):
        os.makedirs('data')
    np.savetxt('data/camera-intrinsics.txt', intrinsic_cam2, delimiter=' ')

    # Setup Model.
    model = RFMobileNetV2Plus(
        n_class=19, in_size=(image_h_seg, image_w_seg), width_mult=1.0,
        out_sec=256, aspp_sec=(12, 24, 36),
        norm_act=partial(ABN, activation=nn.LeakyReLU(inplace=True)))
    model = torch.nn.DataParallel(model, device_ids=[0])
    # Note: the fine-tuned model on KIITI will not be provided.
    if torch.cuda.is_available():
        model = model.cuda()
        pre_weight = torch.load('network/checkpoints/kitti.pkl')
    else:
        pre_weight = torch.load('network/checkpoints/kitti.pkl', map_location='cpu')

    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)
    model.eval()

    print("Performing segmentation...")
    for frame_idx in range(num_of_frames):
        print("Processing frame {:d}/{:d}".format(frame_idx + 1, num_of_frames))

        # Obtain pose.
        pose = dataset.oxts[frame_idx].T_w_imu.dot(np.linalg.inv(dataset.calib.T_cam2_imu))
        np.savetxt('data/frame-{:06d}.pose.txt'.format(frame_idx), pose, delimiter=' ')

        # Obtain Velodyne points and right RGB image.
        velo_points = dataset.get_velo(frame_idx)
        cam2_image = dataset.get_cam2(frame_idx)
        cam2_seg = cam2_image.resize((image_w_seg, image_h_seg), Image.ANTIALIAS)
        cam2_image = cam2_image.resize((image_w_new, image_h_new), Image.ANTIALIAS)

        cam2_image.save('data/frame-{:06d}.color.jpg'.format(frame_idx))

        # Project points to camera.
        cam2_points = dataset.calib.T_cam2_velo.dot(velo_points.T).T

        # Filter out points behind camera
        idx = cam2_points[:, 2] > 0
        velo_points = velo_points[idx]
        cam2_points = cam2_points[idx]

        # Remove homogeneous z.
        cam2_points = cam2_points[:, :3] / cam2_points[:, 2:3]

        # Apply instrinsics.
        cam2_points = intrinsic_cam2.dot(cam2_points.T).T[:, [1, 0]]
        cam2_points = cam2_points.astype(int)

        # Create depth image.
        depth_cam2 = np.zeros((cam2_image.size[1], cam2_image.size[0]))

        # Filter out points out of image boundary
        idx = (cam2_points[:, 0] >= 0) & (cam2_points[:, 0] < depth_cam2.shape[0]) & (cam2_points[:, 1] >= 0) & (cam2_points[:, 1] < depth_cam2.shape[1])
        velo_points = velo_points[idx]
        cam2_points = cam2_points[idx]

        # Project points onto camera image.
        for i in range(cam2_points.shape[0]):
            depth_cam2[cam2_points[i, 0], cam2_points[i, 1]] = np.linalg.norm(velo_points[i, :3])

        # Convert depth to millimeter.
        depth_cam2 = (depth_cam2 * 1000).astype(np.uint16)
        with open('data/frame-{:06d}.rawdepth.png'.format(frame_idx), 'wb') as f:
            writer = png.Writer(width=depth_cam2.shape[1], height=depth_cam2.shape[0], bitdepth=16, greyscale=True)
            writer.write(f, depth_cam2.tolist())

        # Compute semantic segmentation for camera image.
        cam2_seg = np.array(cam2_seg, dtype=np.float32)
        cam2_seg = (cam2_seg - np.mean(cam2_seg, axis=(0, 1))) / 255.0
        cam2_seg = np.expand_dims(cam2_seg.transpose(2, 0, 1), 0)  # HWC -> NCWH
        cam2_seg = torch.from_numpy(cam2_seg).float()
        if torch.cuda.is_available():
            cam2_seg = cam2_seg.cuda()
        with torch.no_grad():
            cam2_seg = F.softmax(model(cam2_seg), dim=1)
            cam2_seg = torch.unsqueeze(cam2_seg.data.max(1)[1], 0)
            cam2_seg = torch.squeeze(F.interpolate(cam2_seg.double(), (image_h_new, image_w_new)), 0).long()
            cam2_seg = np.squeeze(cam2_seg.cpu().numpy(), axis=0)

        # Mask out dynamic objects.
        # Data lable chat: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        dynamic_mask = (cam2_seg >= 11) & (cam2_seg <= 18)
        depth_cam2[dynamic_mask] = 65535
        with open('data/frame-{:06d}.depth.png'.format(frame_idx), 'wb') as f:
            writer = png.Writer(width=depth_cam2.shape[1], height=depth_cam2.shape[0], bitdepth=16, greyscale=True)
            writer.write(f, depth_cam2.tolist())

        # Visualize segmentation results.
        cam2_seg = decode_segmap(cam2_seg).astype(np.uint8)
        cv2.imwrite('data/frame-{:06d}.seg.png'.format(frame_idx), cam2_seg)


if __name__ == "__main__":
    basedir = 'KITTI'
    date = '2011_09_26'
    drive = '0005'
    image_h = 375
    image_w = 1242
    image_ratio = 0.25  # [0.25 ~ 0.5] is recommended for KITTI
    main(basedir, date, drive, image_h, image_w, image_ratio)
