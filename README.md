# Semantic TSDF

Semantic TSDF for self-driving static scene reconstruction.

TSDF (truncated signed distance function) can be used to reconstruct the static scene around a self-driving car. Since it is fast and optimizable by GPU, we prefer to use TSDF rather than fused point cloud to represent the world. However, we must filter out dynamic objects that may hinder our reconstruction. Since motion-based methods will still capture cars that stop for a while at the red light, we would like to use semantic segmentation networks to filter out the objects that are possibly dynamic.

Below is the reconstruction result on KITTI sequence 0106.

![KITTI Sequence 0106](0106_color_.gif)
![Reconstruction Example](0106_tsdf.gif)

Please refer to the LICENSE file for inherited Licenses.

## Dependencies

pytorch 1.0+

Optional: GPU acceleration requires an NVIDA GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyCUDA](https://developer.nvidia.com/pycuda)

```shell
pip install pycuda
```

## Data generation

This step is completed by default. To generate a new data sequence, create a `KITTI` directory and unzip the downloaded raw KITTI data. The directory will look like this

```text
- KITTI
| - 2011_09_26
  | - 2011_09_26_drive_XXXX_extract
  | - 2011_09_26_drive_XXXX_sync
  | - calib_cam_to_cam.txt
  | - calib_imu_to_velo.txt
  | - calib_velo_to_cam.txt
```

Then, simply run

```shell
python data_parser.py
```

Note: depth images are saved as 16-bit PNG in millimeters.

Note: We currently only provide pretrained model on cityscape. For KITTI, you may need to fine-tune yourself. The `kitti.pkl` file provided is only a dummy file.

## Demo

```shell
python demo.py
```

The result is save as a `.ply` file, and we recommend to use `Meshlab` to render it with colors.
