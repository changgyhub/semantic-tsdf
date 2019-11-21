# Semantic-TSDF for Self-driving Static Scene Reconstruction

![Reconstruction Example](example.gif)


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
