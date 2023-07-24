# Reconstruct Point Cloud

`main.py` is the main file to run the reconstruction.


## Dataset

The dataset directory must be organized like this:

```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── Frame
│   ├── frame_0.json
│   ├── frame_1.json
│   └── ...
├── Confidence
│   ├── confidence_0.png
│   ├── confidence_1.png
│   └── ...
├── Label
│   ├── label_0.npy
│   ├── label_1.npy
│   └── ...
```

The default path to the dataset is set to be the working directory. You can specify the path with `-p` option, though:

```
python main.py -p /path/to/dataset
```


## Parameters

In addition to the dataset path, there are several parameters you can specify:

- `-n` or `--num`: The number of pixels per frame used for reconstruction. Default: $5,000$
- `-s` or `--step`: The step size of frames used for reconstruction. Default: $1$
- `-e` or `--extension`: The extension of the rgb/confidence files. Default: `png`


## Saved Files

After running the reconstruction, the following files are saved in the dataset directory.

- Reconstructed point cloud: `point_cloud.ply`
- Labels for each point in the point cloud: `label.npy`


### Reconstructed Point Cloud


The reconstructed point cloud is saved as a ply file:

```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── Frame
│   ├── frame_0.json
│   ├── frame_1.json
│   └── ...
├── Confidence
│   ├── confidence_0.png
│   ├── confidence_1.png
│   ├── ...
├── Label
│   ├── label_0.npy
│   ├── label_1.npy
│   └── ...
├── point_cloud.ply
```

### Labels

Each point in the point cloud has an object (instance) label attached to it. The list of their labels are saved as an `npy` file:


```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── Frame
│   ├── frame_0.json
│   ├── frame_1.json
│   └── ...
├── Confidence
│   ├── confidence_0.png
│   ├── confidence_1.png
│   ├── ...
├── Label
│   ├── label_0.npy
│   ├── label_1.npy
│   └── ...
├── point_cloud.ply
├── label.npy
```
