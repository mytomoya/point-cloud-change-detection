# Extract Point Clouds

`main.py` extracts point clouds from the dataset.

Point clouds in `Before` are extracted according to their object label, while point clouds in `After` are extracted according to the bounding volume of the corresponding point cloud in `Before`.


## Dataset

The dataset directory must be organized like this:

```
dataset
├── Before
│   ├── merged_label.npy
│   └── whole_point_cloud.ply
├── After
│   ├── merged_label.npy
│   └── whole_point_cloud.ply
```

The default path to the dataset is set to be the working directory. You can specify the path with `-p` option, though:

```
python main.py -p /path/to/dataset
```


## Parameters

In addition to the dataset path, there are several parameters you can specify:

- `-l` or `--labels`: The labels of the objects to be extracted. Labels not specified here will be ignored.
- `-m` or `--min`: The minimum number of points in the extracted point cloud. If the extracted point cloud has less points than this, it will be discarded.


## Saved Files

After running the extraction, the following files are saved in the dataset directory.

- Extracted point clouds: `{n}.ply`
- Labels for each point in the point cloud: `{n}.npy`
- Indices of the points in the original point cloud: `{n}.npy`

`{n}` is the id of the point cloud.


```
dataset
├── Before
│   ├── PLY
│   │   ├── 0.ply
│   │   ├── 1.ply
│   │   └── ...
│   ├── Label
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── Index
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── merged_label.npy
│   └── whole_point_cloud.ply
├── After
│   ├── PLY
│   │   ├── 0.ply
│   │   ├── 1.ply
│   │   └── ...
│   ├── Label
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── Index
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── merged_label.npy
│   └── whole_point_cloud.ply
```
