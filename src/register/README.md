# Registration

`main.py` is the entry point of the registration module.

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

The default path to the source dataset is set to be the working directory. You can specify the path with `-p` option, though:


```
python main.py -p /path/to/source_dataset
```


## Parameters

In addition to the dataset path, you can specify the voxel size used for downsampling:

- `-s` or `--size`: The size of the voxel grid. Default: $0.1$.


## Saved Files

The registered point clouds are saved in the dataset directory as `registered.ply`:

```
├── Before
│   ├── merged_label.npy
│   ├── registered.ply
│   └── whole_point_cloud.ply
├── After
│   ├── merged_label.npy
│   ├── registered.ply
│   └── whole_point_cloud.ply
``````
