# Label Merge

`main.py` is the main script for label merging.


## Source Dataset

The source dataset directory must be organized like this:

```
source_dataset
├── point_cloud.ply
├── label.ply
└── unpair.json
```

The default path to the source dataset is set to be the working directory. You can specify the path with `-p` option, though:


```
python main.py -p /path/to/source_dataset
```


## Renamed Labels

When two points have the same label and close enough to each other, they are likely to be the same instance. Such labels are merged into one and saved as `merged_label.npy`:

```
source_dataset
├── point_cloud.ply
├── label.ply
├── merged_label.npy
└── unpair.json
```


## Destination Dataset

The dataset that will finally be used for change detection looks like:


```
destination_dataset
├── merged_labe.npy
└── whole_point_cloud.ply
```

The default path to the destination dataset is set to be the working directory and you can specify the path with `-o` option:


```
python main.py -o /path/to/destination_dataset
```


## Parameters

In addition to the dataset paths, there are several parameters you can specify:

- `-n` or `--neighbors`: The number of neighbors to consider when merging labels. Default: $17$.
- `-d` or `--distance`: The distance threshold for merging labels. Default: $0.02$.
