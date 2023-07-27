# How to Run


## Create Dataset

To create the dataset, put your data in the following structure:

```
dataset
├── Before
│   ├── RGB
│   │   ├── rgb_0.png
│   │   ├── rgb_1.png
│   │   └── ...
│   ├── Frame
│   │   ├── frame_0.json
│   │   ├── frame_1.json
│   │   └── ...
│   └── Confidence
│        ├── confidence_0.png
│        ├── confidence_1.png
│        └── ...
└── After
     ├── RGB
     │   ├── rgb_0.png
     │   ├── rgb_1.png
     │   └── ...
     ├── Frame
     │   ├── frame_0.json
     │   ├── frame_1.json
     │   └── ...
     └── Confidence
          ├── confidence_0.png
          ├── confidence_1.png
          └── ...
```

You can get the dataset with [RGB-D Scan with ARKit](https://github.com/Tomoya-Matsubara/RGB-D-Scan-with-ARKit).


If your dataset is composed of multiple scenes, you can organize your dataset as follows:

```
dataset
├── 1
│   ├── Before
│   └── After
├── 2
│   ├── Before
│   └── After
```



With the dataset prepared, run the following command to create the dataset to apply preprocessing.

```bash
python dataset.py \
    --source /path/to/source-dataset \
    --destination /path/to/processed-dataset \
    --segmentation_step 1 \
    --num_samples 5000 \
    --extension png \
    --reconstruction_step 1 \
    --merge_neighbors 30 \
    --merge_distance_threshold 0.02 \
    --labels book chair \
    --min 1000
```

As a minimum, you need to specify the source and destination paths. The other parameters are optional.

For each preprocessing step, refer to the following `README.md` files:

1. [Panoptic Segmentation](./segmentation)
1. [Reconstruct Point Cloud](./reconstruct)
1. [Merge Instance Label](./merge)
1. [Register Point Clouds](./register)
1. [Extract Point Clouds](./extract)

Instead of running the whole process at once, you can run each step separately. For more information, refer to the above links.


## Change Detection

Once you have prepared the dataset, you can run the change detection with the following command:

```bash
python change_detection.py \
    --path /path/to/processed-dataset \
    --neighbors 10 \
    --threshold 0.01 \
```

- `-p` or `--path`: Path to the processed dataset. It should contain the `Before` and `After` directories.
- `-n` or `--neighbors`: Number of neighbors to use for the change detection.
- `-t` or `--threshold`: Distance threshold for the change detection.

The result will be saved as a confusion matrix with the name `confusion_matrix.svg`.
