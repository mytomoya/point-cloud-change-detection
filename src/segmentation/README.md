## Panoptic Segmentation

`main.py` is the main file for applying panoptic segmentation on images.


## Dataset

The dataset directory must be organized like this:

```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
```

The default path to the dataset is set to be the working directory. You can specify the path with `-p` option, though:

```
python main.py -p /path/to/dataset
```


## Labels

Panoptic segmentation is applied to each RGB image in the dataset and produces a 2D array of labels for each. The array of
$i$-th image is saved as `label_i.npy` in Label directory.

```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── Label
│   ├── label_0.npy
│   ├── label_1.npy
│   └── ...
```


## `unpair.json`

Panoptic segmentation distinguishes between different instances of the same object. However, it does not tell the difference between instances in different images.

`unpair_set` is a set of pairs of instances that must be different.

For example, let us think about the following situation:

- `desk_1` and `desk_2` are detected in Image 1
- `desk_3` is detected in Image 2

In this case, `unpair_set` looks like:

```python
unpair_set = {
    ("desk_1", "desk_2")
}
```

It is worth noting that `desk_3` of Image 2 can be the same as `desk_1` or `desk_2` of Image 1. It is also possible that all of them are different.


`unpair_set` is saved as a JSON file `unpair.json`:

```
dataset
├── RGB
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── Label
│   ├── label_0.npy
│   ├── label_1.npy
│   └── ...
├── unpair.json
```
