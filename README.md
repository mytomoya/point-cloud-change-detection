# point-cloud-change-detection

## Getting Started

Run the following command to launch the docker container:

```bash
docker compose up -d
```

**Note:** Packages are managed by [Poetry](https://python-poetry.org/) inside the docker container. By default, you need to run scripts with

```bash
poetry run python <script>
```

in the container.


## How to Run

The source code is located in the `src` folder.


For details on how to organize the data and run the code, please refer to the [`README.md`](./src/) file in the `src` folder.

If iPhone devices equipped with LiDAR sensors are available, you can use [RGB-D Scan with ARKit](https://github.com/Tomoya-Matsubara/RGB-D-Scan-with-ARKit) to collect the data.
