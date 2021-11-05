# CAR-API: Cityscapes Attributes Recognition API

This is the official api to download and fetch attributes annotations for Cityscapes Dataset.



## Content

- [Installation](#Installation)
- [PyTorch Example](#PyTorch-Example)
- [Code Structure](#Code-Structure)
- [Citation](#Citation)



## Installation

You first need to download Cityscapes dataset.
You can do so by checking [this repo](https://github.com/mcordts/cityscapesScripts.git).

I'm showing here a simple working example to download the data but for further issues please refer to the source repo. Or download from [the official website](https://www.cityscapes-dataset.com/)

1. Install Cityscapes scripts.

```shell
pip install cityscapesScripts
```

2. Run the following script to download Cityscapes dataset.

```shell
csDownload -d [DESTINATION_PATH] PACKAGE_NAME
```

Note: you can also use `-l` option to list all possible packages to download. i.e.

```shell
csDownload -l
```

3. After downloading all required packages, set the environment variable `CITYSCAPES_DATASET` to the location of the dataset.
For example, if the dataset is installed in the path `/home/user/cityscapes/`

```shell
export CITYSCAPES_DATASET="/home/user/cityscapes/"
```

Note: you can also export the previous command to your `~/.bashrc` file for example.

```shell
echo 'export CITYSCAPES_DATASET="/home/user/cityscapes/"' > ~/.bashrc
```

Note2: we actually need the images only. We do not need the labels as it is stored with the attributes annotations as well.

4. Run the following to download the json files of CAR

```shell
python download_CAR.py 
```

You can also select a specific set of cities to download. For example, `--cities "aachen,bremen"` will download attributes files of aachen and bremen cities only.


## PyTorch Example

We provide a simple pytorch example to read the dataset and retrieve a sample of the dataset in [`pytorch_CAR.py`](pytorch_CAR.py).

A sample contains the following:
1. 


## Code Structure



## Citation

```text
@misc{

}
```
