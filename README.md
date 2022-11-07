# CAR-API: Cityscapes Attributes Recognition API

This is the official api to download and fetch attributes annotations for Cityscapes Dataset.



## Content

- [Installation](#Installation)
- [PyTorch Example](#PyTorch-Example)
- [Citation](#Citation)

[comment]: <> (- [Code Structure]&#40;#Code-Structure&#41;)



## Installation

You first need to download Cityscapes dataset.
You can do so by checking [this repo](https://github.com/mcordts/cityscapesScripts.git).

I'm showing here a simple working example to download the data but for further issues please refer to the source repo. Or download from [the official website](https://www.cityscapes-dataset.com/)

1. Install Cityscapes scripts and other required packages.

```shell
$ pip install -r requirements.txt
```

2. Run the following script to download Cityscapes dataset.
If you don't have an account, you will need to [create an account](https://www.cityscapes-dataset.com/register/).

```shell
$ csDownload -d [DESTINATION_PATH] PACKAGE_NAME
```

Note: you can also use `-l` option to list all possible packages to download. i.e.

```shell
$ csDownload -l
```

3. After downloading all required packages, set the environment variable `CITYSCAPES_DATASET` to the location of the dataset.
For example, if the dataset is installed in the path `/home/user/cityscapes/`

```shell
$ export CITYSCAPES_DATASET="/home/user/cityscapes/"
```

Note: you can also export the previous command to your `~/.bashrc` file for example.

```shell
$ echo 'export CITYSCAPES_DATASET="/home/user/cityscapes/"' > ~/.bashrc
```

Note2: we actually need the images only. We do not need the labels as it is stored with the attributes annotations as well.

4. Run the following to download the `json` files of CAR compressed as a single `zip` file extract it and then remove the `zip` file.

```shell
$ python download_CAR.py --url_path "https://DOWNLOAD_LINK_HERE"
```

You can download the annotations through [this link](https://drive.google.com/uc?export=download&id=1kf8JgAYKo_8ePckdsJfpM2g2x_OWRy9D)

For comments and feedback, please email me at `kmetwaly511 [at] gmail [dot] com`.

At this point, you have 4 `json` files; namely `all.json`, `train.json`, `val.json` and `test.json`

[comment]: <> (You can also select a specific set of cities to download. For example, `--cities "aachen,bremen"` will download attributes files of aachen and bremen cities only.)


## PyTorch Example

We provide a pytorch example to read the dataset and retrieve a sample of the dataset in [`pytorch_dataset_CAR.py`](pytorch_dataset_CAR.py).
Please, refer to [`main`](pytorch_dataset_CAR.py#L137-L161).It contains a code that goes through the entire dataset.

An output sample of the dataset class is of custom type `ModelInputItem`. Please refer to [the definiton](dataclasses_CAR.py#L39-L90) of the class for more details about defined methods and variables. 


[comment]: <> (## Code Structure)




## Citation

If you are planning to use this code or the dataset, please cite the work appropriately as follows.

```text
@misc{car_api,
  title = {{CAR}-{API}: an {API} for {CAR} Dataset},
  key = {{CAR}-{API}},
  howpublished = {\url{http://github.com/kareem-metwaly/car-api}},
  note = {Accessed: 2021-11-16}
}

@misc{metwaly2022car,
  title={{CAR} -- Cityscapes Attributes Recognition A Multi-category Attributes Dataset for Autonomous Vehicles}, 
  author={Kareem Metwaly and Aerin Kim and Elliot Branson and Vishal Monga},
  year={2021},
  eprint={2111.08243},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  howpublished = {\url{https://arxiv.org/abs/2111.08243}},
  urldate = {2021-11-17},
}
```
