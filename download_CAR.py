#!/usr/bin/env python

import os
from argparse import ArgumentParser

import wget
from tqdm import tqdm

from utils_CAR import attributes_path, cities


def main():
    parser = ArgumentParser()
    # TODO: add defualt url value here
    parser.add_argument(
        "--url_path", default="http://url", help="URL address to fetch attributes from"
    )
    parser.add_argument("--cities", default=None, help="Cities to be downloaded")
    args = parser.parse_args()
    download_cities = (
        [city.replace(" ", "") for city in args.cities.slplit(",")] if args.cities else cities
    )
    os.makedirs(attributes_path(), exist_ok=True)
    for city in tqdm(download_cities):
        os.chdir(attributes_path())
        file_url = args.url_path + "/" + city + ".json"
        wget.download(file_url)


if __name__ == "__main__":
    main()
