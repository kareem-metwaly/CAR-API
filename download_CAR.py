#!/usr/bin/env python

import os
import zipfile
from argparse import ArgumentParser

import wget

from utils_CAR import attributes_path  # , cities


def main():
    parser = ArgumentParser()
    parser.add_argument("--url_path", required=True, help="URL address to fetch attributes from")
    # parser.add_argument("--cities", default=None, help="Cities to be downloaded")
    args = parser.parse_args()
    # download_cities = (
    #     [city.replace(" ", "") for city in args.cities.slplit(",")] if args.cities else cities
    # )
    os.makedirs(attributes_path(), exist_ok=True)
    # for city in tqdm(download_cities):
    #     os.chdir(attributes_path())
    #     file_url = args.url_path + "/" + city + ".json"
    #     wget.download(file_url)
    os.chdir(attributes_path())
    file_name = wget.download(args.url_path)
    print(
        "\n"
        "The zip file is successfully installed to:\n"
        f"{os.path.join(attributes_path(), file_name)}\n"
        "Extracting ... \n"
    )

    # Extract files
    zip = zipfile.ZipFile(file_name)
    files = zip.namelist()
    zip.extractall()
    zip.close()
    # Delete zip file
    os.remove(file_name)
    print(f"Extraction Completed.\n {', '.join(files)}")


if __name__ == "__main__":
    main()
