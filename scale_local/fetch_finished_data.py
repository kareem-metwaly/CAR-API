import argparse
import json
import os

from scaleapi import ScaleClient, Task, TaskStatus
from tqdm import tqdm

from scale_local.cityscapes_labeling.utils import LIVE_API_KEY, TEST_API_KEY
from utils_CAR import CARInstance


def reform_task(task: Task) -> CARInstance:
    assert task.status == TaskStatus.Completed.value, task.status
    assert len(task.params["layers"]["polygons"]) == 1, task.params["layers"]["polygons"]

    converted_task = CARInstance(
        unique_id=task.metadata["unique_id"],
        category=task.params["layers"]["polygons"][0]["label"],
        polygon_annotations=task.params["layers"]["polygons"][0]["vertices"],
        attributes=task.response["taxonomies"],
        _meta={"batch_name": task.batch},
    )
    return converted_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="cityscapes_attributes")
    parser.add_argument(
        "--batch", type=str, default="", help='the batch to be fetched, set to "" for all'
    )
    parser.add_argument(
        "--reformat",
        type=bool,
        action="store_true",
        help="whether to reformat to work with CARInstance or not",
    )
    parser.add_argument("--path", type=str, default="/home/krm/datasets/car_api/")
    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)
    assert os.path.isdir(args.path)
    file_name = (
        os.path.join(args.path, "all.json")
        if args.batch == ""
        else os.path.join(args.path, args.batch + ".json")
    )
    if os.path.exists(file_name):
        print(f"File found: {file_name}")
        out = input("delete already existing files y/a/[N]? ").lower()
        if out in ["", "n"]:
            raise FileExistsError
        elif out not in ["y", "a"]:
            raise ValueError
        overwrite = "w" if out == "y" else "a"
    else:
        overwrite = "w"

    API_KEY = TEST_API_KEY if args.dryrun else LIVE_API_KEY
    client = ScaleClient(API_KEY)

    client.get_project(args.project)

    batch_name = f"{args.project}::{args.batch}" if args.batch != "" else None

    if batch_name:
        tasks = list(client.get_tasks(project_name=args.project, batch_name=batch_name))
    else:
        tasks = list(client.get_tasks(project_name=args.project))

    # Filter tasks that are not completed.
    tasks = [task for task in tasks if task.status == TaskStatus.Completed.value]

    if args.reformat:
        frames = []
        for task in tqdm(tasks):
            task = reform_task(task)
            frames.append(task.as_dict)
        with open(file_name, overwrite) as f:
            json.dump(frames, f)
    else:
        with open(file_name, overwrite) as f:
            json.dump([t._json for t in tasks], f)


if __name__ == "__main__":
    main()
