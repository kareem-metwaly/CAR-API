import argparse
import concurrent.futures

from scaleapi import ScaleClient, Task
from tqdm import tqdm

from scale_local.cityscapes_labeling.utils import LIVE_API_KEY, TEST_API_KEY, logger, map_unordered


def cancel_task(task: Task) -> Task:
    try:
        task.cancel()
        logger.info(f"{task.id} is cancelled")
    except Exception as err:
        logger.info(err)
        print(err)
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="cityscapes_attributes")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--batch", type=str, default="StartIndex5000Length1000")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    API_KEY = TEST_API_KEY if args.dryrun else LIVE_API_KEY
    client = ScaleClient(API_KEY)

    client.get_project(args.project)

    batch_name = f"{args.project}::{args.batch}" if args.batch else None

    if batch_name:
        tasks = list(client.get_tasks(project_name=args.project, batch_name=batch_name))
    else:
        tasks = list(client.get_tasks(project_name=args.project))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
    gen = map_unordered(executor, cancel_task, iterable=tasks, verbose=True)
    for _ in tqdm(gen):
        pass


if __name__ == "__main__":
    main()
