import argparse
import concurrent.futures
import functools
import time
import typing as t

import scaleapi
from ml_linter_2d.cityscapes_labeling.utils import (  # POLYGON_VERTICAL_OFFSET,
    BASE_HTTP_URL,
    BASE_S3_URL,
    CALLBACK_URL,
    INSTRUCTION_IFRAME,
    LIVE_API_KEY,
    TAXONOMY,
    TEST_API_KEY,
    CsIterator,
    CSMap,
    CsSample,
    UniqueID,
    append_to_json,
    logger,
    map_unordered,
)
from scaleapi import BatchStatus, TaskStatus, TaskType, exceptions
from tqdm import tqdm


def create_task(
    client: scaleapi.ScaleClient,
    sample: CsSample,
    image_url: BASE_HTTP_URL,
    metadata: t.Dict,
    project: str,
    batch: str,
):
    try:
        sample.object.label = CSMap(sample.object.label)
    except KeyError:
        logger.info(f"This key doesn't exist in our map {sample.object.label}")
        return None

    attributes = TAXONOMY.fetch(category=sample.object.label).attributes
    taxonomy = {}
    for attr in attributes:
        attr.description = (
            "N/A" if attr.description == "" else attr.description
        )  # to deal with any possibly empty description
        dct = attr.to_dict()
        taxonomy.update(dct)
    if len(taxonomy) == 0:
        logger.info(
            f"Skipping this sample it doesn't have attributes {sample.unique_id} has label {sample.object.label}"
        )
        return None

    task_config = dict(
        project=project,
        batch=batch,
        # No callback
        callback_url=CALLBACK_URL,
        instruction=INSTRUCTION_IFRAME,
        attachment_type="image",
        attachment=str(image_url),
        metadata=metadata,
        taxonomies=taxonomy,
        layers=dict(
            polygons=[
                dict(
                    label=sample.object.label,
                    vertices=sample.polygon,
                )
            ],
        ),
    )
    try:
        res = client.create_task(task_type=TaskType.Categorization, **task_config)
    except exceptions.ScaleException:
        task_config["metadata"][
            "polygon"
        ] = None  # some polygons are quite big and that craches the code since scaleapi requires the size of metadata to not exceed 10240B
        res = client.create_task(task_type=TaskType.Categorization, **task_config)
    return res


def process_samples(
    sample: t.Union[UniqueID, CsSample],
    client: scaleapi.ScaleClient,
    existing_uniqueids: t.Sequence[UniqueID],
    project_name: str,
    batch_name: str,
) -> t.List[dict]:
    if isinstance(sample, UniqueID):
        sample = CsSample(sample)
        sample.load()

    obj = sample.object
    category = obj.label
    s3_url = BASE_S3_URL / f"{sample.image_path}"
    http_url = BASE_HTTP_URL.with_path(s3_url.path)
    logger.info(f"{sample.unique_id}: {category}")
    if sample.unique_id in existing_uniqueids:
        logger.info(f"{sample.unique_id}: Already exists")
        return []

    metadata = dict(
        unique_id=sample.unique_id.id,
        box=dict(
            top=sample.bbox.top,
            left=sample.bbox.left,
            width=sample.bbox.width,
            height=sample.bbox.height,
        ),
        polygon=sample.polygon,
        # polygon_shift_value=POLYGON_VERTICAL_OFFSET,
    )
    task = create_task(
        client=client,
        sample=sample,
        metadata=metadata,
        image_url=http_url,
        project=project_name,
        batch=batch_name,
    )
    if task is not None:
        logger.info(f"{sample.unique_id}: Created {task.id}")
        return [metadata]
    else:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=1000)
    parser.add_argument(
        "--output", type=str, default="./ml_linter_2d/cityscapes_labeling/output.json"
    )
    parser.add_argument("--project", default="cityscapes_attributes")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--batch", type=str, default="LaunchStartIndex1000Length1000")
    parser.add_argument(
        "--unique_id",
        type=str,
        default="",
        # default="leftImg8bit/train/aachen/aachen_000033_000019_leftImg8bit.png::gtFine/train/aachen/aachen_000033_000019_gtFine_polygons.json::61",
        help="setting this will create a new batch that has a name dervied from the UniqueID and send a single sample of that UniqueID",
    )
    args = parser.parse_args()

    API_KEY = TEST_API_KEY if args.dryrun else LIVE_API_KEY
    client = scaleapi.ScaleClient(API_KEY)

    # Create project if necessary
    if args.project:
        try:
            client.get_project(args.project)
        except exceptions.ScaleException:
            # with open(args.instructions, "r") as f:
            #     instructions = "".join(f.readlines())
            client.create_project(
                project_name=args.project,
                task_type=TaskType.Categorization,
                # params=dict(instruction=instructions),
            )
            logger.info(f"Created new project {args.project}")

    batch_name = f"{args.project}::{args.batch}" if args.batch else None
    # Create batch if necessary
    if batch_name and not args.unique_id:
        try:
            batch = client.create_batch(args.project, batch_name, callback=CALLBACK_URL)
            logger.info(f"Created new batch: {batch_name}")
        except exceptions.ScaleException:
            batch = client.get_batch(batch_name)
            logger.info(f"Got existing batch: {batch_name}")

        # Make sure the batch has not been finalized
        assert BatchStatus(batch.status) == BatchStatus.InProgress, batch.status
    else:
        batch_name = f"TEST_{args.unique_id}_{time.time()}"
        batch = client.create_batch(args.project, batch_name, callback=CALLBACK_URL)
        logger.info(f"Created new batch: {batch_name}")

    cs_samples_generator = CsIterator().generator(start_index=args.start_index)

    # Get UUIDs already annotated
    if batch_name:
        completed = list(
            client.get_tasks(
                project_name=args.project,
                batch_name=batch_name,
                status=TaskStatus.Completed,
            )
        )
        completed.extend(
            list(
                client.get_tasks(
                    project_name=args.project,
                    batch_name=batch_name,
                    status=TaskStatus.Pending,
                )
            )
        )
    else:
        completed = list(client.get_tasks(project_name=args.project, status=TaskStatus.Completed))
        completed.extend(
            list(
                client.get_tasks(
                    project_name=args.project,
                    status=TaskStatus.Pending,
                )
            )
        )

    existing_uniqueids = (
        [UniqueID(id=c._json["metadata"]["unique_id"]) for c in completed]
        if len(completed) > 0
        else []
    )
    if args.unique_id:
        process_samples(
            UniqueID(id=args.unique_id),
            client=client,
            existing_uniqueids=[],
            project_name=args.project,
            batch_name=batch_name,
        )
    else:
        logger.info(f"Number of objects already annotated: {len(existing_uniqueids)}")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
        rows = []
        pbar = tqdm(total=args.num_samples)
        gen = map_unordered(
            executor,
            functools.partial(
                process_samples,
                client=client,
                project_name=args.project,
                batch_name=batch_name,
                existing_uniqueids=existing_uniqueids,
            ),
            iterable=cs_samples_generator,
            verbose=True,
        )

        for i, _rows in enumerate(gen):
            pbar.update()
            rows += _rows
            if i % args.save_freq == 0:
                append_to_json(rows, args.output)
                rows = []
            if i >= args.num_samples:
                break
        if rows:
            append_to_json(rows, args.output)

    # if batch_name:
    #     batch.finalize()


if __name__ == "__main__":
    main()
