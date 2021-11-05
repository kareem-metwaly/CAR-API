import collections.abc
import concurrent.futures
import copy
import glob
import json
import logging
import os
import traceback
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import inquirer
import pandas as pd
import PIL.Image
from cityscapesscripts.helpers.annotation import Annotation, CsObjectType, CsPoly, Point
from cityscapesscripts.viewer.cityscapesViewer import CsLabelType, LabelType
from yarl import URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

# Where to store the images
BASE_S3_URL = URL("s3://scale-static-assets/cityscapes_attributes/dataset/")
BASE_HTTP_URL = URL("https://scale-static-assets.s3-us-west-2.amazonaws.com/")

# TODO: change the link
INSTRUCTION_IFRAME = """
<iframe src="https://docs.google.com/document/d/e/2PACX-1vS3dFuu8zZVybfvKApJyjg3Xh2PfrY0x05PXuk7jc03vg8EalLXFxkTqiBDvCrzYhOhDTpkFj5-yW4C/pub?embedded=true"></iframe>
"""

# No need for callbacks
CALLBACK_URL = "https://127.0.0.1/callback"
LIVE_API_KEY = os.environ["SCALE_LIVE_API_KEY"]
TEST_API_KEY = os.environ["SCALE_TEST_API_KEY"]

POLYGON_VERTICAL_OFFSET = 0  # 20


def shift_polygons(annotations: Annotation):
    for obj in annotations.objects:
        for i, point in enumerate(obj.polygon):
            obj.polygon[i] = Point(x=point.x, y=point.y + POLYGON_VERTICAL_OFFSET)
    return annotations


def submit_next_job(
    executor: concurrent.futures.Executor, fn: t.Callable, iterable: t.Iterator
) -> t.Optional[concurrent.futures.Future]:
    try:
        args = next(iterable)
    except StopIteration:
        return None

    if not isinstance(args, tuple):
        args = (args,)
    fut = executor.submit(fn, *args)
    return fut


def map_unordered(
    executor: concurrent.futures.ThreadPoolExecutor,
    fn: t.Callable,
    iterable: t.Iterator,
    verbose: bool = False,
) -> t.Iterator:
    if not isinstance(iterable, collections.abc.Iterator):
        iterable = iter(iterable)

    # Submit the initial jobs
    # pylint: disable=protected-access
    num_jobs = executor._max_workers
    futs = []
    for _ in range(num_jobs):
        fut = submit_next_job(executor, fn, iterable)
        if fut:
            futs.append(fut)

    while True:
        # Get the next completed future
        try:
            fut = next(concurrent.futures.as_completed(futs))
            futs.remove(fut)
        except StopIteration:
            break

        # Yield the result if there is no error
        try:
            yield fut.result()
        except Exception:
            if verbose:
                traceback.print_exc()

        # Add the next job
        fut = submit_next_job(executor, fn, iterable)
        if fut:
            futs.append(fut)


def append_to_json(rows, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "a+") as f:
        json.dump(rows, f)
        logger.info(f"Wrote {len(rows)} rows to {output}")
        f.write(",\n")


def CsPath():
    assert "CITYSCAPES_DATASET" in os.environ
    return os.environ["CITYSCAPES_DATASET"]


@dataclass
class ChoicesTaxonomy:
    values: t.List[str]


@dataclass
class NumericalTaxonomy:
    min: float
    max: float
    step: float

    def to_dict(self) -> t.Dict[str, float]:
        return {
            "min": self.min,
            "max": self.max,
            "step": self.step,
        }


class AttributeTypeTaxonomy(Enum):
    category = "category"
    number = "number"
    condition = "condition"


@dataclass
class ConditionTaxonomy:
    values: t.List[str]
    taxonomy: str
    choices: ChoicesTaxonomy

    def to_dict(self) -> t.Dict[str, t.Union[str, t.Sequence[str]]]:
        return {
            "choices": self.values,
            "condition": {
                "taxonomy": self.taxonomy,
                "choices": self.choices.values,
            },
        }


@dataclass
class AttributeTaxonomy:
    name: str
    description: str
    value: t.Union[ChoicesTaxonomy, NumericalTaxonomy, ConditionTaxonomy]
    allow_multiple: bool = False
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    @property
    def type(self) -> AttributeTypeTaxonomy:
        if isinstance(self.value, ChoicesTaxonomy):
            return AttributeTypeTaxonomy.category
        elif isinstance(self.value, NumericalTaxonomy):
            return AttributeTypeTaxonomy.number
        elif isinstance(self.value, ConditionTaxonomy):
            return AttributeTypeTaxonomy.condition
        else:
            raise ValueError(f"value attribute has unknown type {type(self.value)}")

    def __post_init__(self):
        assert self.description != ""
        if (
            self.type is AttributeTypeTaxonomy.condition
            or self.type is AttributeTypeTaxonomy.category
        ):
            self.value.values.append("Unclear")

    def to_dict(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        if self.type == AttributeTypeTaxonomy.category:
            return self.category_to_dict()
        elif self.type == AttributeTypeTaxonomy.number:
            return self.number_to_dict()
        elif self.type == AttributeTypeTaxonomy.condition:
            return self.condition_to_dict()
        else:
            raise NotImplementedError(f"This type is not supported {self.type}")

    def category_to_dict(
        self,
    ) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        return {
            self.name: {
                "type": self.type.value,
                "description": self.description,
                "choices": self.value.values,
                "allow_multiple": self.allow_multiple,
            },
        }

    def number_to_dict(self) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        return {
            self.name: {
                "type": self.type.value,
                "description": self.description,
                "min": self.value.min,
                "max": self.value.max,
                "step": self.value.step,
            },
        }

    def condition_to_dict(
        self,
    ) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        output_dict = {
            "type": "category",
            "description": self.description,
            "allow_multiple": self.allow_multiple,
        }
        output_dict.update(self.value.to_dict())
        return {self.name: output_dict}


class AttributesList:
    _items: t.Dict[str, AttributeTaxonomy]

    def __init__(self, items: t.Sequence[AttributeTaxonomy]):
        self._items = {}
        for item in items:
            assert not self.has(item)
            self._items.update({item.name: item})

    def has(self, item):
        return item.name in self._items.keys()

    def __repr__(self):
        return f"AttributesList[{', '.join(self._items.keys())}]"

    def append(self, item: AttributeTaxonomy):
        assert not self.has(item)
        self._items.update({item.name: item})

    def __iter__(self):
        for item in self._items.values():
            yield item

    def __getitem__(self, item: t.Union[str, int]):
        if isinstance(item, str):
            return self._items[item]
        elif isinstance(item, int):
            return next(value for i, value in self._items.values() if i == item)


@dataclass
class CategoryTaxonomy:
    name: str
    description: str
    attributes: AttributesList
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert self.description != ""

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            self.name: {
                "description": self.description,
                "attributes": [attribute.to_dict() for attribute in self.attributes],
            },
        }


@dataclass
class CompleteTaxonomy:
    things: t.Sequence[CategoryTaxonomy]
    stuff: t.Sequence[CategoryTaxonomy]
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "things": [thing.to_dict() for thing in self.things],
            "stuff": [stuff.to_dict() for stuff in self.stuff],
        }

    def __iter__(self) -> CategoryTaxonomy:
        for thing in self.things:
            yield thing
        for stuff in self.stuff:
            yield stuff

    def is_sane(self) -> bool:
        for category in self:
            for attribute in category.attributes:
                if (
                    (
                        attribute.type is AttributeTypeTaxonomy.category
                        and not isinstance(attribute.value, ChoicesTaxonomy)
                    )
                    or (
                        attribute.type is AttributeTypeTaxonomy.number
                        and not isinstance(attribute.value, NumericalTaxonomy)
                    )
                    or (
                        attribute.type is AttributeTypeTaxonomy.condition
                        and not isinstance(attribute.value, ConditionTaxonomy)
                    )
                ):
                    return False
                if attribute.type is AttributeTypeTaxonomy.condition:
                    dependent_attribute_name = attribute.value.taxonomy
                    dependent_attribute_values = attribute.value.choices.values
                    for value in dependent_attribute_values:
                        assert value in category.attributes[dependent_attribute_name].value.values
        return True

    @staticmethod
    def load() -> "CompleteTaxonomy":
        return CompleteTaxonomy(
            things=[
                CategoryTaxonomy(
                    name="Mid-to-Large Vehicle",
                    description="This class contains all possible medium to large vehicles, which basically include all vehicles with 4 wheels. It does not include bicycles, motorcycles and any other small or portable vehicles.",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vehicle Visibility",
                                description="How much of the vehicle is visible?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "0% to 20% is visible",
                                        "21% to 40% is visible",
                                        "41% to 60% is visible",
                                        "60% to 80% is visible",
                                        "81% to 100% is visible",
                                    ],
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Form",
                                description="Vehicle Form Factor; On Rails includes Trams or Trains",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Sedan",
                                        "Hatchback",
                                        "SUV",
                                        "Sport",
                                        "Van",
                                        "Pickup",
                                        "Truck",
                                        "Trailer",
                                        "Bus",
                                        "School Bus",
                                        "On Rails",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is the vehicle towing or being towed",
                                description="Is it towing something? or is it being towed?",
                                value=ChoicesTaxonomy(values=["Towing", "Being Towed", "Neither"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Type",
                                description="What is this vehicle used for?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Police",
                                        "Ambulance",
                                        "Fire",
                                        "Construction",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Light Status",
                                description="whether the vehicle's lights are on or off."
                                "If the vehicle is a school bus then the emergency option implis that the stop sign is on."
                                "Emergency indicate any special type of lights that the vehicle has, for example ambulances, police cars ... etc",
                                value=ChoicesTaxonomy(values=["On", "Off", "Emergency"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Status",
                                description="whether the vehicle is parked, stopped in the road or moving",
                                value=ChoicesTaxonomy(values=["Moving", "Stopped", "Parked"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Direction",
                                description="which part of the vehicle is facing the camera (if it is facing the camera with some angle, pick the one that closely agrees)",
                                value=ChoicesTaxonomy(
                                    values=["Front", "Back", "Left Side", "Right Side"]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Small Vehicle",
                    description="It includes the remaining small or portable vehicles such as bicycles, motorcycles, scooters, ... etc.",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vehicle Type",
                                description='What is the type of the vehicle? "Other" class may include things such as scooters.',
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Bicycle",
                                        "Motorcycle",
                                        "Float Drivable Surface",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Status",
                                description="whether the vehicle is parked, stopped in the road or moving",
                                value=ChoicesTaxonomy(values=["Moving", "Stopped", "Parked"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Has Rider",
                                description="whether there is a rider or not",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Direction",
                                description="which part of the vehicle is facing the camera (if it is facing the camera with some angle, pick the one that closely agrees)",
                                value=ChoicesTaxonomy(
                                    values=["Front", "Back", "Left Side", "Right Side"]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Has Sidecar",
                                description="Has the Vehicle a Sidecar? some vehicles such as motorcycles may have sidecars. This can be used for bicycles as well with an additional cart or something else.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Pedestrian",
                    description="A human object in the image",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Visibility",
                                description="How much of the pedestrian is visible?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "0% to 20% is visible",
                                        "21% to 40% is visible",
                                        "41% to 60% is visible",
                                        "60% to 80% is visible",
                                        "81% to 100% is visible",
                                    ],
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Age",
                                description="Age of the pedestrian",
                                value=ChoicesTaxonomy(values=["Adult", "Child"]),
                            ),
                            AttributeTaxonomy(
                                name="Pedestrian Type",
                                description="Is the pedestrian either a police officer / construction worker or neither of them",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Police Officer",
                                        "Construction Worker",
                                        "Neither",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Activity",
                                description="The posture of the person",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Sitting",
                                        "Lying Down",
                                        "Standing",
                                        "Walking",
                                        "Running",
                                        "Riding",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Using Vehicle",
                                description="Is pedestrian using a vehicle? whether the pedestrian is using a bicycle, motorcycle, scooter, wheelchair or something else",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Pushing Or Dragging",
                                description="whether the pedestrian is pushing something in front of him/her or pulling something behind.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Carrying",
                                description="whether the pedestrian is carrying anything (a child, a backpack).",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Hearing Impaired",
                                description="Is pedestrian hearing impaired? not necessarily due to a disability, but could be due to wearing headphones or any item that may prevent the pedestrian from hearing the surrounding environment.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Blind",
                                description="Is pedestrian blind? may not be clear to you, choose the best of what you believe",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Disabled",
                                description="Is pedestrian disabled? any kind of disability except for being blind or hair impaired. Please do NOT use unclear unless it looks like there is a disability that is not clear to you. Otherwise, probably it will be No.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Traffic Light",
                    description="contains the whole traffic light object not just the lights",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Traffic Light Type",
                                description="What kind of traffic light",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Circle Lamp",
                                        "Forward Arrow (should be an actual arrow pointing forward, not a circle lamp)",
                                        "Right Arrow",
                                        "Left Arrow",
                                        "U-Turn",
                                        "Pedestrian",
                                        "Unknown",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Traffic Light Status",
                                description="Traffic light color? for some traffic lights such as pedestrian Green would represent pedestrians can cross and Red for no. Black means it is neither working nor functioning",
                                value=ChoicesTaxonomy(values=["Green", "Yellow", "Red", "Black"]),
                            ),
                            AttributeTaxonomy(
                                name="Flashing Traffic Light",
                                description="Flashing traffic light? whether the traffic lights are flashing or not. May not be clear, choose what you believe is correct.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Traffic Sign",
                    description="contains the main body of the traffic sign",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Is Electronic",
                                description="is it a fixed sign or an electronic one that can change later",
                                value=ChoicesTaxonomy(values=["Electronic", "Fixed"]),
                            ),
                            AttributeTaxonomy(
                                name="Traffic Sign Type",
                                description="Traffic sign type",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Stop",
                                        "Speed Limit",
                                        "Construction",
                                        "Caution",
                                        "No Stopping",
                                        "No Parking",
                                        "No Turn Right",
                                        "No Turn Left",
                                        "Wrong Way",
                                        "Do Not Enter",
                                        "One Way",
                                        "Barrier",
                                        "Advertisement or Informative",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Speed Limit Value",
                                description="Speed Limit Value? In case of a speed limit sign, what's the max value for speed? set only when the traffic sign is a speed limit, otherwise set to any value",
                                # type=AttributeTypeTaxonomy.condition,
                                value=NumericalTaxonomy(
                                    min=5,
                                    max=400,
                                    step=5,
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Road Lane",
                    description="Just the lanes of the road",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Road Lane Type",
                                description="what kind of lanes?",
                                value=ChoicesTaxonomy(values=["Solid", "Broken", "Other"]),
                            ),
                            AttributeTaxonomy(
                                name="Road Lane Color",
                                description="The color of the lane",
                                value=ChoicesTaxonomy(values=["White", "Yellow"]),
                            ),
                        ]
                    ),
                ),
            ],
            stuff=[
                CategoryTaxonomy(
                    name="Sky",
                    description="The sky including clouds/sun",
                    attributes=AttributesList([]),
                ),
                CategoryTaxonomy(
                    name="Sidewalk",
                    description="sidewalk excluding the road",
                    attributes=AttributesList([]),
                ),
                CategoryTaxonomy(
                    name="Construction",
                    description="any kind of human made objects",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Construction Type",
                                description="what kind of construction",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Building",
                                        "Wall",
                                        "Fence",
                                        "Bridge",
                                        "Tunnel",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Vegetation",
                    description="vegetation above the level of the ground that may prohibit a vehicle from going in its direction",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vegetation Type",
                                description="type of the vegetation",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Trees",
                                        "Hedges",
                                        "Small Bushes",
                                        "All other Kinds Of Vertical Vegetation",
                                        "Other",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Movable Object",
                    description="any thing that may move later in time",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Movable Object Type",
                                description="the type of the movable object",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Traffic Cones",
                                        "Debris",
                                        "Barriers",
                                        "Push-able or Pull-able",
                                        "Animal",
                                        "Umbrella",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Animal On Ground",
                                description="Is Animal on ground? either on ground or otherwise flying or being carried by someone.",
                                value=ConditionTaxonomy(
                                    values=["On Ground", "No"],
                                    taxonomy="Movable Object Type",
                                    choices=ChoicesTaxonomy(values=["Animal"]),
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Animal Moving by Itself",
                                description="Is the animal moving by itself? if being carried set to No",
                                value=ConditionTaxonomy(
                                    values=["Yes", "No"],
                                    taxonomy="Movable Object Type",
                                    choices=ChoicesTaxonomy(values=["Animal"]),
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Static Object",
                    description="any objects that probably will be there for a long time",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Static Object Type",
                                description="the type of the static object",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Bicycle Rack",
                                        "Pole",
                                        "Rail Track",
                                        "Trash (or anything that holds trash)",
                                        "Fence or Barrier",
                                        "Other",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Ground",
                    description="any ground object such as road and parking spots",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Ground Type",
                                description="the type of the ground? Terrain includes grass, soil and sand.",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Terrain",
                                        "Road",
                                        "Pedestrian Sidewalk",
                                        "Curb",
                                        "Parking Lots and Driveways",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Terrain Type",
                                description="Terrain Type? what type of terrain it is. Set only when the type is terrain, otherwise it doesn't matter",
                                value=ConditionTaxonomy(
                                    values=["Grass", "Soil", "Sand", "Other"],
                                    taxonomy="Ground Type",
                                    choices=ChoicesTaxonomy(values=["Terrain"]),
                                ),
                            ),
                        ]
                    ),
                ),
            ],
        )

    def fetch(self, **kwargs) -> t.Union["CategoryTaxonomy", "AttributeTaxonomy"]:
        assert len(kwargs) > 0, kwargs
        taxonomy = copy.deepcopy(self)
        if "category" in kwargs:
            category = kwargs["category"]
            things = [cat for cat in taxonomy.things if cat.name == category]
            stuff = [cat for cat in taxonomy.stuff if cat.name == category]
            if len(things) == 1 and len(stuff) != 1:
                output = things[0]
                output.meta.update({"Is Thing": True})
            elif len(stuff) == 1:
                output = stuff[0]
                output.meta.update({"Is Thing": False})
            else:
                raise ValueError(f"{category} doesn't exist in things nor stuff")

            if "attribute" in kwargs:
                attribute = kwargs["attribute"]
                attrs = [attr for attr in output.attributes if attr.name == attribute]
                assert len(attrs) == 1, attrs
                is_thing = output.meta["Is Thing"]
                output = attrs[0]
                output.meta.update({"Is Thing": is_thing, "category": category})

            return output
        else:
            raise NotImplementedError(kwargs)


class CsTranslator:
    """
    Converts between Cityscapes way of labeling and our way of labeling
    """

    # _cs2ours:
    #   - keys are cityscapes classes, the values corresponds to our taxonomy
    #   - values is a string matching the category in our taxonomy
    _cs2ours = {
        "road": "Ground",
        "sidewalk": "Sidewalk",
        "parking": "Ground",
        "rail track": "Static Object",
        "person": "Pedestrian",
        "rider": "Pedestrian",
        "car": "Mid-to-Large Vehicle",
        "truck": "Mid-to-Large Vehicle",
        "bus": "Mid-to-Large Vehicle",
        "on rails": "Mid-to-Large Vehicle",
        "motorcycle": "Small Vehicle",
        "bicycle": "Small Vehicle",
        "caravan": "Mid-to-Large Vehicle",
        "trailer": "Mid-to-Large Vehicle",
        "building": "Construction",
        "wall": "Construction",
        "fence": "Construction",
        "guard rail": "Movable Object",
        "bridge": "Construction",
        "tunnel": "Construction",
        "pole": "Static Object",
        "pole group": "Static Object",
        "traffic sign": "Traffic Sign",
        "traffic light": "Traffic Light",
        "vegetation": "Vegetation",
        "terrain": "Ground",
        "sky": "Sky",
        "ground": "Ground",
        "dynamic": "Movable Object",
        "static": "Static Object",
    }

    def __init__(self, sanity_check: bool = True):
        if sanity_check:
            self.is_sane()

    def is_sane(self):
        ours = CompleteTaxonomy.load()
        assert ours.is_sane()
        for k, v in self._cs2ours.items():
            ours.fetch(category=v)

    def Cs2Ours(self, name: str):
        return self._cs2ours[name]

    def Ours2Cs(self, name: str):
        # TODO: implement this one as well, it may requires changing the structure as the mapping is many to one not one to one so far
        raise NotImplementedError()


TAXONOMY = CompleteTaxonomy.load()
CSMap = CsTranslator(sanity_check=True).Cs2Ours


def prompt_question(name: str, message: str, choices: list):
    questions = [
        inquirer.List(
            name=name,
            message=message,
            choices=choices,
        ),
    ]
    item = inquirer.prompt(questions)[name]
    return item


@dataclass
class UniqueID:
    id: str  # the id format is as follows: image_path::label_path::object_id if the default delimiter is used
    delimiter: str = "::"

    def __post_init__(self):
        assert self.is_consistent

    def __repr__(self):
        return f"UniqueID({self.id})"

    @staticmethod
    def construct(
        image_path: str, label_path: str, object_id: str, delimiter: str = "::"
    ) -> "UniqueID":
        image_path = (
            image_path.replace(CsPath(), "") if image_path.startswith(CsPath()) else image_path
        )
        label_path = (
            label_path.replace(CsPath(), "") if label_path.startswith(CsPath()) else label_path
        )
        idx = UniqueID(
            id=f"{image_path}{delimiter}{label_path}{delimiter}{object_id}",
            delimiter=delimiter,
        )
        return idx

    @staticmethod
    def from_df(df: pd.DataFrame, append_to_df: bool = False) -> t.List["UniqueID"]:
        output = [
            UniqueID.construct(image_path=image_path, label_path=label_path, object_id=object_id)
            for image_path, label_path, object_id in zip(
                df["image_path"], df["label_path"], df["object_id"]
            )
        ]
        if append_to_df:
            df["unique_id"] = output
        return output

    def decompose(self) -> t.Mapping[str, t.Union[str, int]]:
        vals = self.id.split(self.delimiter)
        assert len(vals) == 3
        return {"image_path": vals[0], "label_path": vals[1], "object_id": int(vals[2])}

    @property
    def image_path(self) -> str:
        return self.id.split(self.delimiter)[0]

    @property
    def label_path(self) -> str:
        return self.id.split(self.delimiter)[1]

    @property
    def object_id(self) -> int:
        return int(self.id.split(self.delimiter)[2])

    @property
    def is_consistent(self) -> bool:
        image_path = self.image_path.split("/")
        label_path = self.label_path.split("/")
        image_prefix = "_".join(image_path[3].split("_")[:-1])
        label_prefix = "_".join(label_path[3].split("_")[:-2])
        return all(
            [
                image_path[1] == label_path[1],  # same split (train, val or test)
                image_path[2] == label_path[2],  # same city
                image_prefix == label_prefix,  # same prefix of the image/json name
            ]
        )

    @property
    def city(self) -> str:
        return self.image_path.split("/")[2]

    @property
    def split(self) -> str:
        return self.image_path.split("/")[1]

    @property
    def gtType(self):
        return self.label_path.split("/")[0]


@dataclass
class BBox:
    top: int
    bottom: int
    left: int
    right: int

    @staticmethod
    def from_polygon(polygon: t.Sequence[t.Tuple[int, int]]) -> "BBox":
        x_vals, y_vals = tuple(zip(*polygon))
        return BBox(top=min(y_vals), bottom=max(y_vals), left=min(x_vals), right=max(x_vals))

    @property
    def width(self):
        return self.right - self.left + 1

    @property
    def height(self):
        return self.bottom - self.top + 1


class CsSample:
    unique_id: UniqueID
    image_path: str
    label_path: str
    object_id: int
    city: str
    split: str
    gtType: CsLabelType
    _data: t.Optional["CsInstance"] = None
    object: t.Optional["CsPoly"] = None
    image: t.Optional[PIL.Image.Image] = None

    def __init__(self, unique_id: UniqueID, load: bool = False):
        self.unique_id = unique_id
        self.image_path = unique_id.image_path
        self.label_path = unique_id.label_path
        self.object_id = unique_id.object_id
        self.city = unique_id.city
        self.split = unique_id.split
        self.gtType = [
            k for k, v in CsInstance().labelTypes.items() if v.description == unique_id.gtType
        ][0]
        if load:
            self.load()

    def __repr__(self):
        return f"CsSample(UniqueID={self.unique_id.id}{', loaded' if self._data else ''})"

    def load(self):
        self._data = CsInstance(self)
        self._data.pickObject(self.object_id)  # clears all annotations but the object of interest
        self.object = self._data.annotation.objects[0]
        self.image = self._data.image

    @property
    def bbox(self):
        if not self._data:
            self.load()
        assert self._data and self.object
        assert len(self._data.annotation.objects) == 1
        return BBox.from_polygon(self.object.toJsonText()["polygon"])

    @property
    def polygon(self):
        if not self._data:
            self.load()
        assert self._data and self.object
        assert len(self._data.annotation.objects) == 1
        return [
            {
                "y": point.y,
                "x": point.x,
            }
            for point in self.object.polygon
        ]


class CsInstance:
    def __init__(self, sample: t.Optional[CsSample] = None, user: bool = False):
        # The filename of the image we currently working on
        self.currentFile = ""
        # The filename of the labels we currently working on
        self.currentLabelFile = ""
        # The path of the images of the currently loaded city
        self.city = ""
        # The name of the currently loaded city
        self.cityName = ""
        # The name of the current split
        self.split = ""
        # Ground truth type
        self.gtType = CsLabelType.NONE
        # The path of the labels. In this folder we expect a folder for each city
        # Within these city folders we expect the label with a filename matching
        # the images, except for the extension
        self.labelPath = ""
        # Filenames of all images in current city
        self.images = []
        # Image extension
        self.imageExt = "_leftImg8bit.png"
        # Ground truth extension
        self.gtExt = "_gt*.json"
        # Current image as QImage
        self._image = PIL.Image.Image()
        self._loadedImageFile = ""
        # Index of the current image within the city folder
        self.idx = 0
        # All annotated objects in current image, i.e. list of csPoly or csBbox
        self.annotation = Annotation()
        # Enable disparity visu in general
        self.enableDisparity = True
        # The filename of the disparity map we currently working on
        self.currentDispFile = ""
        # The disparity image
        self.dispImg = None
        # The disparity search path
        self.dispPath = None
        # Disparity extension
        self.dispExt = "_disparity.png"
        # Available label types
        self.labelTypes = {
            CsLabelType.POLY_FINE: LabelType("gtFine", "gtFine", CsObjectType.POLY),
            CsLabelType.POLY_COARSE: LabelType("gtCoarse", "gtCoarse", CsObjectType.POLY),
            CsLabelType.CS3D_BBOX3D: LabelType("CS3D: 3D Boxes", "gtBbox3d", CsObjectType.BBOX3D),
            CsLabelType.CS3D_BBOX2D_MODAL: LabelType(
                "CS3D: Modal 2D Boxes", "gtBbox3d", CsObjectType.BBOX3D
            ),
            CsLabelType.CS3D_BBOX2D_AMODAL: LabelType(
                "CS3D: Amodal 2D Boxes", "gtBbox3d", CsObjectType.BBOX3D
            ),
            CsLabelType.CITYPERSONS_BBOX2D: LabelType(
                "Citypersons", "gtBboxCityPersons", CsObjectType.BBOX2D
            ),
            CsLabelType.DISPARITY: LabelType("Stereo Disparity", "disparity", CsObjectType.POLY),
        }
        assert not (sample is not None and user is True)
        if user is True:
            self.getCityFromUser()
            self.loadCity()
            self.imageChanged()
        elif sample:
            self.getCityFromConfig(sample)
            self.selectImageFromPath(sample.image_path)

    # Switch to previous image in file list
    # Load the image
    # Load its labels
    def prevImage(self):
        if not self.images:
            return
        if self.idx > 0:
            self.idx -= 1
            self.imageChanged()
        else:
            message = "Already at the first image"
            print(message)
        return

    # Switch to next image in file list
    # Load the image
    # Load its labels
    def nextImage(self):
        if not self.images:
            return
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self.imageChanged()
        else:
            message = "Already at the last image"
            print(message)
        return

    # Switch to a selected image of the file list
    # Ask the user for an image
    # Load the image
    # Load its labels
    def selectImage(self):
        if not self.images:
            return

        dlgTitle = "Select image to load"
        print(dlgTitle)
        items = [os.path.basename(i) for i in self.images]
        item = prompt_question("image", dlgTitle, items)
        idx = items.index(item)
        if idx != self.idx:
            self.idx = idx
            self.imageChanged()

    def selectImageFromPath(self, path: str):
        if not self.images:
            return

        items = [os.path.basename(i) for i in self.images]
        item = os.path.basename(path)
        idx = items.index(item)
        if idx != self.idx:
            self.idx = idx
            self.imageChanged()

    #############################
    # Custom events
    #############################

    def imageChanged(self):
        # Load the first image
        self.loadImage()
        # Load its labels if available
        self.loadLabels()
        # Load disparities if available
        self.loadDisparities()

    #############################
    # File I/O
    #############################

    # Load the currently selected city if possible
    def loadCity(self):
        # clear annotations
        self.annotation = []
        # Search for all *.pngs to get the image list
        self.images = []
        if os.path.isdir(self.city):
            self.images = glob.glob(os.path.join(self.city, "*" + self.imageExt))
            self.images.sort()
            if self.currentFile in self.images:
                self.idx = self.images.index(self.currentFile)
            else:
                self.idx = 0

    # Load the currently selected image
    # Does only load if not previously loaded
    # Does not refresh the GUI
    def loadImage(self):
        # message = "Loading image method"
        if self.images:
            filename = self.images[self.idx]
            filename = os.path.normpath(filename)
            if not filename == self.currentFile:
                self.currentFile = filename
                # self.image = PIL.Image.open(filename).convert("RGB")
                # try:
                #     self.image.verify()
                #     # message = "Read image: {0}".format(filename)
                # except Exception:
                #     message = "Failed to read image: {0}".format(filename)
                #     print(message)

    @property
    def image(self) -> PIL.Image.Image:
        if self._loadedImageFile == self.currentFile:
            return self._image
        else:
            self._image = PIL.Image.open(self.currentFile).convert("RGB")
            self._image.verify()
            self._loadedImageFile = self.currentFile
            return self._image

    # Load the labels from file
    # Only loads if they exist
    # Otherwise the filename is stored and that's it
    def loadLabels(self):
        filename = self.getLabelFilename()
        if not filename:
            self.clearAnnotation()
            return

        # If we have everything and the filename did not change, then we are good
        if self.annotation and filename == self.currentLabelFile:
            return

        # Clear the current labels first
        self.clearAnnotation()

        try:
            self.annotation = Annotation(self.labelTypes[self.gtType].objectType)
            self.annotation.fromJsonFile(filename)
            # FIXME: we are shifting the coords vertically as it doesn't match on scaleapi, will need to be revisited (it now only has the shift issue on the dashborad, it doesn't have it on the actual sumbitted task)
            self.annotation = shift_polygons(self.annotation)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing labels in {0}. Message: {1}".format(filename, e.strerror)
            print(message)

        # Remember the filename loaded
        self.currentLabelFile = filename

    # Load the disparity map from file
    # Only loads if they exist
    def loadDisparities(self):
        if not self.enableDisparity:
            return
        if not self.gtType == CsLabelType.DISPARITY:
            return

        filename = self.getDisparityFilename()
        if not filename:
            self.dispImg = None
            return

        # If we have everything and the filename did not change, then we are good
        if self.dispImg and filename == self.currentDispFile:
            return

        # Clear the current labels first
        self.dispImg = None

        try:
            self.dispImg = PIL.Image.open(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing disparities in {0}. Message: {1}".format(filename, e.strerror)
            print(message)
            self.dispImg = None

        # Remember the filename loaded
        self.currentDispFile = filename

    # Clear the current labels
    def clearAnnotation(self):
        self.annotation = None
        self.currentLabelFile = ""

    @property
    def AvailableLabelTypes(self):
        availableLabelTypes = []
        gtDirs = [os.path.basename(path) for path in glob.glob(os.path.join(CsPath(), "*"))]

        for gtType in self.labelTypes:
            if self.labelTypes[gtType].gtDir in gtDirs:
                if gtType != CsLabelType.DISPARITY or self.enableDisparity:
                    availableLabelTypes.append(self.labelTypes[gtType].description)
        return availableLabelTypes

    # Get the label type to view
    def getLabelTypeFromUser(self):
        if self.cityName == "" or self.split == "":
            return

        # Specify title
        dlgTitle = "Select new label type"
        message = "Select label type for viewing"
        question = "Which label type would you like to view?"
        print(message)

        if self.AvailableLabelTypes:
            item = prompt_question(dlgTitle, question, self.AvailableLabelTypes)

            self.gtType = [k for k, v in self.labelTypes.items() if v.description == item][0]

            self.city = os.path.normpath(
                os.path.join(CsPath(), "leftImg8bit", self.split, self.cityName)
            )
            self.labelPath = os.path.normpath(
                os.path.join(
                    CsPath(),
                    self.labelTypes[self.gtType].gtDir,
                    self.split,
                    self.cityName,
                )
            )
            self.dispPath = os.path.normpath(
                os.path.join(CsPath(), "disparity", self.split, self.cityName)
            )

            self.loadCity()
            self.imageChanged()

    def getLabelTypeFromConfig(self, gtType: CsLabelType):
        if self.cityName == "" or self.split == "":
            return

        if self.AvailableLabelTypes:
            self.gtType = gtType

            self.city = os.path.normpath(
                os.path.join(CsPath(), "leftImg8bit", self.split, self.cityName)
            )
            self.labelPath = os.path.normpath(
                os.path.join(
                    CsPath(),
                    self.labelTypes[self.gtType].gtDir,
                    self.split,
                    self.cityName,
                )
            )
            self.dispPath = os.path.normpath(
                os.path.join(CsPath(), "disparity", self.split, self.cityName)
            )

            self.loadCity()
            self.imageChanged()

    @property
    def AvailableCities(self):
        availableCities = []
        splits = ["train_extra", "train", "val", "test"]
        for split in splits:
            cities = glob.glob(os.path.join(CsPath(), "leftImg8bit", split, "*"))
            cities.sort()
            availableCities.extend([(split, os.path.basename(c)) for c in cities if os.listdir(c)])

        # List of possible labels
        items = [split + ", " + city for (split, city) in availableCities]
        return items

    def ChangeCity(
        self,
        city: str,
        split: str,
        from_user: bool,
        gtType: t.Optional[CsLabelType] = None,
    ):
        self.cityName = city
        self.split = split

        if gtType is not None:
            self.gtType = gtType

        if self.gtType != CsLabelType.NONE:
            self.city = os.path.normpath(
                os.path.join(CsPath(), "leftImg8bit", self.split, self.cityName)
            )
            self.labelPath = os.path.normpath(
                os.path.join(
                    CsPath(),
                    self.labelTypes[self.gtType].gtDir,
                    self.split,
                    self.cityName,
                )
            )
            self.dispPath = os.path.normpath(
                os.path.join(CsPath(), "disparity", self.split, self.cityName)
            )
            self.loadCity()
            self.imageChanged()

        else:
            if from_user:
                self.getLabelTypeFromUser()
            else:
                self.getLabelTypeFromConfig(gtType=gtType)

    def getCityFromUser(self):
        items = self.AvailableCities
        # Specify title
        dlgTitle = "Select new city"
        message = "Select city for viewing"
        question = "Which city would you like to view?"
        print(message)
        if items:
            item = prompt_question(dlgTitle, question, items)
            (split, city) = [str(i) for i in item.split(", ")]
            self.ChangeCity(city=city, split=split, from_user=True)
        else:
            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the Cityscapes root folder\n"
            warning += "or\n"
            warning += " - set CITYSCAPES_DATASET to the Cityscapes root folder\n"
            warning += "       e.g. 'export CITYSCAPES_DATASET=<root_path>'\n"
            print("ERROR!")
            print(warning)
        return

    def getCityFromConfig(self, sample: CsSample):
        items = self.AvailableCities
        city = sample.city
        split = sample.split
        if items:
            self.ChangeCity(city=city, split=split, from_user=False, gtType=sample.gtType)
        else:
            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the Cityscapes root folder\n"
            warning += "or\n"
            warning += " - set CITYSCAPES_DATASET to the Cityscapes root folder\n"
            warning += "       e.g. 'export CITYSCAPES_DATASET=<root_path>'\n"
            print("ERROR!")
            print(warning)
        return

    # Determine if the given candidate for a label path makes sense
    @staticmethod
    def isLabelPathValid(labelPath):
        return os.path.isdir(labelPath)

    # Get the filename where to load labels
    # Returns empty string if not possible
    def getLabelFilename(self):
        # And we need to have a directory where labels should be searched
        if not self.labelPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not self.isLabelPathValid(self.labelPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename(self.currentFile)
        filename = filename.replace(self.imageExt, self.gtExt)
        filename = os.path.join(self.labelPath, filename)
        search = glob.glob(filename)
        if not search:
            return ""
        filename = os.path.normpath(search[0])
        return filename

    # Get the filename where to load disparities
    # Returns empty string if not possible
    def getDisparityFilename(self):
        # And we need to have a directory where disparities should be searched
        if not self.dispPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not os.path.isdir(self.dispPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename(self.currentFile)
        filename = filename.replace(self.imageExt, self.dispExt)
        filename = os.path.join(self.dispPath, filename)
        filename = os.path.normpath(filename)
        return filename

    def pickObject(self, object_id: int):
        self.annotation.objects = [obj for obj in self.annotation.objects if obj.id == object_id]
        assert len(self.annotation.objects) == 1


class Index:
    _current_index: int
    _range: t.Optional[int]

    def __repr__(self):
        return f"Index(current_index at {self._current_index} out of {self._range})"

    @property
    def is_sane(self):
        if self._range is not None:
            if not 0 <= self._current_index < self._range:
                raise IndexError()
        return True

    def __init__(self, range: t.Optional[int], index: int = 0):
        self._current_index = index
        self._range = range
        assert self.is_sane

    def update(self, new_range: int, new_index: int = 0):
        self._current_index = new_index
        self._range = new_range
        assert self.is_sane

    def __add__(self, other: int):
        assert self._range
        current_index = self._current_index + other
        if not self._range > current_index >= 0:
            raise IndexError()
        self._current_index = current_index

    def __eq__(self, other: int):
        assert self._range
        return self._current_index == other

    def __index__(self):
        assert self._range
        return self._current_index

    def increment(self):
        self.__add__(1)

    def reset(self):
        assert self._range
        self._current_index = 0

    def __iter__(self):
        while True:
            yield self.__index__() if self._range else None
            try:
                self.increment()
            except IndexError:
                break

    def __str__(self):
        return str(self._current_index)

    def __int__(self):
        return self._current_index if self._range else None

    def __len__(self):
        return self._range if self._range else 0


class CsIterator:
    _instance: CsInstance
    _available_cities: t.List[t.Tuple[str]]
    _attempt_label_types: t.List[CsLabelType] = [
        CsLabelType.POLY_FINE,
        CsLabelType.POLY_COARSE,
    ]

    def __init__(self):
        self._instance = CsInstance(sample=None, user=False)
        self._available_cities = [i.split(", ") for i in self._instance.AvailableCities]

        self.current_city_split_index = Index(range=len(self._available_cities), index=0)
        self.current_label_type_index = Index(range=len(self._attempt_label_types), index=0)
        self.city_changed()

        self.current_image_index = Index(range=len(self._instance.images), index=0)
        self.current_object_index = Index(range=len(self._instance.annotation.objects), index=0)

    @property
    def current_split(self):
        return self._available_cities[self.current_city_split_index][0]

    @property
    def current_city(self):
        return self._available_cities[self.current_city_split_index][1]

    @property
    def current_label_type(self):
        return self._attempt_label_types[self.current_label_type_index]

    @property
    def current_image(self):
        return self._instance.images[self.current_image_index]

    def image_changed(self):
        self._instance.selectImageFromPath(path=self.current_image)
        self.current_object_index.update(
            new_range=len(self._instance.annotation.objects) if self._instance.annotation else None
        )

    def city_changed(self):
        self._instance.ChangeCity(
            city=self.current_city,
            split=self.current_split,
            from_user=False,
            gtType=self.current_label_type,
        )
        if hasattr(self, "current_image_index"):
            self.current_image_index.update(new_range=len(self._instance.images))

    def label_changed(self):
        self.current_city_split_index.reset()
        self.city_changed()

    def generator(self, start_index: int = 0) -> t.Generator:
        def generate():
            """Cityscapes Generator"""
            counter = 0
            for _ in self.current_label_type_index:
                self.label_changed()
                for _ in self.current_city_split_index:
                    # print(self.current_city_split_index)
                    self.city_changed()
                    for _ in self.current_image_index:
                        # print(self.current_image_index)
                        self.image_changed()
                        for _ in self.current_object_index:
                            # print(self.current_object_index)
                            if counter < start_index:
                                counter = counter + 1
                                continue
                            else:
                                yield CsSample(
                                    unique_id=UniqueID.construct(
                                        image_path=self._instance.currentFile,
                                        label_path=self._instance.currentLabelFile,
                                        object_id=str(self.current_object_index),
                                    ),
                                    load=True,
                                )

        return generate()

    @classmethod
    def len(cls, label_type: t.Optional[CsLabelType]) -> int:
        count = 0
        self = cls()
        for _ in self.current_label_type_index:
            self.label_changed()
            if self.current_label_type == label_type or label_type is None:
                for _ in self.current_city_split_index:
                    self.city_changed()
                    for _ in self.current_image_index:
                        self.image_changed()
                        count += len(self.current_object_index)
        return count

    @classmethod
    def __len__(cls) -> int:
        return cls().len(None)


def main():
    # test = CsIterator().current_object_index
    cs_iterator = CsIterator().generator(start_index=0)
    print(CsIterator.len(label_type=CsLabelType.POLY_FINE))
    print(CsIterator.len(label_type=CsLabelType.POLY_COARSE))
    print(len(CsIterator()))

    # cs_iterator = CsIterator().generator()
    passed = []
    for idx, sample in enumerate(cs_iterator):
        unique_id = sample.unique_id
        print(f"{idx} = {unique_id}")
        if unique_id in passed:
            print("found one")
            print(passed)
            print(idx)
            print(unique_id)
            raise ValueError
        passed.append(unique_id)
        pass

    instance = CsInstance(
        CsSample(
            UniqueID(
                "leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png::gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json::0"
            )
        )
    )
    objects_json = [obj.toJsonText() for obj in instance.annotation.objects]
    print(objects_json)

    frames = [pd.DataFrame.from_dict(object_json) for object_json in objects_json]
    df = pd.concat(frames)
    print(df.columns)

    df = df.rename(columns={"id": "object_id"})
    df = df[["label", "object_id", "polygon"]]
    df["city"] = instance.cityName
    df["label_type"] = instance.labelTypes[instance.gtType].description
    df["label_path"] = instance.currentLabelFile.replace(CsPath(), "")
    df["image_path"] = instance.currentFile.replace(CsPath(), "")
    UniqueID.from_df(df, append_to_df=True)
    print(df.head())

    # df.to_csv("/home/krm/models/ml_linter_2d/cityscapes_labeling/test_sample.csv")

    sample = CsSample(df.iloc[0]["unique_id"])
    sample.load()


if __name__ == "__main__":
    main()
