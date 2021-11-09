import json
import os
import typing as t
from dataclasses import dataclass

import PIL.Image as pil
from PIL import ImageDraw
from PIL.Image import Image

cities = [
    "train/aachen",
    "train/bochum",
    "train/bremen",
    "train/cologne",
    "train/darmstadt",
    "train/dusseldorf",
    "train/erfurt",
    "train/hamburg",
    "train/hanover",
    "train/jena",
    "train/krefeld",
    "train/monchengladbach",
    "train/strasbourg",
    "train/stuttgart",
    "train/tubingen",
    "train/ulm",
    "train/weimar",
    "train/zurich",
    "val/frankfurt",
    "val/lindau",
    "val/munster",
    "test/berlin",
    "test/bielefeld",
    "test/bonn",
    "test/leverkusen",
    "test/mainz",
    "test/munich",
]


def cs_path():
    assert "CITYSCAPES_DATASET" in os.environ
    return os.environ["CITYSCAPES_DATASET"]


def attributes_path():
    return os.path.join(cs_path(), "attributes")


def city_attribute_path(city: str):
    return os.path.join(attributes_path(), city + ".json")


@dataclass
class CARUniqueID:
    id: str  # the id format is as follows: image_path::label_path::object_id if the default delimiter is used
    delimiter: str = "::"

    def __post_init__(self):
        assert self.is_consistent

    def __repr__(self):
        return f"CARUniqueID({self.id})"

    @staticmethod
    def construct(
        image_path: str, label_path: str, object_id: str, delimiter: str = "::"
    ) -> "CARUniqueID":
        image_path = (
            image_path.replace(cs_path(), "") if image_path.startswith(cs_path()) else image_path
        )
        label_path = (
            label_path.replace(cs_path(), "") if label_path.startswith(cs_path()) else label_path
        )
        idx = CARUniqueID(
            id=f"{image_path}{delimiter}{label_path}{delimiter}{object_id}", delimiter=delimiter
        )
        return idx

    # @staticmethod
    # def from_df(df: pd.DataFrame, append_to_df: bool = False) -> t.List["UniqueID"]:
    #     output = [
    #         UniqueID.construct(image_path=image_path, label_path=label_path, object_id=object_id)
    #         for image_path, label_path, object_id in zip(
    #             df["image_path"], df["label_path"], df["object_id"]
    #         )
    #     ]
    #     if append_to_df:
    #         df["unique_id"] = output
    #     return output

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
    def gt_type(self):
        return self.label_path.split("/")[0]


class CARPoint:
    x: int
    y: int

    def __init__(self, **kwargs):
        if "value" in kwargs:
            assert "x" not in kwargs and "y" not in kwargs
            value = kwargs["value"]
            assert set(value.keys()) == {"x", "y"}, value.keys()
            self.x = value["x"]
            self.y = value["y"]
        else:
            assert set(kwargs.keys()) == {"x", "y"}, kwargs.keys()
            self.x = kwargs["x"]
            self.y = kwargs["y"]
        assert isinstance(self.x, int)
        assert isinstance(self.y, int)

    def __repr__(self):
        return f"CARPoint(x={self.x}, y={self.y})"

    @property
    def json(self):
        return json.dumps(self.dict)

    @staticmethod
    def from_json(text: str) -> "CARPoint":
        return CARPoint(value=json.loads(text))

    @property
    def dict(self) -> t.Dict[str, int]:
        return {"x": self.x, "y": self.y}

    @property
    def tuple(self) -> t.Tuple[int, int]:
        return self.x, self.y


class CARPolygon:
    points: t.Sequence[CARPoint]

    def __init__(self, points: t.Sequence[t.Union[t.Mapping[str, int], CARPoint]]):
        if isinstance(points[0], t.Mapping):
            points = [CARPoint(value=point) for point in points]
        self.points = points

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"CARPolygon(points=[{', '.join([str(p) for p in self.points])}])"

    def __getitem__(self, item: int):
        return self.points[item]

    def __iter__(self):
        for p in self.points:
            yield p

    @property
    def json(self):
        return json.dumps(self.dict)

    @staticmethod
    def from_json(text) -> "CARPolygon":
        data = json.loads(text)
        return CARPolygon(points=[CARPoint.from_json(d) for d in data])

    @property
    def dict(self) -> t.Sequence[t.Dict[str, int]]:
        return [p.dict for p in self.points]

    @property
    def list(self) -> t.Sequence[t.Dict[str, int]]:
        return [p.tuple for p in self.points]


class CARInstance:
    unique_id: CARUniqueID
    category: str
    polygon_annotations: CARPolygon
    attributes: t.Mapping[str, str]
    _meta: t.Optional[t.Mapping[str, t.Any]]

    def __init__(
        self,
        unique_id: str,
        category: str,
        polygon_annotations,
        attributes: t.Mapping[str, str],
        _meta: t.Optional[t.Mapping[str, t.Any]] = None,
    ):
        self.unique_id = CARUniqueID(unique_id)
        self.category = category
        self.attributes = attributes
        self._meta = _meta
        self.polygon_annotations = (
            polygon_annotations
            if isinstance(polygon_annotations, CARPolygon)
            else CARPolygon(polygon_annotations)
        )
        assert self.is_valid_types()

    def is_valid_types(self) -> bool:
        assert isinstance(self.unique_id, CARUniqueID), type(self.unique_id)
        assert isinstance(self.category, str), type(self.category)
        assert isinstance(self.polygon_annotations, CARPolygon), self.polygon_annotations
        assert isinstance(self.attributes, t.Mapping), self.attributes
        if self._meta:
            assert isinstance(self._meta, t.Mapping)
        return True

    def __repr__(self):
        return (
            f'CARInstance(unique_id="{self.unique_id.id}", category="{self.category}", '
            f"polygon_annotations={self.polygon_annotations}, attributes={self.attributes})"
        )

    @property
    def json(self):
        return json.dumps(self.dict)

    @staticmethod
    def from_json(text: str) -> "CARInstance":
        data = json.loads(text)
        data["polygon_annotations"] = CARPolygon.from_json(data["polygon_annotations"]).points
        return CARInstance(**data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.json, f)

    @staticmethod
    def load(path: str) -> "CARInstance":
        with open(path, "r") as f:
            data = json.load(f)
        return CARInstance.from_json(data)

    @property
    def dict(self) -> t.Dict[str, t.Any]:
        out = {
            "unique_id": self.unique_id.id,
            "polygon_annotations": self.polygon_annotations.dict,
            "attributes": self.attributes,
            "category": self.category,
        }
        if self._meta:
            out.update({"_meta": self._meta})
        return out

    @staticmethod
    def from_dict(dct) -> "CARInstance":
        return CARInstance(**dct)

    @property
    def image(self) -> Image:
        image_path = os.path.join(cs_path(), self.unique_id.image_path)
        return pil.open(image_path).convert("RGB")

    def show_annotations(self, fill):
        image = self.image
        # font = ImageFont.truetype(size=16)
        draw = ImageDraw.Draw(image)
        draw.polygon(self.polygon_annotations.list, outline=1, fill=fill)
        draw.text((0, 0), self.attributes, fill=(255, 255, 255))
        image.show()


class CARInstances:
    instances: t.List[CARInstance]

    def __init__(self, instances: t.List[CARInstance]):
        self.instances = list(instances)

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __getitem__(self, item: int):
        return self.instances[item]

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return f"CARInstances(instances={self.instances})"

    def __add__(
        self, other: t.Union[CARInstance, "CARInstances", t.Sequence[CARInstance]]
    ) -> "CARInstances":
        if isinstance(other, CARInstance):
            other = [other]
        instances = other.instances if isinstance(other, CARInstances) else other
        self.instances.extend(instances)
        return self

    @staticmethod
    def from_json(text: str) -> "CARInstances":
        instances = [CARInstance.from_dict(d) for d in text]
        return CARInstances(instances=instances)

    @staticmethod
    def load(path: str) -> "CARInstances":
        with open(path, "r") as f:
            data = json.load(f)
        return CARInstances.from_json(data)


if __name__ == "__main__":
    file = "/home/krm/datasets/car_api/all.json"
    x = CARInstances.load(file)
    pass
