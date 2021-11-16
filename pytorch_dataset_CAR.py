import os
import typing as t

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from dataclasses_CAR import AugmentationsConfig, CARDatasetConfig, ModelInputItem, ModelInputItems
from taxonomy_CAR import TAXONOMY
from utils_CAR import CARInstances, TaxonomyCoDec, attributes_path, cs_path, images_crop


class CARDataset(Dataset):
    car_data: CARInstances
    codec: TaxonomyCoDec

    def __init__(
        self,
        configs: CARDatasetConfig,
        mode: t.Literal["train", "test", "val"],
    ):
        super(CARDataset, self).__init__()
        assert mode in [
            "train",
            "val",
            "test",
        ], f"mode = {mode} should be either 'train', 'val' or 'test'"
        self.configs = configs
        self.mode = mode
        self.keep_square = configs.keep_square
        self.path = configs.path if configs.path else cs_path()
        if self.path != cs_path():
            os.environ["CITYSCAPES_DATASET"] = self.path

        self.augmentations = []
        self.augmentations_mask = []
        if configs.augmentations is not None:
            if configs.augmentations.resize is not None:
                self.augmentations.append(
                    T.Resize(size=(configs.augmentations.resize, configs.augmentations.resize))
                )
                self.augmentations_mask.append(
                    T.Resize(size=(configs.augmentations.resize, configs.augmentations.resize))
                )
            self.augmentations.append(T.ToTensor())
            self.augmentations_mask.append(T.ToTensor())
            if configs.augmentations.normalize:
                self.augmentations.append(
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                )
        else:
            self.augmentations.append(T.ToTensor())
            self.augmentations_mask.append(T.ToTensor())

        self.augmentations = T.Compose(self.augmentations)
        self.augmentations_mask = T.Compose(self.augmentations_mask)

        car_json_path = attributes_path(f"{mode}.json")
        self.car_data = CARInstances.load(car_json_path)
        self.codec = TaxonomyCoDec()

        # eliminating samples without attributes
        eliminated = []
        for cat in TAXONOMY:
            if len(cat.attributes) == 0:  # doesn't have attributes
                eliminated.append(cat.name)
        eliminated = {"category_name": eliminated}
        self.car_data = CARInstances(
            sample for sample in self.car_data if sample.category not in eliminated["category_name"]
        )

        if configs.n_samples:
            self.car_data = self.car_data[: configs.n_samples]

    def __getitem__(self, idx: int) -> ModelInputItem:
        for trial in range(10):
            try:
                sample = self.car_data[idx]
                instance_id = torch.tensor(sample.instance_id)
                image = sample.image_nocache
                cls_name = sample.category
                cls_id, attr_vector = self.codec.encode(sample, return_vector=True)
                mask = sample.binary_mask
                cropped_image, cropped_mask = images_crop(
                    images=[image, mask],
                    bbox=sample.polygon_annotations.bbox,
                    keep_square=self.keep_square,
                )

                image = self.augmentations(image)
                mask = self.augmentations_mask(mask)
                cropped_image = self.augmentations(cropped_image)
                cropped_mask = self.augmentations_mask(cropped_mask)
                output = ModelInputItem(
                    image=image,
                    mask=mask,
                    cropped_image=cropped_image,
                    cropped_mask=cropped_mask,
                    category_id=torch.tensor(cls_id),
                    category_name=cls_name,
                    attributes_label=torch.Tensor(attr_vector),
                    id=torch.tensor(idx),
                    instance_id=instance_id,
                )
                if self.configs.scale:
                    output.instances_tensor = torch.from_numpy(
                        sample.instances_matrix(self.configs.scale)
                    ).to(dtype=torch.float)

                return output

            except Exception as err:
                print(f"idx is {idx}")
                print(f"Error is {err}")
                if trial < 9:
                    continue
                else:
                    raise err

    def __len__(self):
        return len(self.car_data)

    @staticmethod
    def collate_fn(batch: t.List[ModelInputItem]) -> ModelInputItems:
        return ModelInputItems.collate(batch)

    @property
    def n_categories(self):
        return self.codec.n_categories

    @property
    def n_attributes(self):
        return self.codec.n_attributes


if __name__ == "__main__":
    config = CARDatasetConfig(
        path=cs_path(),
        n_samples=None,
        augmentations=AugmentationsConfig(resize=224, normalize=False),
        keep_square=False,
        scale=28,
    )
    dataset = CARDataset(configs=config, mode="train")
    collate_fn = dataset.collate_fn
    attributes_len = dataset.codec.n_attributes
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    for i, item in tqdm(enumerate(data_loader), total=len(data_loader)):
        obj = item[0]
        print(dataset.codec.decode(obj.category_id, obj.attributes_label))
        obj.show()
