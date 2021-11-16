import typing as t
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.transforms import ToPILImage


@dataclass
class AugmentationsConfig:
    resize: t.Optional[int]  # the final shape of the cropped image will be resize x resize
    normalize: bool = True  # normalize based on pretrained pytorch models with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]


@dataclass
class CARDatasetConfig:
    path: t.Optional[str]  # should contain root path for Cityscapes
    keep_square: bool  # when cropping should we try to keep aspect ratio while making it square
    augmentations: t.Optional[AugmentationsConfig]
    n_samples: t.Optional[
        int
    ] = None  # set the size of the training/val/test datasets (limiting the length of the dataset to that size)
    scale: t.Optional[int] = None  # if set the output is scaled to that spatial size

    def __hash__(self):
        return hash((hash(item) for item in self.__dict__))

    @staticmethod
    def Model(
        configs: "CARDatasetConfig",
        mode: str,
    ):
        from pytorch_dataset_CAR import CARDataset

        return CARDataset(configs=configs, mode=mode)


@dataclass
class ModelInputItem:
    image: Tensor
    mask: Tensor
    cropped_image: Tensor
    cropped_mask: Tensor
    class_id: Tensor
    class_name: str
    attributes_label: Tensor
    id: Tensor
    instance_id: Tensor
    instances_tensor: t.Optional[Tensor] = None

    def __post_init__(self):
        assert (
            len(self.cropped_mask.shape) == len(self.cropped_image.shape) == 3
        ), f"The cropped image and mask should be a 3D Tensor, got {self.cropped_image.shape} and {self.cropped_mask.shape}"
        assert (
            self.image.shape[-2:] == self.mask.shape[-2:]
        ), f"The size of the mask ({self.mask.shape}) and image ({self.image.shape}) should be equal"
        assert self.mask.shape[0] == 1
        assert self.image.shape[0] == 3
        assert (
            self.cropped_image.shape[-2:] == self.cropped_mask.shape[-2:]
        ), f"The size of the cropped mask ({self.cropped_mask.shape}) and cropped image ({self.cropped_image.shape}) should be equal"
        assert (
            self.cropped_mask.shape[0] == 1
        ), f"The size of the channels of the mask should be 1, got {self.cropped_mask.shape}"
        assert (
            len(self.attributes_label.shape) == 1
        ), f"the attributes label should be a single dimensional tensor, got {self.attributes_label.shape}"
        if self.instances_tensor is not None:
            assert self.instances_tensor.shape[0] == 6, self.instances_tensor.shape
            assert len(self.instances_tensor.shape) == 3, self.instances_tensor.shape

    def to_device(self, device: torch.device):
        return ModelInputItem(
            image=self.image.to(device),
            mask=self.mask.to(device),
            class_name=self.class_name,
            cropped_image=self.cropped_image.to(device),
            cropped_mask=self.cropped_mask.to(device),
            class_id=self.class_id.to(device),
            attributes_label=self.attributes_label.to(device),
            id=self.id.to(device),
            instance_id=self.instance_id.to(device),
            instances_tensor=self.instances_tensor.to(device)
            if self.instances_tensor is not None
            else None,
        )

    def show(self) -> None:
        ToPILImage()(self.image).show()


@dataclass
class ModelInputItems:
    images: Tensor
    masks: Tensor
    cropped_images: Tensor
    cropped_masks: Tensor
    class_ids: Tensor
    class_names: t.List[str]
    attributes_labels: Tensor
    ids: Tensor
    instance_ids: Tensor
    instances_tensor: t.Optional[Tensor] = None

    def __post_init__(self):
        assert (
            len(self.cropped_masks.shape) == len(self.cropped_images.shape) == 4
        ), f"The cropped image and mask should be a 4D Tensor, got {self.cropped_images.shape} and {self.cropped_masks.shape}"
        assert (
            len(self.images.shape) == len(self.masks.shape) == 4
        ), f"The image and mask should be a 4D Tensor, got {self.images.shape} and {self.masks.shape}"
        assert (
            self.images.shape[0] == self.masks.shape[0]
        ), f"The batch size is inconsistent, got {self.images.shape} and {self.masks.shape}"
        assert (
            self.images.shape[-2:] == self.masks.shape[-2:]
        ), f"Image and mask H&W inconsistency, got {self.images.shape} and {self.masks.shape}"
        # assert len(self.images) == len(
        #     self.masks
        # ), f"The number of images and masks should match, got {len(self.images)} and {len(self.masks)}"
        # for image, mask in zip(self.images, self.masks):
        #     assert (
        #         mask.size == image.size
        #     ), f"The size of the mask ({mask.size}) and image ({image.size}) should be equal"
        assert (
            self.cropped_images.shape[-2:] == self.cropped_masks.shape[-2:]
        ), f"The size of the cropped mask ({self.cropped_masks.shape}) and cropped image ({self.cropped_images.shape}) should be equal"
        assert (
            self.cropped_masks.shape[1] == 1
        ), f"The size of the channels of the mask should be 1, got {self.cropped_masks.shape}"
        if self.instances_tensor is not None:
            assert self.instances_tensor.shape[1] == 6, self.instances_tensor.shape
            assert len(self.instances_tensor.shape) == 4, self.instances_tensor.shape
        self.class_ids = self.class_ids.to(dtype=torch.int64)
        assertion_list = [
            len(self.attributes_labels),
            len(self.class_names),
            len(self.images),
            len(self.masks),
            self.cropped_masks.shape[0],
            self.cropped_images.shape[0],
            self.class_ids.shape[0],
            self.attributes_labels.shape[0],
            self.ids.shape[0],
            self.instance_ids.shape[0],
        ]
        if self.instances_tensor is not None:
            assertion_list.append(self.instances_tensor.shape[0])
        assert set(
            int(x) for x in assertion_list
        ), "The batch size should be the same over all elements of the batch"

    @staticmethod
    def collate(model_input_items: t.Sequence[ModelInputItem]) -> "ModelInputItems":
        return ModelInputItems(
            images=torch.stack([item.image for item in model_input_items], dim=0),
            masks=torch.stack([item.mask for item in model_input_items], dim=0),
            cropped_images=torch.stack([item.cropped_image for item in model_input_items], dim=0),
            cropped_masks=torch.stack([item.cropped_mask for item in model_input_items], dim=0),
            class_ids=torch.stack([item.class_id for item in model_input_items], dim=0),
            class_names=[item.class_name for item in model_input_items],
            attributes_labels=torch.stack(
                [item.attributes_label for item in model_input_items], dim=0
            ),
            ids=torch.stack([item.id for item in model_input_items], dim=0),
            instance_ids=torch.stack([item.instance_id for item in model_input_items], dim=0),
            instances_tensor=torch.stack(
                [item.instances_tensor for item in model_input_items], dim=0
            )
            if model_input_items[0].instances_tensor is not None
            else None,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> ModelInputItem:
        return ModelInputItem(
            image=self.images[idx],
            mask=self.masks[idx],
            cropped_image=self.cropped_images[idx],
            cropped_mask=self.cropped_masks[idx],
            class_id=self.class_ids[idx],
            class_name=self.class_names[idx],
            attributes_label=self.attributes_labels[idx],
            id=self.ids[idx],
            instance_id=self.instance_ids[idx],
            instances_tensor=self.instances_tensor[idx]
            if self.instances_tensor is not None
            else None,
        )

    def __iter__(self) -> ModelInputItem:
        for idx in range(len(self)):
            yield self[idx]

    def to_device(self, device: torch.device):
        return ModelInputItems(
            images=self.images.to(device),
            masks=self.masks.to(device),
            class_names=self.class_names,
            cropped_images=self.cropped_images.to(device),
            cropped_masks=self.cropped_masks.to(device),
            class_ids=self.class_ids.to(device),
            attributes_labels=self.attributes_labels.to(device),
            ids=self.ids.to(device),
            instance_ids=self.instance_ids.to(device),
            instances_tensor=self.instances_tensor.to(device)
            if self.instances_tensor is not None
            else None,
        )

    @property
    def classes_set(self) -> t.Set:
        return set(self.class_ids.cpu().tolist())

    @property
    def from_single_class(self) -> bool:
        return len(self.classes_set) == 1

    @property
    def single_class_id(self) -> int:
        assert self.from_single_class
        return self.classes_set.pop()

    @property
    def single_class_name(self) -> str:
        assert self.from_single_class
        return self.class_names[0]
