import os
import lightning as L

from pycocotools.coco import COCO
from dataset import CocoDetection
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor

class DataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        batch_size: int,
        pretrained_model: str,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.pretrained_model = pretrained_model

    def setup(self, stage: str = None):

        if stage is None:
            coco = COCO(os.path.join(self.val_dataset_path, "_annotations.coco.json"))
            cat_ids = coco.getCatIds(supNms=["debs-pillows-lamps-bedstands"])       # remove supercategory as label
            self.num_labels = len(cat_ids)
            print("Number of classes:", self.num_labels)

        if stage == "fit":
            self.image_processor = DetrImageProcessor.from_pretrained(self.pretrained_model)

            self.train_dataset = CocoDetection(
                image_directory_path=self.train_dataset_path,
                image_processor=self.image_processor
            )

            self.val_dataset = CocoDetection(
                image_directory_path=self.val_dataset_path,
                image_processor=self.image_processor
            )

            # remove supercategory as label
            self.num_labels = len(
                self.train_dataset.coco.getCatIds(
                    supNms=["debs-pillows-lamps-bedstands"]
                )
            )

            print("Number of classes:", self.num_labels)

    def collate_fn(self, batch):
        # DETR uses various image sizes during training. Images are padded to the largest resolution in a given 
        # batch, and corresponding binary pixel_mask are created, indicating whether pixels are real or padding
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels
        }

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)