import os
import torchvision

from transformers import DetrImageProcessor

ANNOTATION_FILE_NAME = "_annotations.coco.json"

class CocoDetection(torchvision.datasets.CocoDetection):
    
    def __init__(self, image_directory_path: str, image_processor: DetrImageProcessor):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # load image and COCO annotations
        image, annotations = super(CocoDetection, self).__getitem__(idx)

        # Convert into tensors with DETR-compatible format
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()   # remove batch dimension
        target = encoding["labels"][0]                      # remove batch dimension

        return pixel_values, target