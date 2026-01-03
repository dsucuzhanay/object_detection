def collate_fn(batch, image_processor):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }

def prepare_for_coco_detection(predictions):
    coco_results = []

    for image_id, output in predictions.items():
        if len(output["boxes"]) == 0:
            continue

        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]

        # xyxy → xywh
        boxes = boxes.clone()
        boxes[:, 2:] -= boxes[:, :2]

        for box, score, label in zip(boxes, scores, labels):
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": box.tolist(),
                "score": float(score),
            })

    return coco_results