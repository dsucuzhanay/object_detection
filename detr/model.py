import torch
import lightning as L

from transformers import DetrForObjectDetection

class DeTr(L.LightningModule):

    def __init__(
            self,
            pretrained_model: str,
            num_labels: int,
            batch_size: int,
            lr: float,
            lr_backbone: float,
            weight_decay: float,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"]
        )
        return outputs.loss

    def training_step(self, batch):
        loss = self.common_step(batch)
        self.log("training_loss", loss, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch):
        loss = self.common_step(batch)
        self.log("validation_loss", loss, batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        # DETR uses different learning rate for the backbone
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone
            }
        ]

        return torch.optim.AdamW(params=param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)