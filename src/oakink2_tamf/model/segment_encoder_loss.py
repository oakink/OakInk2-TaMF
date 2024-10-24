import numpy as np
import torch


class SegmentEncoderLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, model_output, batch):
        activation = model_output["activation"]  # (B, nclass)
        category_id = batch["action_label_id"]

        ce_loss = self.cross_entropy_loss(activation, category_id)

        # compute accuracy
        pred = torch.max(activation, dim=1)[1]
        acc = torch.mean((pred == category_id).float())

        loss = 0.0
        loss = loss + ce_loss
        loss_dict = {
            "loss": loss,
            "ce": ce_loss,
            "acc": acc,
        }
        return loss, loss_dict
