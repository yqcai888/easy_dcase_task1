import os
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit
from model.shared import ResNorm


class OverrideEpochStepCallback(pl.callbacks.Callback):
    """
    Override the step axis in Tensorboard with epoch. Just ignore the warning message popped out.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class FreezeEncoderFinetuneClassifier(pl.callbacks.Callback):
    """
    Freeze the encoder of a model while fine-tuning the classifier.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for layer in pl_module.backbone.classifier.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Freeze all parameters
        for param in pl_module.backbone.parameters():
            param.requires_grad = False
        pl_module.backbone.eval()
        # Unfreeze the parameters of the classifier
        for param in pl_module.backbone.classifier.parameters():
            param.requires_grad = True
        pl_module.backbone.classifier.train()


class PredictionWriter(BasePredictionWriter):
    """
    Write the predictions of a pretrained model into a pt file.
    """
    def __init__(self, output_dir, predict_subset, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.predict_subset = predict_subset

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{self.predict_subset}.pt"))


class PostTrainingQuantization(pl.callbacks.Callback):
    """
    Post-train quantization based on Intel® Neural Compressor.
    For more details: https://lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html#model-quantization

    Args:
        tolerable_acc_loss (float): Tolerance of accuracy loss in the accuracy criterion. (default: ``0.01``)
        max_trials (float): Maximum trial number of quantization. (default: ``100``)
    """
    def __init__(self, tolerable_acc_loss=0.01, max_trials=100):
        super().__init__()
        self.tolerable_acc_loss = tolerable_acc_loss
        self.max_trials = max_trials

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = pl_module.backbone
        # Configure the parameters
        accuracy_criterion = AccuracyCriterion(tolerable_loss=self.tolerable_acc_loss)
        tuning_criterion = TuningCriterion(max_trials=self.max_trials)
        conf = PostTrainingQuantConfig(
            approach="static", backend="default", tuning_criterion=tuning_criterion,
            accuracy_criterion=accuracy_criterion
        )
        # Define evaluation functions
        def cal_accuracy(logits, labels):
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == labels).item() / len(labels)
            return acc, pred

        def eval_func(model):
            y_all = []
            y_hat_all = []
            model.eval()
            with torch.no_grad():
                for batch in trainer.val_dataloaders:
                    x = batch[0]
                    labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
                    y = labels[pl_module.class_label]
                    x = x.cpu()
                    y = y.cpu()
                    x = pl_module.spec_extractor(x).unsqueeze(1)
                    y_hat = model(x)
                    y_all.append(y)
                    y_hat_all.append(y_hat)
                y_all = torch.cat(y_all, dim=0)
                y_hat_all = torch.cat(y_hat_all, dim=0)
                accuracy, _ = cal_accuracy(y_hat_all, y_all)
            return accuracy
        # Exclude the ResNorm layer
        prepare_custom_config_dict = {"non_traceable_module_class": [ResNorm]}
        q_model = fit(model=model, conf=conf, calib_func=eval_func, eval_func=eval_func,
                      prepare_custom_config_dict=prepare_custom_config_dict)
        # Save the quantized model to the original directory of trained model
        q_model.save(f"{trainer.log_dir}/quantized_model/")
        quit()