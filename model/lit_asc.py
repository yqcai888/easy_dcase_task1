from typing import Union
import torch
import torch.nn as nn
import torchinfo
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from util.lr_scheduler import exp_warmup_linear_down
from util import ClassificationSummary


class LitAcousticSceneClassificationSystem(L.LightningModule):
    """
    Acoustic Scene Classification system based on LightningModule.
    Backbone model, data augmentation techniques and spectrogram extractor are designed to be plug-and-played.
    Backbone architecture, system complexity, classification report and confusion matrix are shown at test stage.

    Args:
        backbone (L.LightningModule, nn.Module): Deep neural network backbone, e.g. cnn, transformer...
        data_augmentation (dict): A dictionary containing instances of data augmentation techniques (nn.Module), e.g. MixUp, MixStyle, DIR Augmentation, SpecAugmentation. Set each to ``None`` if not use one of them.
        class_label (str): Class label. e.g. scene, device, city. (default:``scene``)
        domain_label (str): Domain label. e.g. scene, device, city. (default:``device``)
        spec_extractor (nn.Module): Spectrogram extractor used to transform 1D waveforms to 2D spectrogram. If ``None``, the input features should be 2D spectrogram.
    """

    def __init__(self,
                 backbone: Union[L.LightningModule, nn.Module],
                 data_augmentation: dict,
                 class_label: str = "scene",
                 domain_label: str = "device",
                 spec_extractor: nn.Module = None):
        super().__init__()
        # Save the hyperparameters for Tensorboard visualization, 'backbone' and 'spec_extractor' are excluded.
        self.save_hyperparameters(ignore=['backbone', 'spec_extractor'])
        self.backbone = backbone
        self.data_aug = data_augmentation
        self.class_label = class_label
        self.domain_label = domain_label
        self.cla_summary = ClassificationSummary(class_label, domain_label)
        self.spec_extractor = spec_extractor

        # Save data during testing for statistical analysis
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': []}
        # Input size of a 4D sample (1, 1, F, T), used for generating model profile.
        self._test_input_size = None

    @staticmethod
    def accuracy(logits, labels):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        return acc, pred

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Load a batch of waveforms with size (N, X)
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Choose class label
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug['dir_aug']
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y = mix_up(x, y)
        # Get the predicted labels
        y_hat = self(x)
        # Calculate the loss and accuracy
        if mix_up is not None:
            pred = torch.argmax(y_hat, dim=1)
            train_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(
                y_hat, y[1])
            corrects = (mix_up.lam * torch.sum(pred == y[0]) + (1 - mix_up.lam) * torch.sum(
                pred == y[1]))
            train_acc = corrects.item() / len(x)
        else:
            train_loss = F.cross_entropy(y_hat, y)
            train_acc, _ = self.accuracy(y_hat, y)
        # Log for each epoch
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Get the input size of feature for measuring model profile
        self._test_input_size = (1, 1, x.size(-2), x.size(-1))
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})

        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        return test_acc

    def on_test_epoch_end(self):
        tensorboard = self.logger.experiment
        # Summary the model profile
        model = torch.nn.Sequential(self.backbone, self.cla_layer)
        model_profile = torchinfo.summary(model, input_size=self._test_input_size)
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
        # Convert the summary to string
        model_summary = str(model_profile)
        model_summary += f'\n MACC:\t \t {macc / 1e6:.3f}M'
        model_summary += f'\n Params:\t \t {params / 1e3:.3f}K\n'
        model_summary = model_summary.replace('\n', '<br/>').replace(' ', '&nbsp;').replace('\t', '&emsp;')
        tensorboard.add_text('model_summary', model_summary)
        # Generate a classification report table
        tab_report = self.cla_summary.get_table_report(self._test_step_outputs)
        tensorboard.add_text('classification_report', tab_report)
        # Generate an confusion matrix figure
        cm = self.cla_summary.get_confusion_matrix(self._test_step_outputs)
        tensorboard.add_figure('confusion_matrix', cm)

    def predict_step(self, batch):
        x = batch[0]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        return self(x)


class LitAscWithKnowledgeDistillation(LitAcousticSceneClassificationSystem):
    """
    ASC system with knowledge distillation.

    Args:
        temperature (float): A higher temperature indicates a softer distribution of pseudo-probabilities.
        kd_lambda (float): Weight to control the balance between kl loss and label loss.
        logits_index (int): Index of the logits in Dataset, as multiple logits may be used during training. (default: ``-1``)
    """
    def __init__(self, temperature, kd_lambda, logits_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.kd_lambda = kd_lambda
        self.logits_index = logits_index
        # KL Divergence loss for soft targets
        self.kl_div_loss = torch.nn.KLDivLoss(log_target=True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Load soft labels
        teacher_logits = batch[self.logits_index]
        y_soft = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        # Load hard labels
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug['dir_aug']
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y, y_soft = mix_up(x, y, y_soft)
        # Get the predicted labels
        y_hat = self(x)
        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.temperature, dim=-1)
        # Calculate the loss and accuracy
        if mix_up is not None:
            label_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(y_hat, y[1])
            kd_loss = mix_up.lam * self.kl_div_loss(y_hat_soft, y_soft[0]) + (1 - mix_up.lam) * self.kl_div_loss(y_hat_soft, y_soft[1])
        else:
            label_loss = F.cross_entropy(y_hat, y)
            kd_loss = self.kl_div_loss(y_hat_soft, y_soft)
        kd_loss = kd_loss * (self.temperature ** 2)
        loss = self.kd_lambda * label_loss + (1 - self.kd_lambda) * kd_loss
        self.log_dict({'loss': loss, 'label_loss': label_loss, 'kd_loss': kd_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


class LitAscWithWarmupLinearDownScheduler(LitAcousticSceneClassificationSystem):
    """
    ASC system with warmup-linear-down scheduler. Adapted from: https://github.com/fschmid56/cpjku_dcase23/tree/main
    """
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithTwoSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with two customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, milestones, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithThreeSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with three customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, scheduler3: LRSchedulerCallable, milestones, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler3 = self.scheduler3(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}