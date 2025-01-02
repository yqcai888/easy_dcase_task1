# Easy DCASE Task 1 - Train Your Model with a YAML file

Author: **Yiqiang Cai** (yiqiang.cai21@student.xjtlu.edu.cn), *Xi'an Jiaotong-Liverpool University*

## Task Description

The task 1 of DCASE Challenge focuses on Acoustic Scene Classification (ASC), recognizing the environment in which an audio recording was captured, such as streets, parks, or airports. For a detailed description of the challenge and this task, please visit the [DCASE website](https://dcase.community/challenge2024/). The main challenges of this task are summarized as below:
1. Domain Shift: Unseen devices exist in test set. (2020~)
2. Short Duration: The duration of audio recordings reduced from 10s (\~2021) to 1s (2022\~).
3. Low Complexity: Limited model parameters (128K INT8) and computational overheads (30 MMACs). (2022~)
4. Data Efficiency: Train model with fewer data, specifically 5%, 10%, 25%, 50% and 100%. (2024~)

## System Description

This repository provides an easy way to train your models on the datasets of DCASE task 1. The example system (TF-SepNet + BEATs teacher) won the **Judges' Award** for DCASE2024 Challenge Task1. Corresponding paper has been accepted by DCASE Workshop 2024 and available [here](https://arxiv.org/abs/2408.14862).

1. All configurations of model, dataset and training can be done via a simple YAML file.
2. Entire system is implemented using [PyTorch Lightning](https://lightning.ai/).
3. Logging is implemented using [TensorBoard](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html#tensorboardlogger). ([Wandb API](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) is also supported.)
4. Various task-related techniques have been included.
   * 3 Spectrogram Extractor: [Cnn3Mel](https://dcase-repo.github.io/dcase_util/generated/dcase_util.features.MelExtractor.html?highlight=mel#dcase_util.features.MelExtractor), [CpMel](https://github.com/fschmid56/cpjku_dcase23/tree/main), [BEATsMel](https://github.com/microsoft/unilm/tree/master/beats)
   * 3 High-performing Backbones: [BEATs](https://arxiv.org/pdf/2212.09058), [TF-SepNet](https://ieeexplore.ieee.org/abstract/document/10447999), [BC-ResNet](https://arxiv.org/abs/2106.04140).
   * 4 Plug-and-played Data Augmentation Techniques: [MixUp](https://arxiv.org/abs/1710.09412), [FreqMixStyle](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Schmid_27.pdf), [SpecAugmentation](https://arxiv.org/abs/1904.08779), [Device Impulse Response Augmentation](https://arxiv.org/pdf/2305.07499).
   * 2 Model Compression Methods: [Post-training Quantization](https://lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html#model-quantization), [Knowledge Distillation](https://github.com/fschmid56/cpjku_dcase23/tree/main).

## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n dcase_t1
conda activate dcase_t1
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip install torch torchvision torchaudio
```

4. Install requirements:

```
pip install -r requirements.txt
```

5. Download and extract the [TAU Urban Acoustic Scenes 2020 Mobile, Development dataset](https://zenodo.org/records/3819968), [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421) and [Microphone Impulse Response](https://micirp.blogspot.com/?m=1) according to your needs.  The directory should be placed in the **parent path** of code directory.

You should end up with a directory that contains, among other files, the following:
* ../TAU-urban-acoustic-scenes-2020-mobile-development/development/audio/: A directory containing 230,35 audio files in *wav* format.
* ../TAU-urban-acoustic-scenes-2022-mobile-development/development/audio/: A directory containing 230,350 audio files in *wav* format.
* ../microphone_impulse_response/: A directory containing 67 impulse response files in *wav* format.

6. Several default configuration yaml files are provided in config/. The training procedure can be started by running the following command:
```
python main.py fit --config config/tfsepnet_train.yaml
```

You can select or revise the config files under code-dcase-2024/config/... or directly override individual arguments in the command line:
```
python main.py fit --config config/tfsepnet_train.yaml --trainer.max_epochs 30
python main.py fit --config config/tfsepnet_train.yaml --optimizer.lr 0.006
python main.py fit --config config/tfsepnet_train.yaml --data.audio_dir ../TAU-urban-acoustic-scenes-2020-mobile-development/development
python main.py fit --config config/tfsepnet_train.yaml --data.train_subset split100
```

7. Test model:
```
python main.py test --config config/tfsepnet_test.yaml --ckpt_path path/to/ckpt
```

8. View results:
```
tensorboard --logdir log/tfsepnet_train # Check training results
tensorboard --logdir log/tfsepnet_test # Check testing results
```
Then results will be available at [localhost port 6006](http://127.0.0.1:6006/).

9. Quantize model:
```
python main.py validate --config config/tfsepnet_quant.yaml --ckpt_path path/to/ckpt
```

## Check Available Arguments

The available arguments and documentations can be shown by command line.
1. To see the available commands type:
```
python main.py --help
```

2. View all available options with the --help argument given after the subcommand:
```
python main.py fit --help
```

3. View the documentations and arguments of a specific class:
```
python main.py fit --model.help LitAcousticSceneClassificationSystem
python main.py fit --data.help DCASEDataModule
```
4. View the documentations of a specific argument:
```
python main.py fit --model.help LitAcousticSceneClassificationSystem --model.init_args.backbone.help TFSepNet
```

## Fine-tune BEATs for ASC
For convenience, please download the [checkpoints](https://github.com/microsoft/unilm/tree/master/beats) into the path: model/beats/checkpoints/.

1. Freeze encoder and fine-tune classifier of self-supervised pre-trained BEATs, __BEATs (SSL)*__:
```
python main.py fit --config config/beats_ssl_star.yaml
```
2. Unfrozen fine-tune the self-supervised pre-trained BEATs, __BEATs (SSL)__:
```
python main.py fit --config config/beats_ssl.yaml
```
3. Unfrozen fine-tune the self-supervised pre-trained BEATs with additional supervised fine-tuning on AudioSet, __BEATs (SSL+SL)__:
```
python main.py fit --config config/beats_ssl+sl.yaml
```
4. Test model:
```
python main.py test --config config/beats_test.yaml
```
5. Get predictions from fine-tuned BEATs:
```
python main.py predict --config config/beats_predict.yaml
```

## Knowledge Distillation
Before knowledge distillation, make sure that the logits of teacher model have been generated and placed in your preferred directory. Alternatively, we also provided [logits of fine-tuned BEATs](https://github.com/yqcai888/easy_dcase_task1/releases/tag/v0.0.1) for easier implementation. Please download and extract them into log/ when use. Input the path of logits files into config/tfsepnet_kd.yaml. If use more than one logit, the logits will be averaged as teacher ensemble.

```
    logits_files:
      - log/beats_ssl_star/predictions_split*.pt
      - log/beats_ssl/predictions_split*.pt
      - log/beats_ssl+sl/predictions_split*.pt
```
Distilling knowledge from fine-tuned BEATs to TF-SepNet:
```
python main.py fit --config config/tfsepnet_kd.yaml
```

## Customize Your System

Deploy your model in model/backbones/ and inherit the _BaseBackbone:
```
class YourModel(_BaseBackbone):
...
```
Implement new spectrogram extractor in util/spec_extractor/ and inherit the _SpecExtractor:
```
class NewExtractor(_SpecExtractor):
...
```
Declare new data augmentation method in util/data_augmentation/ and inherit the _DataAugmentation:  
(Changes also need to be made in model/lit_asc/LitAcousticSceneClassificationSystem)
```
class NewAugmentation(_DataAugmentation):
...
```

More instructions can be found on [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)

## Citation
If you find our code helps, we would appreciate using the following citation:
```
@inproceedings{Cai2024workshop,
    author = "Cai, Yiqiang and Li, Shengchen and Shao, Xi",
    title = "Leveraging Self-Supervised Audio Representations for Data-Efficient Acoustic Scene Classification",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    month = "October",
    year = "2024",
    pages = "21--25",
}
```

