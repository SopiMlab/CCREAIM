# CCREAIM

CCREAIM aims to explore musician and AI interactions in a live setting, utilizing transformer's attention weights to provide the musician with potentially interesting information about the motivation behind the AI's decisions.

This codebase includes the training and usage of models suitable for that task.

## Code structure

Here's the directory structure of the parts relevant to the model's training and usage:
```
main.py
live.py
ccreaim/
        utils/
        model/
        process/
cfg/
    hydra/
        launcher/
    runs/
        test/
        live/
        train/
notebooks/
```
`main.py` is used for training and non-live testing of the model, while `live.py` is used for the live setting. `ccreaim`-directory contains code relevant to the training of the model, as well as some useful utils. `cfg`-directory contains hydra configurations for both live and non-live situations.

## The model

The model does prediction for the next token index in a sequence, for a predefined bank of audio samples. Essentially, the model is a causal decoder-only transformer, so for training it uses a causal mask to prevent peeking into the future at training time and at inference the model returns "given some unfinished sequence, for each sample in the library, what is the likelihood that this sample is the next one in the sequence".

These likelihoods will be used to create musically interesting combinations of the samples, and the results of those combinations are fed back to the transformer recurrently to create a continuation for some audio input.

## Usage

`conda env create -f ccreaim.yml` to create the conda environment.

### Creating a dataset
\[*This interface would be good to improve before publishing the code*\]

To create a dataset, use `ccreaim/utils/create_feature_dataset.py`. Before running the script, some variables (mostly paths to directories/output paths) in the file should be modified. A directory with enough audio data (`vocab_size*sample_len_seconds`) should be created beforehand and specified in the code. All of the details are commented in the code itself.

An accompanying script `ccreaim/utils/create_sample_bank.py` should be run on the same audio directory, for inference purposes.

### Running a training

To run a training, run `main.py` with relevant hydra configs.

An example command to launch trainings:

`nohup python main.py hydra/launcher=slurm runs=train/bank-classifier_system data.data_tar=/tmp/some_training_data.tar resources.timeout_min=120 hyper.transformer.dropout=0.5 hyper.learning_rate=1e-4,1e-5 hyper.epochs=80 logging.checkpoint=30 &`

This command runs `main.py` with configs inherited from `cfg/base.yaml`, but overridden first with values from `cfg/hydra/launcher/slurm.yaml` and `cfg/runs/train/bank-classifier_system.yaml`, then again overridden with the values defined in the command. As a result, *two trainings are started*: one with learning rate 1e-4 and one with learning rate 1e-5, otherwise identical. The logs and checkpoints for model weights will be output in locations defined as hydra configs in `base.yaml`.

### Inference

Inference is done in a standard way with the transformer, `notebooks/test_model_bank.ipynb` has examples for how to run these. Notably, the cacheing does not work currently for some situations, so for now the standard transformer is used by counterintuitively setting `model.train()` during inference, and setting the training specific hyperparameter dropout to zero.

### Live
A live session can be started with `python live.py runs=live/some_live_config.yaml`.
Currently, we have the following options:
* `python live.py runs=live/bank-classifier-samples.yaml` 
    * Go to ccreaim/utils/dataset.py. Go to the BankTransformerDataset __init__ function and change the following row: 
        * `self.item_path_list = util.get_sample_path_list(self.data_root, self.ext)`
* `python live.py runs=live/bank-classifier-maestro.yaml` 
    * Go to ccreaim/utils/dataset.py. Go to the BankTransformerDataset __init__ function and change the following row: 
        * `self.item_path_list = util.get_sample_path_list_orig(self.data_root, self.ext)`

You change the length of the live session from `live.py` by setting the `NUM_ITER` parameter to anything you want.
Around 1250 iterations is one minute.

### Attention Visualization

In order to visualize the attention after running a live session with `live.py`, go into the `attention_visualization` folder and run: `python -m attention_visualization`.

```bash
cd attention_visualization
python -m attention_visualization
```

## Additional details

### Hydra

This project uses [Hydra](https://hydra.cc/) as a configuration tool. Hydra uses hierarchical configuration, where the configuration is composed from multiple sources. For our project, `base.yaml`/`live.yaml` holds the basic configuration, and different files in `cfg/runs/*` in tandem with the command line add to and override that basic configuration. Our configuration utilizes [Submitit Launcher Plugin](https://hydra.cc/docs/plugins/submitit_launcher/) to interact with a Slurm job scheduler directly, this can be employed by adding the `hydra/launcher=slurm` command line option when launching a training/testing. More info on Hydra usage [here](https://hydra.cc/docs/intro/).

### Wandb

The project uses [Weights and Biases](https://wandb.ai/site), or wandb for short, for monitoring and storing training data. To use it, login to wandb in command line with `wandb login`, then start a training. By default, the configuration includes wandb logging, but if you want to exclude a training from wandb, use `logging.wandb=false`. More info on wandb usage [here](https://docs.wandb.ai/quickstart).

### Feature extraction

For feature extraction, [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) is used. More info on usage [here](https://github.com/tyiannak/pyAudioAnalysis/wiki).

### Data augmentation

For data augmentation, [PyTorch Audio Augmentations](https://github.com/Spijkervet/torchaudio-augmentations) is used.
