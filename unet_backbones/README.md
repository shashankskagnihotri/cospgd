# UNet Backbone with ConvNeXt_tiny backbone

We make changes to the github repo by [Berkay Mayalı (mberkay0)](https://github.com/mberkay0/pretrained-backbones-unet)
And include CosPGD, SegPGD and PGD attacks.

## Requirements

python == 3.10.6

create a conda environment: `conda create -n unet python=3.10.6`

activate conda environment: `conda activate unet`

install pytorch: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

install all required libraries: `pip install -r requirements.txt`

## Dataset: Cityscapes

1. Make dirs to download the [CityScapes](https://www.cityscapes-dataset.com/) dataset in datasets/data/cityspaces.
    ```
    mkdir datasets/data
    mkdir datasets/data/cityscapes
    cd datasets/data/cityscapes
    ```

2. Then download the files:
    
>    1. [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
>    2. [gtCoarse.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=2)
>    3. [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
    

3. unzip the downloaded files to the respective folder names:
    ```
    unzip gtFine_trainvaltest.zip -d gtFine
    unzip gtCoarse.zip -d gtCoarse
    unzip leftImg8bit_trainvaltest.zip -d leftImg8bit
    ```

## Steps for running the code

1. Run train.py to train a model:
    ```
    python train.py --encoder [name of encoder to use]
    ```

2. Run attack.py to perform adversarial attacks. 
    ```
    python attack.py --encoder [name of the encoder used for training] --attack [CosPGD, SegPGD, PGD] --iterations [number of attack steps] --targeted ["True", "False"] --norm [Lipschitz continous norm to use] --alpha [attack lr] --epsilon [permissible epsilon] --path [path to the trained model to attack]
    ```

## More information

If you wish to use the pretrained model used for our evaluations, it has been provided as a [Google Drive Link here](https://drive.google.com/file/d/1TgNepNU17_HGAg6f0PaZmbRDJZCb3iaf/view?usp=sharing).

We provide bash scripts all_attacks.sh and attack.sh to perform adversarial attacks using a slurm cluster.
One can update the `--path` in attack.py to the path of the model to be attacked.