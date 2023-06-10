# UNet Backbone with ConvNeXt_tiny backbone

We make changes to the github repo by [Berkay Mayalı (mberkay0)](https://github.com/mberkay0/pretrained-backbones-unet)
And include CosPGD, SegPGD and PGD attacks.

## STEPS

1. Run train.py to train a model:
    ```
    python train.py --encoder [name of encoder to use]
    ```

2. Run attack.py to perform adversarial attacks. 
    ```
    python attack.py --encoder [name of the encoder used for training] --attack [CosPGD, SegPGD, PGD] --iterations [number of attack steps] --targeted ["True", "False"] --norm [Lipschitz continous norm to use] --alpha [attack lr] --epsilon [permissible epsilon]
    ```