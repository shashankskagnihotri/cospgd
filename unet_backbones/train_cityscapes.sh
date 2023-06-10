#!/bin/bash
#echo "Training convnext_tiny on cityscapes Epochs 100 Small decoder True"
#python -W ignore train.py --epochs 100 --dataset cityscapes --small_decoder "True"

echo "Training convnext_tiny on cityscapes Epochs 100 Small decoder False"
python -W ignore train.py --epochs 100 --dataset cityscapes --small_decoder "False"

echo "Training convnext_tiny on cityscapes Epochs 150  Small decoder True"
python -W ignore train.py --epochs 150 --dataset cityscapes --small_decoder "True"

echo "Training convnext_tiny on cityscapes Epochs 150  Small decoder False"
python -W ignore train.py --epochs 150 --dataset cityscapes --small_decoder "False"