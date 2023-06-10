import sys
sys.path.append('unet_backbones')
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer
from backbones_unet.utils.reproducibility import set_seed

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import torchvision
from torchvision.transforms import Normalize

from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et

from metrics import StreamSegMetrics

import json
import logging
import numpy as np

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--small_decoder', default=True, type=bool)
    parser.add_argument('-en', '--encoder', type=str, default='convnext_tiny', choices={'convnext_atto', 'convnext_atto_ols', 'convnext_base', 
                                                                                'convnext_base_384_in22ft1k', 'convnext_base_in22ft1k', 
                                                                                'convnext_base_in22k', 'convnext_femto', 'convnext_femto_ols', 
                                                                                'convnext_large', 'convnext_large_384_in22ft1k', 'convnext_large_in22ft1k', 
                                                                                'convnext_large_in22k', 'convnext_nano', 'convnext_nano_ols', 'convnext_pico', 
                                                                                'convnext_pico_ols', 'convnext_small', 'convnext_small_384_in22ft1k', 'convnext_small_in22ft1k', 
                                                                                'convnext_small_in22k', 'convnext_tiny', 'convnext_tiny_384_in22ft1k', 
                                                                                'convnext_tiny_hnf', 'convnext_tiny_in22ft1k', 'convnext_tiny_in22k', 
                                                                                'convnext_xlarge_384_in22ft1k', 'convnext_xlarge_in22ft1k', 'convnext_xlarge_in22k', 
                                                                                'cs3darknet_focus_l', 'cs3darknet_focus_m', 'cs3darknet_l', 'cs3darknet_m', 'cs3darknet_x', 
                                                                                'cs3edgenet_x', 'cs3se_edgenet_x', 'cs3sedarknet_l', 'cs3sedarknet_x', 'cspdarknet53', 'cspresnet50', 
                                                                                'cspresnext50', 'darknet53', 'darknetaa53', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenetblur121d', 'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'eca_nfnet_l0', 'eca_nfnet_l1', 'eca_nfnet_l2', 'eca_resnet33ts', 'eca_resnext26ts', 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50t', 'ecaresnet101d', 'ecaresnet269d', 'ecaresnetlight', 'edgenext_base', 'edgenext_small', 'edgenext_small_rw', 'edgenext_x_small', 'edgenext_xx_small', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_el', 'efficientnet_el_pruned', 'efficientnet_em', 'efficientnet_es', 'efficientnet_es_pruned', 'efficientnet_lite0', 'efficientnetv2_rw_m', 'efficientnetv2_rw_s', 'efficientnetv2_rw_t', 'ese_vovnet19b_dw', 'ese_vovnet39b', 'fbnetc_100', 'fbnetv3_b', 'fbnetv3_d', 'fbnetv3_g', 'gc_efficientnetv2_rw_t', 'gcresnet33ts', 'gcresnet50t', 'gcresnext26ts', 'gcresnext50ts', 'gernet_l', 'gernet_m', 'gernet_s', 'ghostnet_100', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 'hardcorenas_a', 'hardcorenas_b', 'hardcorenas_c', 'hardcorenas_d', 'hardcorenas_e', 'hardcorenas_f', 'hrnet_w18', 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'lambda_resnet26t', 'lambda_resnet50ts', 'lcnet_050', 'lcnet_075', 'lcnet_100', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mnasnet_100', 'mnasnet_small', 'mobilenetv2_050', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 'mobilenetv3_large_100', 'mobilenetv3_large_100_miil', 'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_rw', 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100', 'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs', 'mobilevitv2_050', 'mobilevitv2_075', 'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_150_384_in22ft1k', 'mobilevitv2_150_in22ft1k', 'mobilevitv2_175', 'mobilevitv2_175_384_in22ft1k', 'mobilevitv2_175_in22ft1k', 'mobilevitv2_200', 'mobilevitv2_200_384_in22ft1k', 'mobilevitv2_200_in22ft1k', 'nf_regnet_b1', 'nf_resnet50', 'nfnet_l0', 'regnetv_040', 'regnetv_064', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320', 'regnetz_040', 'regnetz_040h', 'regnetz_b16', 'regnetz_c16', 'regnetz_c16_evos', 'regnetz_d8', 'regnetz_d8_evos', 'regnetz_d32', 'regnetz_e8', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4', 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e', 'resnest200e', 'resnest269e', 'resnet10t', 'resnet14t', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet26t', 'resnet32ts', 'resnet33ts', 'resnet34', 'resnet34d', 'resnet50', 'resnet50_gn', 'resnet50d', 'resnet51q', 'resnet61q', 'resnet101', 'resnet101d', 'resnet152', 'resnet152d', 'resnet200d', 'resnetaa50', 'resnetblur50', 'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420', 'resnetv2_50', 'resnetv2_50d_evos', 'resnetv2_50d_gn', 'resnetv2_50x1_bit_distilled', 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k', 'resnetv2_101', 'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k', 'resnetv2_152x2_bit_teacher', 'resnetv2_152x2_bit_teacher_384', 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k', 'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k', 'resnext26ts', 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'semnasnet_075', 'semnasnet_100', 'seresnet33ts', 'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26ts', 'seresnext50_32x4d', 'seresnext101_32x8d', 'seresnext101d_32x8d', 'seresnextaa101d_32x8d', 'skresnet18', 'skresnet34', 'skresnext50_32x4d', 'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'tf_efficientnetv2_l', 'tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21k', 'tf_efficientnetv2_m', 'tf_efficientnetv2_m_in21ft1k', 'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_s', 'tf_efficientnetv2_s_in21ft1k', 'tf_efficientnetv2_s_in21k', 'tf_efficientnetv2_xl_in21ft1k', 'tf_efficientnetv2_xl_in21k', 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'tinynet_a', 'tinynet_b', 'tinynet_c', 'tinynet_d', 'tinynet_e', 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet50_2', 'wide_resnet101_2', 'xception41', 'xception41p', 'xception65', 'xception65p', 'xception71'},
                        help='Encoder')
# * Dataset parameters
    parser.add_argument('--dataset', default='cityscapes', type=str, help='dataset to train/eval on')
    parser.add_argument("--download", action='store_true', default=False, help="download datasets")
    parser.add_argument('--crop_size', default=512, type=int, help='crop_size for training')
    parser.add_argument('--crop_val', action='store_true', default=False, help='To crop  val images or not')
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

# * Save dir
    parser.add_argument('--save', default='results/neurips', type=str, help='directory to save models')   
    

    # * Loss
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='Loss Criteria')
    
    # * Large transposed convolution kernels, plots and FGSM attack    
    parser.add_argument('-it', '--iterations', type=int, default=1,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-at', '--attack', type=str, default='cospgd', choices={'fgsm', 'cospgd', 'segpgd', 'pgd'},
                        help='Which adversarial attack')
    parser.add_argument('-ep', '--epsilon', type=float, default=0.03,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-nr', '--norm', type=str, default="inf", choices={'inf', 'two', 'one'},
                        help='lipschitz continuity bound to use')
    parser.add_argument('-tar', '--targeted', type=str, default="False", choices={'False', 'True'},
                        help='use a targeted attack or not')
    parser.add_argument('-pt', '--path', type=str, default='pretrained_model/best_model.pt',
                        help='Path of pretrained model to be adversarially attacked')
    parser.add_argument('-m', '--mode', type=str, default='adv_attack', choices={'adv_attack', 'adv_train', 'train', 'test'},
                        help='What to do?')

    return parser


def get_logger(save_folder):
    log_path = str(save_folder) + '/log.log'
    logging.basicConfig(filename=log_path, filemode='a')
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'pascalvoc2012':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def main(args):
    """ device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) """
    set_seed(args.seed)

    dataset_path={"pascalvoc2012": {"num_classes":21, "data_root": "datasets/data/VOCdevkit/VOC2012", "crop_size":256},
            "cityscapes": {"num_classes":19, "data_root": "datasets/data/cityscapes", "crop_size":512}}
    args.num_classes = dataset_path[args.dataset]["num_classes"]

    if 'small_decoder_True' in args.path:
        args.small_decoder = True
    elif 'small_decoder_False' in args.path:
        args.small_decoder = False

    if args.iterations ==1:
        args.alpha = args.epsilon

    save_path = os.path.join(args.save, args.dataset, args.encoder, "small_deocder_"+str(args.small_decoder), "epochs_"+str(args.epochs), "lr_"+str(args.lr), args.mode, args.attack, 'iterations' + str(args.iterations), 'alpha_' + str(args.alpha), 'eps_'+ str(args.epsilon), "L_"+args.norm, 'targeted_'+args.targeted)
    args.save_path = save_path
    model_path = os.path.join(save_path , "model")
    json_path = os.path.join(save_path, "losses.json")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(args.save_path)

    tmp = args.targeted
    if tmp == 'True':
        args.targeted = True
    elif tmp =='False':
        args.targeted = False   


    args.data_root = dataset_path[args.dataset]["data_root"] 
    args.crop_size = dataset_path[args.dataset]["crop_size"] 

    for arg, value in sorted(vars(args).items()):
        logger.info("{}: {}".format(arg, value))

    #train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path)
    #val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path)
    train_dataset, val_dataset = get_dataset(args)

    if 'train' in args.mode:
        train_loader = DataLoader(train_dataset, batch_size=2, num_workers=24, shuffle=True)
    else:
        train_loader = None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=24, shuffle=True)

    model = Unet(
        #backbone='convnext_base', # backbone network name
        backbone=args.encoder,
        small_decoder=args.small_decoder,
        in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
        num_classes=dataset_path[args.dataset]["num_classes"],
        #num_classes=1,            # output channels (number of classes in your dataset)
    )

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    model = nn.Sequential(Normalize(mean = mean, std = std), model)

    if args.path is not None:
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])       

    if args.mode != "adv_attack" or args.mode != "test":
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    else:
        optimizer = None

    criterion = {"cross_entropy": nn.CrossEntropyLoss(ignore_index=255, reduction="none"), "dice_loss": DiceLoss(reduction=None)} 

    metrics = StreamSegMetrics(dataset_path[args.dataset]["num_classes"])
    actual_metrics = StreamSegMetrics(dataset_path[args.dataset]["num_classes"]) if args.targeted else None
    initial_metrics = StreamSegMetrics(dataset_path[args.dataset]["num_classes"]) if args.targeted else None

    trainer = Trainer(
        model,                    # UNet model with pretrained backbone
        criterion = criterion[args.loss],
        #criterion=DiceLoss(),     # loss function for model convergence
        optimizer=optimizer,      # optimizer for regularization
        epochs=args.epochs,               # number of epochs for model training
        metrics = metrics,
        actual_metrics = actual_metrics,
        initial_metrics = initial_metrics,
        logger = logger,
        model_save_path = os.path.join(model_path, "best_model.pt"),
        args=args
    )

    trainer.fit(train_loader, val_loader)        
    
    torch.save({"epoch": args.epochs, "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": trainer.train_losses_,
                "val_loss": trainer.val_losses_}, 
                os.path.join(model_path, "final_model.pt"))

    if args.targeted:
        losses = {"encoder:": args.encoder,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "attack:": args.attack,
                    "iterations:": args.iterations,
                    "epsilon": args.epsilon,
                    "alpha": args.alpha,
                    "targeted": str(args.targeted),
                    "norm": args.norm,
                    "train loss": trainer.train_losses_.detach().cpu().tolist(), 
                    "val loss": trainer.val_losses_.detach().cpu().tolist(), 
                    "wrt target": trainer.metrics.get_results(),
                    "wrt initial": trainer.initial_metrics.get_results(),
                    "wrt actual": trainer.actual_metrics.get_results()}
    else:
        losses = {"encoder:": args.encoder,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "attack:": args.attack,
                    "iterations:": args.iterations,
                    "epsilon": args.epsilon,
                    "alpha": args.alpha,
                    "targeted": str(args.targeted),
                    "norm": args.norm,
                    "train loss": trainer.train_losses_.detach().cpu().tolist(), 
                    "val loss": trainer.val_losses_.detach().cpu().tolist(), 
                    "wrt gt": trainer.metrics.get_results(),
                    }
    json_losses = json.dumps(losses, indent=4)
    with open(json_path, "w") as f:
        f.write(json_losses)
    
    
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser('UNet training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    if args_.iterations == 1:
        args_.alpha = args_.epsilon
    main(args_)