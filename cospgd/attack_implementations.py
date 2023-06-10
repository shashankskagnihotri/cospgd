# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:31:16 2023

@author: sjung_local
"""

import torch

class Attack:
    def __init__(self):
        pass
    
    @staticmethod
    def step_inf(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            targeted,
            clamp_min = 0,
            clamp_max = 1,
            grad_scale = None
        ):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = alpha*data_grad.sign()
        if targeted:
            sign_data_grad *= -1
        if grad_scale is not None:
            sign_data_grad *= grad_scale
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + sign_data_grad
        # Adding clipping to maintain [0,1] range
        delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image
    
    @staticmethod
    def step_l2(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            targeted,
            clamp_min = 0,
            clamp_max = 1,
            grad_scale = None
        ):
        # normalize gradients
        if targeted:
            data_grad *= -1
        data_grad = Attack.lp_normalize(
            data_grad,
            p = 2,
            epsilon = 1.0,
            decrease_only = False
        )
        if grad_scale is not None:
            data_grad *= grad_scale
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + alpha*data_grad
        # clip to l2 ball
        delta = Attack.lp_normalize(
            noise = perturbed_image - orig_image,
            p = 2,
            epsilon = epsilon,
            decrease_only = True
        )
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image
    
    @staticmethod
    def lp_normalize(
            noise,
            p,
            epsilon = None,
            decrease_only = False
        ):
        if epsilon is None:
            epsilon = torch.tensor(1.0)
        denom = torch.norm(noise, p=p, dim=(-1, -2, -3))
        denom = torch.maximum(denom, torch.tensor(1E-12)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if decrease_only:
            denom = torch.maximum(denom/epsilon, torch.tensor(1))
        else:
            denom = denom / epsilon
        return noise / denom
    
    @staticmethod
    def init_linf(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
        ):
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images
    
    @staticmethod
    def init_l2(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
        ):
        noise = torch.FloatTensor(images.shape).uniform_(-1, 1).to(images.device)
        noise = Attack.lp_normalize(
            noise = noise,
            p = 2,
            epsilon = epsilon,
            decrease_only = False
        )
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images
    
    @staticmethod
    def segpgd_scale(
            predictions,
            labels,
            loss,
            iteration,
            iterations,
            targeted
        ):
        lambda_t = iteration/(2*iterations)
        output_idx = torch.argmax(predictions, dim=1)
        if targeted:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    lambda_t*loss,
                    (1-lambda_t)*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        else:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    (1-lambda_t)*loss,
                    lambda_t*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        return loss
    
    @staticmethod
    def cospgd_scale(
            predictions,
            labels,
            loss,
            num_classes,
            targeted
        ):
        one_hot_target = torch.nn.functional.one_hot(
            torch.clamp(labels, labels.min(), num_classes-1),
            num_classes = num_classes
        ).permute(0,3,1,2)
        cossim = torch.nn.functional.cosine_similarity(
            torch.nn.functional.softmax(predictions, dim=1),
            one_hot_target,
            dim = 1
        )
        if targeted:
            cossim = 1 - cossim
        loss = cossim.detach() * loss
        return loss