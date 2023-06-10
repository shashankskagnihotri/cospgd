import warnings
import torch
import math
import sys
from tqdm import tqdm, trange

import torch.nn.functional as F


class Trainer:
    """
    Trainer class that eases the training of a PyTorch model.
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    epochs : int
        The total number of iterations of all the training 
        data in one cycle for training the model.
    scaler : torch.cuda.amp
        The parameter can be used to normalize PyTorch Tensors 
        using native functions more detail:
        https://pytorch.org/docs/stable/index.html.
    lr_scheduler : torch.optim.lr_scheduler
        A predefined framework that adjusts the learning rate 
        between epochs or iterations as the training progresses.
    Attributes
    ----------
    train_losses_ : torch.tensor
        It is a log of train losses for each epoch step.
    val_losses_ : torch.tensor
        It is a log of validation losses for each epoch step.
    """
    def __init__(
        self, 
        model, 
        criterion, 
        optimizer,
        epochs,
        metrics=None,
        initial_metrics=None,
        actual_metrics=None,    
        logger=None,
        model_save_path=None,
        args=None,
        scaler=None,
        lr_scheduler=None, 
        device=None,        
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.device = self._get_device(device)
        self.epochs = epochs
        self.logger = logger
        self.model = model.to(self.device)
        self.metrics = metrics
        self.model_save_path = model_save_path
        self.mIoU= 0.0
        self.mode = args.mode
        self.epsilon = args.epsilon
        #self.alpha = -1*args.alpha if self.targeted else args.alpha
        self.alpha = args.alpha
        self.iterations = args.iterations
        self.attack = args.attack
        self.num_classes = args.num_classes
        self.norm = args.norm
        self.targeted = args.targeted
        self.batch_size = None
        self.initial_metrics = initial_metrics
        self.actual_metrics = actual_metrics
        
    def save_ckpt(self, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.train_losses_,
                "val_loss": self.val_losses_,
                "mIoU": self.mIoU}, 
                self.model_save_path)
    
    # FGSM attack code
    def fgsm_attack(self, perturbed_image, data_grad, orig_image):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image        
        if self.targeted:
            sign_data_grad *= -1
        perturbed_image = perturbed_image.detach() + self.alpha*sign_data_grad
        # Adding clipping to maintain [0,1] range
        if self.norm == 'inf':
            delta = torch.clamp(perturbed_image - orig_image, min = -1*self.epsilon, max=self.epsilon)
        elif self.norm == 'two':
            delta = perturbed_image - orig_image
            delta_norms = torch.norm(delta.view(self.batch_size, -1), p=2, dim=1)
            factor = self.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def fit(self, train_loader, val_loader):
        """
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        """
        # attributes  
        self.train_losses_ = torch.zeros(self.epochs)
        self.val_losses_ = torch.zeros(self.epochs)
        # ---- train process ----
        for epoch in trange(1, self.epochs + 1, desc='Traning Model on {} epochs'.format(self.epochs)):
            # train
            get_score = True
            if 'train' in self.mode:
                get_score = True if epoch%10 ==0 else False
                self._train_one_epoch(train_loader, epoch)
            if not self.mode == 'adv_attack' and get_score :    
                # validate
                self._evaluate(val_loader, epoch)
            if self.mode == 'adv_attack' :    
                self.adv_attack(val_loader, epoch)
                        
            if get_score:
                score = self.metrics.get_results()
                if self.logger != None:
                    string = "epoch: " + str(epoch) + "   "
                    for item in score:
                        string += item + ": {}    ".format(score[item])
                    self.logger.info(string)
                if self.mode == 'adv_attack' or self.mode == 'test':
                    break
                if score["Mean IoU"] > self.mIoU:
                    self.mIoU = score["Mean IoU"]
                    self.save_ckpt(epoch)
    
    def _train_one_epoch(self, data_loader, epoch):
        self.model.train()
        losses = torch.zeros(len(data_loader))
        with tqdm(data_loader, unit=" training-batch", colour="green") as training:
            for i, (images, labels) in enumerate(training):
                training.set_description(f"Epoch {epoch}")
                images, labels = images.to(self.device), labels.to(self.device)
                # forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    #import ipdb;ipdb.set_trace()
                    preds = self.model(images)
                    #import ipdb;ipdb.set_trace()
                    if "CrossEntropyLoss" in str(type(self.criterion)):
                        loss = self.criterion(preds.float(), labels.long())
                    else:
                        loss = self.criterion(preds.float(), labels.float())
                if not math.isfinite(loss):
                    msg = f"Loss is {loss}, stopping training!"
                    warnings.warn(msg)
                    sys.exit(1)
                # remove gradient from previous passes
                self.optimizer.zero_grad()
                # backprop
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                # parameters update
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                training.set_postfix(loss=loss.item())
                losses[i] = loss.item()
        
            self.train_losses_[epoch - 1] = losses.mean()
    
    @torch.inference_mode()
    def _evaluate(self, data_loader, epoch):
        self.model.eval()
        self.metrics.reset()
        losses = torch.zeros(len(data_loader))
        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels) in enumerate(evaluation):
                evaluation.set_description(f"Validation")
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                if "CrossEntropyLoss" in str(type(self.criterion)):
                    loss = self.criterion(preds.float(), labels.long())
                else:
                    loss = self.criterion(preds.float(), labels.float())
                self.metrics.update(labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())
                self.val_losses_[epoch - 1] = loss.item()
                evaluation.set_postfix(loss=loss.item())
                losses[i] = loss.item()

            self.val_losses_[epoch - 1] = losses.mean()    
    
    @torch.enable_grad()
    def adv_attack(self, data_loader, epoch):
        self.model.eval()
        self.metrics.reset()
        if self.targeted:
            self.actual_metrics.reset()
            self.initial_metrics.reset()
        losses = torch.zeros(len(data_loader))
        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels) in enumerate(evaluation):
                evaluation.set_description(f"Validation")                
                images, labels = images.to(self.device), labels.to(self.device)
                orig_labels = labels.clone()
                if self.targeted:
                    labels = torch.ones_like(labels)
                orig_image = images.clone()
                
                with torch.no_grad():
                    orig_preds = self.model(images)
                if 'pgd' in self.attack:
                    if self.norm == 'inf':
                        images = images + torch.FloatTensor(images.shape).uniform_(-1*self.epsilon, self.epsilon).to(images.device)
                        #images = torch.clamp(images + torch.FloatTensor(images.shape).uniform_(-1*self.epsilon, self.epsilon).to(images.device), min=0, max=1)
                    elif self.norm == 'two':
                        adv_images = images.clone().detach()
                        self.batch_size = len(images)
                        delta = torch.empty_like(adv_images).normal_()
                        d_flat = delta.view(adv_images.size(0), -1)
                        n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
                        r = torch.zeros_like(n).uniform_(0, 1)
                        delta *= r/n*self.epsilon
                        images = torch.clamp(adv_images + delta, min=0, max=1).detach() 
                #images.retain_grad()
                images.requires_grad=True
                preds = self.model(images)
                if "CrossEntropyLoss" in str(type(self.criterion)):
                    loss = self.criterion(preds.float(), labels.long())
                else:
                    loss = self.criterion(preds.float(), labels.float())
                for t in range(self.iterations):                    
                    if self.attack == 'cospgd':
                        one_hot_target = F.one_hot(torch.clamp(labels, labels.min(), self.num_classes-1), num_classes=self.num_classes).permute(0,3,1,2)
                        cossim = F.cosine_similarity(F.softmax(preds, dim=1), one_hot_target, dim=1)
                        if self.targeted:
                            cossim = 1 - cossim
                        loss = cossim.detach() * loss
                    elif self.attack == 'segpgd':
                        lambda_t = t/(2*self.iterations)
                        output_idx = torch.argmax(preds, dim=1)
                        if self.targeted:
                            loss=torch.sum(torch.where(output_idx==labels,lambda_t*loss, (1-lambda_t)*loss))/(preds.shape[-2]*preds.shape[-1])
                        else:
                            loss=torch.sum(torch.where(output_idx==labels, (1-lambda_t)*loss, lambda_t*loss))/(preds.shape[-2]*preds.shape[-1])
                    loss = loss.mean()
                    loss.backward()
                    #def fgsm_attack(self, perturbed_image, data_grad, orig_image):
                    images = self.fgsm_attack(images, images.grad, orig_image)
                    images.requires_grad = True
                    preds = self.model(images)
                    if "CrossEntropyLoss" in str(type(self.criterion)):
                        loss = self.criterion(preds.float(), labels.long())
                    else:
                        loss = self.criterion(preds.float(), labels.float())

                loss = loss.mean()
                self.metrics.update(labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())
                if self.targeted:
                    self.actual_metrics.update(orig_labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())
                    self.initial_metrics.update(orig_preds.detach().max(dim=1)[1].cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())                
                self.val_losses_[epoch - 1] = loss.item()
                evaluation.set_postfix(loss=loss.item())
                losses[i] = loss.item()

            self.val_losses_[epoch - 1] = losses.mean()

    def _get_device(self, _device):
        if _device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {device}"
            warnings.warn(msg)
            return device
        return _device