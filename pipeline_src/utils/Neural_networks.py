"""
This script defines several neural network models and utility functions for image segmentation and background estimation using PyTorch and PyTorch Lightning. It includes implementations of U-Net++ architectures, a convolutional neural network wrapper, and functions for performing generalized likelihood ratio tests (GLRT) on image data.

Classes:
- **Segment_NE**: A PyTorch Lightning module for image segmentation tasks using models from `segmentation_models_pytorch`. It handles training, validation, and testing steps, and computes metrics like Intersection over Union (IoU).

- **ConvolutionalNeuralNet**: A wrapper class for training convolutional neural networks. It includes methods for training the network, saving checkpoints, and making predictions.

- **Convblock**: A convolutional block consisting of a convolutional layer, batch normalization, and ReLU activation function.

- **Unet_pp**: An implementation of the U-Net++ architecture for image segmentation tasks. It utilizes multiple levels of downsampling and upsampling with skip connections.

- **Unet_pp_timeseries**: A variant of the U-Net++ architecture designed to handle time series data with multiple input channels. It includes an initial convolutional layer that adapts to the number of input channels.

Functions:
- **downsample(in_channels, out_channels)**: Creates a downsampling layer using a convolutional layer followed by batch normalization and ReLU activation.

- **upsample(in_channels, out_channels)**: Creates an upsampling layer using a transposed convolutional layer followed by batch normalization and ReLU activation.

- **convblock(in_channels, out_channels)**: Returns a convolutional block (`Convblock`) with the specified input and output channels.

- **last_convblock(in_channels, out_channels)**: Returns the final convolutional block in the network.

- **first_conv(in_channels, out_channels)**: Returns the first convolutional layer of the network, adapting based on the number of input channels.

- **glrtfunction(...)**: Performs the Generalized Likelihood Ratio Test (GLRT) on a set of image samples to detect the presence of a signal in noisy data. It utilizes models for Gaussian point spread functions (PSF) and computes likelihood ratios.

Dependencies:
- **PyTorch**: An open-source machine learning library for Python.
- **PyTorch Lightning**: A lightweight PyTorch wrapper for high-performance AI research.
- **Segmentation Models PyTorch** (`segmentation_models_pytorch`): A library with pre-implemented segmentation models.
- **NumPy**: A fundamental package for scientific computing with Python.
- **Torchvision.transforms**: Common image transformations for computer vision.
- **TQDM**: A fast, extensible progress bar for loops in Python.

Usage:
- The script can be used to train and evaluate image segmentation models on datasets where background estimation is necessary.
- The `Segment_NE` class can be instantiated with the desired architecture and encoder to perform segmentation tasks.
- The `ConvolutionalNeuralNet` class provides an interface for training custom convolutional neural networks.
- The `glrtfunction` can be used for statistical testing in image analysis, particularly in applications like single-molecule localization microscopy.
"""

# import necessary packages
import copy

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
import torchvision.transforms as T
class Segment_NE(pl.LightningModule):

    # initialization method (arch = model to be created, encoder_name = model to be used)
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):

        # initialize from parent class
        super().__init__()

        # define model attribute of object
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocess normalization parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)

        # preprocess normalization buffers for image
        # for dividing by the standard deviation
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # for subtracting the mean
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # define loss function
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    # compute mask
    def forward(self, image):
        # normalize image using buffers
        image = (image - self.mean) / self.std

        # pass image to model and return computed mask
        mask = self.model(image)
        return mask

    # ensure correctness of input data, applies thresholding to predicted mask,
    # and calculates evaluation metrics
    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # aggregates metrics and does more calculations for monitoring model's performance
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    # computes loss/metrics during each training step
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # aggregates metrics and other metric calculations at end of each training epoch
    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    # computes loss/metrics during each validation step
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    # aggregates metrics and other metric calculations at end of each validation epoch
    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    # computes loss/metrics during each test step
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    # aggregates metrics and other metric calculations at end of each testing epoch
    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    # optimizes parameters of model and defines learning rate
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)



class ConvolutionalNeuralNet():
    def __init__(self, network, learning, save_path, preweights=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning)
        self.preweights = preweights
        self.save_path = save_path

    def train(self, loss_function, epochs, batch_size, training_set, validation_set):
        # creating log
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        }

        # defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)

        # initializing network weights
        if self.preweights is None:
            self.network.apply(init_weights)
        else:
            checkpoint = torch.load(self.preweights)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.network.eval()
        # creating dataloaders
        train_loader = DataLoader(training_set, batch_size,shuffle=True)
        val_loader = DataLoader(validation_set, batch_size,shuffle=True)

        # setting convnet to training mode

        self.network.train()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_losses = []

            # training
            print('training...')
            for images, labels in tqdm(train_loader):
                # sending data to device
                images, labels = images.to(device), labels.to(device)
                # resetting gradients
                self.optimizer.zero_grad()
                # making predictions
                predictions = self.network(images)
                # computing loss
                loss = loss_function(predictions, labels)
                log_dict['training_loss_per_batch'].append(loss.item())
                train_losses.append(loss.item())
                # computing gradients
                loss.backward()
                # updating weights
                self.optimizer.step()

            with torch.no_grad():
                print('deriving training accuracy...')
                # computing training accuracy
                train_accuracy = 0
                log_dict['training_accuracy_per_epoch'].append(train_accuracy)

            # validation
            print('validating...')
            val_losses = []

            # setting convnet to evaluation mode
            self.network.eval()

            with torch.no_grad():
                # for images, labels in tqdm(val_loader):
                #     # sending data to device
                #     images, labels = images.to(device), labels.to(device)
                #     # making predictions
                #     predictions = self.network(images)
                #     # computing loss
                #     val_loss = loss_function(predictions, labels)
                #     log_dict['validation_loss_per_batch'].append(val_loss.item())
                #     val_losses.append(val_loss.item())
                # computing accuracy
                print('deriving validation accuracy...')
                log_dict['validation_accuracy_per_epoch'].append(0)

            train_losses = np.array(train_losses).mean()
            val_losses = np.array(val_losses).mean()

            print(f'training_loss: {train_losses}  training_accuracy: ' +
                  f'{train_accuracy}  validation_loss: {round(val_losses, 4)} ' +
                  f'validation_accuracy: {0}\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, self.save_path)

        return log_dict

    def predict(self, x):
        return self.network(x)

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

def downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def convblock(in_channels, out_channels):
    return Convblock(in_channels, out_channels)

def last_convblock(in_channels, out_channels):
    return Convblock(in_channels, out_channels)
class Unet_pp(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_sample_1 = downsample(1, 2 ** 3)
        self.down_sample_2 = downsample(2 ** 3, 2 ** 4)
        self.down_sample_3 = downsample(2 ** 4, 2 ** 5)

        self.upsample1_0 = upsample(2 ** 3, 1)
        self.upsample2_0 = upsample(2 ** 4, 2 ** 3)
        self.upsample1_1 = upsample(2 ** 3, 1)

        self.upsample3_0 = upsample(2 ** 5, 2 ** 4)
        self.upsample2_1 = upsample(2 ** 4, 2 ** 3)
        self.upsample1_2 = upsample(2 ** 3, 1)

        self.convblock0_1 = convblock(2, 1)
        self.convblock1_1 = convblock(2 * 2 ** 3, 2 ** 3)
        self.convblock0_2 = convblock(3, 1)

        self.convblock2_1 = convblock(2 * 2 ** 4, 2 ** 4)
        self.convblock1_2 = convblock(3 * 2 ** 3, 2 ** 3)
        self.convblock0_3 = last_convblock(4, 1)

    def forward(self, image):
        x0_0 = image
        x1_0 = self.down_sample_1(x0_0)
        x2_0 = self.down_sample_2(x1_0)
        x3_0 = self.down_sample_3(x2_0)

        x0_1 = self.convblock0_1(torch.cat([x0_0, self.upsample1_0(x1_0)], dim=1))
        x1_1 = self.convblock1_1(torch.cat([x1_0, self.upsample2_0(x2_0)], dim=1))
        x0_2 = self.convblock0_2(torch.cat([x0_0, x0_1, self.upsample1_1(x1_1)], dim=1))

        x2_1 = self.convblock2_1(torch.cat([x2_0, self.upsample3_0(x3_0)], dim=1))
        x1_2 = self.convblock1_2(torch.cat([x1_1, x1_0, self.upsample2_1(x2_1)], dim=1))
        x0_3 = self.convblock0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample1_2(x1_2)], dim=1))
        #transfrom = torchvision.transforms.GaussianBlur(5, sigma=(1.5, 1.5))
        #x0_3 = transfrom(x0_3)
        return x0_3

from utils.psf_fit_utils import LM_MLE_with_iter, Gaussian2D_IandBg, Gaussian2D_Bg, gauss_psf_2D_I_Bg, gauss_psf_2D_Bg, VectorPSF_2D
from tqdm import tqdm

from typing import Optional
#@torch.jit.script
def glrtfunction(smp_arr, batch_size:int, bounds, initial_arr, roisize:int,sigma:float, tol, lambda_:float=1e-5, iterations:int=30,
                 bg_constant:Optional[torch.Tensor]=None,use_pos:Optional[bool]=False, vector:Optional[bool]=False, GT=False):
    n_iterations = smp_arr.size(0) // batch_size + int(smp_arr.size(0) % batch_size > 0)

    loglik_bg_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    loglik_int_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    traces_bg_all = torch.zeros((smp_arr.size(0),iterations+1,1), device=smp_arr.device)
    traces_int_all = torch.zeros((smp_arr.size(0),iterations+1,2), device=smp_arr.device)



    modelIbg = Gaussian2D_IandBg(roisize, sigma)
    modelbg  = Gaussian2D_Bg(roisize, sigma)
    mle_Ibg = LM_MLE_with_iter(modelIbg, lambda_=lambda_, iterations=iterations,
                           param_range_min_max=bounds[[2, 3], :], tol=tol)
    bg_params = bounds[3, :]
    bg_params = bg_params[None, ...]
    mle_bg = LM_MLE_with_iter(modelbg, lambda_=lambda_, iterations=iterations, param_range_min_max=bg_params,
                           tol=tol[:1])
    for batch in range(n_iterations):
        smp_ = smp_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :, :]
        initial_ = initial_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :]
        if bg_constant is not None:
            bg_constant_batch = bg_constant[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :, :]
            if not GT:
                std_dev = bg_constant_batch.std(dim=(-2, -1))

                # Calculate the mean value along the last two axes
                mean_val = bg_constant_batch.mean(dim=(-2, -1), keepdim=True).expand_as(bg_constant_batch)
                # Create a mask where the standard deviation is less than 2
                mask = (std_dev < 4).unsqueeze(-1).unsqueeze(-1).expand_as(bg_constant_batch)
                # Use the mask to replace slices with their mean value where the condition is met
                bg_constant_batch = torch.where(mask, mean_val, bg_constant_batch)
        else:
            bg_constant_batch = None

        if use_pos:
            pos = initial_[:, :2]
        else:
            pos = None
        with torch.no_grad():  # when no tensor.backward() is used

            # setup model and compute Likelhood for hypothesis I and Bg



            # mle = LM_MLE(model, lambda_=1e-3, iterations=40,
            #              param_range_min_max=param_range[[2, 3], :], traces=True)
            if vector:
                #mle = torch.compile(mle)
                test = 0  # select if single gpus
            else:
                mle_Ibg = torch.jit.script(mle_Ibg)  # select if single gpus

            if vector == True and pos == None:
                pos_in = torch.ones_like(initial_[:, [2, 3]])*roisize/2
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in,
                                                                 bg_only=False)
                mu_iandbg, _ = modelIbg.forward(params_, bg_constant_batch, pos_in)
            elif vector == True:
                pos_in = copy.copy(pos)
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in, bg_only=False)
                mu_iandbg, _ = modelIbg.forward(params_, bg_constant_batch, pos_in)
            else:
                pos_in = copy.copy(pos)
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in)
                mu_iandbg, _ = gauss_psf_2D_I_Bg(params_, roisize, sigma, bg_constant_batch, pos_in)




            bg = initial_[:, 3]
            bg = bg[..., None]




            if vector:
                #mle = torch.compile(mle)
                test = 0  # select if single gpus
            else:
                mle_bg = torch.jit.script(mle_bg)


            # setup model and compute Likelhood for hypothesis Bg
            if vector == True and pos == None:
                pos_in = torch.ones_like(initial_[:, [2, 3]]) * roisize / 2
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_, bg, bg_constant_batch, pos_in,
                                                                     bg_only=True)
                mu_bg, _ = model.forward(params_, bg_constant_batch, pos_in,     bg_only=True)
            elif vector == True:
                pos_in = copy.copy(pos)
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_, bg, bg_constant_batch, pos_in,
                                                                     bg_only=True)
                mu_bg, _ = model.forward(params_, bg_constant_batch, pos_in, bg_only=True)
            else:
                pos_in = copy.copy(pos)
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_[:, :, :], bg, bg_constant_batch, pos_in)
                mu_bg, _ = gauss_psf_2D_Bg(params_bg_, roisize, sigma, bg_constant_batch, pos)







            loglik_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_bgonly))] = loglik_bgonly
            loglik_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg))] = loglik_I_andbg
            traces_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_bgonly,[1,0,2])
            traces_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_iandbg,[1,0,2])

    ratio = 2 * (loglik_int_all - loglik_bg_all)
    #ratio[torch.isnan(ratio)] = 0
    return ratio, loglik_int_all, loglik_bg_all, mu_iandbg,mu_bg, traces_bg_all, traces_int_all

def first_conv(in_channels, out_channels):
    if in_channels==1:
        return nn.Sequential(
        )

    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )


class Unet_pp_timeseries(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.first_conv = first_conv(channels, 1)
        self.down_sample_1 = downsample(1, 2 ** 3)
        self.down_sample_2 = downsample(2 ** 3, 2 ** 4)
        self.down_sample_3 = downsample(2 ** 4, 2 ** 5)

        self.upsample1_0 = upsample(2 ** 3, 1)
        self.upsample2_0 = upsample(2 ** 4, 2 ** 3)
        self.upsample1_1 = upsample(2 ** 3, 1)

        self.upsample3_0 = upsample(2 ** 5, 2 ** 4)
        self.upsample2_1 = upsample(2 ** 4, 2 ** 3)
        self.upsample1_2 = upsample(2 ** 3, 1)

        self.convblock0_1 = convblock(2, 1)
        self.convblock1_1 = convblock(2 * 2 ** 3, 2 ** 3)
        self.convblock0_2 = convblock(3, 1)

        self.convblock2_1 = convblock(2 * 2 ** 4, 2 ** 4)
        self.convblock1_2 = convblock(3 * 2 ** 3, 2 ** 3)
        self.convblock0_3 = last_convblock(4, 1)

    def forward(self, image):
        x0_0 = image
        x0_0 = self.first_conv(x0_0)

        x1_0 = self.down_sample_1(x0_0)
        x2_0 = self.down_sample_2(x1_0)
        x3_0 = self.down_sample_3(x2_0)

        x0_1 = self.convblock0_1(torch.cat([x0_0, self.upsample1_0(x1_0)], dim=1))
        x1_1 = self.convblock1_1(torch.cat([x1_0, self.upsample2_0(x2_0)], dim=1))
        x0_2 = self.convblock0_2(torch.cat([x0_0, x0_1, self.upsample1_1(x1_1)], dim=1))

        x2_1 = self.convblock2_1(torch.cat([x2_0, self.upsample3_0(x3_0)], dim=1))
        x1_2 = self.convblock1_2(torch.cat([x1_1, x1_0, self.upsample2_1(x2_1)], dim=1))
        x0_3 = self.convblock0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample1_2(x1_2)], dim=1))
        #transfrom = torchvision.transforms.GaussianBlur(5, sigma=(1.5, 1.5))
        #x0_3 = transfrom(x0_3)
        transform = T.GaussianBlur(kernel_size=(7, 7), sigma=(1.5, 1.5))
        blurred_img = transform(x0_3)
        bg_estimated = blurred_img * 1
        return bg_estimated
