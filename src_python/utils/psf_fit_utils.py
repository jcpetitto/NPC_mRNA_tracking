"""
This script provides a comprehensive set of classes and functions for performing maximum likelihood estimation (MLE)
using the Levenberg-Marquardt (LM) algorithm, specifically tailored for fitting Gaussian point spread functions (PSF)
and performing background estimation in image data. The implementation leverages PyTorch for tensor operations and
supports both analytical and numerical derivative computations.

Modules and Dependencies:
- **os**: Operating system interfaces.
- **torch**: PyTorch library for tensor computations and deep learning.
- **numpy**: Numerical operations on arrays.
- **matplotlib.pyplot**: Plotting and visualization.
- **math**: Mathematical functions.
- **time**: Time-related functions.
- **typing**: Type hinting.
- **matplotlib.pyplot**: Visualization.
- **torch.jit**: Just-In-Time compilation for optimized performance.
- **utils.psf_fit_utils**: Custom utility functions for PSF fitting (assumed to be available).
- **tqdm**: Progress bar for loops.

Classes:
- **LM_MLE_with_iter (torch.nn.Module)**:
    Implements the Levenberg-Marquardt algorithm for MLE with iterative updates.

    **Attributes**:
    - `model`: The model to be fitted, which should return expected values and Jacobians.
    - `param_range_min_max`: Tensor specifying the minimum and maximum bounds for each parameter.
    - `iterations`: Number of iterations for the LM algorithm.
    - `lambda_`: Damping parameter for the LM update.
    - `tol`: Tolerance for convergence.

    **Methods**:
    - `forward(smp, initial, bg_constant=None, pos=None, bg_only=False)`: Performs MLE to estimate parameters.

- **npcfit_class (torch.nn.Module)**:
    Wrapper class for performing PSF fitting using the `npc_channel` function.

    **Attributes**:
    - `points`: Tensor containing points along the line for fitting.

    **Methods**:
    - `forward(x, good_array)`: Applies the `npc_channel` to the specified points.

- **VectorPSF_2D (torch.nn.Module)**:
    Models a vectorized 2D Point Spread Function (PSF).

    **Attributes**:
    - `VPSF`: An instance of a PSF model providing the `poissonrate` method.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF and its Jacobian.

- **Gaussian2DFixedSigmaPSF (torch.nn.Module)**:
    PSF model with fixed sigma.

    **Attributes**:
    - `roisize`: Region of interest size.
    - `sigma`: Fixed standard deviation of the Gaussian.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF with fixed sigma.

- **Gaussian2DFixedSigmaPSFFixedPos (torch.nn.Module)**:
    PSF model with fixed sigma and fixed position.

    **Attributes**:
    - `roisize`: Region of interest size.
    - `sigma`: Fixed standard deviation of the Gaussian.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF with fixed sigma and position.

- **Gaussian2D_IandBg (torch.nn.Module)**:
    PSF model that fits both intensity and background.

    **Attributes**:
    - `roisize`: Region of interest size.
    - `sigma`: Standard deviation of the Gaussian.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF fitting intensity and background.

- **Gaussian2D_Bg (torch.nn.Module)**:
    PSF model that fits only the background.

    **Attributes**:
    - `roisize`: Region of interest size.
    - `sigma`: Standard deviation of the Gaussian.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF fitting only the background.

- **Gaussian_flexsigma (torch.nn.Module)**:
    PSF model with flexible sigma.

    **Attributes**:
    - `roisize`: Region of interest size.

    **Methods**:
    - `forward(x, bg_constant=None, pos=None, bg_only=False)`: Computes the PSF with flexible sigma.

- **MLE_new (torch.nn.Module)**:
    Another implementation of the LM algorithm for MLE.

    **Attributes**:
    - `model`: The model to be fitted.

    **Methods**:
    - `forward(initial, smp, param_range_min_max, iterations, lambda_)`: Performs MLE to estimate parameters.
    - `forward_spline(smp, initial, const_=None)`: Performs spline-based MLE.

- **LM_MLE_forspline (torch.nn.Module)**:
    LM MLE implementation specifically for spline fitting.

    **Attributes**:
    - `model`: The spline model to be fitted.
    - `param_range_min_max`: Parameter bounds.
    - `iterations`: Number of LM iterations.
    - `lambda_`: Damping parameter.

    **Methods**:
    - `forward(smp, initial, const_=None)`: Performs spline-based MLE.

- **LM_MLE_forspline_new (torch.nn.Module)**:
    An updated version of the LM MLE for spline fitting.

    **Attributes**:
    - `model`: The spline model to be fitted.

    **Methods**:
    - `forward(initial, smp, param_range_min_max, iterations, lambda_)`: Performs updated spline-based MLE.

- **LM_MLE (torch.nn.Module)**:
    Base LM MLE implementation.

    **Attributes**:
    - `model`: The model to be fitted.
    - `param_range_min_max`: Parameter bounds.
    - `iterations`: Number of LM iterations.
    - `lambda_`: Damping parameter.

    **Methods**:
    - `forward(smp, initial, const_=None)`: Performs MLE using the LM algorithm.
    - `forward_spline(smp, initial, const_=None)`: Performs spline-based MLE.

Functions:
- **d_fun(x, mu, sigma, A, Offset, scaling, factor, mu2)**:
    Computes the numerical derivative of the `intensity_func_sig` with respect to `mu2` using finite differences.

    **Parameters**:
    - `x`: Input tensor.
    - `mu`, `sigma`, `A`, `Offset`, `scaling`, `factor`, `mu2`: Parameters of the intensity function.

    **Returns**:
    - Numerical derivative tensor.

- **intensity_func_sig(x, mu, sigma, A, Offset, scaling, factor, mu2)**:
    Computes the intensity function combining a Gaussian and a sigmoid component.

    **Parameters**:
    - `x`: Input tensor.
    - `mu`, `sigma`, `A`, `Offset`, `scaling`, `factor`, `mu2`: Parameters of the intensity function.

    **Returns**:
    - Computed intensity tensor.

- **npc_channel(theta, dist_alongline_int)**:
    Computes the predicted intensity and its derivatives for non-parametric channel fitting.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `dist_alongline_int`: Distance along the line tensor.

    **Returns**:
    - Predicted intensity tensor.
    - Derivative tensor.

- **compute_numerical_derivatives(theta, epsilon, deriv_analytical, numpixels, bg_constant)**:
    Computes numerical derivatives using central differences.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `epsilon`: Small perturbation value.
    - `deriv_analytical`: Analytical derivative tensor.
    - `numpixels`: Number of pixels.
    - `bg_constant`: Background constant tensor.

    **Returns**:
    - Numerical derivative tensor.

- **gauss_psf_2D(theta, numpixels, bg_constant=None)**:
    Computes a 2D Gaussian PSF and its Jacobian.

    **Parameters**:
    - `theta`: Parameter tensor of shape [batchsize, 5], containing [x, y, N, bg, sigma].
    - `numpixels`: Number of pixels in each dimension.
    - `bg_constant`: Optional background constant tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `deriv`: Jacobian tensor.

- **show_tensor(img)**:
    Displays a tensor as an image using Napari.

    **Parameters**:
    - `img`: Image tensor to display.

- **gauss_psf_2D_fixed_sigma(theta, roisize, sigma, bg_constant=None)**:
    Computes a 2D Gaussian PSF with fixed sigma.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `roisize`: Region of interest size.
    - `sigma`: Fixed standard deviation.
    - `bg_constant`: Optional background constant tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.

- **gauss_psf_2D_fixed_sigma_fixed_pos(theta, roisize, sigma, bg_constant=None, pos=None)**:
    Computes a 2D Gaussian PSF with fixed sigma and position.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `roisize`: Region of interest size.
    - `sigma`: Fixed standard deviation.
    - `bg_constant`: Optional background constant tensor.
    - `pos`: Optional position tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.

- **gauss_psf_2D_flex_sigma(theta, roisize, bg_constant=None, pos=None)**:
    Computes a 2D Gaussian PSF with flexible sigma.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `roisize`: Region of interest size.
    - `bg_constant`: Optional background constant tensor.
    - `pos`: Optional position tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.

- **gauss_psf_2D_I_Bg(theta, roisize, sigma, bg_constant=None, pos=None)**:
    Computes a 2D Gaussian PSF fitting both intensity and background.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `roisize`: Region of interest size.
    - `sigma`: Standard deviation.
    - `bg_constant`: Optional background constant tensor.
    - `pos`: Optional position tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.

- **gauss_psf_2D_Bg(theta, roisize, sigma, bg_constant=None, pos=None)**:
    Computes a 2D Gaussian PSF fitting only the background.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `roisize`: Region of interest size.
    - `sigma`: Standard deviation.
    - `bg_constant`: Optional background constant tensor.
    - `pos`: Optional position tensor.

    **Returns**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.

- **compute_crlb(mu, jac, skip_axes=[])**:
    Computes the Cramér-Rao Lower Bound (CRLB) from the expected value and per-pixel derivatives.

    **Parameters**:
    - `mu`: Expected value tensor of shape [N, H, W].
    - `jac`: Jacobian tensor of shape [N, H, W, coords].
    - `skip_axes`: List of axes to skip in the computation.

    **Returns**:
    - `crlb`: CRLB tensor of shape [N, num_parameters].

- **likelihood_v2(image, mu, dmudtheta)**:
    Computes the log-likelihood, gradient, and Hessian matrix for the given image data.

    **Parameters**:
    - `image`: Observed image tensor.
    - `mu`: Expected value tensor.
    - `dmudtheta`: Derivative of `mu` with respect to parameters.

    **Returns**:
    - `logL`: Log-likelihood tensor.
    - `gradlogL`: Gradient of the log-likelihood.
    - `HessianlogL`: Hessian matrix of the log-likelihood.

- **MLE_instead_lmupdate(cur, mu, jac, smp, lambda_, param_range_min_max)**:
    Performs an LM update step using the Hessian and gradient from the likelihood.

    **Parameters**:
    - `cur`: Current parameter estimates.
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.
    - `smp`: Sample tensor.
    - `lambda_`: Damping parameter.
    - `param_range_min_max`: Parameter bounds tensor.

    **Returns**:
    - Updated parameter estimates.
    - Scale factor.

- **lm_alphabeta(mu, jac, smp)**:
    Computes the alpha and beta matrices for the LM algorithm.

    **Parameters**:
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.
    - `smp`: Sample tensor.

    **Returns**:
    - `alpha`: Alpha matrix tensor.
    - `beta`: Beta tensor.

- **lm_update(cur, mu, jac, smp, lambda_, param_range_min_max, scale_old=torch.Tensor(1))**:
    Performs the LM update step to adjust the current parameter estimates.

    **Parameters**:
    - `cur`: Current parameter estimates.
    - `mu`: Expected value tensor.
    - `jac`: Jacobian tensor.
    - `smp`: Sample tensor.
    - `lambda_`: Damping parameter.
    - `param_range_min_max`: Parameter bounds tensor.
    - `scale_old`: Previous scale tensor.

    **Returns**:
    - Updated parameter estimates.
    - Scale factor.

- **compute_numerical_derivatives(theta, epsilon, deriv_analytical, numpixels, bg_constant)**:
    Computes numerical derivatives of the PSF with respect to parameters using central differences.

    **Parameters**:
    - `theta`: Parameter tensor.
    - `epsilon`: Small perturbation value.
    - `deriv_analytical`: Analytical derivative tensor.
    - `numpixels`: Number of pixels.
    - `bg_constant`: Background constant tensor.

    **Returns**:
    - Numerical derivative tensor.

- **show_tensor(img)**:
    Displays a tensor as an image using Napari.

    **Parameters**:
    - `img`: Image tensor to display.

Usage:
--------
This script is designed for applications involving image analysis where Gaussian PSF fitting and background estimation are required.
It is particularly useful in fields like microscopy, where precise localization of points in noisy data is essential.

**Example Workflow**:
1. **Define the PSF Model**:
    ```python
    psf_model = Gaussian2D_IandBg(roisize=64, sigma=1.5)
    mle = LM_MLE_with_iter(model=psf_model, param_range_min_max=bounds, iterations=30, lambda_=1e-5, tol=tol)
    ```
2. **Prepare Sample Data**:
    ```python
    samples = torch.randn(batch_size, H, W)  # Replace with actual image data
    initial_params = torch.tensor([...])      # Initial parameter estimates
    ```
3. **Perform MLE**:
    ```python
    estimates, loglik, traces = mle(smp=samples, initial=initial_params, bg_constant=None, pos=None)
    ```
4. **Compute CRLB**:
    ```python
    crlb = compute_crlb(mu, jac)
    ```
5. **Visualize Results**:
    ```python
    show_tensor(estimates[0])
    ```

Notes:
--------
- Ensure that all dependencies, especially custom modules like `utils.psf_fit_utils`, are correctly installed and accessible.
- The script is optimized for GPU usage with PyTorch. Ensure that CUDA is available for best performance.
- The `torch.jit.script` decorators indicate functions that can be compiled for optimized performance but may have limitations on supported operations.
- The LM algorithm implementations (`LM_MLE`, `LM_MLE_with_iter`, etc.) are designed to be flexible for different PSF models and fitting scenarios.
- The PSF models assume specific parameterizations; ensure that input parameters align with these expectations.
- The `compute_crlb` function calculates the Cramér-Rao Lower Bound, providing a theoretical lower bound on the variance of unbiased estimators.
- The `likelihood_v2` function computes the log-likelihood and its derivatives, essential for gradient-based optimization in MLE.
- The `show_tensor` function utilizes Napari for visualization, which should be installed separately.

Best Practices:
-----------------
- Validate input data shapes and parameter bounds before performing MLE to avoid runtime errors.
- Monitor the convergence of the LM algorithm by examining the `traces` output.
- Utilize GPU acceleration for large batch sizes to expedite computations.
- Handle potential numerical instabilities, such as division by zero, by appropriately clamping or regularizing input tensors.
- Customize tolerance (`tol`) and damping parameter (`lambda_`) based on the specific dataset and fitting requirements.

"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple, Union, Optional
from torch import Tensor


def d_fun(x, mu, sigma, A, Offset, scaling, factor, mu2):
    h = 1e-5
    return (intensity_func_sig(x, mu, sigma, A, Offset, scaling, factor, mu2 +h) - intensity_func_sig(x, mu, sigma, A,
                                                                                                  Offset, scaling,
                                                                                                  factor , mu2)) / ( h)


# @torch.jit.script
def intensity_func_sig(x, mu, sigma, A, Offset, scaling, factor, mu2):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2) + factor / (1.0 + np.exp(-1.0 * scaling * (x - mu2))) + Offset


class LM_MLE_with_iter(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float, tol):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()
        self.tol = tol
        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)


        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, initial, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        dev = smp.device
        smp = torch.clamp(smp,1e-4)
        cur = initial * 1
        traces_list = torch.zeros(cur.size(),device=dev)
        loglik = torch.empty(cur.size(dim=0))
        mu = torch.zeros(smp.size(),device=dev)
        scale = torch.zeros(cur.size(),device=dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], cur.size()[1])).to(dev)

        scale = torch.zeros(cur.size(), device=dev)
        traces = torch.zeros((self.iterations + 1, cur.size()[0], cur.size()[1]), device=dev)
        traces[0, :, :] = cur

        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * self.tol[None, ...].repeat([cur.size()[0], 1]).to(dev)
        good_array = torch.ones(cur.size()[0], dtype=torch.bool).to(dev)
        delta = torch.ones(cur.size()).to(dev)#.type(torch.float)
        bool_array = torch.ones(cur.size(), dtype=torch.bool).to(dev)
        i = 0
        flag_tolerance = 0


        while (i < self.iterations) and (flag_tolerance == 0):
            if bg_constant is not None:
                bg_constant_temp=bg_constant[good_array, ...]
            else:
                bg_constant_temp = None
            if pos is not None:
                pos_temp = pos[good_array, ...]
            else:
                pos_temp = None
            temp1, temp2 =  self.model.forward(cur[good_array,...], bg_constant_temp, pos_temp,bg_only)
            mu[good_array,...]= temp1
            jac[good_array, ...] = temp2
            # cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
            #                                                      jac[good_array, :, :], smp[good_array, :], lambda_,
            #                                                      param_range_min_max, scale[good_array, :])
            cur[good_array, :], scale[good_array, :] = MLE_instead_lmupdate(cur[good_array, :], mu[good_array, :, :],
                                                                            jac[good_array, :, :, :],
                                                                            smp[good_array, :, :],
                                                                            self.lambda_, self.param_range_min_max)

            #mu, jac = self.model(cur, const_)
            #cur[good_array, ...], scale[good_array, ...] = lm_update(cur[good_array, ...], mu[good_array, ...]
                                #, jac[good_array, ...], smp[good_array, ...], self.lambda_, self.param_range_min_max, scale[good_array, ...])
            loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))

            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])
            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :])#.type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != cur.size(-1)
            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        return cur, loglik, traces
#@torch.jit.script
def npc_channel(theta, dist_alongline_int):
    dist_alongline = dist_alongline_int  # [None,...]
    mu, sigma, A, Offset, scaling, factor, mu2 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6]

    sigma = sigma[..., None]
    A = A[..., None]
    Offset = Offset[..., None]
    scaling = scaling[..., None]
    factor = factor[..., None]
    mu = mu[..., None]
    mu2 = mu2[..., None]

    ypred = A * torch.exp(-(dist_alongline - mu) ** 2 / 2 / sigma ** 2) + factor / (
            1.0 + torch.exp(-1.0 * scaling * (dist_alongline - mu2))) + Offset

    x = dist_alongline * 1

    dfdmu = A * (x - mu) / (sigma ** 2) * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    test = d_fun(x.cpu().detach().numpy(), mu.cpu().detach().numpy(), sigma.cpu().detach().numpy(), A.cpu().detach().numpy(), Offset.cpu().detach().numpy(), scaling.cpu().detach().numpy(), factor.cpu().detach().numpy(), mu2.cpu().detach().numpy())
    dfdsigma = A * (x - mu) ** 2 / (sigma ** 3) * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    dfdA = torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    dfdOffset = torch.ones(dfdA.size(), device=theta.device)
    dfds = (factor * (x - mu) * torch.exp(-scaling * (x - mu))) / (1 + torch.exp(-scaling * (x - mu))) ** 2
    dfdfactor = 1 / (1 + torch.exp(-scaling * (x - mu)))
    dfdmu2 = - (
            factor * scaling * torch.exp(-scaling * (x - mu))) / (1 + torch.exp(-scaling * (x - mu))) ** 2
    check = dfdmu2.cpu().detach().numpy()
    deriv = torch.stack((dfdmu, dfdsigma, dfdA, dfdOffset, dfds, dfdfactor,dfdmu2), -1)

    return ypred[..., None].type(torch.cuda.FloatTensor), deriv[..., None, :].type(torch.cuda.FloatTensor)


class npcfit_class(torch.nn.Module):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def forward(self, x, good_array):
        return npc_channel(x, self.points[good_array])


#######################################################################################
class VectorPSF_2D(torch.nn.Module):
    def __init__(self, VPSF):
        super().__init__()
        self.VPSF = VPSF


    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool =False):


        if pos is not None:
            x_ori = x*1
            x_new = torch.cat((pos,x_ori), dim=1)

        else:
            x_new = x*1

        x_new[:, 0:2] = (x_new[:, 0:2] - (self.VPSF.Mx / 2)) * self.VPSF.pixelsize
        x_new[:, [1, 0]] = x_new[:, [0, 1]]
        # insert z estim (0s)
        tensor_with_column = torch.cat((x_new[:, :2], torch.zeros_like(x_new[:,0])[...,None], x_new[:, 2:]), dim=1)
        if bg_only:

            tensor_with_column[:,3] = 0
            tensor_with_column = torch.cat((tensor_with_column, x_ori),
                                           dim=1)

        mu, jac = self.VPSF.poissonrate(tensor_with_column, bg_constant)  # needs to return mu and jac

        x_new[:, 0:2] = x_new[:, 0:2] / self.VPSF.pixelsize + self.VPSF.Mx / 2
        jac[:,:,:,[0,1]] =  jac[:,:,:,[1,0]]* self.VPSF.pixelsize
        # get rid of z
        jac = jac[:,:,:,[0,1,3,4]]
        if pos is not None:
            jac = jac[:,:,:,[2,3]]
        if bg_only == True:
            jac = jac[...,-1][...,None]
        #swap jac x and y
        return mu, jac


################################################################################
def compute_numerical_derivatives( theta, epsilon, deriv_analytical, numpixels, bg_constant):
    # Initialize numerical derivatives tensor
    deriv_numerical = torch.zeros_like(deriv_analytical)

    # Loop over parameters
    for i in range(5):
        # Add epsilon to parameter i
        theta_plus = theta.clone()
        theta_plus[:, i] += epsilon

        # Compute mu for theta_plus
        mu_plus, _ = gauss_psf_2D(theta_plus, numpixels, bg_constant)

        # Subtract epsilon from parameter i
        theta_minus = theta.clone()
        theta_minus[:, i] -= epsilon

        # Compute mu for theta_minus
        mu_minus, _ = gauss_psf_2D(theta_minus, numpixels, bg_constant)

        # Compute numerical derivative for parameter i
        deriv_numerical[..., i] = (mu_plus - mu_minus) / (2 * epsilon)

    return deriv_numerical
@torch.jit.script
def gauss_psf_2D(theta, numpixels: int, bg_constant:Optional[torch.Tensor]=None):
    """
    theta: [x,y,N,bg,sigma].T
    """
    numpixels = torch.tensor(numpixels, device=theta.device)
    pi = 3.141592653589793  # torch needs to include pi
    pi = torch.tensor(pi, device=theta.device)
    x, y, N, bg, sigma = theta[:, 0]-0.5, theta[:, 1]-0.5, theta[:, 2], theta[:, 3], theta[:, 4]
    pixelpos = torch.arange(0, numpixels, device=theta.device)

    OneOverSqrt2PiSigma = (1.0 / (torch.sqrt(2 * pi) * sigma))[:, None, None]
    OneOver2Sigmasqrd = (1.0 / (torch.tensor(2, device=theta.device) * sigma**2))[:, None, None]
    OneOver2SigmasqrdSqrtpi = (1.0 / (torch.tensor(2, device=theta.device) * sigma**2 *torch.sqrt(pi)))[:, None, None]

    OneOverSqrt2PiSigmasqrd = (1.0 / (torch.sqrt(2 * pi) * sigma ** 2))[:, None, None]

    # Pixel centers
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]
    Xexp0 = (Xc - x[:, None, None] + 0.5) * OneOver2Sigmasqrd
    Xexp1 = (Xc - x[:, None, None] - 0.5) * OneOver2Sigmasqrd

    Xexp0sig = (Xc - x[:, None, None] + 0.5) * OneOverSqrt2PiSigmasqrd
    Xexp1sig = (Xc - x[:, None, None] - 0.5) * OneOverSqrt2PiSigmasqrd

    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOver2SigmasqrdSqrtpi* (torch.exp(-Xexp1 ** 2 ) - torch.exp(-Xexp0 ** 2))
    # dEx_dSigma = (torch.exp(-Xexp1 ** 2) * Xexp1 - torch.exp(-Xexp0 ** 2) * Xexp0) / torch.sqrt(pi)
    dEx_dSigma = -Xexp0sig * torch.exp(-Xexp0 ** 2) + Xexp1sig * torch.exp(-Xexp1 ** 2)

    Yexp0 = (Yc - y[:, None, None] + 0.5) * OneOver2Sigmasqrd
    Yexp1 = (Yc - y[:, None, None] - 0.5) * OneOver2Sigmasqrd

    Yexp0sig = (Yc - y[:, None, None] + 0.5) * OneOverSqrt2PiSigmasqrd
    Yexp1sig = (Yc - y[:, None, None] - 0.5) * OneOverSqrt2PiSigmasqrd

    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy =   OneOver2SigmasqrdSqrtpi*(torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))
    # dEy_dSigma = (torch.exp(-Yexp1 ** 2) * Yexp1 - torch.exp(-Yexp0 ** 2) * Yexp0) / torch.sqrt(pi)
    dEy_dSigma = -Yexp0sig * torch.exp(-Yexp0 ** 2) + Yexp1sig * torch.exp(-Yexp1 ** 2)

    if bg_constant is not None:
        mu = N[:, None, None] * Ex * Ey + bg_constant/torch.amax(bg_constant, dim=(-1, -2))[...,None,None] * bg[...,None,None]
        dmu_bg = bg_constant/torch.amax(bg_constant, dim=(-1, -2))[...,None,None]
    else:
        mu = N[:, None, None] * Ex * Ey + bg[:, None, None]
        dmu_bg = 1 + mu * 0
    dmu_x = N[:, None, None] * Ey * dEx
    dmu_y = N[:, None, None] * Ex * dEy
    dmu_I = Ex * Ey

    dmu_sigma = N[:, None, None] * (Ex * dEy_dSigma + dEx_dSigma * Ey)

    deriv = torch.stack((dmu_x, dmu_y, dmu_I, dmu_bg, dmu_sigma), -1)
    # deriv_num = compute_numerical_derivatives(theta, 1e-4, deriv, numpixels, bg_constant)
    # show_tensor(deriv[...,0])
    # show_tensor(deriv_num[..., 0])
    return mu, deriv

def show_tensor(img):
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())
@torch.jit.script
def gauss_psf_2D_fixed_sigma(theta, roisize: int, sigma: float,bg_constant:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    theta_ = torch.cat((theta, sigma_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., :-1]

@torch.jit.script
def gauss_psf_2D_fixed_sigma_fixed_pos(theta, roisize: int, sigma: float,bg_constant:Optional[torch.Tensor]=None,pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    theta_ = torch.cat((theta, sigma_), -1)
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((pos, theta_), -1)
    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 2:-1]



@torch.jit.script
def gauss_psf_2D_flex_sigma(theta, roisize: int, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    mu, jac = gauss_psf_2D(theta, roisize)
    return mu, jac[..., :]


@torch.jit.script
def gauss_psf_2D_I_Bg(theta, roisize: int, sigma: float, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((theta, sigma_), -1)
    theta_ = torch.cat((pos, theta_), -1)
    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 2:-1]


@torch.jit.script
def gauss_psf_2D_Bg(theta, roisize: int, sigma: float, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    I = torch.zeros((len(theta), 1), device=theta.device)
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((theta, sigma_), -1)
    theta_ = torch.cat((I, theta_), -1)
    theta_ = torch.cat((pos, theta_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 3:-1]


class Gaussian2DFixedSigmaPSF(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_fixed_sigma(x, self.roisize, self.sigma, bg_constant)

class Gaussian2DFixedSigmaPSFFixedPos(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_fixed_sigma_fixed_pos(x, self.roisize, self.sigma, bg_constant, pos)

class Gaussian2D_IandBg(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_I_Bg(x, self.roisize, self.sigma, bg_constant, pos)


class Gaussian2D_Bg(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None , pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_Bg(x, self.roisize, self.sigma, bg_constant,pos)


class Gaussian_flexsigma(torch.nn.Module):
    def __init__(self, roisize):
        super().__init__()
        self.roisize = roisize

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None , pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_flex_sigma(x, self.roisize)


def compute_crlb(mu: Tensor, jac: Tensor, *, skip_axes: List[int] = []):
    """
    Compute crlb from expected value and per pixel derivatives.
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)

    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[..., axes]

    sample_dims = tuple(np.arange(1, len(mu.shape)))

    fisher = torch.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
    fisher = fisher / mu[..., None, None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = torch.zeros((len(mu), naxes), device=mu.device)
    crlb[:, axes] = torch.sqrt(torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2))
    return crlb
@torch.jit.script
def likelihood_v2(image, mu, dmudtheta):

    sample_dims = [-2, -1]
    sample_dimsjac = [-3, -2]
    varfit = 0
    # calculation of weight factors
    keps = 1e3 * 2.220446049250313e-16

    mupos = (mu > 0) * mu + (mu < 0) * keps

    weight = (image - mupos) / (mupos + varfit)
    dweight = (image + varfit) / (mupos + varfit) ** 2
    num_params = dmudtheta.size()[-1]

    # log-likelihood, gradient vector and Hessian matrix
    logL = torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), sample_dims)
    gradlogL = torch.sum(weight[..., None] * dmudtheta, sample_dimsjac)
    HessianlogL = torch.zeros((gradlogL.size(0), num_params, num_params))
    for ii in range(num_params):
        for jj in range(num_params):
            HessianlogL[:, ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj], sample_dims)

    return logL, gradlogL, HessianlogL

#@torch.jit.script
def MLE_instead_lmupdate(cur, mu, jac, smp, lambda_: float, param_range_min_max):
    """
    Separate some of the calculations to speed up with jit script
    """
    #
    merit, grad, Hessian = likelihood_v2(smp, mu, jac)
    diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
    b = torch.eye(diag.size(1))
    # hessian can be approximated by J^T*J
    c = diag.unsqueeze(2).expand(diag.size(0),diag.size(1),diag.size(1))

    diag_full = c * b
    # matty = Hessian + lambda_ * diag_full

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + lambda_ * diag_full
    Bmat = Bmat.to(cur.device)

    # try:
    #     dtheta = torch.linalg.solve(-Bmat, grad)
    # except:
    eye = torch.eye(Bmat.size(1), device=Bmat.device, dtype=Bmat.dtype)
    Bmat += eye * 1e-6
    dtheta = torch.linalg.solve(-Bmat, grad)
    dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
    cur = cur + dtheta

    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))

    scale = 1
    return cur, scale

@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    # assert np.array_equal(smp.shape, mu.shape)
    sampledims = [i for i in range(1, len(smp.shape))]

    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp * invmu ** 2

    jacm = torch.matmul(jac[..., None], jac[..., None, :])
    alpha = jacm * af[..., None, None]
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp * invmu - 1)[..., None]).sum(sampledims)
    return alpha, beta


#@torch.jit.script
def lm_update(cur, mu, jac, smp, lambda_: float, param_range_min_max, scale_old=torch.Tensor(1)):
    """
    Separate some of the calculations to speed up with jit script
    """
    alpha, beta = lm_alphabeta(mu, jac, smp)
    scale_old = scale_old.to(device=cur.device)
    K = cur.shape[-1]

    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)
        scale = (alpha * alpha).sum(1)
        #scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent

        if scale_old.size() != torch.Size([1]):
            scale = torch.maximum(scale,scale_old)

        # assert torch.isnan(scale).sum()==0
        alpha += lambda_ * scale[:, :, None] * torch.eye(K, device=smp.device)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K, device=smp.device)[None]

    steps = torch.linalg.solve(alpha, beta)
    #steps[torch.isnan(steps)] = (cur[torch.isnan(steps)]+0.1)*0.9
    assert torch.isnan(cur).sum() == 0
    assert torch.isnan(steps).sum() == 0

    cur = cur + steps
    # if Tensor.dim(param_range_min_max) == 2:
    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))
    # elif Tensor.dim(param_range_min_max) == 3:
    #
    # cur = torch.maximum(cur, param_range_min_max[:, :, 0])
    # cur = torch.minimum(cur, param_range_min_max[:, :, 1])
    # else:
    #     raise 'check bounds'
    if scale_old.size() != torch.Size([1]):
        return cur, scale
    else:
        return cur, scale

class MLE_new(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model
    def forward(self, initial, smp, param_range_min_max,iterations:int, lambda_:float):
        """
            model:
                function that takes parameter array [batchsize, num_parameters]
                and returns (expected_value, jacobian).
                Expected value has shape [batchsize, sample dims...]
                Jacobian has shape [batchsize, sample dims...,num_parameters]

            initial: [batchsize, num_parameters]

            return value is a tuple with:
                estimates [batchsize,num_parameters]
                traces [iterations, batchsize, num_parameters]
        """
        dev = initial.dev
        # if not isinstance(initial, torch.Tensor):
        #     initial = torch.Tensor(initial).to(smp.device)
        cur = (initial * 1)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #     param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        traces = torch.zeros((iterations + 1, cur.size()[0], cur.size()[1]), device=dev)
        traces[0, :, :] = cur

        assert len(smp) == len(initial)
        scale = torch.zeros(cur.size(), device=dev)
        tol_ = torch.ones(cur.size()[1]).to(dev)*0.1
        mu = torch.zeros(smp.size()).to(dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],cur.size()[1])).to(dev)
        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tol_[None, ...].repeat([cur.size()[0], 1])
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(dev)
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)
        i = 0
        flag_tolerance = 0

        while (i < iterations) and (flag_tolerance == 0):
            mu[good_array, :], jac[good_array, :, :] = self.model.forward(cur[good_array, :])
            logL, gradlogL, HessianlogL = likelihood_v2(smp[good_array, :],mu[good_array, :], jac[good_array, :, :])
            # cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
            #                                                      jac[good_array, :, :], smp[good_array, :], lambda_,
            #                                                      param_range_min_max, scale[good_array, :])
            cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
                                                          gradlogL[good_array, :, :], smp[good_array, :], lambda_,
                                                          param_range_min_max, scale[good_array, :])
            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != cur.size()[1]

            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
        loglik[torch.isinf(loglik)]=1e-20
        return cur,loglik, traces


class LM_MLE_forspline(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()

        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, initial, const_: Optional[Tensor] = None):
        cur = initial * 1
        scale = torch.zeros(cur.size(), device=smp.dev)
        for i in range(self.iterations):
            mu, jac = self.model(cur, const_)
            cur,scale = lm_update(cur, mu, jac, smp, self.lambda_, self.param_range_min_max, scale)

        return cur

class LM_MLE_forspline_new(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model
    def forward(self, initial, smp, param_range_min_max,iterations:int, lambda_:float):
        """
            model:
                function that takes parameter array [batchsize, num_parameters]
                and returns (expected_value, jacobian).
                Expected value has shape [batchsize, sample dims...]
                Jacobian has shape [batchsize, sample dims...,num_parameters]

            initial: [batchsize, num_parameters]

            return value is a tuple with:
                estimates [batchsize,num_parameters]
                traces [iterations, batchsize, num_parameters]
        """
        dev = initial.dev
        # if not isinstance(initial, torch.Tensor):
        #     initial = torch.Tensor(initial).to(smp.device)
        cur = (initial * 1)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #     param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        traces = torch.zeros((iterations + 1, cur.size()[0], cur.size()[1]), device=dev)
        traces[0, :, :] = cur

        assert len(smp) == len(initial)
        scale = torch.zeros(cur.size(), device=dev)
        toll = torch.tensor([0.01,0.01,1,1,0.01,1,0.1], device=dev)
        tol_ = toll
        mu = torch.zeros(smp.size()).to(dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],cur.size()[1])).to(dev)
        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tol_[None, ...].repeat([cur.size()[0], 1])
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(dev)
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)
        i = 0
        flag_tolerance = 0

        while (i < iterations) and (flag_tolerance == 0):
            mu[good_array, :], jac[good_array, :, :] = self.model.forward(cur[good_array, :], good_array)

            cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
                                                                 jac[good_array, :, :], smp[good_array, :], lambda_,
                                                                 param_range_min_max, scale[good_array, :])

            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != cur.size()[1]

            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
        loglik[torch.isinf(loglik)]=1e-20
        return cur,loglik, traces



class LM_MLE(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()

        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, initial, const_: Optional[Tensor] = None):
        cur = initial * 1
        loglik = torch.empty(cur.size(dim=0))
        mu = torch.zeros(cur.size())
        dev = cur.device
        scale = torch.zeros(cur.size(), device=dev)
        for i in range(self.iterations):
            mu, jac = self.model(cur, const_)
            cur, scale = lm_update(cur, mu, jac, smp, self.lambda_, self.param_range_min_max, scale)

            loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))

        # if chi:
        #     return cur, loglik, (smp - mu)**2/mu
        # else:

        return cur, loglik, (smp - mu) ** 2 / mu

    def forward_spline(self, smp, initial, const_: Optional[Tensor] = None):
        cur = initial * 1
        scale = torch.zeros(cur.size(), device=smp.dev)
        for i in range(self.iterations):
            mu, jac = self.model(cur, const_)
            cur, scale = lm_update(cur, mu, jac, smp, self.lambda_, self.param_range_min_max, scale)

        return cur

