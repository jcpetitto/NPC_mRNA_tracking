"""
This script provides implementations of various models and fitting procedures using PyTorch,
including functions for computing Richards curves, Gaussians, and their combinations,
as well as classes for performing Levenberg-Marquardt optimization for model fitting.

Functions:
- richards_curve_gaussian: Compute a combined Richards curve and Gaussian function and their derivatives.
- intensity_func_sig: Compute an intensity function combining a Gaussian and a scaled sigmoid.
- linear_gaussian_channel: Compute a linear function plus a Gaussian and their derivatives.
- numerical_derivative: Compute the numerical derivative of a function with respect to a parameter.
- numerical_derivative_linear: Compute the numerical derivative of a function with respect to a parameter.
- sigmoid_gaussian_channel: Compute a function combining a sigmoid and a Gaussian, and their derivatives.
- npc_channel: Compute the NPC (Richards curve plus Gaussian) model and its derivatives.
- lm_alphabeta: Compute the alpha and beta matrices for the Levenberg-Marquardt algorithm.
- lm_update: Perform an update step in the Levenberg-Marquardt algorithm.

Classes:
- LinearGaussianFitClass: PyTorch Module for fitting the Linear Gaussian model.
- SigmoidGaussianFitClass: PyTorch Module for fitting the Sigmoid Gaussian model.
- npcfit_class: PyTorch Module for fitting the NPC model.
- LM_MLE_forspline_new: Levenberg-Marquardt Maximum Likelihood Estimation for spline fitting.

The script is designed to facilitate the fitting of complex models to data using PyTorch,
providing both the models and the optimization algorithms necessary for parameter estimation.
"""

import torch

def richards_curve_gaussian(t, A, K, B,C, nu, Q, M, mu, sigma, amplitude, offset):
    """
        Compute a combined Richards curve and Gaussian function and their derivatives.

        Parameters:
        -----------
        t : torch.Tensor
            Independent variable.
        A : float
            Lower asymptote of the Richards curve.
        K : float
            Upper asymptote of the Richards curve.
        B : float
            Growth rate of the Richards curve.
        C : float
            Affects near which asymptote maximum growth occurs in the Richards curve.
        nu : float
            Shape parameter of the Richards curve.
        Q : float
            Affects near which asymptote maximum growth occurs in the Richards curve.
        M : float
            The time of maximum growth of the Richards curve.
        mu : float
            Mean of the Gaussian.
        sigma : float
            Standard deviation of the Gaussian.
        amplitude : float
            Amplitude of the Gaussian.
        offset : float
            Vertical offset of the combined function.

        Returns:
        --------
        combined_curve : torch.Tensor
            The combined Richards curve and Gaussian function evaluated at `t`.
        dA, dK, dB, dC, dnu, dQ, dM, dmu, dsigma, damplitude, d_offset : torch.Tensor
            Derivatives of the combined function with respect to each parameter.
        """
    exponent_richards = -B * (t - M)
    denominator_richards = C + Q * torch.exp(exponent_richards)
    power_richards = 1 / nu
    richards_curve = A + (K - A) / denominator_richards**power_richards

    exponent_gaussian = -(t - mu)**2 / (2 * sigma**2)
    gaussian_curve = amplitude * torch.exp(exponent_gaussian)

    combined_curve = richards_curve + gaussian_curve + offset

    # Compute derivatives
    dA = 1 - denominator_richards**(-power_richards)
    dK = denominator_richards**(-power_richards)
    dB = (K - A) * (t - M) * Q * torch.exp(exponent_richards) / (nu * denominator_richards**(power_richards + 1))
    # dC = -(K - A) * Q * torch.exp(exponent_richards) / (nu * denominator_richards ** (power_richards + 1))
    dC = -(K - A)  / (nu * denominator_richards ** (power_richards + 1))
    # dnu = (K - A) * torch.log(1 + Q * torch.exp(exponent_richards)) / (nu**2 * denominator_richards**(1 / nu))
    dnu = (K - A) * torch.log(C + Q * torch.exp(exponent_richards)) / (nu ** 2 * denominator_richards ** (1 / nu))
    dQ = -(K - A) * torch.exp(exponent_richards) / (nu * denominator_richards**(power_richards + 1))
    dM = -(K - A) * Q * B * torch.exp(exponent_richards) / (nu * denominator_richards**(power_richards + 1))
    dmu = amplitude * (t - mu) / sigma**2 * torch.exp(exponent_gaussian)
    dsigma = amplitude * (t - mu)**2 / sigma**3 * torch.exp(exponent_gaussian)
    damplitude = torch.exp(exponent_gaussian)
    d_offset = torch.ones_like(damplitude)
    return combined_curve, dA, dK, dB, dC,dnu, dQ, dM, dmu, dsigma, damplitude,d_offset


# @torch.jit.script
def intensity_func_sig(x, mu, sigma, A, Offset, scaling, factor, mu2):
    return A * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + factor / (
                1.0 + torch.exp(-1.0 * scaling * (x - mu2))) + Offset

def linear_gaussian_channel(theta, dist_alongline_int):
    dist_alongline = dist_alongline_int  # [None,...]

    slope = theta[:, 0][..., None]
    intercept = theta[:, 1][..., None]
    mu = theta[:, 2][..., None]
    sigma = theta[:, 3][..., None]
    amplitude = theta[:, 4][..., None]
    offset = theta[:, 5][..., None]
    offset = torch.zeros_like(offset) + 1e-5
    x = dist_alongline * 1

    exponent_gaussian = -0.5 * ((x - mu) / sigma) ** 2
    linear_part = slope * x + intercept
    gaussian_part = amplitude * torch.exp(exponent_gaussian) + offset

    ypred = linear_part + gaussian_part
    dslope = x
    dintercept = torch.ones_like(x)
    dmu = amplitude * (x - mu) / sigma**2 * torch.exp(exponent_gaussian)
    dsigma = amplitude * (x - mu)**2 / sigma**3 * torch.exp(exponent_gaussian)
    damplitude = torch.exp(exponent_gaussian)
    doffset = torch.ones_like(damplitude)*1e-6

    deriv = torch.stack((dslope, dintercept, dmu, dsigma, damplitude, doffset), -1)
    return ypred[..., None].type(torch.cuda.FloatTensor), deriv[..., None, :].type(torch.cuda.FloatTensor)
def numerical_derivative(theta, dist_alongline_int, idx, h=0.01):
    theta_h = theta.clone()
    theta_h[:, idx] += h
    ypred_h, _ = sigmoid_gaussian_channel(theta_h, dist_alongline_int)
    ypred, _ = sigmoid_gaussian_channel(theta, dist_alongline_int)
    num_deriv = (ypred_h - ypred) / h
    return num_deriv

def numerical_derivative_linear(theta, dist_alongline_int, idx, h=0.01):
    theta_h = theta.clone()
    theta_h[:, idx] += h
    ypred_h, _ = linear_gaussian_channel(theta_h, dist_alongline_int)
    ypred, _ = linear_gaussian_channel(theta, dist_alongline_int)
    num_deriv = (ypred_h - ypred) / h
    return num_deriv


def sigmoid_gaussian_channel(theta, dist_alongline_int):
    dist_alongline = dist_alongline_int  # [None,...]

    A = theta[:, 0][..., None]
    B = theta[:, 1][..., None]
    mu_sigmoid = theta[:, 2][..., None]  # New parameter for the sigmoid center
    mu_gaussian = theta[:, 3][..., None]  # Parameter for the Gaussian center
    sigma = theta[:, 4][..., None]
    amplitude = theta[:, 5][..., None]
    offset = theta[:, 6][..., None]

    x = dist_alongline * 1

    sigmoid_part = A / (1 + torch.exp(-B * (x - mu_sigmoid)))
    gaussian_part = amplitude * torch.exp(-0.5 * ((x - mu_gaussian) / sigma) ** 2) + offset

    ypred = sigmoid_part + gaussian_part
    dA = 1 / (1 + torch.exp(-B * (x - mu_sigmoid)))
    dB = A * (x - mu_sigmoid) * torch.exp(-B * (x - mu_sigmoid)) / ((1 + torch.exp(-B * (x - mu_sigmoid))) ** 2)
    dmu_sigmoid = -A * B * torch.exp(-B * (x - mu_sigmoid)) / ((1 + torch.exp(-B * (x - mu_sigmoid))) ** 2)
    dmu_gaussian = amplitude * torch.exp(-0.5 * ((x - mu_gaussian) / sigma) ** 2) * (x - mu_gaussian) / (sigma ** 2)
    dsigma = amplitude * torch.exp(-0.5 * ((x - mu_gaussian) / sigma) ** 2) * ((x - mu_gaussian) ** 2) / (sigma ** 3)
    damplitude = torch.exp(-0.5 * ((x - mu_gaussian) / sigma) ** 2)
    doffset = torch.ones_like(damplitude)

    deriv = torch.stack((dA, dB, dmu_sigmoid, dmu_gaussian, dsigma, damplitude, doffset), -1)
    return ypred[..., None].type(torch.cuda.FloatTensor), deriv[..., None, :].type(torch.cuda.FloatTensor)

class LinearGaussianFitClass(torch.nn.Module):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def forward(self, x, good_array):
        return linear_gaussian_channel(x, self.points[good_array])

class SigmoidGaussianFitClass(torch.nn.Module):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def forward(self, x, good_array):
        return sigmoid_gaussian_channel(x, self.points[good_array])
#@torch.jit.script
def npc_channel(theta, dist_alongline_int):


    dist_alongline = dist_alongline_int  # [None,...]

    A = theta[:, 0][..., None]
    K = theta[:, 1][..., None]
    B = theta[:, 2][..., None]
    C = theta[:, 3][..., None]
    nu = theta[:, 4][..., None]
    Q = theta[:, 5][..., None]
    M = theta[:, 6][..., None]
    mu = theta[:, 7][..., None]
    sigma = theta[:, 8][..., None]
    amplitude = theta[:, 9][..., None]
    offset = theta[:, 10][..., None]
    offset = torch.zeros_like(offset)+ 1e-5
    x = dist_alongline * 1

    ypred, dA, dK, dB,dC, dnu, dQ, dM, dmu, dsigma, damplitude,doffset = richards_curve_gaussian(x, A, K, B,C, nu, Q, M, mu, sigma, amplitude, offset)
    doffset=torch.zeros_like(doffset) + 1e-5
    deriv = torch.stack((dA, dK, dB,dC, dnu, dQ, dM, dmu, dsigma, damplitude, doffset), -1)

    # plt.plot(np.linspace(0,200,100), check[0,:], label='theoretical')
    # plt.plot(np.linspace(0, 200, 100), test[0,:], label='reality')
    # plt.legend()
    # plt.show()
    return ypred[..., None].type(torch.cuda.FloatTensor), deriv[..., None, :].type(torch.cuda.FloatTensor)


# !!!: for npc refinement
class npcfit_class(torch.nn.Module):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def forward(self, x, good_array):
        return npc_channel(x, self.points[good_array])

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
    steps = torch.zeros(cur.size()).to(device=cur.device)
    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)
        scale = (alpha * alpha).sum(1)
        # scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent
        #
        # if scale_old.size() != torch.Size([1]):
        #     scale = torch.maximum(scale, scale_old)

        # assert torch.isnan(scale).sum()==0
        alpha += lambda_[:,None,None] * scale[:, :, None] * torch.eye(K, device=smp.device)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K, device=smp.device)[None]

    try:
        steps = torch.linalg.solve(alpha, beta)
    except:
        steps=cur * 0.9



    #cur[torch.isnan(cur)] = 0.1
    steps[torch.isnan(steps)] = 0.1

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


# !!!: for npc refinement
class LM_MLE_forspline_new(torch.nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
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
        dev = 'cuda'
        # if not isinstance(initial, torch.Tensor):
        #     initial = torch.Tensor(initial).to(smp.device)
        cur = (initial * 1)
        cur_temp = (initial * 1)
        # if not isinstance(param_range_min_max, torch.Tensor):
        #     param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        traces = torch.zeros((iterations + 1, cur.size()[0], cur.size()[1]), device=dev)
        traces[0, :, :] = cur

        assert len(smp) == len(initial)
        scale = torch.zeros(cur.size(), device='cuda')
        if cur.size(1) == 11:
            toll = torch.tensor([1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-2,1e-2,1e-1,1e-1, 1e-1], device='cuda')
        elif cur.size(1) == 6:
            toll = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2, 1e-1, 1e-1], device='cuda')
        elif cur.size(1) == 7:
            toll = torch.tensor([1e-1, 1e-1,1e-2, 1e-2, 1e-2, 1e-1, 1e-1], device='cuda')
            
        tol_ = toll
        
        mu = torch.zeros(smp.size()).to(dev)
        
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],cur.size()[1])).to(dev)
        
        mu_new = torch.zeros(smp.size()).to(dev)
        
        jac_new = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],cur.size()[1])).to(dev)
        
        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tol_[None, ...].repeat([cur.size()[0], 1])
        
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        
        lambda_arr = torch.ones(cur.size()[0]).to(dev)*lambda_
        
        delta = torch.ones(cur.size()).to(dev)
        
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)
        
        lambda_fac = 1
        
        i = 0
        flag_tolerance = 0
        
        while (i < iterations) and (flag_tolerance == 0):
            
            if cur.size(1) == 7:
                pass
            else:
                cur[good_array, -1] = 0 # for offset == 0
                
            mu[good_array, :], jac[good_array, :, :] = self.model.forward(cur[good_array, :],
                                                                          good_array)

            merit_start = torch.sum(abs(mu[good_array, :]-smp[good_array, :]),dim=(-2,-1))
            cur_temp[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :],
                                                                      mu[good_array, :],
                                                                      jac[good_array, :, :],
                                                                      smp[good_array, :],
                                                                      lambda_arr[good_array],
                                                                      param_range_min_max,
                                                                      scale[good_array, :]
                                                                      )
            
            mu_new[good_array, :], jac_new[good_array, :, :] = self.model.forward(cur_temp[good_array, :],
                                                                                 good_array)

            merit_nodiv = torch.sum(abs(mu_new[good_array, :] - smp[good_array, :]), dim=(-2,-1))
            
            cur_temp[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :],
                                                                      mu[good_array, :],
                                                                      jac[good_array, :, :],
                                                                      smp[good_array, :],
                                                                      lambda_arr[good_array]/lambda_fac,
                                                                      param_range_min_max,
                                                                      scale[good_array, :])

            mu_new[good_array, :], jac_new[good_array, :, :] = self.model.forward(cur_temp[good_array, :],
                                                                                  good_array)
            
            merit = torch.sum(abs(mu_new[good_array, :] - smp[good_array, :]), dim=(-2, -1))
            
            lambda_arr[good_array][torch.logical_and(merit_start < merit_nodiv, merit_start < merit)] = \
                lambda_arr[good_array][torch.logical_and(merit_start < merit_nodiv,merit_start < merit)] * lambda_fac
            
            lambda_arr[good_array][merit_start > merit_nodiv] = \
                lambda_arr[good_array][merit_start > merit_nodiv] / lambda_fac
            
            cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :],
                                                                 mu[good_array, :],
                                                                     jac[good_array, :, :],
                                                                     smp[good_array, :],
                                                                     lambda_arr[good_array] / lambda_fac,
                                                                     param_range_min_max,
                                                                     scale[good_array, :])

            if cur.size(1) == 7:
                pass
            else:
                cur[good_array, -1] = 0  # for offset == 0
                
            traces[i + 1, good_array, :] = cur[good_array, :]
            
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            
            bool_array_np = bool_array.detach().cpu().numpy()
            
            delta_np = delta.detach().cpu().numpy()
            
            test = torch.sum(bool_array, dim = 1)
            
            good_array = test != cur.size()[1]

            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            
            i = i + 1
        
        loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
        
        loglik[torch.isinf(loglik)] = 1e-20
        
        return cur, loglik, traces
    
    



