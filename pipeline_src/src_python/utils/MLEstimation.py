import torch
import theseus as th

# -- NPC Fit Prediction -- #
def model_error_func(optim_vars: tuple[th.Variable, ...], aux_vars: tuple[th.Variable, ...], model_func: callable) -> torch.Tensor:
    """
    Computes the error vector for the model fit.
    Args:
        optim_vars: A tuple containing the variables to be optimized (theta).
        aux_vars: A tuple containing auxiliary data (distances, intensities).
        model_func: callable function to use for the model
    Returns:
        torch.Tensor: The error vector (prediction - measurement).
    """
    theta_var = optim_vars[0]
    distances, intensities = aux_vars
    
    # Use your existing vectorized model to get the prediction
    prediction = model_func(distances, theta_var)    
    
    residual = prediction - intensities.tensor
    return residual

def richards_curve_gaussian_vectorized(t, theta):
    t_tensor = t.tensor
    A, K, B, C, nu, Q, M, mu, sigma, amplitude, offset = theta.tensor.split(1, dim=-1)

    # --- ADD/MODIFY THESE LINES ---
    # Clamp strictly positive parameters to a small, safe minimum.
    # This is the fix for the Cholesky error.
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    nu = torch.clamp(nu, min=epsilon)
    C = torch.clamp(C, min=epsilon) # C is in the log, can't be <= 0
    
    # Also clamp the denominator base to avoid (negative_number)^(fraction)
    exponent_richards = -B * (t_tensor - M)
    denominator_richards = C + Q * torch.exp(exponent_richards)
    denominator_clamped = torch.clamp(denominator_richards, min=epsilon)
    # --- END OF MODIFICATIONS ---

    power_richards = 1 / nu
    # Use the clamped denominator
    richards_curve = A + (K - A) / denominator_clamped**power_richards

    exponent_gaussian = -(t_tensor - mu)**2 / (2 * sigma**2) # sigma is now safe
    gaussian_curve = amplitude * torch.exp(exponent_gaussian)
    
    combined_curve = richards_curve + gaussian_curve + offset
    # This clamp is good, keep it
    combined_curve = torch.clamp(combined_curve, min=1e-9)

    # --- ALL DERIVATIVES, using clamped values ---
    dA = 1 - denominator_clamped**(-power_richards) # Use clamped
    dK = denominator_clamped**(-power_richards) # Use clamped
    dB = (K - A) * (t_tensor - M) * Q * torch.exp(exponent_richards) / (nu * denominator_clamped**(power_richards + 1)) # Use clamped
    dC = -(K - A)  / (nu * denominator_clamped ** (power_richards + 1)) # Use clamped
    dnu = (K - A) * torch.log(denominator_clamped) / (nu ** 2 * denominator_clamped ** (1 / nu)) # Use clamped
    dQ = -(K - A) * torch.exp(exponent_richards) / (nu * denominator_clamped**(power_richards + 1)) # Use clamped
    dM = -(K - A) * Q * B * torch.exp(exponent_richards) / (nu * denominator_clamped**(power_richards + 1)) # Use clamped
    dmu = amplitude * (t_tensor - mu) / sigma**2 * torch.exp(exponent_gaussian) # sigma is safe
    dsigma = amplitude * (t_tensor - mu)**2 / sigma**3 * torch.exp(exponent_gaussian) # sigma is safe
    damplitude = torch.exp(exponent_gaussian)
    d_offset = torch.ones_like(damplitude)
    # --- END DERIVATIVES ---

    deriv = torch.stack((dA, dK, dB, dC, dnu, dQ, dM, dmu, dsigma, damplitude, d_offset), -1).squeeze(-2)
    
    return combined_curve.squeeze(-1)

def gaussian_linear_vectorized(t_var, theta_var):
    t = t_var.tensor
    theta = theta_var.tensor
    
    slope, intercept, mu, sigma, amplitude = theta.split(1, dim=-1)
    
    linear_part = slope * t + intercept
    exponent_gaussian = -(t - mu)**2 / (2 * sigma**2)
    gaussian_part = amplitude * torch.exp(exponent_gaussian)
    
    prediction = linear_part + gaussian_part
    return torch.clamp(prediction, min=1e-9).squeeze(-1)

def construct_gaus_lin_initial_guess(dist_tensor, intensity_tensor, device):
    """ Creates a 5-parameter initial guess for the Gaussian+Linear model. """
    
    # Heuristics for a simpler model
    slope = (intensity_tensor[:, -1] - intensity_tensor[:, 0]) / (dist_tensor[:, -1] - dist_tensor[:, 0])
    intercept = intensity_tensor[:, 0] - slope * dist_tensor[:, 0]
    
    # Estimate amplitude as max intensity above the baseline
    baseline = slope.unsqueeze(-1) * dist_tensor + intercept.unsqueeze(-1)
    amp = torch.max(intensity_tensor - baseline, dim=1).values
    amp = torch.clamp(amp, min=1.0)

    # Find the position of the max intensity
    mu_indices = torch.argmax(intensity_tensor - baseline, dim=1)
    mu = dist_tensor[torch.arange(dist_tensor.shape[0]), mu_indices]

    # Create the guess tensor
    init_guess = torch.zeros((dist_tensor.shape[0], 5), device=device)
    init_guess[:, 0] = slope
    init_guess[:, 1] = intercept
    init_guess[:, 2] = mu
    init_guess[:, 3] = torch.tensor(1.0, device=device)  # Initial sigma
    init_guess[:, 4] = amp

    return init_guess

# def construct_rich_gaus_initial_guess(
#     dist_along_norm_tensor: torch.Tensor,
#     intensity_init_tensor: torch.Tensor,
#     line_length: float,
#     num_parameters: int,
#     intensity_params: list,
#     device: torch.device
# ) -> torch.Tensor:
#     """
#     Constructs an 11-parameter initial guess for the Richards+Gaussian model.
#     """
    
#     # 1. Get batch size
#     batch_size = dist_along_norm_tensor.shape[0]

#     # 2. Create the output tensor
#     init_guess = torch.zeros((batch_size, num_parameters), device=device)

#     # 3. Heuristics for parameters
#     # A (lower asymptote) = min intensity
#     A = torch.min(intensity_init_tensor, dim=1).values
    
#     # K (upper asymptote) = max intensity
#     K = torch.max(intensity_init_tensor, dim=1).values
    
#     # M (Richards inflection) & mu (Gaussian center)
#     # Start them at the center of the line
#     center_pos = line_length / 2.0
    
#     # sigma (Gaussian width)
#     # Start with a reasonable default, e.g., 1.0 pixel
#     sigma_default = 1.0

#     # 4. Fill the tensor
#     init_guess[:, 0] = A
#     init_guess[:, 1] = K
#     init_guess[:, 2] = torch.tensor(intensity_params[0], device=device)  # B
#     init_guess[:, 3] = torch.tensor(intensity_params[1], device=device)  # C
#     init_guess[:, 4] = torch.tensor(intensity_params[2], device=device)  # nu
#     init_guess[:, 5] = torch.tensor(intensity_params[3], device=device)  # Q
#     init_guess[:, 6] = torch.tensor(center_pos, device=device)           # M
#     init_guess[:, 7] = torch.tensor(center_pos, device=device)           # mu
#     init_guess[:, 8] = torch.tensor(sigma_default, device=device)        # sigma
    
#     # amplitude (Gaussian amplitude)
#     # Estimate as difference between max (K) and min (A)
#     amplitude = K - A
#     init_guess[:, 9] = torch.clamp(amplitude, min=1.0) # Ensure it's positive
    
#     # offset (Linear offset)
#     init_guess[:, 10] = torch.tensor(intensity_params[6], device=device) # offset
    
#     return init_guess

# --- PUT THIS IN utils/MLEstimation.py ---
# (You'll need to add 'import torch' at the top of the file if it's not there)


def construct_rich_gaus_initial_guess(
    dist_along_norm_tensor: torch.Tensor,
    intensity_init_tensor: torch.Tensor,
    line_length: float,
    num_parameters: int,
    intensity_params: list,
    device: torch.device
) -> torch.Tensor:
    """
    Constructs an 11-parameter initial guess for the Richards+Gaussian model.
    """
    
    # 1. Get batch size
    batch_size = dist_along_norm_tensor.shape[0]

    # 2. Create the output tensor
    init_guess = torch.zeros((batch_size, num_parameters), device=device)

    # 3. Heuristics for parameters
    # A (lower asymptote) = min intensity
    A = torch.min(intensity_init_tensor, dim=1).values
    
    # K (upper asymptote) = max intensity
    K = torch.max(intensity_init_tensor, dim=1).values
    
    # M (Richards inflection) & mu (Gaussian center)
    # Start them at the center of the line
    center_pos = line_length / 2.0
    
    # sigma (Gaussian width)
    # Start with a reasonable default, e.g., 1.0 pixel
    sigma_default = 1.0

    # 4. Fill the tensor
    init_guess[:, 0] = A
    init_guess[:, 1] = K
    init_guess[:, 2] = torch.tensor(intensity_params[0], device=device)  # B
    init_guess[:, 3] = torch.tensor(intensity_params[1], device=device)  # C
    init_guess[:, 4] = torch.tensor(intensity_params[2], device=device)  # nu
    init_guess[:, 5] = torch.tensor(intensity_params[3], device=device)  # Q
    init_guess[:, 6] = torch.tensor(center_pos, device=device)           # M
    init_guess[:, 7] = torch.tensor(center_pos, device=device)           # mu
    init_guess[:, 8] = torch.tensor(sigma_default, device=device)        # sigma
    
    # amplitude (Gaussian amplitude)
    # Estimate as difference between max (K) and min (A)
    amplitude = K - A
    init_guess[:, 9] = torch.clamp(amplitude, min=1.0) # Ensure it's positive
    
    # offset (Linear offset)
    init_guess[:, 10] = torch.tensor(intensity_params[6], device=device) # offset
    
    return init_guess

# --- THE MODEL REGISTRY ---
# Select a model using a string from config_options
MODEL_REGISTRY = {
    "richards_gaussian": richards_curve_gaussian_vectorized,
    "gaussian_linear": gaussian_linear_vectorized
}

# class ModelFitCost(th.CostFunction):
    #     # 1. Accept theta again in the constructor
    #     def __init__(self, dist_tensor, intensity_tensor, cost_weight: th.CostWeight, theta: th.Vector):
    #         super().__init__(cost_weight, name="model_fit_cost")
    #         self.distances = dist_tensor
    #         self.intensities = intensity_tensor
    #         self.theta = theta
    #         self.register_optim_vars(["theta"])

    #     def error(self) -> torch.Tensor:
    #         theta_var = list(self.optim_vars)[0]
    #         prediction, _ = richards_curve_gaussian_vectorized(self.distances, theta_var.tensor)
    #         return prediction - self.intensities

    #     def jacobians(self) -> tuple[list[torch.Tensor], torch.Tensor]:
    #         theta_var = list(self.optim_vars)[0]
            
    #         _, jacobian = richards_curve_gaussian_vectorized(self.distances, theta_var.tensor)
    #         return [theta_var], jacobian

    #     def dim(self) -> int:
    #         return self.intensities.shape[1]

    #     def to(self, *args, **kwargs):
    #         super().to(*args, **kwargs)
    #         self.distances = self.distances.to(*args, **kwargs)
    #         self.intensities = self.intensities.to(*args, **kwargs)
    #         self.theta = self.theta.to(*args, **kwargs)
    #         return self

    #     def _copy_impl(self, **kwargs):
    #         return ModelFitCost(self.distances.clone(),
    #                             self.intensities.clone(),
    #                             self.weight.copy(),
    #                             self.theta.copy())


# -- Likely Obsolete -- #
# TODO make sure these can be deleted

# NPC_Predictor class initializes a model that is utilized by the LM_MLE_forspline model's forward method
class NPC_FitPredictor(torch.nn.Module):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def forward(self, x):
        return npc_channel(x, self.points)
    
def npc_channel(theta, dist_alongline_int):
#   Takes the distance along the normal line and the intensity
#   Breaks down the intensity tensor into the individual components
#       used for computation (ie. assigns them by position to named variables)
#   Computes the NPC (Richards curve plus Gaussian) model and its derivatives

    # A = theta[:, 0].unsqueeze(-1)
    # K = theta[:, 1].unsqueeze(-1)
    # B = theta[:, 2].unsqueeze(-1)
    # C = theta[:, 3].unsqueeze(-1)
    # nu = theta[:, 4].unsqueeze(-1)
    # Q = theta[:, 5].unsqueeze(-1)
    # M = theta[:, 6].unsqueeze(-1)
    # mu = theta[:, 7].unsqueeze(-1)
    # sigma = theta[:, 8].unsqueeze(-1)
    # amplitude = theta[:, 9].unsqueeze(-1)
    # offset = theta[:, 10].unsqueeze(-1)
    
# ???: is there a reason this would have been done?
#   offset = torch.zeros_like(offset) + 1e-5
     
    x = dist_alongline_int
    
    A, K, B, C, nu, Q, M, mu, sigma, amplitude, offset = theta.split(1, dim=-1)

    ypred, dA, dK, dB, dC, dnu, dQ, dM, dmu, dsigma, damplitude, doffset = \
        richards_curve_gaussian(x, A, K, B,C, nu, Q, M, mu, sigma, amplitude, offset)

# ???: is there a reason this would have been done?
#    doffset = torch.zeros_like(doffset) + 1e-5
    
    deriv = torch.stack((dA, dK, dB, dC, dnu, dQ, dM, dmu, dsigma, damplitude, doffset), -1)

    # plt.plot(np.linspace(0,200,100), check[0,:], label='theoretical')
    # plt.plot(np.linspace(0, 200, 100), test[0,:], label='reality')
    # plt.legend()
    # plt.show()
# !!!: these were torch.cuda.FloatTensor; what is the best option re: assigning these a type? particularly to have non-CUDA option but also a CUDA option if it and the option for paralell processing exists.

    return ypred.unsqueeze(-1), deriv.unsqueeze(-2)

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

    # Clamp the output to a very small positive number for numerical stabiity; keeping the values within range mathematically for the loss function as well as physically (negative intensity not being "a thing" physically, even if noise -> calibration or the optimization process may introduce negative values, thus "breaking" a loss function such as PoissonNLLLoss)
    combined_curve = torch.clamp(combined_curve, min=1e-9)

    # Compute derivatives
    dA = 1 - denominator_richards**(-power_richards)
    dK = denominator_richards**(-power_richards)
    dB = (K - A) * (t - M) * Q * torch.exp(exponent_richards) / \
        (nu * denominator_richards**(power_richards + 1))
        
    # ???: why the change? (more generalized form?)
    # dC = -(K - A) * Q * torch.exp(exponent_richards) / \
    #        (nu * denominator_richards ** (power_richards + 1))
    dC = -(K - A)  / (nu * denominator_richards ** (power_richards + 1))
    
    # ???: why the change? (more generalized form?)
    # dnu = (K - A) * torch.log(1 + Q * torch.exp(exponent_richards)) / \
    #    (nu**2 * denominator_richards**(1 / nu))
    dnu = (K - A) * torch.log(C + Q * torch.exp(exponent_richards)) / \
        (nu ** 2 * denominator_richards ** (1 / nu))
    
    dQ = -(K - A) * torch.exp(exponent_richards) / \
        (nu * denominator_richards**(power_richards + 1))
    dM = -(K - A) * Q * B * torch.exp(exponent_richards) / \
        (nu * denominator_richards**(power_richards + 1))
    dmu = amplitude * (t - mu) / sigma**2 * torch.exp(exponent_gaussian)
    dsigma = amplitude * (t - mu)**2 / sigma**3 * torch.exp(exponent_gaussian)
    damplitude = torch.exp(exponent_gaussian)
    d_offset = torch.ones_like(damplitude)
    
    return combined_curve, dA, dK, dB, dC, dnu, dQ, dM, dmu, dsigma, damplitude, d_offset

# Levenberg-Marquardt Maximum Likelihood Estimation for spline fitting
class LM_MLE_forSpline(torch.nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self._fit_model = model # instance of a fit class (e.g. NPC_FitPredictor)
        
    # ???: re: iterations, should this be a max w/ a cut out sooner if change isn't observed between iterations?
    def forward(self, initial, smp, param_range_min_max, iterations:int, lambda_:float, device, correct_offset = False):
        # initial - initial guess (tensor)
        # smp - sample_data (tensor)
        # param_range_min_max - provides bounds for parameter values
        # iterations: number of iterations to run
        # lambda_ : hyperparameter for regularization (dampening)
        #               Gradient Descent (large λ) and Gauss-Newton (small λ)

        current_params_tensor = initial.clone() # current est. of best fit paramaters
        temp_params = initial.clone()   # proposed est of best fit paramaters
        
        # history log for paramaters
        param_history_tensor = torch.zeros(iterations + 1,
                                           current_params_tensor.size()[0],
                                           current_params_tensor.size()[1],
                                           dtype = torch.float32,
                                           device = device)
        param_history_tensor[0, :, :] = current_params_tensor

        assert len(smp) == len(initial)
        
        scale = torch.zeros(current_params_tensor.size())
        
        # creating a template (tolerance_template) for the per-parameter convergence
        #   tolerance; this it is model specific and accounts for comparitive 
        #   differences in sensitivity to change as well as in scale
        if current_params_tensor.size(1) == 11:
            tolerance_template = torch.tensor([1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                               1e-1, 1e-2, 1e-2, 1e-1, 1e-1,
                                               1e-1],
                                dtype = torch.float32,
                                device = device)
            
        elif current_params_tensor.size(1) == 6:
            tolerance_template = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2, 1e-1,
                                               1e-1],
                                dtype = torch.float32,
                                device = device)
            
        elif current_params_tensor.size(1) == 7:
            tolerance_template = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2, 1e-2,
                                               1e-1, 1e-1],
                                dtype = torch.float32,
                                device = device)
            
        tol_ = tolerance_template
        tolerance_tensor = torch.ones((current_params_tensor.size()[0],
                          current_params_tensor.size()[1])).to(device) * \
            tol_[None, ...].repeat([current_params_tensor.size()[0], 1])
            # tolerance_tensor - full tolerance tensor for the entire batch
            #                    dim (batch_size, num_params)
        
        mu_tensor = torch.zeros(smp.size()).to(device)
        mu_tensor_new = torch.zeros(smp.size()).to(device)
        
        jac_tensor = torch.zeros((smp.size()[0], smp.size()[1],
                           smp.size()[2], current_params_tensor.size()[1])).to(device)
        jac_tensor_new = torch.zeros((smp.size()[0], smp.size()[1],
                               smp.size()[2], current_params_tensor.size()[1])).to(device)

        
        active_mask = torch.ones(current_params_tensor.size()[0]).to(device).type(torch.bool)
        # active_mask - tracks fit re: what has yet to converge; dim (batch_size,)
        
        lambda_arr = torch.ones(current_params_tensor.size()[0]).to(device)*lambda_
        # lambda_arr - contains lambda_ for each fit in the batch
        
        delta_tensor = torch.ones(current_params_tensor.size()).to(device)
        # difference in parameters between iterations
        bool_tensor = torch.ones(current_params_tensor.size()).to(device).type(torch.bool)

        lambda_fac = 1
        
        # -- Main Optimization Loop -- #        
        i = 0 # track iterations
        flag_tolerance = 0 # 0 if there are there fits yet to converge

        while (i < iterations) and (flag_tolerance == 0):
            if current_params_tensor.size(1) == 7:
                pass
            else:
                if correct_offset:
                # ???: is this offset necessary??
                    current_params_tensor[active_mask, -1] = 0 #for offset == 0
                    
            mu_tensor[active_mask, :], jac_tensor[active_mask, :, :] = \
                self._fit_model.forward(current_params_tensor[active_mask, :],
                                        active_mask)

            merit_start = \
                torch.sum(abs(mu_tensor[active_mask, :] - smp[active_mask, :]),
                          dim = (-2, -1))
                
            temp_params[active_mask, :], scale[active_mask, :] = \
                lm_update(current_params_tensor[active_mask, :],
                          mu_tensor[active_mask, :],
                          jac_tensor[active_mask, :, :],
                          smp[active_mask, :],
                          lambda_arr[active_mask],
                          param_range_min_max,
                          scale[active_mask, :])
                
            mu_tensor_new[active_mask, :], jac_tensor_new[active_mask, :, :] = \
                self._fit_model.forward(temp_params[active_mask, :],
                                   active_mask)

            merit_nodiv = \
                torch.sum(abs(mu_tensor_new[active_mask, :] - smp[active_mask, :]),
                          dim = (-2, -1))
                
            temp_params[active_mask, :], scale[active_mask, :] = \
                lm_update(current_params_tensor[active_mask, :],
                          mu_tensor[active_mask, :],
                          jac_tensor[active_mask, :, :],
                          smp[active_mask, :],
                          lambda_arr[active_mask]/lambda_fac,
                          param_range_min_max,
                          scale[active_mask, :])

            mu_tensor_new[active_mask, :], jac_tensor_new[active_mask, :, :] = \
                self._fit_model.forward(temp_params[active_mask, :],
                                   active_mask)
                
            merit = torch.sum(abs(mu_tensor_new[active_mask, :] - smp[active_mask, :]),
                              dim = (-2, -1))
            
            lambda_arr[active_mask][torch.logical_and(merit_start < merit_nodiv,
                                                     merit_start < merit)] = \
                lambda_arr[active_mask][torch.logical_and(merit_start < merit_nodiv,
                                                         merit_start < merit)] \
                    * lambda_fac
            
            lambda_arr[active_mask][merit_start > merit_nodiv] = \
                lambda_arr[active_mask][merit_start > merit_nodiv] / lambda_fac
                
            current_params_tensor[active_mask, :], scale[active_mask, :] = \
                lm_update(current_params_tensor[active_mask, :],
                          mu_tensor[active_mask, :],
                          jac_tensor[active_mask, :, :],
                          smp[active_mask, :],
                          lambda_arr[active_mask] / lambda_fac,
                          param_range_min_max,
                          scale[active_mask, :])

            if (correct_offset and current_params_tensor.size(1) == 7):
                pass
            else:
                current_params_tensor[active_mask, -1] = 0  # for offset == 0
                
            param_history_tensor[i + 1, active_mask, :] = current_params_tensor[active_mask, :]
            # difference between parameters via param_history_tensor
            #   compare prior (i) to current (i+1), which works because of the 
            #   construction of the history tensor
            delta_tensor[active_mask, :] = torch.absolute(param_history_tensor[i, active_mask, :] - \
                                                  param_history_tensor[i+1, active_mask, :])

            bool_tensor[active_mask] = \
                (delta_tensor[active_mask, :] < tolerance_tensor[active_mask, :]).type(torch.bool)
                
            # bool_array_np = bool_tensor.detach().cpu().numpy()
            # delta_np = delta_tensor.detach().cpu().numpy()
            test = torch.sum(bool_tensor, dim = 1)
            active_mask = test != current_params_tensor.size()[1]

            if torch.sum(active_mask) == 0:
                flag_tolerance = 1
                
            i = i + 1
            
        loglik_tensor = torch.sum(smp * torch.log(mu_tensor / smp),
                           dim = (1, 2)) - torch.sum(mu_tensor - smp, dim = (1, 2))
        loglik_tensor[torch.isinf(loglik_tensor)] = 1e-20
        
        return current_params_tensor, loglik_tensor, param_history_tensor


# -- outside of class definitions -- #
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
    except ValueError:
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