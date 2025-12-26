import torch
from typing import Optional
import copy 

from utils.psf_fit_utils import LM_MLE_with_iter, Gaussian2D_IandBg, Gaussian2D_Bg, gauss_psf_2D_I_Bg, gauss_psf_2D_Bg



#@torch.jit.script
def glrtfunction(smp_arr, batch_size:int, bounds, initial_arr, roi_size:int, sigma:float, tol, lambda_:float=1e-5, iterations:int=30, bg_constant:Optional[torch.Tensor]=None, use_pos:Optional[bool]=False, vector:Optional[bool]=False, GT=False):
    n_iterations = smp_arr.size(0) // batch_size + int(smp_arr.size(0) % batch_size > 0)

    loglik_bg_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    loglik_int_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    traces_bg_all = torch.zeros((smp_arr.size(0),iterations+1,1), device=smp_arr.device)
    traces_int_all = torch.zeros((smp_arr.size(0),iterations+1,2), device=smp_arr.device)

    modelIbg = Gaussian2D_IandBg(roi_size, sigma)
    modelbg  = Gaussian2D_Bg(roi_size, sigma)
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
                pos_in = torch.ones_like(initial_[:, [2, 3]])*roi_size/2
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
                mu_iandbg, _ = gauss_psf_2D_I_Bg(params_, roi_size, sigma, bg_constant_batch, pos_in)




            bg = initial_[:, 3]
            bg = bg[..., None]




            if vector:
                #mle = torch.compile(mle)
                test = 0  # select if single gpus
            else:
                mle_bg = torch.jit.script(mle_bg)


            # setup model and compute Likelhood for hypothesis Bg
            if vector == True and pos == None:
                pos_in = torch.ones_like(initial_[:, [2, 3]]) * roi_size / 2
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
                mu_bg, _ = gauss_psf_2D_Bg(params_bg_, roi_size, sigma, bg_constant_batch, pos)







            loglik_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_bgonly))] = loglik_bgonly
            loglik_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg))] = loglik_I_andbg
            traces_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_bgonly,[1,0,2])
            traces_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_iandbg,[1,0,2])

    ratio = 2 * (loglik_int_all - loglik_bg_all)
    #ratio[torch.isnan(ratio)] = 0
    return ratio, loglik_int_all, loglik_bg_all, mu_iandbg,mu_bg, traces_bg_all, traces_int_all
