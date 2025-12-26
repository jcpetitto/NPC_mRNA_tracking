import os
import pickle
import json
import torch

from imaging_pipeline import ImagingPipeline
from utils.particle_detection import detect_spots_with_glrt
from utils.Neural_networks import Unet_pp_timeseries as Unet_pp
from tools.utility_functions import extract_cropped_images

local_config_path = '/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/config_local.json'

test_FoV_id = '0120'

test_mrna_path = f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/example_data/BMY823_7_25_23_aqsettings1_batchC/FoV_{test_FoV_id}/RNAgreen{test_FoV_id}.tiff'

test_crop_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/yeast_output/tracking_experiments/init_fit/ch1_crop_BMY823_BMY823_7_25_23_aqsettings1_batchC.json'

bg_model_path = '/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/example_data/trained_networks/model_wieghts_background_psf.pth'



with open(test_crop_path, 'r') as file:
    test_crop_dictionary = json.load(file)

test_pipeline = ImagingPipeline(local_config_path)
device = test_pipeline.get_device()
config = test_pipeline.retrieve_pipe_config("particle detection")
glrt_config = config['glrt_multichannel']

# test_mrna_img = tifffile.imread(test_mrna_path)
test_ne_label_crops = test_crop_dictionary['0120']

test_cropped_ne_imgs = extract_cropped_images(test_mrna_path, glrt_config['frame_range_particle'], test_ne_label_crops, mean_img = False, pad_to_size = None)




# ??? background model requires config, maybe background_model is instantiated in the function instead of being a parameter?
background_model = Unet_pp(glrt_config['number_channel']).to(device)
background_model.load_state_dict(torch.load(os.path.join(config['directories']['model root'], glrt_config['model_bg']), map_location=device)['model_state_dict'])
background_model.eval()

detect_spots_results = {}
for ne_label, mRNA_img in test_cropped_ne_imgs.items():
    spots = detect_spots_with_glrt(mRNA_img, background_model, device, config, ne_label)
    detect_spots_results.update({f'{ne_label}': spots})

    # with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/detect_spots_results_{ne_label}.pkl', 'wb') as file:
    #     pickle.dump(detect_spots_results, file)

# ??? How to visualize these results??
