"""
Created on Fri May  9 12:55:24 2025

@author: jctourtellotte
"""

# import external packages
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# import supporting pipeline classes (if any)
from utils.utility_functions import apply_to_dir_of_FoVs, list_update_or_replace, select_from_existing, pull_entry_by_value
from utils.responsivity import detector_responsivity_determ
from utils.img_registration import image_registration, compute_drift
from utils.npc_detect_initial import detect_npc

# from utils.npc_spline_refinement import NESplineRefiner
# TODO update ALL times a path is used to load something to os.join or what have you
class ImageProcessor:

    # --- Primary Image Processing Functions --- #
    def __init__(self, config_dict, device = torch.device('cpu'), threshold_regist = 0.5, save_figures=False, save_movies = False):
        self._cfg = config_dict
        # self._cfg['threshold_reg'] = 0.5
        # self._cfg['save_figures'] = False
        # self._cfg['save_movies'] = False

        self._current_device = device

        # setting up references to items in the FoV collection established by the config file
        self._FoV_collection_dict = self._create_FoV_collection_dict(self._cfg['FoV_collection_path'])

        # Bounds and initial guess for GLRT detector
        self._bounds_glrt = [
            [0, self._cfg['roisize'] - 1],  # x bounds
            [0, self._cfg['roisize']],      # y bounds
            [0, 1e9],                 # Photons bounds
            [0, 1e6]                  # Background bounds
        ]

        self._initial_guess = [
            self._cfg['roisize'] / 2,       # x initial guess
            self._cfg['roisize'] / 2,       # y initial guess
            0,                        # Photons initial guess
            60                        # Background initial guess
        ]


        self._reg_diff = []
        self._drift_correction = []
        self._ne_init_fit = []
        self._ne_cropped_imgs = []
        self._init_ne_bsplines = []
        self._ne_refined_fit = []
        

        print('Image Processor initiated')

    # representation given when an instance of the object is printed
    def __repr__(self):
        return f"---Image Processer Configuration---\nImageProcessor(configuration={self._cfg})"

    # Getters & Setters
    # FoV Dictionary
    def _create_FoV_collection_dict(self, directory_path):
        return apply_to_dir_of_FoVs(directory_path, self._create_FoV_dict_entry)

    def _set_FoV_collection_dict(self, dictionary_list):
        self._FoV_collection_dict = dictionary_list

    def _get_FoV_collection_dict(self):
        return self._FoV_collection_dict

    # !!!: FIX NAMING: NOT cell BUT FoV  (will require major overhaul to file structure as well)

    def _create_FoV_dict_entry(self, fn_params = {'FoV_id': None}):
        digits = str(fn_params['FoV_id'])
        prefix = self._cfg['FoV_prefix']
        result_folder = self._cfg['resultsdir']
        path_dict_entry = {'FoV_id': f'{digits}',
                           'FoV_collection_path': self._cfg['FoV_collection_path'],
                           'resultsdir': f'/{prefix}{digits}/{result_folder}',
                           'imgs': {
                               'fn_reg_npc1': f'{prefix}{digits}/BF1red{digits}.tiff',
                               'fn_reg_rnp1': f'{prefix}{digits}/BF1green{digits}.tiff',
                               'fn_reg_npc2': f'{prefix}{digits}/BF2red{digits}.tiff',
                               'fn_reg_rnp2': f'{prefix}{digits}/BF2green{digits}.tiff',
                               'fn_track_rnp': f'{prefix}{digits}/RNAgreen{digits}.tiff',
                               'fn_track_npc': f'{prefix}{digits}/NEred{digits}.tiff'
                         }
                     }

        return path_dict_entry
    
    # ???: Want (maybe) - method to load externally created dictionary?

    # Detector Responsivity Determination
    def _set_drd_stats(self, cal_stats):
        self._drd_stats = cal_stats

    def _get_drd_stats(self):
        return self._drd_stats
    
    def _set_drd_meanvar(self, meanvar):
        self._drd_meanvar = meanvar

    def _get_drd_meanvar(self):
        return self._drd_meanvar

    # Registration
    def _set_reg_diff(self, new_reg_diff_list, add_to_existing = True):
        self._reg_diff = list_update_or_replace(self._get_reg_diff(),
                                                         new_reg_diff_list,
                                                         add_to_existing)

    def _get_reg_diff(self):
        return self._reg_diff

    # Drift Correction
    def _set_drift_correction(self, new_drift_list, add_to_existing = True):
        self._drift_correction = list_update_or_replace(self._get_drift_correction(),
                                                         new_drift_list,
                                                         add_to_existing)

    def _get_drift_correction(self):
        return self._drift_correction

    # Initial NPC Fitting

    def _set_ne_init_fit(self, new_npc_init_list, add_to_existing = True):
        self._ne_init_fit = list_update_or_replace(self._get_ne_init_fit(),
                                                         new_npc_init_list,
                                                         add_to_existing)

    def _get_ne_init_fit(self):
        return self._ne_init_fit
    
    def _set_ne_cropped_imgs(self, new_ne_cropped_img_list, add_to_existing = True):
        self._ne_cropped_imgs = list_update_or_replace(self._get_ne_cropped_imgs(),
                                                         new_ne_cropped_img_list,
                                                         add_to_existing)
    def _get_ne_cropped_imgs(self):
        return self._ne_cropped_imgs
    
    def _set_init_ne_bsplines(self, new_init_bsplines_list, add_to_existing = True):
            self._init_ne_bsplines = list_update_or_replace(self._get_init_ne_bsplines(),
                                                            new_init_bsplines_list,
                                                            add_to_existing)
    def _get_init_ne_bsplines(self):
        return self._init_ne_bsplines

    # Refined NPC Fitting

    def _set_ne_refined_fit(self, new_npc_refined_list, add_to_existing = True):
        self._ne_refined_fit = list_update_or_replace(self._get_ne_refined_fit(),
                                                         new_npc_refined_list,
                                                         add_to_existing)

    def _get_ne_refined_fit(self):
        return self._ne_refined_fit

    # --- Pipes --- #

    def determine_responsivity(self):
        # Detector Responsivity Determination
        #   uses bright (gain) and dark (offset) image pair
        #   return calibration statistics as well as mean/variance
            drd_stats, meanvar_plt_stats = \
                detector_responsivity_determ(self._cfg['bright'],self._cfg['dark'])
                
            self._set_drd_stats(drd_stats)
            self._set_drd_meanvar(meanvar_plt_stats)


    # ???: registration is going to happen AFTER the bounding boxes re: NE are determined
    
    # sets initial fit in motion and directs the results to the appropriate class variables
    def run_init_ne_detection(self, FoV_list=[], add_to_existing = True):
        # if FoV_list is non-empty (ie. contains 1+ FoV_id strings) then this runs for that subset of FoV_ids, otherwise, this step is run for all FoV loaded into existing FoV collection for this ImageProcessor instance
        print("--- Detecting Nuclear Envelope ----")
        if len(FoV_list) != 0:
            FoV_for_npc_detection = \
                select_from_existing(self._get_FoV_collection_dict(), FoV_list)
        else:
            FoV_for_npc_detection = self._get_FoV_collection_dict()
        
        # 1. Run the initial fit (masks -> initial splines)
        ne_init_fit_results, ne_img_crop_results, ne_init_bspline_results = \
            self._run_ne_init_fit(FoV_for_npc_detection)
        
        self._set_ne_init_fit( ne_init_fit_results ) # store initial fit
        # ???: Have report generation available for this data?
        self._set_ne_cropped_imgs( ne_img_crop_results )

        self._set_init_ne_bsplines( ne_init_bspline_results )

    # orchestrates the steps for initial fitting for each entry in an FoV_dict
    def _run_ne_init_fit(self, FoV_dict):
        frame_range = self._cfg['ne_fit']['frame_range']
        ne_trained_model = self._cfg['model_NE']
        current_device = self._current_device
        masking_threshold = self._cfg['ne_fit']['masking_threshold']
        bbox_dim = self._cfg['ne_fit']['bbox_dim']
        use_merged_clusters = self._cfg['ne_fit']['use_merged_clusters']
        max_merge_dist = self._cfg['ne_fit']['max_merge_dist']
        plot_test_imgs = self._cfg['ne_fit']['plot_test_imgs']
        

        ne_init_fit_list = [] # list to store initial fit dicts
        ne_img_crop_list = [] # list to store cropped images coordinates based on initial fit
        ne_bspline_list = [] # list to store initial bspline objects
        
        # run initial detection for each FoV
        for entry in FoV_dict:
            FoV_id = entry['FoV_id']
            track_path = os.path.join(entry['FoV_collection_path'], entry['imgs']['fn_track_npc'])
            if not os.path.exists(track_path):
                print(f" --> SKIPPING: No npc channel track found for {FoV_id}.") 
                continue
            else:
                init_ne_fit, ne_img_crop, ne_bsplines = \
                    detect_npc(
                        img_track_path = track_path,
                        frame_range = frame_range,
                        NE_model = ne_trained_model,
                        device = current_device,
                        FoV_id = FoV_id,
                        masking_threshold = masking_threshold,
                        bbox_dim = bbox_dim,
                        use_merged_clusters = use_merged_clusters,
                        max_merge_dist = max_merge_dist,
                        plot_test_imgs = plot_test_imgs
                        )
                #TODO: catch if none detected
                if (len(init_ne_fit) == 0):
                    print(f" --> NOTE: No NE detected for {FoV_id}.") 
                    continue
                else:
                    ne_init_fit_list.append({'FoV_id': entry['FoV_id'],
                                            'initial_fit': init_ne_fit
                                            })
                    ne_img_crop_list.append({'FoV_id': entry['FoV_id'],
                                            'cropped_img':ne_img_crop
                                            })
                    ne_bspline_list.append({'FoV_id': entry['FoV_id'],
                                            'bsplines': ne_bsplines
                                            })

        return ne_init_fit_list, ne_img_crop_list, ne_bspline_list


    # def run_ne_refinement(self, FoV_list=[], add_to_existing = True):
    #     # 2. Refine the initial fit (intensity profiles -> refined splines)
    #     if len(FoV_list) != 0:
    #         FoV_for_npc_refinement = \
    #             select_from_existing(self._get_FoV_collection_dict(), FoV_list)
    #     else:
    #         FoV_for_npc_refinement = self._get_FoV_collection_dict()

    #     ne_refined_fit_results = \
    #         self._run_ne_refine_fit(
    #             FoV_dict = FoV_for_npc_refinement,
    #             ne_initial_fit_dict = self._get_ne_init_fit(),
    #             ne_bspline_dict = self._get_init_ne_bsplines(),
    #             ne_crop_box_dict = self._get_ne_cropped_imgs()
    #             )
        
    #     self._set_ne_refined_fit( ne_refined_fit_results ) # store refined fit
    #     # TODO: Have report generation available for this data?




#     def _run_ne_refine_fit(self, FoV_dict, ne_initial_fit_dict, ne_bspline_dict, ne_crop_box_dict):

#         ne_refine_bsplines_list = []

#         for entry in FoV_dict:
#             FoV_id = entry['FoV_id']
#             img_path = str(entry['FoV_collection_path'] + entry['imgs']['fn_track_npc'])
#             crop_box_entries = pull_entry_by_value(FoV_id, ne_crop_box_dict, 'FoV_id')
#             init_fit_bsplines = pull_entry_by_value(FoV_id, ne_bspline_dict, 'FoV_id')
            
# # TODO: add back registration to process using the crop box values BUT not here, after refining the fit
# #            reg_entries = pull_entry_by_value(FoV_id, self._get_reg_diff(), 'FoV_id')

#             current_FoV_spline_refiner = \
#                 NESplineRefiner(
#                     img_path,
#                     FoV_ne_crop_box_entry = crop_box_entries,
#                     config_dict = self._cfg,
#                     device = self._current_device
#                     )
#             ne_refined_bsplines = current_FoV_spline_refiner.refine_initial_bsplines(initial_bsplines = init_fit_bsplines, testing_mode = True)

# # TODO: catch if none detected
#             ne_refine_bsplines_list.append({'FoV_id': FoV_id, 'refined_fit': ne_refined_bsplines})

#         return ne_refine_bsplines_list




    # ???: Add a method for adding FoV dictionary entries
    
    
    # image registration
    # want to register the difference between a set of red frames and set of green frames
    # per ne label per FoV <- what we need for translating individual NE
    # AND/OR
    # either the whole FoV (for comparison)

    def register_images(self, FoV_list=[], add_to_existing = True, detailed_output = False):
        # registers the images in a directory
        # apply image_registration to every member of the _FoV_collection_dict
        #   (default)
        # OR
        # FoV_list - use to register a subset of the existing FoV's in the
        #               collection dictionary
        # add_to_exisiting - can either add FoV registration to the existing
        #                       list (default) OR overwrite the list

        if len(FoV_list) != 0:
            FoV_to_register = select_from_existing(self._get_FoV_collection_dict(), FoV_list)
            FoV_crop_dimensions = select_from_existing(self._get_ne_cropped_imgs(), FoV_list)
        else:
            FoV_to_register = self._get_FoV_collection_dict()
            FoV_crop_dimensions = self._get_ne_cropped_imgs()

        reg_diff_list = []

        for FoV_dict_entry in FoV_to_register:
            current_fov_id = FoV_dict_entry['FoV_id']

            # Find the corresponding dictionary in FoV_crop_dimensions
            FoV_crop_dim_entry = next( (item for item in FoV_crop_dimensions \
                                        if item["FoV_id"] == current_fov_id),
                                      None
            )
            # returns dictionary entry, so append will add to the list
            #   (rather than using expand)
            current_FoV_reg, current_ne_labels_reg = \
                image_registration(
                    FoV_dict_entry,
                    FoV_crop_dim_entry,
                    self._cfg['registration']['frame_range'],
                    self._cfg['registration']['frames_per_average'],
                    self._cfg['registration']['upsample_factor'],
                    self._cfg['registration']['upscale_factor'],
                    detailed_output
                    )
            
            reg_diff_list.append(
                {
                    'FoV_id': current_fov_id,
                    'FoV_reg_data': current_FoV_reg,
                    'ne_label_registration': current_ne_labels_reg
                }
                )

        self._set_reg_diff(reg_diff_list, add_to_existing)
    #???: !
    #???: registration diff -> throw out if less than threshold value
    #???: ! 
    
    def run_drift_correction(self, FoV_list=[], add_to_existing = True):
        # registration entry has to exist for all FoV in list to correct for drift
        print("--- Correcting Drift ----")
        # FoV_to_correct = select_from_existing(self._get_reg_diff())
        # [dict_item['FoV_id'] for dict_item in self.get_FoV_collection_dict() \
        #              if FoV_to_correct['FoV_id'] in FoV_to_correct]
        if len(FoV_list) != 0:
            FoV_to_correct = select_from_existing(self._get_FoV_collection_dict(), FoV_list)
        else:
            FoV_to_correct = self._get_FoV_collection_dict()

        drift_correct_list = []

        for FoV_dict_entry in FoV_to_correct:
            # returns dictionary entry, so append will add to the list
            #   (rather than using expand)
            drift_correct_list.append(compute_drift(FoV_dict_entry, self._cfg['drift_bins']))

        self._set_drift_correction(drift_correct_list, add_to_existing)

    def estimate_precision(self):
        print("--- Estimating Precision for Graphing ---")
        # !!!: Create this function




    # ???: run this
    def check_files(self, path_dict, dircheck = False):
        err = False
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_reg_npc1']):
            err = True
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_reg_rnp1']):
            err = True
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_reg_npc2']):
            err = True
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_reg_rnp2']):
            err = True
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_track_rnp']):
            err = True
        if not os.path.exists(path_dict['path'] + path_dict['img']['fn_track_npc']):
            err = True
        if dircheck==True:
            if len(os.listdir(path_dict['resultsdir'])) != 0:
                err = True
        return err