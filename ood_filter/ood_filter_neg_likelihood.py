import os
import json
import torch
import torchvision.models as models
# from fastai.vision import *
# from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import logging
import sys
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
from data import Transform_Normalization
torch.set_printoptions(threshold=10_000)
from engine.feature_extractor import MyFeatureExtractor



class OOD_filter_neg_likelihood:
    def __init__(self, timm_model=None, timm_pretrained=True, num_classes=1,
        sam_model=None):
        """ OOD filter constructor
        Params
        :timm_model (str) -> model name from timm library
        :timm_pretrained (bool) -> whether to load a pretrained model or not
        :num_classes (int) -> number of classes in the ground truth
        """
        self.num_classes = num_classes
        # create a model for feature extraction
        feature_extractor = MyFeatureExtractor(
            timm_model, timm_pretrained, num_classes
        ).to('cuda')
        self.feature_extractor = feature_extractor
        self.sam_model = sam_model

        # create the default transformation
        if feature_extractor.is_transformer:
            trans_norm = Transform_Normalization(
                    size=feature_extractor.input_size, 
                    force_resize=True, keep_aspect_ratio=False
                )
        else:
            trans_norm = Transform_Normalization(
                    size=33, force_resize=False, keep_aspect_ratio=True
                )
        self.trans_norm = trans_norm

    def run_filter(self, 
        labeled_loader, 
        unlabeled_loader,  
        dir_filtered_root = None, 
        ood_thresh = 0.0, 
        ood_hist_bins = 15):
        """
        Params
        :labeled_loader: path for the first data bunch, labeled data
        :unlabeled_loader: unlabeled data
        :dir_filtered_root: path for the filtered data to be stored
        :ood_thresh: ood threshold to apply
        :path_reports_ood: path for the ood filtering reports

        Return
        :NULL -> the output of this method is a json file save in a directory.
        """      
        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []
        for batch in labeled_loader:
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from bounding boxes)
                imgs_f, labels_f = self.get_foreground(batch, idx, self.trans_norm)

                labeled_imgs += imgs_f
                labeled_labels += labels_f
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]
        # get all features maps using: the extractor + the imgs
        feature_maps_list = self.get_all_features(labeled_imgs)

        # 2. Calculating the histograms from the labeled data
        # go over each dimension and get values from the whole dataset
        (
            histograms_all_features_labeled,    # e.g. 512 x 15
            buckets_all_features_labeled        # e.g. 512 x 15
        ) = self.calculate_hist_dataset(feature_maps_list, ood_hist_bins)

        # go through each batch unlabeled
        likelihoods_final_all_obs = 0
        # epsilon to avoid Inf results in logarithm
        eps = 0.0000000001
        # keep track of the img id for every sample created by sam
        imgs_ids = []
        imgs_box_coords = []
        imgs_scores = []

        # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
        for (current_batch_num_unlabeled, batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):
            unlabeled_imgs = []

            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from sam)
                imgs_s, box_coords, scores = self.sam_model.get_unlabeled_sam(
                    batch, idx, self.trans_norm
                )
                unlabeled_imgs += imgs_s

                # accumulate SAM info (inferences)
                imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
                imgs_box_coords += box_coords
                imgs_scores += scores

            # get all features maps using: the extractor + the imgs
            featuremaps_list = self.get_all_features(unlabeled_imgs)
            featuremaps = torch.stack(featuremaps_list) # e.g. [387 x 512]

            # init buffer with dims
            num_obs_unlabeled_batch = featuremaps.shape[0]
            likelihoods_all_obs_all_dims = torch.zeros(
                num_obs_unlabeled_batch, self.feature_extractor.features_size
            )
            # go  through each dimension, and calculate the likelihood for the whole unlabeled dataset
            for dimension in range(self.feature_extractor.features_size):
                # calculate the histogram for the given feature, in the labeled dataset
                hist_dim_obs_batch_labeled = histograms_all_features_labeled[dimension, :] #e.g. [512 x 15] -> [15]
                bucks_dim_obs_batch_labeled = buckets_all_features_labeled[dimension, :]   #e.g. [512 x 15] -> [15]
                # take only the values of the current feature for all the observations
                vals_feature_all_obs_unlabeled = featuremaps[:, dimension] # e.g. [387 x 512] -> [387]
                # fetch the bucket indices for all the observations, for the current feature
                min_buckets_all_obs_unlabeled = self.find_closest_bucket_all_obs( # e.g. [387]
                    vals_feature_all_obs_unlabeled, bucks_dim_obs_batch_labeled
                )

                # evaluate likelihood for the specific
                likelihoods_all_obs_dim_unlabeled = self.get_prob_values_all_obs( # e.g. [387]
                    min_buckets_all_obs_unlabeled, hist_dim_obs_batch_labeled
                )
                # squeeze to eliminate an useless dimension
                likelihoods_all_obs_all_dims[:, dimension] = likelihoods_all_obs_dim_unlabeled.squeeze() # e.g. [387 x 512]

            # calculate the log of the sum of the likelihoods for all the dimensions, obtaining a score per observation
            #THE LOWER THE BETTER
            likelihoods_all_obs_batch = -1 * torch.sum(torch.log(likelihoods_all_obs_all_dims + eps), 1) # e.g. [387]
            # store the likelihood for all the observations
            if (current_batch_num_unlabeled == 0):
                likelihoods_final_all_obs = likelihoods_all_obs_batch
            else:
                likelihoods_final_all_obs = torch.cat((likelihoods_final_all_obs, likelihoods_all_obs_batch), 0)

        # once we got all the batches of the unlabeled data...
        #----------------------------------------
        num_bins = 30
         # calculate the histogram of the likelihoods
        (histogram_likelihoods, buckets_likelihoods) = np.histogram(
            likelihoods_final_all_obs.numpy(), bins=num_bins, range=None, weights=None, density=None
        )
        #----------------------------------------

        # store per file scores
        scores = []
        for j in range(0, likelihoods_final_all_obs.shape[0]):
            scores += [likelihoods_final_all_obs[j].item()]
        
        thresh = self.get_threshold(scores, ood_thresh)
        num_selected = 0
        # final mask
        selected_samples = [0] * len(scores)
        for i in range(0, len(scores)):
            if(scores[i] <= thresh):
                num_selected += 1
                selected_samples[i] =  1

        # Save boxes that are in the mask
        results = []
        for index,val in enumerate(selected_samples):
            if val == 1:
                image_result = {
                    'image_id': imgs_ids[index],
                    'category_id': 1, # fix this
                    'score': imgs_scores[index],
                    'bbox': imgs_box_coords[index],
                }
                results.append(image_result)

        if len(results) > 0:
            # write output
            results_file = f"{dir_filtered_root}/bbox_results.json"
            if os.path.isfile(results_file):
                os.remove(results_file)
            json.dump(results, open(results_file, 'w'), indent=4)


    def get_prob_values_all_obs(self, min_buckets_all_obs, histogram_norm):
        """ Evaluate the histogram values according to the buckets mapped previously
        Params
        :min_buckets_all_obs: selected buckets according to the feature values. Eg. [404]
        :histogram_norm: normalized histogram
        
        Return
        :the likelihood values, according to the histogram evaluated
        """
        # put in a matrix of one column
        min_buckets_all_obs = min_buckets_all_obs.unsqueeze(dim=0).transpose(0, 1)
        # repeat the histograms to perform substraction
        repeated_histograms = histogram_norm.repeat(1, min_buckets_all_obs.shape[0]) # e.g. [1 x 6060]
        repeated_histograms = repeated_histograms.view(-1, histogram_norm.shape[0]) # e.g. [1 x 6060] -> [404 x 15]
        # evaluate likelihood for all observations
        likelihoods_all_obs = repeated_histograms.gather(1, min_buckets_all_obs) # e.g. [404 x 1]
        return likelihoods_all_obs
            
    def find_closest_bucket_all_obs(self, vals_feature_all_obs, buckets):
        """ Finds the closest bucket position, according to a set of values (from features) received
        Params
        :vals_feature_all_obs (tensor) -> values of features received, to map to the buckets. E.g. [387]
        :buckets (tensor) -> buckets of the previously calculated histogram. E.g. [15]
        
        Return 
        :the list of bucket numbers closest to the buckets received
        """
        # create repeated map to do a matrix substraction, unsqueezeing and transposing the feature values for all the observations
        vals_feature_all_obs = vals_feature_all_obs.unsqueeze(dim=0).transpose(0, 1) # e.g. [387] -> [387 x 1]
        # rep mat
        repeated_vals_dim_obs = vals_feature_all_obs.repeat(1, buckets.shape[0]) # e.g. [387 x 1] -> [387 x 15]
        repeated_vals_dim_obs = repeated_vals_dim_obs.view(-1, buckets.shape[0]) # e.g. [387 x 1] -> [387 x 15]
        # do substraction
        substracted_all_obs = torch.abs(repeated_vals_dim_obs - buckets)
        # find the closest bin per observation (one observation per row)
        min_buckets_all_obs = torch.argmin(substracted_all_obs, 1)
        return min_buckets_all_obs


    def calculate_hist_dataset(self, feature_maps_list, num_bins=15, plot=False):
        """ Calculate feature histogram for the dataset
        Params
        :feature_maps (List<tensor>) -> all feature maps in the labeled dataset.
        :num_bins (int) -> num of bins in the histogram.
        :plot (bool) -> create plot or not.
        Return 
        :histogram
        """
        dimensions = self.feature_extractor.features_size
        histograms_all_features_labeled = torch.zeros((dimensions, num_bins))
        buckets_all_features_labeled = torch.zeros((dimensions, num_bins))

        feature_maps = torch.stack(feature_maps_list)
        for dimension in range(0, dimensions):
            # get data just from one dimension
            dim_data = feature_maps[:, dimension].numpy()
            
            # calculate the histograms
            (hist1, bucks1) = np.histogram(
                dim_data, bins=num_bins, 
                range=None, weights=None, density=False
            )
            # manual normalization, np doesnt work
            hist1 = hist1 / hist1.sum()
            # instead of bin edges, get bin mean
            bucks1 = np.convolve(bucks1, [0.5, 0.5], mode='valid')
            hist1 = torch.tensor(np.array(hist1))
            bucks1 = torch.tensor(bucks1)
            # accumulate
            histograms_all_features_labeled[dimension, :] = hist1
            buckets_all_features_labeled[dimension, :] = bucks1
        return (histograms_all_features_labeled, buckets_all_features_labeled) 
        

    def get_threshold(self, scores, percent_to_filter):
        """ Get the threshold according to the list of observations and the percent of data to filter
        Params
        :percent_to_filter (float) -> value from 0 to 1
        Return
        :the threshold
        """
        new_scores_no_validation = scores.copy()
        #percent_to_filter is from  0 to 1
        new_scores_no_validation.sort()
        num_to_filter = int(percent_to_filter * len(new_scores_no_validation))
        threshold = new_scores_no_validation[num_to_filter]
        return threshold
        
    def get_all_features(self, images):
        """Extract feature vectors from the images.
        Params
        :images (List<tensor>) images to be used to extract features
        """
        features = []
        # get feature maps from the images
        with torch.no_grad():
            for img in images:
                t_temp = self.feature_extractor(img.unsqueeze(dim=0).to('cuda'))
                features.append(t_temp.squeeze().cpu())
        return features

    def get_foreground(self, batch, idx, transform):
        """ From a batch and its index get samples 
        Params
        :batch (<tensor, >)
        """
        imgs = []
        labels = []
        # batch[0] has the images    
        image = batch[0][idx].cpu().numpy().transpose(1,2,0)
        img_pil = Image.fromarray(image)

        # batch[1] has the metadata and bboxes
        # tensor dim where the boxes are, is: [100x4]
        # where no box is present, the row is [-1,-1,-1,-1]. So, get all boxes and classes as:
        bbox_indx = (batch[1]['bbox'][idx].sum(axis=1)>0).nonzero(as_tuple=True)[0].cpu()
        boxes = batch[1]['bbox'][idx][bbox_indx].cpu()
        classes = batch[1]['cls'][idx][bbox_indx].cpu() 

        # ITERATE: BBOX
        for idx_bbox, bbox in enumerate(boxes):
            #FOR SOME REASON, THE COORDINATES COME:
            # [y1,x1,y2,x2] and need to be translated to: [x1,y1,x2,y2]
            bbox = bbox[[1,0,3,2]]
            bbox = bbox.numpy()

            # get img
            crop = img_pil.crop(bbox)  
            new_sample = transform.preprocess(crop)
            labels.append(classes[idx_bbox].item())
            imgs.append(new_sample)
        return imgs, labels


        
