import torch
import pandas as pd
import ast
import numpy as np
import h5py
from torchvision import transforms
import os
from PIL import Image
from collections import defaultdict
from itertools import chain as chain
import random


def collate_fn_override(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    batch_input, length_of_sequences, batch_target, video_id_arr = zip(*data)
    
    time_sequence_length = max(length_of_sequences)
    batch_input_tensor = torch.zeros((len(batch_input), np.shape(batch_input[0])[1], time_sequence_length),
                                     dtype=torch.float32)
    batch_target_tensor = torch.ones(len(batch_input), time_sequence_length, dtype=torch.long) * (-100)
    
    mask = torch.zeros(len(batch_input), time_sequence_length, dtype=torch.int)

    for i in range(len(batch_input)):
        # print("BatchTarget: {}, length_of_sequences: {}".format(np.shape(batch_target[i])[0], length_of_sequences[i]))
        assert np.shape(batch_target[i])[0] == np.shape(batch_input[i])[0]
#         print("BatchInput: {}, length_of_sequences: {}".format(np.shape(batch_input[i])[0], length_of_sequences[i]))      
#         assert np.shape(batch_input[i])[0] == length_of_sequences[i]
              
        batch_input_tensor[i, :, :np.shape(batch_input[i])[0]] = torch.from_numpy(batch_input[i].T)
        batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
        mask[i, :np.shape(batch_target[i])[0]] = torch.ones(np.shape(batch_target[i])[0])

    return batch_input_tensor, torch.tensor(length_of_sequences, dtype=torch.int), batch_target_tensor, mask, video_id_arr# , torch.stack(labels_present_arr)


class Breakfast(torch.utils.data.Dataset):
    def __init__(self, args, fold, fold_file_name, sample_rate=1):
        self.fold = fold
        self.args = args
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.base_dir_name = args.features_file_name
        self.sample_rate = sample_rate
        self.frames_format = "{}/{:06d}.jpg"
        self.ground_truth_files_dir = args.ground_truth_files_dir
        df=pd.read_csv(args.label_id_csv)
        self.action_id_to_name = {}
        self.action_name_to_id = {}
        for i, ele in df.iterrows():
            self.action_id_to_name[ele.label_id] = ele.label_name
            self.action_name_to_id[ele.label_name] = ele.label_id

        self.data = self.make_data_set(fold_file_name)

    def make_data_set(self, fold_file_name):
        data = open(fold_file_name).read().split("\n")[0:-1]
        data_arr = []
        num_video_not_found = 0
        
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            if not os.path.exists(os.path.join(self.base_dir_name, video_id + ".npy")):
                print("Not found video with id", os.path.join(self.base_dir_name, video_id + ".npy"))
                num_video_not_found += 1
                continue

            ele_dict = {'video_id': video_id}
            
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")

            with open(filename, 'r') as f:
                recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
                f.close()
                
            recog_content = [self.action_name_to_id[e] for e in recog_content]
            recog_content = np.array(recog_content, dtype=int)
            
            if self.sample_rate > 1:
                recog_content = recog_content[::self.sample_rate]
            
            ele_dict["labels"] = recog_content
            data_arr.append(ele_dict)

        print("Number of videos logged in {} fold is {}".format(self.fold, len(data_arr)))
        print("Number of videos not found in {} fold is {}".format(self.fold, num_video_not_found))
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        ele_dict = self.data[index]
        
        image_path = os.path.join(self.base_dir_name, ele_dict['video_id'] + ".npy")
        elements = np.load(image_path).T
        
        if self.sample_rate > 1:
            elements = elements[::self.sample_rate]

        return elements, elements.shape[0], ele_dict["labels"], ele_dict['video_id']
    
    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


def collate_fn_override_wtd(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    batch_input, length_of_sequences, batch_target, video_id_arr, weights = zip(*data)
    
    time_sequence_length = max(length_of_sequences)
    batch_input_tensor = torch.zeros((len(batch_input), np.shape(batch_input[0])[1], time_sequence_length),
                                     dtype=torch.float32) 
    posterior_weights = torch.zeros((len(batch_input), time_sequence_length, weights[0].shape[1]),
                                     dtype=torch.float32)
    batch_target_tensor = torch.ones(len(batch_input), time_sequence_length, dtype=torch.long) * (-100)
    
    mask = torch.zeros(len(batch_input), time_sequence_length, dtype=torch.int)

    for i in range(len(batch_input)):
        assert np.shape(batch_target[i])[0] == np.shape(batch_input[i])[0]
        batch_input_tensor[i, :, :np.shape(batch_input[i])[0]] = torch.from_numpy(batch_input[i].T)
        posterior_weights[i, :np.shape(batch_input[i])[0], :] = weights[i]
        batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
        mask[i, :np.shape(batch_target[i])[0]] = torch.ones(np.shape(batch_target[i])[0])

    return batch_input_tensor, torch.tensor(length_of_sequences, dtype=torch.int), batch_target_tensor, mask, video_id_arr, posterior_weights


class BreakfastWithWeights(Breakfast):
    def __init__(self, args, fold, fold_file_name, sample_rate=1):
        super(BreakfastWithWeights, self).__init__(args, fold, fold_file_name, sample_rate=sample_rate)
        self.weight_folder = os.path.join(self.args.output_dir, 'posterior_weights')
        
    def __getitem__(self, idx):
        ret = self.getitem(idx)
        vidid = ret[-1]
        weight_fname = os.path.join(self.weight_folder, f'{vidid}.wt')
        if os.path.exists(weight_fname):
            weights = torch.load(weight_fname)
        else:
            weights = torch.zeros((ret[1], self.args.num_class))
        return ret[0], ret[1], ret[2], ret[3], weights
