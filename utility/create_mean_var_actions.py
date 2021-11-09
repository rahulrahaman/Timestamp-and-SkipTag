from collections import defaultdict
import pickle
import numpy as np
import sys

groundtruthDir = sys.argv[1]  # '/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/'
semi_supervised_5_per_file_name = sys.argv[2]  # "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/semi_supervised/train.split3_amt0.05.bundle"
outfile_dict = sys.argv[3]  # "data/breakfast_meanvar_actions.pkl"
sample_rate = int(sys.argv[4])  # 2 for 50salads, 1 for rest


def get_mean_var_actions(labels_arr):
    action_len_dict = defaultdict(list)
    
    prev_ele = None
    start = 0
    for i, ele in enumerate(labels_arr):
        if prev_ele is not None and prev_ele != ele:
            action_len_dict[prev_ele].append(i - start)
            start = i
        prev_ele = ele
    
    action_mean_var_dict = {}
    for ele in action_len_dict.keys():
        action_mean_var_dict[ele] = (np.mean(action_len_dict[ele]), np.std(action_len_dict[ele]))
    return action_mean_var_dict


labels_arr = []

all_files_names = open(semi_supervised_5_per_file_name).read().split("\n")[0:-1]
print("Number of files taken for calculating action mean = ", len(all_files_names))

for video_id_name in all_files_names:
    gd_labels = open(groundtruthDir + video_id_name).read().split("\n")[0:-1]
    gd_labels = np.array(gd_labels)
    if sample_rate > 1:
        gd_labels = gd_labels[::sample_rate]
    labels_arr.append(gd_labels)


labels_arr = np.concatenate(labels_arr)
mean_var_actions = get_mean_var_actions(labels_arr)

pickle.dump(mean_var_actions, open(outfile_dict, "wb"))
