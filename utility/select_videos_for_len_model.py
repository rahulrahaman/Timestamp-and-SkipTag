import numpy as np
import pickle
import glob
from collections import defaultdict
import os

splits = {1, 2, 3, 4}

ground_truth_dir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/"

for split in [1]:#splits:
    all_train_dataset = open("/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/splits/all_files.txt").read().split("\n")[0:-1]

    activity_with_vid_dict = defaultdict(list)
    for filename in all_train_dataset:
        video_id = filename.split(".txt")[0]
        main_act = video_id.split("_")[-1]

        activity_with_vid_dict[main_act].append(filename)


    amt_data = 4 # int(data_per * len(activity_with_vid_dict[activity])) + 1
    uniq_labels = []
    while len(np.unique(uniq_labels)) < 48:
        uniq_labels = []
        selected_vids = []
        total_data = 0.0 # data_per * len(all_train_dataset)
        for activity in activity_with_vid_dict.keys():
            vids = np.random.choice(activity_with_vid_dict[activity], size=amt_data)
            total_data += amt_data 
            selected_vids.extend(vids)
            temp_labels = []
            for vid in vids:
                labels = open(os.path.join(ground_truth_dir, vid)).read().split("\n")[0:-1]
                temp_labels.extend(np.unique(labels))
            uniq_labels.extend(np.unique(temp_labels))
        print("Number of uniq labels found = ", len(np.unique(uniq_labels)))


    semi_supervised_train_dataset = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/semi_supervised/train.amt{}vids.bundle".format(amt_data)
    with open(semi_supervised_train_dataset, "w") as wfp:
        wfp.write("\n".join(selected_vids))
