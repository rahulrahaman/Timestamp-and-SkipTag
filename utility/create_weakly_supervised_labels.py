import numpy as np
import pickle

single_timestamp_indexes = np.load('data/breakfast_annotation_all.npy', allow_pickle=True).item()

breakfast_weakly_supervised_labels_dict = {}
groundtruthDir = '/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/'

for video_id_name in single_timestamp_indexes.keys():
    gd_labels = open(groundtruthDir + video_id_name).read().split("\n")[0:-1]
    gd_labels = np.array(gd_labels)

    selected_labels = gd_labels[single_timestamp_indexes[video_id_name]]

    breakfast_weakly_supervised_labels_dict[video_id_name] = selected_labels.tolist()

outfile_dict  = "data/breakfast_weaklysupervised_labels.pkl"

pickle.dump(breakfast_weakly_supervised_labels_dict, open(outfile_dict, "wb"))
