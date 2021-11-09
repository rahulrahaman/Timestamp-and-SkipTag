import numpy as np
import pickle
import sys

# =================================
# Code to form the single frame dictionary from the mstcn single frame files
# Example command: python utility/convert_timestamp_annot.py data/gtea_annotation_all.npy
#                         /mnt/ssd/all_users/dipika/ms_tcn/data/gtea/groundTruth/ data/gtea_single_frame.pkl 1
# =================================

annota_selected_frame = np.load(sys.argv[1], allow_pickle=True).item()

groundtruth_dir = sys.argv[2]
dump_file_name = sys.argv[3]
sample_rate = int(sys.argv[4])

new_selected_frame_dict = {}

for filename in annota_selected_frame.keys():
    selected_frame_indices = annota_selected_frame[filename]
    
    gd_labels = np.array(open(groundtruth_dir + filename, "r").read().split("\n")[0:-1])
    if sample_rate > 1:
        gd_labels = gd_labels[::sample_rate]

    selected_frames_labels = gd_labels[selected_frame_indices].tolist()

    new_selected_frame_dict[filename] = [(ele1, ele2) for ele1, ele2 in zip(selected_frame_indices, selected_frames_labels)]

pickle.dump(new_selected_frame_dict, open(dump_file_name, "wb"))
