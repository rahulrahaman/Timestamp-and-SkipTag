import pickle
import glob
import numpy as np
import sys

# =================================
# Code to form the randomly chosen frames, this is different from single frames
# Example command: python utility/form_boundaryframes_dict.py
#                  /mnt/ssd/all_users/dipika/ms_tcn/data/gtea/groundTruth/ data/gtea_boundary_annotations.pkl 1
# =================================

groundTruth = sys.argv[1]
output_dump_file = sys.argv[2]
sample_rate = int(sys.argv[3])


def get_boundary(labels_arr):
    unique_ids = []
    
    prev_ele = None
    start = 0
    for i, ele in enumerate(labels_arr):
        if prev_ele is not None and prev_ele != ele:
            unique_ids.append(i - 1)
            start = i
        prev_ele = ele
    
    unique_ids.append(len(labels_arr) - 1)
    return unique_ids


boundary_dict = {}

for filename in glob.glob(groundTruth + "/*.txt"):
    video_id = filename.split("/")[-1]
    data = open(filename).read().split("\n")[0:-1]
    data = np.array(data)
    if sample_rate > 1:
        data = data[::sample_rate]
    boundary = get_boundary(data)
    boundary_dict[video_id] = boundary
    
pickle.dump(boundary_dict, open(output_dump_file, "wb"))
