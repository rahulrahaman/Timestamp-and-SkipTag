import pickle
import numpy as np
import glob
import sys

groundTruthDir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/"
dumy_seg_dir = sys.argv[1]
correct = 0.0
total = 0.0

for video_id in glob.glob(groundTruthDir + "/*"):
    video_id = video_id.split("/")[-1]
    ground_labels = open(groundTruthDir + video_id).read().split("\n")[0:-1]
    ground_labels = np.array(ground_labels)

    dumy_labels = open(dumy_seg_dir + video_id).read().split("\n")[0:-1]
    dumy_labels = np.array(dumy_labels)

    assert len(dumy_labels) == len(ground_labels)

    correct += np.sum(ground_labels == dumy_labels)
    total += len(ground_labels)

print("Total correct pivots labels selected = ", correct * 100.0 / total)
