import pickle
import numpy as np

groundTruthDir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/"

selected_frame_dict = pickle.load(open("data/breakfast_len_assum_annotations.pkl", "rb"))
correct = 0.0
total = 0.0

for video_id in selected_frame_dict.keys():
    ground_labels = open(groundTruthDir + video_id).read().split("\n")[0:-1]
    ground_labels = np.array(ground_labels)

    selected_frames_index = [ele[0] for ele in selected_frame_dict[video_id]]
    selected_frames_labels = np.array([ele[1] for ele in selected_frame_dict[video_id]])

    ground_selected_labels = ground_labels[selected_frames_index]
    
    correct += np.sum(ground_selected_labels == selected_frames_labels)
    total += len(ground_selected_labels)

print("Total correct pivots labels selected = ", correct * 100.0 / total)
