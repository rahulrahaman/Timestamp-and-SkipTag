import pickle
import numpy as np

groundTruthDir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/"
weakly_supervised_labels_dict = pickle.load(open("data/breakfast_weaklysupervised_labels.pkl", "rb"))

action_meanvar_dict = pickle.load(open("data/breakfast_meanvar_actions.pkl", "rb"))
output_dir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/length_segmentation_output/"
output_dump_pickle_file = "data/breakfast_len_assum_annotations.pkl"

selectedFrames_dict = {}

correct = 0.0
total = 0.0


def get_selected_labels(labels_arr):
    unique_ids = []
    
    prev_ele = None
    start = 0
    for i, ele in enumerate(labels_arr):
        if prev_ele is not None and prev_ele != ele:
            select_item = ((start + i) // 2, prev_ele) # np.random.randint(start, i, 1)[0]
            unique_ids.append(select_item)
            start = i
        prev_ele = ele
    
    select_item = ((start + len(labels_arr)) // 2, ele) # np.random.randint(start, len(labels_arr), 1)[0]
    unique_ids.append(select_item)
    return unique_ids


for video_id_file in weakly_supervised_labels_dict.keys():
    weakly_labels = weakly_supervised_labels_dict[video_id_file]

    length_arr = np.array([action_meanvar_dict[label][0] for label in weakly_labels])
    
    sum_length = np.sum(length_arr)

    length_arr_norm = length_arr / sum_length

    ground_labels = np.array(open(groundTruthDir + video_id_file).read().split("\n")[0:-1])

    total_vid_len = len(ground_labels)

    estimated_weakly_labels = []
    for i, len_est in enumerate(length_arr_norm):
        est_l = [weakly_labels[i]] * int(len_est * total_vid_len)
        estimated_weakly_labels.extend(est_l)

    total_estimated_weakly_len = len(estimated_weakly_labels)

    if total_estimated_weakly_len > total_vid_len:
        estimated_weakly_labels = estimated_weakly_labels[:total_vid_len]
    elif total_estimated_weakly_len < total_vid_len:
        estimated_weakly_labels.extend([estimated_weakly_labels[-1]] * (total_vid_len - total_estimated_weakly_len))

    assert(len(estimated_weakly_labels) == total_vid_len) 
    correct += np.sum(np.array(estimated_weakly_labels) == ground_labels)
    total += total_vid_len


    selectedFrames = get_selected_labels(estimated_weakly_labels)
    selectedFrames_dict[video_id_file] = selectedFrames

    estimated_labels_str = "\n".join(estimated_weakly_labels) + "\n"
    with open(output_dir + video_id_file, "w") as fp:
        fp.write(estimated_labels_str)

pickle.dump(selectedFrames_dict, open(output_dump_pickle_file, 'wb'))
print("Segmentation accuracy = ", correct * 100.0 / total)
    
