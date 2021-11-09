import pickle
import glob
import numpy as np
import sys

def get_selected_labels(labels_arr):
    unique_ids = []
    
    prev_ele = None
    start = 0
    for i, ele in enumerate(labels_arr):
        if prev_ele is not None and prev_ele != ele:
            select_item = np.random.randint(start, i, 1)[0]
            unique_ids.append(select_item)
            start = i
        prev_ele = ele
    
    select_item = np.random.randint(start, len(labels_arr), 1)[0]
    unique_ids.append(select_item)
    start_point = np.random.choice([0, 1], 1)[0]

    gap_unique_ids = []
    for idx in range(start_point, len(unique_ids), 2):
        gap_unique_ids.append(unique_ids[idx])
    return gap_unique_ids, unique_ids


def main():
    selectedFrames_dict = {}
    count = 0
    annota_selected_frame = np.load(sys.argv[3], allow_pickle=True).item()
    for file_name in annota_selected_frame.keys():
        data = open(sys.argv[1] + "/" + file_name).read().split("\n")[0:-1]
        data = np.array(data)

        full_selected_frames = annota_selected_frame[file_name]
        gd_labels = data[full_selected_frames]

        select_frames_with_labels = []
        for select_idx, gd_idx in zip(full_selected_frames, gd_labels):
            select_frames_with_labels.append((select_idx, gd_idx))

        start_point = np.random.choice([0, 1], 1)[0]
        flag_first = False
        flag_last = False
        if start_point != 0:
            flag_first = True

        gap_unique_ids = []
        for idx in range(start_point, len(select_frames_with_labels), 2):
            gap_unique_ids.append(select_frames_with_labels[idx])

        if gap_unique_ids[-1][0] != select_frames_with_labels[-1][0]:
            flag_last = True

        if count % 20 == 0:
            print("original selected frames,", select_frames_with_labels)
            print("Reduced seleted frames = ", gap_unique_ids)
            print("Is First Missing ", flag_first)
            print("Is Last Missing ", flag_last)

        selectedFrames_dict[file_name] = (gap_unique_ids, flag_first, flag_last, select_frames_with_labels)
        count += 1 

    pickle.dump(selectedFrames_dict, open(sys.argv[2], "wb"))
 

if __name__ == '__main__':
    main()
