import pickle
import glob
import numpy as np
import sys
import numpy as np

# =================================
# Code to form the randomly chosen frames, this is different from single frames
# Example command: python utility/form_selected_frames_dict.py
#                  /mnt/ssd/all_users/dipika/ms_tcn/data/gtea/groundTruth/ 16 gtea 1
# =================================

groundTruth = sys.argv[1]
select_per_video = int(sys.argv[2])
dataset = sys.argv[3]
sample_rate = int(sys.argv[4])


def get_selected_labels(total_len, num_part):
    
    select_arr = np.linspace(0, total_len, num_part + 1).astype(int)
    uniq_l = []

    for i, ele in enumerate(select_arr[:-1]):
        select_one = np.random.randint(ele, select_arr[i+1], 1)[0]
        uniq_l.append(int(select_one.item()))
    
    return uniq_l
# get_selected_labels(ab)


def main():
    selectedFrames_dict = {}
    count = 0
    for file_n in glob.glob(groundTruth + "/*txt"):
        video_id = file_n.split("/")[-1].split(".txt")[0]
        data = open(file_n).read().split("\n")[0:-1]
        data = np.array(data)
        
        if sample_rate > 1:
            data = data[::sample_rate]
        
        total_len = len(data)
        selectedFrames = get_selected_labels(total_len, select_per_video)
        gd_labels = data[selectedFrames]
    
        select_frames_with_labels = []
        for select_idx, gd_idx in zip(selectedFrames, gd_labels):
            select_frames_with_labels.append((select_idx, gd_idx))

        if count % 10 == 0:
            print(select_frames_with_labels)
        count += 1

        selectedFrames_dict[file_n.split("/")[-1]] = select_frames_with_labels

    pickle.dump(selectedFrames_dict, open(f"data/{dataset}_random{select_per_video}frame_selection.pkl", "wb"))
        

if __name__ == '__main__':
    main()
