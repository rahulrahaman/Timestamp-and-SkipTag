import glob
import numpy as np
import pickle

twfinch_clus_output = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/results/twfinch_cluster/"
all_numpy_files = glob.glob(twfinch_clus_output + "*npy")
weakly_supervised_labels = pickle.load(open('data/breakfast_weaklysupervised_labels.pkl', "rb"))
gd_dir = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/groundTruth/"
twfinch_seg_out = "/mnt/ssd/all_users/dipika/ms_tcn/data/breakfast/results/twfinch_cluster_segments/"

def get_corrected_seg(segment):
    prev_seg = -1
    new_seg = []
    for seg_ele in segment:
        if seg_ele < prev_seg:
            seg_ele = prev_seg
        new_seg.append(seg_ele)
        prev_seg = seg_ele
    return np.array(new_seg)

for np_f in all_numpy_files:
    seg_o = np.load(np_f)
    video_id = np_f.split("/")[-1].split("_before_hungarian.npy")[0]
    weakly_labels = np.array(weakly_supervised_labels[video_id + ".txt"])

    clusts_in_seg = np.unique(seg_o)
   
    num_clus = len(weakly_labels)

    correct_seg = get_corrected_seg(seg_o)

    seg_out = weakly_labels[correct_seg].tolist()

    gd_seg = open(gd_dir + video_id + ".txt").read().split("\n")[0:-1]

    if len(gd_seg) != len(seg_out):
        print("Error")
        print(f"Diff in len {len(gd_seg) - len(seg_out)}")


    output_str = "\n".join(seg_out) + "\n"
    with open(twfinch_seg_out + video_id + ".txt", "w") as fp:
        fp.write(output_str)    

#    for i in correct_seg:
#        seg_out.append
'''
# Some verification codes
    if num_clus != len(np.unique(correct_seg)):
        print("Num of clusters from weakly supervised label ", num_clus)
        print("Unique seg output", np.unique(correct_seg))
        print("Segmentation out", correct_seg.tolist())

    for i in range(num_clus):
        positions_vec = np.where(correct_seg == i)[0]
        
        diff = positions_vec[1:] - positions_vec[:-1]

        if np.any(diff != 1):
            correct_seg(seg_o, )
            print("Not what we want")
            print(f"For video {video_id}, and weaakly labels {weakly_labels[i]} found many segments")
            print(seg_o)
            print(i)


'''

    
