import tensorflow as tf
###---Number-of-GPU
DATA_CSV='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Liver_Train_March22_2020_UPDTRBM.csv'
PATH_TO_SAVE_TFRECORDS='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Liver_tfrecords/Train_tfrecords2/'
NAME_OF_PATH_CSV='Liver_Train_128X128X128_patches'
PATH_TO_SAVE_MIDDLE_SLICE_PNG='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Liver_tfrecords/Train_tfrecords_mask/'
SAVING_PNG_OF_THE_PATCH='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Liver_tfrecords/Train_tfrecords_EXP2/'
COLORMAP='gist_stern'
ALPHA_VALUES_CT=1
ALPHA_VALUES_MASK=0.5
###--------------------tfrecords Generator Parameters
PATCH_PARAMS = {'n_examples': 1,
                 'example_size': [96, 128, 128],
                 'extract_examples': True}
NUM_OF_CTS_IN_SINGLE_TFRECORDS=1
