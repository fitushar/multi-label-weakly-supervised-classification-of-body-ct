import tensorflow as tf
###---Number-of-GPU
DATA_CSV='/Local/nobackup/Lung_CenterPatch_2mm_April20_P224/Lung_Train_March27_2020_UPDTRBM.csv'
PATH_TO_SAVE_TFRECORDS='/Local/nobackup/Lung_CenterPatch_2mm_April20_P224/Lung_tfrecords/Train_tfrecords/'
NAME_OF_PATH_CSV='Lung_Train_224_160x160_patches'
PATH_TO_SAVE_MIDDLE_SLICE_PNG='/Local/nobackup/Lung_CenterPatch_2mm_April20_P224/Lung_tfrecords/Train_tfrecords_mask/'
SAVING_PNG_OF_THE_PATCH='/Local/nobackup/Lung_CenterPatch_2mm_April20_P224/Lung_tfrecords/Train_tfrecords_png/'
COLORMAP='gist_stern'
ALPHA_VALUES_CT=1
ALPHA_VALUES_MASK=0.5

###--------------------tfrecords Generator Parameters
PATCH_PARAMS = {'n_examples': 1,
                 'example_size': [224, 160, 160],
                 'extract_examples': True}
NUM_OF_CTS_IN_SINGLE_TFRECORDS=1
