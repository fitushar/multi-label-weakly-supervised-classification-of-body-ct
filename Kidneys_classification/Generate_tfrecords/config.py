import tensorflow as tf

##---ploting-hyp
COLORMAP='gist_stern'
ALPHA_VALUES_CT=1
ALPHA_VALUES_MASK=0.5
###--------------------tfrecords Generator Parameters
PATCH_PARAMS = {'n_examples': 1,
                 'example_size': [96, 128, 128],
                 'extract_examples': True}
NUM_OF_CTS_IN_SINGLE_TFRECORDS=1

###########----Train----######
DATA_CSV='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kidney_Train_March19_2020_UPDTRBM.csv'
PATH_TO_SAVE_TFRECORDS='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Train_tfrecords/'
NAME_OF_PATH_CSV='Kidney_Train_96x128x128_patches'
PATH_TO_SAVE_MIDDLE_SLICE_PNG='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Train_tfrecords_mask/'
SAVING_PNG_OF_THE_PATCH='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Train_tfrecords_png/'

'''
###########----Val----######
DATA_CSV='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kidney_Val_March19_2020_UPDTRBM.csv'
PATH_TO_SAVE_TFRECORDS='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Val_tfrecords/'
NAME_OF_PATH_CSV='Kidney_Val_96x128x128_patches'
PATH_TO_SAVE_MIDDLE_SLICE_PNG='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Val_tfrecords_mask/'
SAVING_PNG_OF_THE_PATCH='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Val_tfrecords_png/'

###########----Test----######
DATA_CSV='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kidney_Test_March19_2020_UPDTRBM.csv'
PATH_TO_SAVE_TFRECORDS='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Test_tfrecords/'
NAME_OF_PATH_CSV='Kidney_Test_96x128x128_patches'
PATH_TO_SAVE_MIDDLE_SLICE_PNG='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Test_tfrecords_mask/'
SAVING_PNG_OF_THE_PATCH='/Local/nobackup/Kidney_CenterPatch_2mm_April11_2020/Kindey_tfrecords/Test_tfrecords_png/'
'''
