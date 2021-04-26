import tensorflow as tf
from loss_funnction_And_matrics import*
import math 
###---Number-of-GPU
NUM_OF_GPU=2
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1"]
###----Resume-Training
RESUME_TRAINING=0
RESUME_TRAIING_MODEL='/image_data/Scripts/April_Model/Lung_Clf_April22_2020/Lung_Model_April22_2020/'
TRAINING_INITIAL_EPOCH=0

##Network Configuration
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(224,160,160, 1)
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
##Training Hyper-Parameter
##Training Hyper-Parameter
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()
BATCH_SIZE=6
TRAINING_STEP_PER_EPOCH=math.ceil((3514)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((759)/BATCH_SIZE)
TRAING_EPOCH=300
NUMBER_OF_PARALLEL_CALL=6
PARSHING=4*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='val_loss'
TRAINING_SAVE_MODEL_PATH='/image_data/Scripts/April_Model/Lung_Clf_April22_2020/Lung_Model_April22_2020/'
TRAINING_CSV='Lung_Model_April22_2020.csv'
LOG_NAME="Lung_Log_April22_2020"
MODEL_SAVING_NAME="LungML_{val_loss:.2f}_{epoch}.h5"


####
TRAINING_TF_RECORDS='/image_data/nobackup/Lung_CenterPatch_2mm_April20_P224/all_tf/Train_tfrecords/'
VALIDATION_TF_RECORDS='/image_data/nobackup/Lung_CenterPatch_2mm_April20_P224/all_tf/Val_tfrecords/'
