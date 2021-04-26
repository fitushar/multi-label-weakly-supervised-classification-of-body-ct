import tensorflow as tf
from loss_funnction_And_matrics import*
import math 
###---Number-of-GPU
NUM_OF_GPU=2
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1","gpu:2"]

###----Resume-Training
RESUME_TRAINING=0
RESUME_TRAIING_MODEL='/Local/April_Model_2020/Liver_Clf_April11_2020/Liver_Model_April11_2020i/'
TRAINING_INITIAL_EPOCH=0

##Network Configuration
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(96,128,128, 1)
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
BATCH_SIZE=9
TRAINING_STEP_PER_EPOCH=math.ceil((3081)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((650)/BATCH_SIZE)
TRAING_EPOCH=300
NUMBER_OF_PARALLEL_CALL=6
PARSHING=3*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='val_loss'
TRAINING_SAVE_MODEL_PATH='/Local/April_Model_2020/Liver_Clf_April11_2020/Liver_Model_April11_2020i/'
TRAINING_CSV='Liver_Model_April11_2020i.csv'
LOG_NAME="Liver_Log_April11_2020i"
MODEL_SAVING_NAME="LiverML_{val_loss:.2f}_{epoch}.h5"


####
TRAINING_TF_RECORDS='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Train_tfrecords/'
VALIDATION_TF_RECORDS='/Local/nobackup/Liver_CenterPatch_2mm_April11_2020/Val_tfrecords/'
