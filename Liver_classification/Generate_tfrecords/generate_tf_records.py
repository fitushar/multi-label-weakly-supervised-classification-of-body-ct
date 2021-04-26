from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sys
import cv2
from tfrecords_utilities import*
from config import*
import math
import random
from scipy import ndimage
random.seed(3)


###Generate_tf_records_for_training
def generate_tf_records_from_csv(csv,path,infoname,png_path):

    read_csv=pd.read_csv(csv,keep_default_na=False,na_values=[])
    patch_params=PATCH_PARAMS

    get_chunk=flow_from_df(read_csv,NUM_OF_CTS_IN_SINGLE_TFRECORDS)
    Number_of_elements_in_df=len(read_csv)


    how_many_times_to_run_the_iteration=math.ceil(Number_of_elements_in_df/NUM_OF_CTS_IN_SINGLE_TFRECORDS)
    print('the Generator ot be called -->>{}--times'.format(how_many_times_to_run_the_iteration))

    df_lists=[]

    for i in range(how_many_times_to_run_the_iteration):
        chank=next(get_chunk)
        df_lists.append(chank)

    print('------>> Starting creation of tfrecord')
    tf_records_name=[]
    tf_recoeds_path=[]
    for i in range(2000,len(df_lists)):
        df=df_lists[i]
        number_tf=i
        tf_name='ct_liver_train_{}_'.format(i)
        save_path=path+tf_name
        am=png_path
        print('Starting--Processes--for-creating---{}'.format(tf_name))
        creat_tfrecord(df,patch_params,save_path,am)
        tf_records_name.append(tf_name)
        tf_recoeds_path.append(save_path)
        print('Saved--tf_recoeds:---{}'.format(tf_name))
    info_name=infoname+'.csv'
    tf_info=pd.DataFrame(list(zip(tf_records_name,tf_recoeds_path)),columns=['name','path'])
    tf_info.to_csv(info_name, encoding='utf-8', index=False)

    return

if __name__ == '__main__':

    csv= DATA_CSV
    path=PATH_TO_SAVE_TFRECORDS
    name=NAME_OF_PATH_CSV
    png_path=PATH_TO_SAVE_MIDDLE_SLICE_PNG
    generate_tf_records_from_csv(csv,path,name,png_path)
