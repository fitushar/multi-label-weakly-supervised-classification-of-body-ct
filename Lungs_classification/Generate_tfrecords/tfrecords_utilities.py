from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from Preprocessing_utlities import extract_class_balanced_example_array
from Preprocessing_utlities import resize_image_with_crop_or_pad
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from Preprocessing_utlities import resample_img2mm
from Preprocessing_utlities import normalise_zero_one
from Preprocessing_utlities import normalise_one_one
from Preprocessing_utlities import Get_Center_from_a_binary_mask
#from Preprocessing_utlities import extract_Lung_center_patch_updated
from Preprocessing_utlities import extract_Lung_center_patch_updated_problemsSolvedCenter
from matplotlib import pyplot as plt
from config import*


def Save_pngAndmask(a_id,a_lbl,img1,mask1,path):

    center_slice=64
    #RIO_mask= ((mask1 == 2)|(mask1 == 3)|(mask1 == 5)|(mask1 == 7)|(mask1 == 8)).astype(np.int32)
    #mask1=RIO_mask*mask1
    #####Getting Name
    p_id=a_id
    #p_id=str(p_id)
    #p_id=p_id.split("b'")
    #p_id=p_id[1].split("'")
    #p_id=p_id[0].split('_0')
    print(p_id)

    ####-----Decoding----Lbl
    p_lbl=a_lbl
    print(p_lbl)
    decoding_lbl=[]
    if p_lbl[0]==1:
        decoding_lbl.append('Atelectasis')
    if p_lbl[1]==1:
        decoding_lbl.append('Nodule')
    if p_lbl[2]==1:
        decoding_lbl.append('Emphysema')
    if p_lbl[3]==1:
        decoding_lbl.append('Effusion')
    if p_lbl[4]==1:
        decoding_lbl.append('Normal')
    print(decoding_lbl)


    f, axarr = plt.subplots(2,3,figsize=(128,128));
    f.suptitle('Id-->{}\n Label-->{}\n'.format(p_id,decoding_lbl),fontsize = 128)

    ########--------------------------Row---1---------###############
    img_plot = axarr[0][0].imshow(np.squeeze(img1[0, :, :]), cmap='gray',origin='lower');
    axarr[0][0].axis('off')
    axarr[0][0].set_title('First-Slice',fontsize = 128)

    middle_plot = axarr[0][1].imshow(np.squeeze(img1[center_slice, :, :]), cmap='gray',origin='lower');
    axarr[0][1].axis('off')
    axarr[0][1].set_title('Middle-Slice',fontsize = 128)

    last_plot = axarr[0][2].imshow(np.squeeze(img1[122, :, :]), cmap='gray',origin='lower');
    axarr[0][2].axis('off')
    axarr[0][2].set_title('5th-Last-Slice',fontsize = 128)

    ########--------------------------Row---2---------###############
    mask_plot = axarr[1][0].imshow(np.squeeze(img1[0, :, :]), cmap='gray',alpha=ALPHA_VALUES_CT,origin='lower');
    mask_plot = axarr[1][0].imshow(np.squeeze(mask1[0, :, :]), cmap=COLORMAP,alpha=ALPHA_VALUES_MASK,origin='lower');
    axarr[1][0].axis('off')
    axarr[1][0].set_title('Mask First-Slice',fontsize = 128)

    mask_middle_plot = axarr[1][1].imshow(np.squeeze(img1[center_slice, :, :]), cmap='gray',alpha=ALPHA_VALUES_CT,origin='lower');
    mask_middle_plot = axarr[1][1].imshow(np.squeeze(mask1[center_slice, :, :]), cmap=COLORMAP,alpha=ALPHA_VALUES_MASK,origin='lower');
    axarr[1][1].axis('off')
    axarr[1][1].set_title('Mask Middle-Slice',fontsize = 128)

    mask_last_plot = axarr[1][2].imshow(np.squeeze(img1[122, :, :]), cmap='gray',alpha=ALPHA_VALUES_CT,origin='lower');
    mask_last_plot = axarr[1][2].imshow(np.squeeze(mask1[122, :, :]), cmap=COLORMAP,alpha=ALPHA_VALUES_MASK,origin='lower');
    axarr[1][2].axis('off')
    axarr[1][2].set_title('Mask 5th-Last-Slice',fontsize = 128)

    png_name=path+p_id+'.png'
    plt.savefig(png_name)
    plt.close()
    return

def Lung_Saving_with_Slice(IMG,lbl,sub_id,path):
    #

    ####-----Decoding----Lbl
    p_lbl=lbl
    print(p_lbl)
    decoding_lbl=[]
    if p_lbl[0]==1:
        decoding_lbl.append('Atelectasis')
    if p_lbl[1]==1:
        decoding_lbl.append('Lung Nodule(Nodule/Mass)')
    if p_lbl[2]==1:
        decoding_lbl.append('Emphysema')
    if p_lbl[3]==1:
        decoding_lbl.append('Effusion')
    if p_lbl[4]==1:
        decoding_lbl.append('Lung Normal')
    print(decoding_lbl)

    fig=plt.figure(figsize=(128, 128))
    fig.suptitle('Label-->{}'.format(decoding_lbl),fontsize =128)

    columns =8
    rows =16 
    for i in range(1, columns*rows +1):
        img_slice=i-1
        img1 = IMG[img_slice,:,:]
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(img1[:,:]),cmap='gray',origin='lower')
        plt.axis('off')
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        #plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path+sub_id+'.png')
    plt.close()

    return




########################-------Fucntions for tf records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def flow_from_df(dataframe: pd.DataFrame, chunk_size):
    for start_row in range(0, dataframe.shape[0], chunk_size):
        end_row  = min(start_row + chunk_size, dataframe.shape[0])
        yield dataframe.iloc[start_row:end_row, :]


def creat_tfrecord(df,extraction_perameter,tf_name,path):

    read_csv=df.as_matrix()
    patch_params = extraction_perameter

    img_list=[]
    mask_list=[]
    lbl_list=[]
    id_name=[]

    for Data in read_csv:
        img_path = Data[4]
        subject_id = img_path.split('/')[-1].split('.')[0]
        Subject_lbl=Data[5:10]
        print(Subject_lbl.shape)

        print('Subject ID-{}'.format(subject_id))
        print('Labels--{}'.format(Subject_lbl))

        #Img
        img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        #img_sitk = resample_img2mm(img_sitk)
        image    = sitk.GetArrayFromImage(img_sitk)
        #image    = np.clip(image, -200., 500.).astype(np.float32)
        #image    = normalise_one_one(image)

        #Mask
        mask_fn = str(Data[10])
        mask = sitk.ReadImage(mask_fn,sitk.sitkInt32)
        #mask = resample_img2mm(mask,is_label=True)
        mask = sitk.GetArrayFromImage(mask)
        print('CT-Shape---{}'.format(image.shape))
        print('Mask-Shape---{}'.format(mask.shape))

        patch_name =bytes(subject_id, 'utf-8')

    #ct_patch,mask_patch=extract_Kidney_center_patch(image ,mask,subject_id,path)
    #ct_patch,mask_patch=extract_Kidney_center_patch_updated(image ,mask,subject_id,path)
    ct_patch,mask_patch=extract_Lung_center_patch_updated_problemsSolvedCenter(image ,mask,subject_id,path)
    Save_pngAndmask(subject_id,Subject_lbl,ct_patch,mask_patch,SAVING_PNG_OF_THE_PATCH)
    #Lung_Saving_with_Slice(ct_patch,Subject_lbl,subject_id,SAVING_PNG_OF_THE_PATCH)


    print('This Rfrecords will contain--{}--Pathes--of-size--{}'.format(len(id_name),patch_params['example_size']))

    record_mask_file =tf_name+'{}.tfrecords'.format(subject_id)
    with tf.io.TFRecordWriter(record_mask_file) as writer:
        feature = {'label1': _int64_feature(Subject_lbl[0]),
                       'label2': _int64_feature(Subject_lbl[1]),
                       'label3': _int64_feature(Subject_lbl[2]),
                       'label4': _int64_feature(Subject_lbl[3]),
                       'label5': _int64_feature(Subject_lbl[4]),
                        'image':_bytes_feature(ct_patch.tostring()),
                        'mask':_bytes_feature(mask_patch.tostring()),
                        'Height':_int64_feature(patch_params['example_size'][0]),
                        'Weight':_int64_feature(patch_params['example_size'][1]),
                        'Depth':_int64_feature(patch_params['example_size'][2]),
                        'label_shape':_int64_feature(5),
                        'Sub_id':_bytes_feature(patch_name)
                        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    return
