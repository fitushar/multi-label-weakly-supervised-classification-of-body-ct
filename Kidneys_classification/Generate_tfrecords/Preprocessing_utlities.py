from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import SimpleITK as sitk
from scipy import ndimage
import pandas as pd
import math
from skimage.filters import threshold_mean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import cv2
import math



def resample_img2mm(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


def flip(imagelist, axis=1):
    """Randomly flip spatial dimensions
    Args:
        imagelist (np.ndarray or list or tuple): image(s) to be flipped
        axis (int): axis along which to flip the images
    Returns:
        np.ndarray or list or tuple: same as imagelist but randomly flipped
            along axis
    """

    # Check if a single image or a list of images has been passed
    was_singular = False
    if isinstance(imagelist, np.ndarray):
        imagelist = [imagelist]
        was_singular = True

    # With a probility of 0.5 flip the image(s) across `axis`
    do_flip = np.random.random(1)
    if do_flip > 0.5:
        for i in range(len(imagelist)):
            imagelist[i] = np.flip(imagelist[i], axis=axis)
    if was_singular:
        return imagelist[0]
    return imagelist


def add_gaussian_offset(image, sigma=0.1):
    """
    Add Gaussian offset to an image. Adds the offset to each channel
    independently.
    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from
    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    offsets = np.random.normal(0, sigma, ([1] * (image.ndim - 1) + [image.shape[-1]]))
    image += offsets
    return image


def add_gaussian_noise(image, sigma=0.05):
    """
    Add Gaussian noise to an image
    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from
    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    image += np.random.normal(0, sigma, image.shape)
    return image


def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [1].
    [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis", in Proc. of the
        International Conference on Document Analysis and Recognition, 2003.
    Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5
    Args:
        image (np.ndarray): image to be deformed
        alpha (list): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (list): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation
    Returns:
        np.ndarray: deformed image
    """

    assert len(alpha) == len(sigma), \
        "Dimensions of alpha and sigma are different"

    channelbool = image.ndim - len(alpha)
    out = np.zeros((len(alpha) + channelbool, ) + image.shape)

    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        array = (np.random.rand(*image.shape) * 2 - 1)
        out[jj] = gaussian_filter(array, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]

    # Map mask to indices
    shapes = list(map(lambda x: slice(0, x, None), image.shape))
    grid = np.broadcast_arrays(*np.ogrid[shapes])
    indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

    # Transform image based on masked indices
    transformed_image = map_coordinates(image, indices, order=0,
                                        mode='reflect').reshape(image.shape)

    return transformed_image

def extract_class_balanced_example_array(image,
                                         label,
                                         example_size=[1, 64, 64],
                                         n_examples=1,
                                         classes=2,
                                         class_weights=None):
    """Extract training examples from an image (and corresponding label) subject
        to class balancing. Returns an image example array and the
        corresponding label array.

    Args:
        image (np.ndarray): image to extract class-balanced patches from
        label (np.ndarray): labels to use for balancing the classes
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total
        classes (int or list or tuple): number of classes or list of classes
            to extract

    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
            images with the shape [batch, example_size..., image_channels]
    """
    assert image.shape[:-1] == label.shape, 'Image and label shape must match'
    assert image.ndim - 1 == len(example_size), \
        'Example size doesnt fit image size'
    #assert all([i_s >= e_s for i_s, e_s in zip(image.shape, example_size)]), \
        #'Image must be larger than example shape'
    rank = len(example_size)



    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)


    if class_weights is None:
        n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(n_examples / n_classes))
    else:
        assert len(class_weights) == n_classes, \
            'Class_weights must match number of classes'
        class_weights = np.array(class_weights)
        n_ex_per_class = np.round((class_weights / class_weights.sum()) * n_examples).astype(int)

    # Compute an example radius to define the region to extract around a
    # center location
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0),
                               np.ceil(np.array(example_size) / 2.0))),
                      dtype=np.int)

    class_ex_images = []
    class_ex_lbls = []
    min_ratio = 1.
    for c_idx, c in enumerate(classes):
        # Get valid, random center locations belonging to that class
        idx = np.argwhere(label == c)

        ex_images = []
        ex_lbls = []

        if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
            class_ex_images.append([])
            class_ex_lbls.append([])
            continue

        # Extract random locations
        r_idx_idx = np.random.choice(len(idx),
                                     size=min(n_ex_per_class[c_idx], len(idx)),
                                     replace=False).astype(int)
        r_idx = idx[r_idx_idx]

        # Shift the random to valid locations if necessary
        r_idx = np.array(
            [np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                           ex_rad[dim][0]) for dim in range(rank)])
             for r in r_idx])

        for i in range(len(r_idx)):
            # Extract class-balanced examples from the original image
            slicer = [slice(r_idx[i][dim] - ex_rad[dim][0], r_idx[i][dim] + ex_rad[dim][1]) for dim in range(rank)]

            ex_image = image[slicer][np.newaxis, :]

            ex_lbl = label[slicer][np.newaxis, :]

            # Concatenate them and return the examples
            ex_images = np.concatenate((ex_images, ex_image), axis=0) \
                if (len(ex_images) != 0) else ex_image
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) \
                if (len(ex_lbls) != 0) else ex_lbl

        class_ex_images.append(ex_images)
        class_ex_lbls.append(ex_lbls)

        ratio = n_ex_per_class[c_idx] / len(ex_images)
        min_ratio = ratio if ratio < min_ratio else min_ratio

    indices = np.floor(n_ex_per_class * min_ratio).astype(int)

    ex_images = np.concatenate([cimage[:idxs] for cimage, idxs in zip(class_ex_images, indices)
                                if len(cimage) > 0], axis=0)
    ex_lbls = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices)
                              if len(clbl) > 0], axis=0)

    return ex_images, ex_lbls

def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


def extract_random_example_array(image_list,example_size=[1, 64, 64],n_examples=1):

    """Randomly extract training examples from image (and a corresponding label).
        Returns an image example array and the corresponding label array.
    Args:
        image_list (np.ndarray or list or tuple): image(s) to extract random
            patches from
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total
    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
        images with the shape [batch, example_size..., image_channels]
    """

    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size) or image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (i.ndim - 1 == image_list[0].ndim or i.ndim == image_list[0].ndim or i.ndim + 1 == image_list[0].ndim),\
                'Example size doesnt fit image size'

            assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
                'Image shapes must match'

    rank = len(example_size)

    # Extract random examples from image and label
    valid_loc_range = [image_list[0].shape[i] - example_size[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
               if valid_loc_range[dim] > 0
               else np.zeros(n_examples, dtype=int) for dim in range(rank)]

    examples = [[]] * len(image_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim])
                  for dim in range(rank)]

        for j in range(len(image_list)):
            ex_image = image_list[j][slicer][np.newaxis]
            # Concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_image), axis=0) \
                if (len(examples[j]) != 0) else ex_image

    if was_singular:
        return examples[0]
    return examples

def Get_Center_from_a_binary_mask(PNGIMG):
    # Load the image and convert it to grayscale:
    image = cv2.imread(PNGIMG)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    # Find contours:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Draw contours:
    cv2.drawContours(image, contours, 0, (0, 255, 0), 2)
    # Calculate image moments of the detected contour
    M = cv2.moments(contours[0])
    CX=round(M['m10'] / M['m00'])
    CY=round(M['m01'] / M['m00'])
    return CX,CY


def extract_Kidney_center_patch_updated(i,l,NameforCenterSlicePNG,pathpng):

    ###Getting the label
    mask=l
    img=i
    try:
        ##--Making Description list
        Kidney_slice_number=[]
        Kidney_slice_Dice=[]
        Kidney_slice_count=[]
        #----------------------------------------------------#
        ###-------Getting the center slice information-----###
        #----------------------------------------------------#
        for i in range(mask.shape[0]):
            roi_slice=mask[i, :, :]
            K= ((roi_slice == 7)|(roi_slice == 8)).astype(np.int32)
            slice_flatten=K.flatten()
            Kidney_idx=np.argwhere(slice_flatten==1)
            Kidney_slice_Dice.append(len(Kidney_idx))
            Kidney_slice_number.append(i)
            if (len(Kidney_idx)!=0):
                Kidney_slice_count.append(1)
            else:
                Kidney_slice_count.append(0)
        #--Making dataframe
        Kidney_dice_And_slice=pd.DataFrame(list(zip(Kidney_slice_number,Kidney_slice_Dice,Kidney_slice_count)),columns=['s_n','s_d','Livr_slice_Count'])
        Kidney_dice_And_slice
        max_dice=np.max(Kidney_slice_Dice)
        where_is_max_dice=np.argwhere(Kidney_slice_Dice==max_dice)
        if (len(where_is_max_dice)>1):
            select_center_slice_index=math.ceil(len(where_is_max_dice)/2)
            ##---Center_Slice---
            Center_slice=(where_is_max_dice[select_center_slice_index])
            Center_slice=Center_slice[0]
            print(Center_slice)
        else:
            Center_slice=where_is_max_dice[0]
            Center_slice=Center_slice[0]
            print(Center_slice)

        ####----
        only_Kidney_slices=Kidney_dice_And_slice[(Kidney_dice_And_slice['Livr_slice_Count']==1) & (Kidney_dice_And_slice['s_d']>=100) ]

        Kidney_first_slice=(only_Kidney_slices['s_n'].iloc[0])
        Kidney_last_slice=(only_Kidney_slices['s_n'].iloc[-1])

           #-------------------------------------------------------------#
           #---------- taking the center slice to get the Kidney Center---#
           #-------------------------------------------------------------#
        mask_a=mask[Center_slice, :, :]
        #mask_a=mask_ai[0,:,:]
        print('mask_shape')
        print(mask_a.shape)
        Kidney_mask= ((mask_a == 7)|(mask_a == 8)).astype(np.int32)
        body_mask = ((mask_a != 0)).astype(np.int32)
        ##-thresholding mask
        thresh = threshold_mean(Kidney_mask)
        binary = Kidney_mask > thresh

        ##-- to remove over segmented region
        binary=ndimage.binary_erosion(binary,structure=np.ones((3,3))).astype(int)
        ##-- Covering holes
        binary=ndimage.binary_opening(binary).astype(int)
        binary= ndimage.binary_closing(binary).astype(int)
        ## Saving image
        Png_name=pathpng+NameforCenterSlicePNG+'.png'
        cv2.imwrite(Png_name,binary*255)
        ###---Getting Center of the binary Kidney Mask
        #cx,cy=Get_Center_from_a_binary_mask(Png_name)
        cx=math.ceil((mask_a.shape[0]/2))
        cy=math.ceil((mask_a.shape[1]/2))
        if (Center_slice < 48):

            y_lower_bound = int(cy) - 64
            if (y_lower_bound <0):
                center_y_lr=0
            else:
                center_y_lr=y_lower_bound
            # Upper boundary y-center
            y_uper_bound  = int(cy) + 64
            if (y_uper_bound >img.shape[1]):
                center_y_upr=img.shape[1]
            else:
                center_y_upr=y_uper_bound
        ##Fixing x-boundary
            # Lower Boundary of Y-Center
            x_lower_bound = int(cx) - 64
            if (x_lower_bound <0):
                center_x_lr=0
            else:
                center_x_lr=x_lower_bound
            # Uper boundary of x-center
            x_uper_bound  = int(cx) + 64
            if (x_uper_bound >img.shape[2]):
                center_x_upr=img.shape[2]
            else:
                center_x_upr=x_uper_bound

            slicer = [slice(Kidney_first_slice,  Kidney_last_slice),slice(center_y_lr, center_y_upr),slice(center_x_lr, center_x_upr)]
            ex_image = img[slicer]
            ex_lbl = mask[slicer]
            ex_image=resize_image_with_crop_or_pad(ex_image, [96,128,128], mode='constant',constant_values=0)
            ex_lbl=resize_image_with_crop_or_pad(ex_lbl, [96,128,128], mode='constant',constant_values=0)
            print('----Using method-alternative----')

        else:
            y_lower_bound = int(cy) - 64
            if (y_lower_bound <0):
                center_y_lr=0
            else:
                center_y_lr=y_lower_bound
            # Upper boundary y-center
            y_uper_bound  = int(cy) + 64
            if (y_uper_bound >img.shape[1]):
                center_y_upr=img.shape[1]
            else:
                center_y_upr=y_uper_bound
            ##Fixing x-boundary
            # Lower Boundary of Y-Center
            x_lower_bound = int(cx) - 64
            if (x_lower_bound <0):
                center_x_lr=0
            else:
                center_x_lr=x_lower_bound
                # Uper boundary of x-center
            x_uper_bound  = int(cx) + 64
            if (x_uper_bound >img.shape[2]):
                center_x_upr=img.shape[2]
            else:
                center_x_upr=x_uper_bound
            slicer = [slice(Center_slice-48,  Center_slice+48),slice(center_y_lr, center_y_upr),slice(center_x_lr, center_x_upr)]
            ex_image = img[slicer]
            ex_lbl = mask[slicer]
            ex_image=resize_image_with_crop_or_pad(ex_image, [96,128,128], mode='constant',constant_values=0)
            ex_lbl=resize_image_with_crop_or_pad(ex_lbl, [96,128,128], mode='constant',constant_values=0)
            print('---Using --Kidney--Center--Method---')
    except:
            mask= ((mask == 7)|(mask == 8)).astype(np.int32)
            print("Going Class balanced Policy")
            patch_size=[96,128,128]
            img_shape=img.shape

            ###----padding_data_if_needed
           #####----z dimention-----######
            if (patch_size[0] >=img_shape[0]):
                dimention1=patch_size[0]+10
            else:
                dimention1=img_shape[0]

            #####----x dimention-----######
            if (patch_size[1] >=img_shape[1]):
                 dimention2=patch_size[1]+10
            else:
                dimention2=img_shape[1]

            #####----Y dimention-----######
            if (patch_size[2] >=img_shape[2]):
                 dimention3=patch_size[2]+10
            else:
                dimention3=img_shape[2]
            print('------before padding image shape--{}-----'.format(img.shape))
            image=resize_image_with_crop_or_pad(img, [dimention1,dimention2,dimention3], mode='constant',constant_values=0)
            mask=resize_image_with_crop_or_pad(mask, [dimention1,dimention2,dimention3], mode='constant',constant_values=0)
            print('######before padding image shape--{}#####'.format(image.shape))



            img_shape=image.shape
            image= np.expand_dims(image, axis=3)

            images,masks = extract_class_balanced_example_array(
                        image,mask,
                        example_size=[96,128,128],
                        n_examples=1,
                        classes=1,class_weights=[1])

            ex_image=images[0][:,:,:,0]
            ex_lbl=masks[0][:,:,:]

    return ex_image, ex_lbl
