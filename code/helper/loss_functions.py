import keras.metrics
import tensorflow as tf
import numpy as np

debug = True

def threshold_tensor(tensor):
    thresholded = tf.nn.softsign(tensor - 0.5)
    thresholded = thresholded * 3
    thresholded = (thresholded + 1) / 2
    return thresholded

def convolve_and_threshold(tensor, conv_filter):
    filtered = tf.nn.conv2d(tensor, conv_filter, strides=[1,1,1,1], padding='SAME')
    thresholded = threshold_tensor(filtered)
    return thresholded

def crazyloss(y_true, y_pred, dim1, dim2):
    # y_true and y_pred are TF tensors
    
    y_true = tf.Print(y_true, [y_true], 'y_true: ')

    slicer_img_1c_size = tf.constant([-1,dim1,dim2,1])
    slicer_img_3c_size = tf.constant([-1,dim1,dim2,3])
    
    slicer_img_begin = tf.constant([0,0,0,0])
    slicer_seed1_begin = tf.constant([0,0,0,3])
    slicer_seed2_begin = tf.constant([0,0,0,4])
    
    # get image and seeds
    y_true_img = tf.slice(y_true, slicer_img_begin, slicer_img_3c_size)
    seed1 = tf.slice(y_true, slicer_seed1_begin, slicer_img_1c_size)
    seed2 = tf.slice(y_true, slicer_seed2_begin, slicer_img_1c_size)
    
    # slice images in layers
    slicer_img_nuc1_begin = tf.constant([0,0,0,1])
    slicer_img_nuc2_begin = tf.constant([0,0,0,2])
    
    # 1 where we have the respective nucleus in y_true
    nuc1 = tf.slice(y_true_img, slicer_img_nuc1_begin, slicer_img_1c_size)
    nuc2 = tf.slice(y_true_img, slicer_img_nuc2_begin, slicer_img_1c_size)
    
    nuc1_pred = tf.slice(y_pred, slicer_img_nuc1_begin, slicer_img_1c_size)
    nuc2_pred = tf.slice(y_pred, slicer_img_nuc2_begin, slicer_img_1c_size)
    
    # get nucleus mask
    nucleus_maks = nuc1 - nuc2
    nucleus_maks = tf.Print(nucleus_maks, [nucleus_maks], "mask: ")
    
    conv_filter = tf.reshape(tf.constant([[0,1,0],[1,1,1],[0,1,0]], dtype=tf.float32), shape=(3, 3, 1, 1))
    
    # calc loss on nuc 1
    
    # start with seed
    region1 = tf.reshape((nuc1_pred * seed1), shape=(-1, dim1, dim2, 1))
    
    # grow region
    for i in range(np.maximum(dim1, dim2)):
        region1 = convolve_and_threshold(region1, conv_filter)

    loss1 = tf.reduce_mean(tf.abs(tf.reduce_sum(region1 * nucleus_maks, axis=[1,2])),axis=0)

    # calc loss on nuc 2
    region2 = tf.reshape((nuc2_pred * seed2), shape=(-1, dim1, dim2, 1))
    
    # grow region
    for i in range(np.maximum(dim1, dim2)):
        region2 = convolve_and_threshold(region2, conv_filter)
        
    loss2 = tf.reduce_mean(tf.abs(tf.reduce_sum(region2 * nucleus_maks, axis=[1,2])),axis=0)
   
    return - (loss1 + loss2)


def crazyloss_one_thresh(y_true_coll, y_pred, dim1, dim2):
    # y_true and y_pred are TF tensors
    
    y_true_coll = tf.Print(y_true_coll, [y_true], 'y_true: ')

    slicer_img_1c_size = tf.constant([-1,dim1,dim2,1])
    slicer_img_3c_size = tf.constant([-1,dim1,dim2,3])
    
    # slice images in layers
    slicer_img_bg_begin = tf.constant([0,0,0,0])
    slicer_img_nuc1_begin = tf.constant([0,0,0,1])
    slicer_img_nuc2_begin = tf.constant([0,0,0,2])
    slicer_seed1_begin = tf.constant([0,0,0,3])
    slicer_seed2_begin = tf.constant([0,0,0,4])
    
    # get image and seeds
    y_true_img = tf.slice(y_true, slicer_img_bgbegin, slicer_img_3c_size)
    seed1 = tf.slice(y_true, slicer_seed1_begin, slicer_img_1c_size)
    seed2 = tf.slice(y_true, slicer_seed2_begin, slicer_img_1c_size)
    
    # 1 where we have the respective nucleus in y_true
    nuc1 = tf.slice(y_true_img, slicer_img_nuc1_begin, slicer_img_1c_size)
    nuc2 = tf.slice(y_true_img, slicer_img_nuc2_begin, slicer_img_1c_size)
    
    
    # get nucleus mask
    nucleus_maks = nuc1 - nuc2
    nucleus_maks = tf.Print(nucleus_maks, [nucleus_maks], "mask: ")
    
    conv_filter = tf.reshape(tf.constant([[0,1,0],[1,1,1],[0,1,0]], dtype=tf.float32), shape=(3, 3, 1))
    thresholded_output_nuc1 = threshold_tensor(nuc1_pred)
    thresholded_output_nuc2 = threshold_tensor(nuc2_pred)
    
    # calc loss on nuc 1
    
    # start with seed
    region1 = tf.reshape((nuc1_pred * seed1), shape=(-1, dim1, dim2, 1))
    
    # grow region
    for i in range(np.maximum(dim1, dim2)):
        region1 = convolve_and_threshold(region1, conv_filter)

    loss1 = tf.reduce_mean(tf.abs(tf.reduce_sum(region1 * nucleus_maks, axis=[1,2])),axis=0)

    # calc loss on nuc 2
    region2 = tf.reshape((nuc2_pred * seed2), shape=(-1, dim1, dim2, 1))
    
    # grow region
    for i in range(np.maximum(dim1, dim2)):
        region2 = convolve_and_threshold(region2, conv_filter)
        
    loss2 = tf.reduce_mean(tf.abs(tf.reduce_sum(region2 * nucleus_maks, axis=[1,2])),axis=0)
   
    return - (loss1 + loss2)