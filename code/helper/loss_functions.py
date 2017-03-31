import keras.metrics
import tensorflow as tf

debug = True

def crazyloss(y_true, y_pred, dim1, dim2):
    # y_true and y_pred are TF tensors

    slicer_img_begin = tf.constant([0,0,0,0])
    slicer_img_size = tf.constant([-1,dim1,dim2,3])
    
    slicer_seeds_begin = tf.constant([0,0,0,3])
    slicer_seeds_size = tf.constant([-1,dim1,dim2,1])
    
    y_true_img = tf.slice(y_true, slicer_img_begin, slicer_img_size)
    y_true_seeds = tf.slice(y_true, slicer_seeds_begin, slicer_seeds_size)
    
    y_true_img = tf.Print(y_true_img, [y_true_seeds], 'y_true_seeds')
    
    return keras.metrics.categorical_crossentropy(y_true_img, y_pred)