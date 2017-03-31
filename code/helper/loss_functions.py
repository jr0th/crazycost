import keras.metrics
import tensorflow as tf

debug = True

def crazyloss(y_true, y_pred):
    # y_true and y_pred are TF tensors

    return keras.metrics.categorical_crossentropy(y_true, y_pred)
    nb_classes = y_pred.get_shape()[2].value

    # calculate frequency over batch
    freq = tf.reduce_mean(y_true, axis = [0, 1])

    if flag_debug:    
        freq = tf.Print(freq, [freq], message="Freq: ")
    
    # calculate weights
    weight = tf.divide(1, freq)
    if flag_debug:    
        weight = tf.Print(weight, [weight], message="Weight (unnormalized): ")
    
    # normalize weights
    weight = tf.divide(weight, tf.reduce_sum(weight))
    weight = tf.multiply(weight, nb_classes)    
    if flag_debug:    
        weight = tf.Print(weight, [weight], message="Weight (normalized): ")

    # enlarge vector to prepare for multiplication    
    weight = tf.reshape(weight, [1, 1, nb_classes])

    # multiply true labels with weights to get weight for each pixel 
    mask = tf.multiply(y_true, weight)
    mask = tf.reduce_max(mask, axis = 2)

    loss =  tf.multiply(keras.metrics.categorical_crossentropy(y_pred, y_true), mask)
    
    return loss
