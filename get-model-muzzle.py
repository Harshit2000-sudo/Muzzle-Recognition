import numpy as np 
import time
import os

import tensorflow as tf
from tensorflow.keras import backend, layers, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
#####################################################################################

def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    
    for i in range(len(pretrained_model.layers)-10):
        pretrained_model.layers[i].trainable = False

    
    input_= layers.Input(input_shape, name="Anchor_Input")
    x= pretrained_model(input_)
    x= layers.Flatten()(x)
    x= layers.Dense(512, activation='relu')(x)
    x= layers.BatchNormalization()(x)
    x= layers.Dense(256, activation="relu")(x)
    x= layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    embed= Model(inputs=input_, outputs=x, name="Embedding")
    return embed

##################################################################
class MahalanobisDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def mdl(self,a,b):
        
        vec1= a
        vec2= b
        in_co =tf.concat((vec1, vec2),0)+1e-4

        # calculate the covariance matrix of vec1
        cov = tf.linalg.inv(tf.linalg.diag(tf.math.reduce_variance(tf.transpose(in_co), axis=1)))
    
        # calculate the Mahalanobis distance between vec1 and vec2
        diff = vec1 - vec2
        mdist = tf.sqrt(tf.matmul(tf.matmul(diff, cov), diff, transpose_b=True))
        return mdist
        
    def call(self, anchor, positive, negative):
        
        apd_nan= self.mdl(anchor, positive)
        value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(apd_nan)), dtype=tf.float32)
        apdnon_nan = tf.math.multiply_no_nan(apd_nan, value_not_nan)+1e-6
        
        apn_nan= self.mdl(anchor, negative)
        value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(apn_nan)), dtype=tf.float32)
        apnnon_nan = tf.math.multiply_no_nan(apn_nan, value_not_nan)+1e-6
        
        ap_dis= tf.reduce_sum(apdnon_nan,-1)
        ap_neg= tf.reduce_sum(apnnon_nan,-1)
        return (ap_dis, ap_neg)


class DistanceLayer(layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    

def get_siamese_network(input_shape = (450, 450, 2), dis_func=  "e"):
    
    encoder = get_encoder((450, 450, 3))
    

    in_= layers.Conv2D(3,(3,3),padding='same', activation="relu",
                       input_shape=input_shape, name= "channel_transform")
    
    
    # Input Layers for the images
    anchor_input   = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")
    
    in_anc= in_(anchor_input)
    in_pos= in_(positive_input)
    in_neg= in_(negative_input)
    
    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(in_anc)
    encoded_p = encoder(in_pos)
    encoded_n = encoder(in_neg)
    
    # A layer to compute ?f(A) - f(P)?� and ?f(A) - f(N)?�
    if dis_func=="e":
        distances = DistanceLayer()(encoded_a,
                                    encoded_p,
                                    encoded_n
                                   )
    elif dis_func=="m":
        distances = MahalanobisDistanceLayer()(encoded_a,
                                    encoded_p,
                                    encoded_n
                                   )
    
    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )
    return siamese_network

##################################################################

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

#########################################################################################