# CITE-Tensorflow
This work is the Tensorflow implementation for our APIN 2023 paper:      
Jian, M., Wang, T., Zhou, S. et al. Compatible intent-based interest modeling for personalized recommendation. Appl Intell 53, 27519–27536 (2023).
DOI: https://doi.org/10.1007/s10489-023-04981-y

# Environment
Python==3.7.0
Tensorflow-gpu==2.4.0
keras==2.8.0

 # Dataset 
 We provide datasets: MovieLens-1M, Gowalla, and Yelp2018 in file ./Data. 
 We also provide pretrained user/item embeddings in flie ./pretrain_parameters
 

# Noting
1) If you want to reproduce this work in Tensorflow 1.x, please modify：

model.py:
import tensorflow.compat.v1 as tf &&& tf.disable_v2_behavior() ------> import tensorflow as tf 

initializer = tf.keras.initializers.glorot_normal() ------> initializer = tf.contrib.layers.xavier_initializer(uniform=False)

2) If 'train loss == nan' when you reproduce our work with other datasets,  please modify:

model.py:
self.mf_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings) ------> self.mf_loss = self.create_ssm_loss(self.u_g_embeddings, self.pos_i_g_embeddings)

But different loss functions may have impact on different datasets w.r.t recommendation performance

3) Please use pretrained user/item embeddings to obtain better recommendation performance mentioned in parameters subsection of the paper.
   Other expermental details please refer to parameters subsection in the paper.
  
