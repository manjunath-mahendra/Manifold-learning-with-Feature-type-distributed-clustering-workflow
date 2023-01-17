#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from fdc.fdc import feature_clustering,value
import seaborn as sns



def neural_network(n_features,hidden_dim1,hidden_dim2,out_emb_size,act1,act2,loss):
    np.random.seed(42)
    tf.random.set_seed(42)
    model=keras.Sequential([
         keras.layers.Dense(hidden_dim1,input_dim=n_features,activation=act1),
         keras.layers.Dense(hidden_dim1,activation=act2),
         keras.layers.Dense(out_emb_size)])
    model.compile(optimizer="adam" ,
              loss=loss, 
              metrics=['mse'])
    return model    


# In[3]:


#Function for decoding the encoded cluster labels
def label_decoder(label_dataframe):
    label_array=np.array(label_dataframe)
    decoded_labels=[]
    for i in label_array:
        max_val=np.argmax(i)
        decoded_labels.append(max_val)
    return decoded_labels


# In[4]:


def plotting(train_data_high_dim, predicted_data_high_dim, decoded_train_label, predicted_train_label,count):
    #concatinating training and predicted high dimensional embedding
    concatenated_5dim=pd.concat([train_data_high_dim,predicted_data_high_dim])
    
    #UMAP on concatinated embedding
    two_dim_viz=feature_clustering(15, 0.1, 'euclidean', concatenated_5dim, 0)
    
    #Concatinating decoded cluster labels of training fold and predicted testing fold
    concatenated_cluster_labels=np.concatenate([np.array(decoded_train_label),np.array(predicted_train_label)+len(np.unique(predicted_train_label))])
    
    two_dim_viz['Cluster']= concatenated_cluster_labels
    
    
    #Setting dark colors for training folds     
    darkerhues=['lightcoral','cornflowerblue','orange','mediumorchid', 'lightseagreen','olive', 'chocolate','steelblue']
    colors_set2=[]
    for i in range(len(np.unique(predicted_train_label))):
        colors_set2.append(darkerhues[i])
    
    #Concatinating dark colors for training folds and corresponding light colors for testing folds
    colors_set2=colors_set2+["lightpink", 'skyblue', 'wheat', "plum","paleturquoise",  "lightgreen",  'burlywood','lightsteelblue']
    
    print('Vizualization for FDC for training fold (shown in dark hue) '+str(count+1) + 'and predicted clusters from neural network on testing fold (shown in corresponding light hues) '+str(count+1))
    
    #visualizing the clusters of both training and testing folds
    sns.lmplot( x="UMAP_0", y="UMAP_1", data=two_dim_viz, fit_reg=False, legend=False, hue='Cluster', scatter_kws={"s": 3},palette=sns.set_palette(sns.color_palette(colors_set2))) 
    plt.show()
    


# In[5]:


def cluster_wise_F1score(ref_list,pred_list):
    def safeDiv(a, b):
        if b != 0:
            return a / b
        return 0.0
    
    F1_score_list = []
    Geometric_mean_list = []
    cluster_score_list = []
    true_positive_total = 0
    for i in np.unique(ref_list):
        indices = [j for j,val in enumerate(ref_list) if val == i]
        true_positive = 0
        for index in indices:
            if i == pred_list[index]:
                true_positive += 1
        true_positive_total += true_positive
        
        precision = safeDiv(true_positive, pred_list.count(i))
        recall = safeDiv(true_positive, len(indices))
        F1_score = safeDiv(2.0 * precision * recall, precision + recall)
        GM = np.sqrt(precision * recall)
        cluster_score = recall * 100.0
        
        print("F1_Score of cluster "+str(i)+" is {}".format(F1_score))
        print("Geometric mean of cluster "+str(i)+" is {}".format(GM))
        print("Correctly predicted data points in cluster "+str(i)+" is {}%".format(cluster_score))
        print("\n")
        F1_score_list.append((ref_list.count(i)/len(ref_list))*F1_score)
        Geometric_mean_list.append((ref_list.count(i)/len(ref_list))*GM)
        cluster_score_list.append((ref_list.count(i)/len(ref_list))*cluster_score)

    #correctly_predicted = safeDiv(100.0 * true_positive_total, len(ref_list))

    print("weigted average F1_Score of all clusters is {}".format(np.sum(F1_score_list)))
    print("weighted average Geometric mean of all clusters is {}".format(np.sum(Geometric_mean_list)))
    print("weighted average of Correctly predicted data points in all clusters is {}%".format(np.sum(cluster_score_list)))


# In[7]:


class Neural_Network_model:
    def __init__(self,X_train,X_test,y0,y1):
        self.X_train=X_train
        self.X_test=X_test
        self.y0=y0
        self.y1=y1
        
    def NN_1(self,input_layer=None,hidden_layer_1=None,hidden_layer_2=None,output_layer=None,activation_1=None,activation_2=None,loss=None):
        input_layer=value(input_layer,len(self.X_train[0]))
        hidden_layer_1=value(hidden_layer_1,int(0.6*len(self.X_train[0])))
        hidden_layer_2=value(hidden_layer_2,int(0.36*len(self.X_train[0])))
        output_layer=value(output_layer,len(self.y0[0]))
        activation_1=value(activation_1,"relu")
        activation_2=value(activation_2,"sigmoid")
        loss=value(loss,"mse")
        model_1=neural_network(input_layer,hidden_layer_1,hidden_layer_2,output_layer,activation_1,activation_2,loss)
        history=model_1.fit(self.X_train,self.y0,epochs=30,batch_size=8)
        print('\n')
        print('Training history across epochs for training fold ')
        plt.plot(history.history['mse'],'r')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.show()
        
        predicted_high_dim=pd.DataFrame(model_1.predict(self.X_test), columns=['c'+str(i+1) for i in range(np.shape(self.y0)[1])])
        predicted_low_dim=feature_clustering(30,0.01, "euclidean", predicted_high_dim, False)
        
        return predicted_high_dim,predicted_low_dim
        
    def NN_2(self,input_layer=None,hidden_layer_1=None,hidden_layer_2=None,output_layer=None,activation_1=None,activation_2=None,loss=None):
        input_layer=value(input_layer,len(self.X_train[0]))
        hidden_layer_1=value(hidden_layer_1,int(0.6*len(self.X_train[0])))
        hidden_layer_2=value(hidden_layer_2,int(0.36*len(self.X_train[0])))
        output_layer=value(output_layer,len(self.y1[0]))
        activation_1=value(activation_1,"relu")
        activation_2=value(activation_2,"sigmoid")
        loss=value(loss,"mse")
        model_2=neural_network(input_layer,hidden_layer_1,hidden_layer_2,output_layer,activation_1,activation_2,loss)
        history=model_2.fit(self.X_train,self.y1,epochs=30,batch_size=8)
        print('\n')
        print('Training history across epochs for training fold ')
        plt.plot(history.history['mse'],'r')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.show()
        
        #predicting testing fold to get encoded cluster labels using trained model_2
        predicted_clusters=pd.DataFrame(model_2.predict(self.X_test))
        
        #Decoding predicted encoded cluster labels
        decoded_predicted_clusters=label_decoder(predicted_clusters)
        return decoded_predicted_clusters
    
    
    def GB_reg(self):
        regr = MultiOutputRegressor(GradientBoostingRegressor(random_state=42)).fit(self.X_train, self.y0)
        reg_predicted_high_dim=pd.DataFrame(regr.predict(self.X_test), columns=['c'+str(i+1) for i in range(np.shape(self.y0)[1])])
        predicted_low_dim=feature_clustering(30,0.01, "euclidean", reg_predicted_high_dim, False)
        return reg_predicted_high_dim,predicted_low_dim
    
    
    def GB_clf(self):
        
        
        clf=GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=42)
        clf.fit(self.X_train, self.y1)
        clf_predicted_clusters=clf.predict(self.X_test)
        return clf_predicted_clusters
        


# In[ ]:




