import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import Counter
import random
class cluster_class:
    
    def __init__(self,K):
        '''
        Create a cluster classifier object
        '''
        kmeans=KMeans(n_clusters=K, random_state=0)
        #Unfit cluster model initialized
        self.value=kmeans
        #Fit cluster model initialized
        self.fit_mod=None
        #Mapping between clusters and predicted class labels initialized 
        self.lab=None

    def fit(self, X,Y):
        '''
        Learn a cluster classifier object
        '''
        cl_model=self.value.fit(X)
        Z=cl_model.labels_
        Z_i=np.bincount(Z)
        Z_ii=np.nonzero(Z_i)[0]
        #Array of tuples of cluster no. and corresponding frequency of data cases
        freq_arr=zip(Z_ii,Z_i[Z_ii])
        K=len(freq_arr)
        res_raw=[]
        for i in range(K):
            #Checking whether current cluster has no data cases
            if(freq_arr[i][1]==0):
                #Assign random label from Y
                res_raw.append(random.choice(Y))
            else:
                index_arr=np.where(Z==i)[0]
                #labels in current cluster
                Tar_Y=[Y[i] for i in index_arr]
                count=Counter(Tar_Y)
                #Checking if cluster has only 1 type of label
                if (len(count.most_common())==1):
                    #Assign corresponding label
                    res_raw.append(count.most_common(1)[0][0])
                #Checking if first two most common labels have same frequency
                elif (count.most_common(1)[0][1]==count.most_common(2)[1][1]):
                    #Assign label by randomly breaking tie
                    rand_val=random.choice([count.most_common(1)[0][0],count.most_common(2)[1][0]])
                    res_raw.append(rand_val)
                else:
                    #Assign label with highest frequency
                    res_raw.append(count.most_common(1)[0][0])
        self.fit_mod=cl_model
        self.lab=res_raw
        
    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''
        tr_cl_mod=self.fit_mod
        tr_arr=self.lab
        
        labels=[]
        #Predicting cluster placement for all data cases
        pred_cl_i=tr_cl_mod.predict(X)
        for i in pred_cl_i:
            #predicting label for each data case based on assigned cluster no.
            labels.append(tr_arr[i])
        return np.asarray(labels)
    
    def score(self,X,Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''          
        Yhat = self.predict(X)
        return 1-accuracy_score(Y,Yhat)
        