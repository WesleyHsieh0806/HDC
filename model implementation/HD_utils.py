''' 
Hyperdimensional (HD) binary classifier package 
'''
import sys
import numpy as np
import random
import keras
import json
from math import floor
from scipy.spatial import distance
from scipy.stats import bernoulli
from scipy.stats import mode
from multiprocessing import Pool
import time

__author__ = "Frank Chuang"
__email__ = "frankchuang@access.ee.ntu.edu.tw"
__date__ = "2019.05.19"


class Dynamic_HD_model():
    def __init__(self, n_features, n_classes, HD_dim, n_levels, BINARY=True): 
        '''
        Description:    Initialization of HD classifier    
        INPUTS:
            n_features: input feature dimension  
            n_levels:   number of level for CiM
            n_classes:  Number of classes 
            HD_dim:     HD dimension   
        '''
        # Parameters
        self.n_features = n_features
        self.n_classes = n_classes  
        self.HD_dim = HD_dim    
        self.n_levels = n_levels 
        self.BINARY = BINARY
        self.n_train_data = 0
        self.initMemories()
        
    def fit_AM(self, train_datas, labels): 
        '''
        Description:    train an associative memory from scratch   
        INPUTS:
            train_datas: training data, shape(n_samples, n_features)
            labels:     labels for training data, shape(n_samples,)  
        ''' 
        n_samples= train_datas.shape[0]
        self.n_train_data+=n_samples
        self.max = np.amax(train_datas, axis=0)
        self.min = np.amin(train_datas, axis=0)
        self.value_range = self.max-self.min
        self.level_range = self.value_range/self.n_levels   
        n_labels = np.zeros(self.n_classes, dtype = int)
        
        for i in range(n_samples):  
            # Spatial Encoder
            spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)     
            val_idx = np.floor((train_datas[i] - self.min) / (self.level_range+1e-20)).astype(int)
            # the max value in test data may larger tham self.max => val_idx > n_level
            val_idx = np.clip(val_idx, 0, self.n_levels-1)
            spatial_array = self.iM ^ self.CiM[val_idx]
            hd_vec = np.sum(spatial_array, axis=0)             
            self.Integer_AM[labels[i]] += hd_vec
            n_labels[labels[i]] += self.n_features
        
        for class_idx in range(self.n_classes):
            self.Integer_AM[class_idx] = self.Integer_AM[class_idx] - n_labels[class_idx]/2 # bipolar function
            self.Binary_AM[class_idx] = (self.Integer_AM[class_idx] >= 0)*1 
        self.mask = self.Binary_AM[0] ^ self.Binary_AM[1] # find different bit
        
        
    def online_fit_AM(self, train_datas, labels, t1=1, t2=0, RETRAIN=True, RETRAIN_BIN=True, RETRAIN_INT=False): 
        '''
        Description:    train an associative memory incrementally
        INPUTS:
            train_datas: training data, shape(n_samples, n_features)
            labels:     labels for training data, shape(n_samples,) 
            t1:         if predict!=label, Binary_HD_scores[label] > t1 => train
            t2:         if predict==label, Binary_HD_scores[label] < t2 => train
            RETRAIN:    whether use retrain method to increase label distance
            RETRAIN_BIN:retrain based on Binary HDC prediction
            RETRAIN_INT:retrain based on Integer HDC prediction
        ''' 
        n_samples= train_datas.shape[0] 
        n_labels = np.zeros(self.n_classes, dtype = int)
        
        if RETRAIN_BIN==True:
            # Get score from Binary HDC
            y_pred, Integer_HD_scores, Binary_HD_scores, n_pass_data = self.predict_AM(train_datas, threshold=0, MASK=False)
            HD_scores=Binary_HD_scores # 0~1
            
        if RETRAIN_INT==True:
            # Get score from Integer HDC
            y_pred, Integer_HD_scores, Binary_HD_scores, n_pass_data = self.predict_AM(train_datas, threshold=1, MASK=False)
            HD_scores=((np.array(Integer_HD_scores)+1)/2).tolist() # -1~1 -> 0~1 
        
        for i in range(n_samples):
            if y_pred[i]!=labels[i] and HD_scores[i][labels[i]]>t1: 
                self.n_train_data+=1
                # Fit one sample
                spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)     
                val_idx = np.floor((train_datas[i] - self.min) / (self.level_range+1e-20)).astype(int)
                val_idx = np.clip(val_idx, 0, self.n_levels-1)
                spatial_array = self.iM ^ self.CiM[val_idx]
                hd_vec = np.sum(spatial_array, axis=0)     
                # retrain method
                self.Integer_AM[labels[i]] += hd_vec   
                n_labels[labels[i]] += self.n_features  
                if RETRAIN:
                    self.Integer_AM[y_pred[i]] -= hd_vec
                    n_labels[y_pred[i]] -= self.n_features
            elif y_pred[i]==labels[i] and HD_scores[i][labels[i]]<t2: 
                self.n_train_data+=1
                # Fit one sample
                spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)     
                val_idx = np.floor((train_datas[i] - self.min) / (self.level_range+1e-20)).astype(int)
                val_idx = np.clip(val_idx, 0, self.n_levels-1)
                spatial_array = self.iM ^ self.CiM[val_idx]
                hd_vec = np.sum(spatial_array, axis=0)             
                self.Integer_AM[labels[i]] += hd_vec
                n_labels[labels[i]] += self.n_features
                
        # May change with each sample???!!!!! Integer change than Binary change??????
        for class_idx in range(self.n_classes):
            self.Integer_AM[class_idx] = self.Integer_AM[class_idx] - n_labels[class_idx]/2
            self.Binary_AM[class_idx] = (self.Integer_AM[class_idx] >= 0)*1 
        self.mask = self.Binary_AM[0] ^ self.Binary_AM[1]         
            
    def predict_AM(self, samples, threshold=0, MASK=False):
        '''
        Description : Predict multiple samples    
        INPUTS :
            samples: feature sample, shape(n_samples, n_features)    
            theshold: for dynamic inference 0~1. 0: Binary HDC, 1: Integer HDC
        OUTPUTS :
            y_hat: Vector of estimated output class, shape (n_samples)
        '''      
        n_workers = 10
        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(samples) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(samples)
                else:
                    batch_end = (len(samples) // n_workers) * (i + 1)
                
                batch = samples[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples_AM, args=(batch, threshold, MASK))
            pool.close()
            pool.join()
        
        y_hats = []
        Integer_HD_scores = []
        Binary_HD_scores = []
        pass_datas = []
        inference_times = []
        for result in results:
            y_hat, Integer_HD_score, Binary_HD_score, pass_data, inference_time = result.get()  
            y_hats += y_hat  
            Integer_HD_scores += Integer_HD_score
            Binary_HD_scores += Binary_HD_score
            pass_datas += [pass_data]
            inference_times += inference_time
            
        return y_hats, Integer_HD_scores, Binary_HD_scores, sum(pass_datas), sum(inference_times)/len(samples)
            
    def preprocess_samples_AM(self, samples, threshold, MASK=False):
        n_samples, _ = samples.shape
        Binary_HD_score = np.zeros((n_samples,self.n_classes))   
        Integer_HD_score = np.zeros((n_samples,self.n_classes))        
        y_hat = np.zeros(n_samples, dtype = int)        
        n_pass_data = 0
        inference_time=[]
        for sample_idx in range(n_samples):        
            # Transform feature vector to binary HD vector
            binary_hd_vec, integer_hd_vec = self.transform_AM(samples[sample_idx])  
                
            # calculate HD score for every test class 
            # ===============================================
            start = time.time()
            
            for test_class in range(self.n_classes):
                Binary_HD_score[sample_idx, test_class] = 1-self.ham_dist(binary_hd_vec, self.Binary_AM[test_class])               

            # Dynamic Inference
            sort_idx = np.argsort(Binary_HD_score[sample_idx]) # small -> big
            first_highest_idx = sort_idx[-1]
            second_highest_idx = sort_idx[-2]
            sim_diff = Binary_HD_score[sample_idx, first_highest_idx] - Binary_HD_score[sample_idx, second_highest_idx]
#             assert sim_diff >= 0            
            if sim_diff >= threshold:
                y_hat[sample_idx] = np.argmax(Binary_HD_score[sample_idx])
            else:                
                for test_class in range(self.n_classes):
                    Integer_HD_score[sample_idx, test_class] = self.cos_similarity(integer_hd_vec, self.Integer_AM[test_class])
#                     if MASK:
#                         Integer_HD_score[sample_idx, test_class] = self.cos_similarity(integer_hd_vec*self.mask,\
#                                                                                        self.Integer_AM[test_class]*self.mask)
#                     else:
#                         Integer_HD_score[sample_idx, test_class] = self.cos_similarity(integer_hd_vec, self.Integer_AM[test_class])       
                y_hat[sample_idx] = np.argmax(Integer_HD_score[sample_idx])
                n_pass_data += 1
            
            end = time.time()
            inference_time.append(end-start)
            # ===============================================

        return (y_hat.tolist(), Integer_HD_score.tolist(), Binary_HD_score.tolist(), n_pass_data, inference_time)
    
    def transform_AM(self, sample): 
        '''
        Description : transform features into hypervectors   
        INPUTS :
            sample: feature sample, shape(n_features)    
        OUTPUTS :
            hd_vec: hypervecotes, shape(HD_dim)   
        '''
        spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)        
        val_idx = np.floor((sample - self.min) / (self.level_range+1e-20)).astype(int)
        val_idx = np.clip(val_idx, 0, self.n_levels-1)
        spatial_array = self.iM ^ self.CiM[val_idx]
        
        binary_hd_vec, _ = mode(spatial_array, axis=0) # majority function   
        integer_hd_vec = np.sum(spatial_array, axis=0) - self.n_features/2 # bipolar
        
        return binary_hd_vec.flatten(), integer_hd_vec
    
    def initMemories(self):
        '''
        Description : initialize the item Memory, continuous item memory and associative memory    
        INPUT:
        OUTPUT:
            iM: item memory for ID of features
            CiM:continuous item memory for value of features
        '''
        self.iM = np.zeros((self.n_features, self.HD_dim), dtype=int) 
        self.CiM = np.zeros((self.n_levels, self.HD_dim), dtype=int)
        # Item Memory: Orthognal mapping
        for i in range(self.n_features):
            self.iM[i] = self.genRandomHV()
            
        # Continuous Memory: Continuous mapping
        initHV = self.genRandomHV()
        currentHV = initHV
        randomIdx = np.arange(self.HD_dim)
        random.shuffle(randomIdx)
        n_flip_bits = floor(self.HD_dim/2/(self.n_levels-1))
        for i in range(self.n_levels):
            self.CiM[i] = currentHV
            startIdx = i*n_flip_bits
            endIdx = (i+1)*n_flip_bits 
            currentHV[randomIdx[startIdx:endIdx]] = self.invert(currentHV[randomIdx[startIdx:endIdx]])
        
        # Assotiative memory (one vector per class)
        self.Binary_AM = np.zeros((self.n_classes, self.HD_dim), dtype=int) 
        self.Integer_AM = np.zeros((self.n_classes, self.HD_dim)) 

    def clean_AM(self):
        '''
        Description : set AM = 0          
        '''
        self.Binary_AM = np.zeros((self.n_classes, self.HD_dim), dtype=int) 
        self.Integer_AM = np.zeros((self.n_classes, self.HD_dim)) 
        
    def reset_n_train_data(self):
        self.n_train_data=0
    
    def genRandomHV(self):
        '''
        Description : generate a random vector with zero mean        
        OUTPUTS :
            randomHV: generated random vector     
        '''
        assert self.HD_dim%2 == 0
        
        randomHV = np.zeros(self.HD_dim, dtype=int) 
        randomIdx = np.arange(self.HD_dim)
        random.shuffle(randomIdx)
        randomHV[randomIdx[:floor(self.HD_dim/2)]] = 1
        randomHV[randomIdx[floor(self.HD_dim/2):]] = 0
        
        return randomHV
    
    def ham_dist(self,vec_a,vec_b): 
        '''
        Description : calculate relative hamming distance    
        INPUTS :
            vec_a: first vector, shape (HD_dim,)
            vec_b: second vector, shape (HD_dim,)          
        OUTPUTS :
            relative hamming distance     
        '''
        return distance.hamming(vec_a,vec_b)  
    
    def cos_similarity(self,vec_a,vec_b): 
        '''
        Description : calculate relative cos_similarity  
        INPUTS :
            vec_a: first vector, shape (HD_dim,)
            vec_b: second vector, shape (HD_dim,)          
        OUTPUTS :
            relative cos_similarity     
        '''
        return np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b)+1e-10)  
        
    def invert(self,vec_a): 
        '''
        Description : invert binary vector    
        INPUTS :
            vec_a: input vector, shape (HD_dim,)        
        OUTPUTS : 
            inverted vetor
        '''
        return 1-vec_a
    
    def permutation(self, HD_sample): 
        '''
        Description : Circular permutation of HD_vector by right shift one bit    
        INPUTS :
            HD_sample: accumulated HD vector, shape (HD_dim,)        
        OUTPUTS :
            HD_sample: permuted HD sample    
        '''
        return np.roll(HD_sample, 1)
        
    def save_model(self, SAVE_PATH, MODEL_NAME='HDC_model.json'):
        print('saving model to {}'.format(SAVE_PATH+MODEL_NAME))
        
        model_dict={}
        model_dict['IM']=self.iM.tolist()
        model_dict['CiM']=self.CiM.tolist()
        model_dict['BIN_AM']=self.Binary_AM.tolist()
        model_dict['INT_AM']=self.Integer_AM.tolist()
        model_dict['max']=self.max.tolist()
        model_dict['min']=self.min.tolist()
        model_dict['value_range']=self.value_range.tolist()
        model_dict['level_range']=self.level_range.tolist()  
        
        json.dump(model_dict, open(SAVE_PATH+MODEL_NAME, 'w'))
    
    def load_model(self, LOAD_PATH, MODEL_NAME='HDC_model.json'):
        print('loading model from {}'.format(LOAD_PATH+MODEL_NAME))
        
        model_dict=json.load(open(LOAD_PATH+MODEL_NAME))
        
        self.iM=np.array(model_dict['IM'])
        self.CiM=np.array(model_dict['CiM'])
        self.Binary_AM=np.array(model_dict['BIN_AM'])
        self.Integer_AM=np.array(model_dict['INT_AM'])
        self.max=np.array(model_dict['max'])
        self.min=np.array(model_dict['min'])
        self.value_range=np.array(model_dict['value_range'])
        self.level_range=np.array(model_dict['level_range'])


class HD_model():
    def __init__(self, n_features, n_classes, HD_dim, n_levels, BINARY=True): 
        '''
        Description:    Initialization of HD classifier    
        INPUTS:
            n_features: input feature dimension  
            n_levels:   number of level for CiM
            n_classes:  Number of classes 
            HD_dim:     HD dimension   
            BINARY:     1 -> quantize to binary ; 0 -> non-binarized
        '''
        # Parameters
        self.n_features = n_features
        self.n_classes = n_classes  
        self.HD_dim = HD_dim    
        self.n_levels = n_levels 
        self.BINARY = BINARY
    
    def fit_AM(self, train_data, labels): 
        '''
        Description:    train an associative memory based on input training data   
        INPUTS:
            train_data: training data, shape(n_samples, n_features)
            labels:     labels for training data, shape(n_samples,)    
        ''' 
        n_samples= train_data.shape[0]
        self.max = np.amax(train_data, axis=0)
        self.min = np.amin(train_data, axis=0)
        self.value_range = self.max-self.min
        self.level_range = self.value_range/self.n_levels   
        n_labels = np.zeros(self.n_classes, dtype = int)
        
        if self.BINARY: # binarized HDC
            for i in range(n_samples):  
                #Spatial Encoder
                spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)     
                val_idx = np.floor((train_data[i] - self.min) / self.level_range).astype(int)
                # the max value in test data may larger than self.max => val_idx > n_level
                val_idx = np.clip(val_idx, 0, self.n_levels-1)
                spatial_array = self.iM ^ self.CiM[val_idx]
                hd_vec, _ = mode(spatial_array, axis=0) # majority function   
                self.AM[labels[i]] += hd_vec.flatten()
                n_labels[labels[i]] += 1
        
            for class_idx in range(self.n_classes):
                self.AM[class_idx] = 1*(self.AM[class_idx] >= floor(n_labels[class_idx]/2)) # majority function
        else: # non-binarized HDC
            for i in range(n_samples):  
                #Spatial Encoder
                spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)     
                val_idx = np.floor((train_data[i] - self.min) / self.level_range).astype(int)
                # the max value in test data may larger tham self.max => val_idx > n_level
                val_idx = np.clip(val_idx, 0, self.n_levels-1)
                spatial_array = self.iM ^ self.CiM[val_idx]
                hd_vec = np.sum(spatial_array, axis=0)             
                self.AM[labels[i]] += hd_vec
                n_labels[labels[i]] += self.n_features
            
            for class_idx in range(self.n_classes):
                self.AM[class_idx] = self.AM[class_idx] - n_labels[class_idx]/2 # bipolar function
    
            
    def predict_AM(self, samples):
        '''
        Description : Predict multiple samples    
        INPUTS :
            samples: feature sample, shape(n_samples, n_features)    
        OUTPUTS :
            y_hat: Vector of estimated output class, shape (n_samples)
            HD_score: Vector of similarities [0,1], shape (n_samples, n_classes)     
        '''      
        n_workers = 10
        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(samples) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(samples)
                else:
                    batch_end = (len(samples) // n_workers) * (i + 1)
                
                batch = samples[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples_AM, args=(batch,))
            pool.close()
            pool.join()
        
        y_hat = []
        HD_score = []
        for result in results:
            predict, score = result.get()
            y_hat += predict  
            HD_score += score          
            
        return y_hat, HD_score   
        
    def preprocess_samples_AM(self, samples):
        n_samples, _ = samples.shape
        HD_score = np.zeros((n_samples,self.n_classes))        
        y_hat = np.zeros(n_samples, dtype = int)
        for sample_idx in range(n_samples):        
            # Transform feature vector to binary HD vector
            hd_vec = self.transform_AM(samples[sample_idx])             
            # calculate HD score for every test class 
            for test_class in range(self.n_classes):
                if self.BINARY:
                    HD_score[sample_idx, test_class] = 1-self.ham_dist(hd_vec, self.AM[test_class]) 
                else:
                    HD_score[sample_idx, test_class] = self.cos_similarity(hd_vec, self.AM[test_class]) 
            # Estimated Class is the one with maximum HD_score 
            y_hat[sample_idx] = np.argmax(HD_score[sample_idx])
        return (y_hat.tolist(), HD_score.tolist())
    
    def transform_AM(self, sample): 
        '''
        Description : transform features into hypervectors   
        INPUTS :
            sample: feature sample, shape(n_features)    
        OUTPUTS :
            hd_vec: hypervecotes, shape(HD_dim)   
        '''
        spatial_array = np.zeros((self.n_features, self.HD_dim), dtype = int)        
        val_idx = np.floor((sample - self.min) / self.level_range).astype(int)
        val_idx = np.clip(val_idx, 0, self.n_levels-1)
        spatial_array = self.iM ^ self.CiM[val_idx]
        if self.BINARY:
            hd_vec, _ = mode(spatial_array, axis=0) # majority function   
        else:
            hd_vec = np.sum(spatial_array, axis=0) - self.n_features/2 # bipolar function     
        return hd_vec.flatten()

    def initMemories(self):
        '''
        Description : initialize the item Memory, continuous item memory and associative memory    
        INPUT:
        OUTPUT:
            iM: item memory for ID of features
            CiM:continuous item memory for value of features
        '''
        self.iM = np.zeros((self.n_features, self.HD_dim), dtype=int) 
        self.CiM = np.zeros((self.n_levels, self.HD_dim), dtype=int)
        # Item Memory: Orthognal mapping
        for i in range(self.n_features):
            self.iM[i] = self.genRandomHV()
            
        # Continuous Memory: Continuous mapping
        initHV = self.genRandomHV()
        currentHV = initHV
        randomIdx = np.arange(self.HD_dim)
        random.shuffle(randomIdx)
        n_flip_bits = floor(self.HD_dim/2/(self.n_levels-1))
        for i in range(self.n_levels):
            self.CiM[i] = currentHV
            startIdx = i*n_flip_bits
            endIdx = (i+1)*n_flip_bits 
            currentHV[randomIdx[startIdx:endIdx]] = self.invert(currentHV[randomIdx[startIdx:endIdx]])
        
        # Assotiative memory (one vector per class)
        self.AM = np.zeros((self.n_classes, self.HD_dim), dtype=int) 

    def clean_AM(self):
        '''
        Description : set AM = 0          
        '''
        # Assotiative memory (one vector per class)
        self.AM = np.zeros((self.n_classes, self.HD_dim), dtype=int)
    
    def genRandomHV(self):
        '''
        Description : generate a random vector with zero mean        
        OUTPUTS :
            randomHV: generated random vector     
        '''
        if self.HD_dim%2 == 1:
            print('Dimension is odd!!')
        else:
            randomHV = np.zeros(self.HD_dim, dtype=int) 
            randomIdx = np.arange(self.HD_dim)
            random.shuffle(randomIdx)
            randomHV[randomIdx[:floor(self.HD_dim/2)]] = 1
            randomHV[randomIdx[floor(self.HD_dim/2):]] = 0
            
            return randomHV
    
    def ham_dist(self,vec_a,vec_b): 
        '''
        Description : calculate relative hamming distance    
        INPUTS :
            vec_a: first vector, shape (HD_dim,)
            vec_b: second vector, shape (HD_dim,)          
        OUTPUTS :
            relative hamming distance     
        '''
        return distance.hamming(vec_a,vec_b)  
        
    def cos_similarity(self,vec_a,vec_b): 
        '''
        Description : calculate relative cos_similarity  
        INPUTS :
            vec_a: first vector, shape (HD_dim,)
            vec_b: second vector, shape (HD_dim,)          
        OUTPUTS :
            relativecos_similarity     
        '''
        return np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))  
 
    def invert(self,vec_a): 
        '''
        Description : invert binary vector    
        INPUTS :
            vec_a: input vector, shape (HD_dim,)        
        OUTPUTS : 
            inverted vetor
        '''
        return 1-vec_a
    
    def permutation(self, HD_sample): 
        '''
        Description : Circular permutation of HD_vector by right shift one bit    
        INPUTS :
            HD_sample: accumulated HD vector, shape (HD_dim,)        
        OUTPUTS :
            HD_sample: permuted HD sample    
        '''
        return np.roll(HD_sample, 1)