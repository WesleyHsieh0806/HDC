# -*- coding=utf-8 -*-
import time
import numpy as np
import pandas as pd
from math import floor
import sys
from multiprocessing import Pool
import multiprocessing as mul
from sklearn.decomposition import PCA
# you can use HDC.help() to realize what parameters are necessary to be filled in
# In this version, we partition CIM levels based on the amount of information
# In this version, we also update PCA functions which we can project the features of data into lower dimension by.

''' Author:謝承延 
    Dataset: ISOLET
'''
np.random.seed(0)


class HDC:

    # attritbue :
    # nof_feature
    # nof_class
    # nof_dimension
    # level
    # IM_vector
    # CIM_vector
    # Prototype_vector
    # self.maximum = np.max(x, axis=0) train data 每個 feature的最大值
    # self.minimum = np.min(x, axis=0)
    # self.difference = self.maximum - self.minimum
    def __init__(self, dim=10000, nof_class=0, nof_feature=0, level=21, PCA_projection=False):
        ''' initialize some necessary attribute and data'''
        # 1.feature數量(how many IM vector?) -->not necessary
        # 2.vector dimension
        # 3.class數量(how many prototype vector)
        self.level = int(level)
        self.nof_feature = int(nof_feature)
        self.nof_dimension = int(dim)
        self.nof_class = int(nof_class)
        # determine whether use PCA to project the features or not
        self.PCA_projection = PCA_projection

    def train(self, x, y):
        ''' use train data x y to train prototype vector'''
        # x,y須為ndarray
        # 此步驟 需要創建IM(字典) CIM(字典) Prototype vector(字典)
        if self.PCA_projection:
            x = self.PCA(x)
        self.nof_feature = len(x[0])
        self.init_IM_vector()
        self.init_CIM_vector()
        self.init_prototype_vector()

        # Obtain the partition points of CIM
        self.partition_points = self.partition_CIM_by_value(
            self.level, len(x), x)

        spatial_vector = np.zeros(
            (len(x), 1, self.nof_dimension)).astype(int)
        # 因為要將x每個feature的value根據數值切成level個等級 所以要記住某個範圍的數值
        # 以level=21為例子 假如數值範圍是0~20 就是0是一個level
        # 但這裡實作 我打算將0~20/21 當作level 0
        # 要將maximum這些存下來 才能用在test
        self.maximum = np.max(x, axis=0)
        self.minimum = np.min(x, axis=0)
        self.difference = self.maximum - self.minimum + 1e-8

        self.encoder_spatial_vector(x, spatial_vector, y)

        for CLASS in range(1, self.nof_class+1):
            # 這裡是train的最後了 把prototype vector再度變回1 -1
            self.Prototype_vector[CLASS][self.Prototype_vector[CLASS] > 0] = 1
            self.Prototype_vector[CLASS][self.Prototype_vector[CLASS] < 0] = -1

    def PCA(self, x):
        print(
            "Project {}-dim features to dimension".format(len(x[0])), end=' ')
        self.pca = PCA(n_components=32)
        self.pca.fit(x)
        x = self.pca.fit_transform(x)
        print("{}".format(len(x[0])))
        return x

    def PCA_test(self, x):
        x = self.pca.transform(x)
        return x

    def test(self, test_x):
        ''' return the predicted y array(class) '''
        # 首先要將test data經過同樣encoder 並產生query vector
        if self.PCA_projection:
            test_x = self.PCA_test(test_x)
        query_vector = np.zeros(
            (len(test_x), 1, self.nof_dimension)).astype(int)
        self.y_pred = np.zeros((len(test_x), 1))
        # 因為要將x每個feature的value根據數值切成level個等級 所以要記住某個範圍的數值
        # 以level=21為例子 假如數值範圍是0~20 就是0是一個level
        # 但這裡實作 我打算將0~20/21 當作level 0

        ''' encoding and prediction'''
        # 利用multiprocessing 的pool 去做多進程(多個CPU核心去做運算)
        # Pool() 代表利用cpu最大核心數量去跑
        # 用 starmap function他就會自動分配資料讓核心去跑 並回傳每次function的結果成一個list
        # 這裡要注意用多核心去跑的時候 即使在function裡面改了self.pred的value 也不會改動到self.pred的value
        # 可以在裡面改self.y_pred[0][0] 並在這裡print看看就可知道
        start = time.time()
        pool = Pool()

        self.y_pred = np.array([pool.starmap_async(self.encoder_query_vector, [
            (test_x[data, :], query_vector[data, :], data) for data in range(len(test_x))]).get()]).reshape((len(test_x), 1))
        pool.close()
        pool.join()
        end = time.time()

        return self.y_pred

    def result_to_csv(self, output='./result.csv'):
        '''output the result of prediction as csv file'''
        with open(output, 'w') as f:
            f.write('data,class\n')
            for data in range(len(self.y_pred)):
                f.write('{},{}\n'.format(str(data), self.y_pred[data][0]))

    def accuracy(self, y_true, y_pred=None):
        '''return the accuracy of the prediction'''
        same = 0
        if y_pred is None:
            # if y_pred is not given, use self.y_pred we obtain in test function
            if self.y_pred:
                y_pred = self.y_pred

        for data in range(len(y_true)):
            if y_pred[data, 0] == y_true[data, 0]:
                same += 1
        return same / len(y_pred)

    def cosine_similarity(self, Query_vector, Prototpye_vector):
        '''return cos(A,B)=|A'*B'|=|C| C is the sum of element'''
        # in paper, they normalize the A B to A' B', but here I am not going to normalize it
        # 這個function只處理1對1的cosine similarity
        cos_sim = np.dot(Query_vector, Prototpye_vector.T)
        return cos_sim

    def most_similar_class(self, Query_vector):
        '''return the number of class(1~self.nof_class)which is the most similar to query_vector'''
        maximum = -100
        max_class = -1
        for Class in range(1, self.nof_class+1):
            similarity = self.cosine_similarity(
                Query_vector, self.Prototype_vector[Class])

            if similarity > maximum:
                maximum = similarity
                max_class = Class

        return max_class

    def init_IM_vector(self):
        ''' 創建feature數量個vector element 為bipolar(1,-1)'''
        self.IM_vector = {}
        for i in range(1, self.nof_feature+1):
            # np.random.choice([1,-1],size) 隨機二選一 size代表選幾次
            self.IM_vector['feature'+str(i)] = np.random.choice(
                [1, -1], self.nof_dimension).reshape(1, self.nof_dimension).astype(int)

    def init_CIM_vector(self):
        ''' slice continuous signal into 21 self.level '''
        # 每往上一個self.level就改 D/2/(self.level-1)個bit

        self.CIM_vector = {}
        nof_change = self.nof_dimension//(2*(self.level-1))
        if self.nof_dimension/2/(self.level-1) != floor(self.nof_dimension/2/(self.level-1)):
            print("warning! D/2/(level-1) is not an integer,", end=' ')
            print(
                "change the dim so that the maximum CIM vector can be orthogonal to the minimum CIM vector")

        self.CIM_vector[0] = np.random.choice(
            [1, -1], self.nof_dimension).reshape(1, self.nof_dimension).astype(int)

        for lev in range(1, self.level):
            # 每個level要改D/2/(level-1)個bit 並且從 D/2/(level-1) * (lev-1)開始改
            # 這裡用到的觀念叫做deep copy 非常重要
            # 只copy value而不是像python assign一樣是share 物件
            self.CIM_vector[lev] = self.CIM_vector[lev-1].copy()
            for index in range(nof_change * (lev-1), nof_change * (lev)):

                self.CIM_vector[lev][0][index] *= -1

    def init_prototype_vector(self):
        '''construct prototype vector'''
        if self.nof_class <= 0:
            print("number of class should pe positive integer!")
            sys.exit(2)
        # 創建nof_class數量個prototype vector  element 為0 並在之後慢慢加上去
        self.Prototype_vector = {}
        # 為配合ISOLET的DATASET 將class vector改成編號1~class
        for i in range(1, self.nof_class+1):
            self.Prototype_vector[i] = np.zeros(
                [1, self.nof_dimension]).astype(int)

    def partition_CIM_by_value(self, number_of_CIM, number_of_data, train_x):
        ''' Return a dictionary records the partition points of each feature\n
            partition_point[feautre] = [point1, point2, ..., point n_of_CIM-1]\n
            with all values smaller or equal to the i-th point belonging to the i-th part
        '''

        # number_of_CIM: how many sections should the CIM be partitioned into
        # we need 20 points to partition CIM into 21 parts
        # e.g.number_of_data/number_of_CIM | number_of_data/number_of_CIM | number_of_data/number_of_CIM| |
        # all values smaller or equal to the i-th point belong to the i-th part
        # sort the value first
        # the i-th point should be number_of_data/number_of_CIM * i-th smallest value (i:1~number_of_CIM-1)

        feature_partition_point = {}
        for feature in range(0, len(train_x[0])):
            # feature_partition_point[i] = [number_of_CIM-1 partition points]
            feature_partition_point[feature] = []
            this_feature = train_x[:, feature]
            # sort the feature value: this_feature_index[i-1] means the index of the i-th smallest value in this_feature
            this_feature_index = np.argsort(this_feature)
            for i in range(1, number_of_CIM):
                # print(number_of_data//number_of_CIM*i-1)
                feature_partition_point[feature].append(
                    this_feature[this_feature_index[number_of_data//number_of_CIM*i-1]])

        return feature_partition_point

    def encoder_query_vector(self, test_x, query_vector, data):
        ''' construct the query vector of each data, and construct the prediction result for each data in y_pred array'''

        for feature in range(1, self.nof_feature+1):
            # 套用新的CIM level切法(每一區CIM level要有相同data資訊量)
            # 若該feature的value <= 第i+1個 partition points(self.partition[feature-1][i]) i:0~level-2
            # 代表lev=i
            lev = (-np.inf)
            for i in range(self.level-1):
                if test_x[feature-1] <= self.partition_points[feature-1][i]:
                    lev = i
                    break
            if lev == (-np.inf):
                # 進入這裡代表level屬於最後一區塊
                # 例如:  0 | 1 | 2 |....| self.level-2 | 這裡
                lev = (self.level-1)

            query_vector += self.IM_vector['feature' +
                                           str(feature)] * self.CIM_vector[0 + lev]

            if feature == 1 and (self.nof_feature % 2) == 0:
                # 若feature數量為偶數 需要再補上E1*V1*E2*V2這個向量
                # 用跟上面相同的方法去找到LEV
                LEV = (-np.inf)
                for i in range(self.level-1):
                    if test_x[feature] <= self.partition_points[feature][i]:
                        LEV = i
                        break
                if LEV == (-np.inf):
                    # 進入這裡代表level屬於最後一區塊
                    # 例如:  0 | 1 | 2 |....| self.level-2 | 這裡
                    LEV = (self.level-1)

                query_vector += self.IM_vector['feature' +
                                               str(feature)] * self.CIM_vector[0+lev] * self.IM_vector['feature' + str(feature+1)] * self.CIM_vector[0+LEV]
        # 這裡有更動 先將query vector做binarize 可能導致accuracy下降
        query_vector[query_vector > 0] = 1
        query_vector[query_vector < 0] = -1
        y_pred = self.most_similar_class(query_vector)

        return y_pred

    def encoder_spatial_vector(self, x, spatial_vector, y):
        '''contruct spatial vector and prototyper vector'''

        for data in range(0, len(x)):
            print("[{}/{}] Dimension:{} Level:{}".format(data,
                                                         len(x), self.nof_dimension, self.level), end='\r')
            # data會是0~最後 字典的key會叫做'featurei'
            for feature in range(1, self.nof_feature + 1):
                # 套用新的CIM level切法(每一區CIM level要有相同data資訊量)
                # 若該feature的value <= 第i+1個 partition points(self.partition[feature-1][i]) i:0~level-2
                # 代表lev=i
                lev = (-np.inf)
                for i in range(self.level-1):
                    if x[data, feature-1] <= self.partition_points[feature-1][i]:
                        lev = i
                        break
                if lev == (-np.inf):
                    # 進入這裡代表level屬於最後一區塊
                    # 例如:  0 | 1 | 2 |....| self.level-2 | 這裡
                    lev = (self.level-1)

                # 每一筆data都會形成一個 Spatial vector S = IM1*CIM1 + IM2*CIM2 + ... 1 2 3代表feature號碼
                spatial_vector[data] += self.IM_vector['feature' +
                                                       str(feature)] * self.CIM_vector[0+lev]

                if feature == 1 and (self.nof_feature % 2) == 0:
                    # 若feature數量為偶數 需要再補上E1*V1*E2*V2這個向量
                    # 用跟上面相同的方法去找到LEV
                    LEV = (-np.inf)
                    for i in range(self.level-1):
                        if x[data, feature] <= self.partition_points[feature][i]:
                            LEV = i
                            break
                    if LEV == (-np.inf):
                        # 進入這裡代表level屬於最後一區塊
                        # 例如:  0 | 1 | 2 |....| self.level-2 | 這裡
                        LEV = (self.level-1)

                    spatial_vector[data] += self.IM_vector['feature' +
                                                           str(feature)] * self.CIM_vector[0+lev] * self.IM_vector['feature' +
                                                                                                                   str(feature+1)] * self.CIM_vector[0+LEV]

            # 以y的數值 決定這是哪一個class (1~self.nof_class)
            whichclass = int(y[data])
            self.Prototype_vector[whichclass] += spatial_vector[data]

    def help():
        '''model usage instruction'''
        print("The necessary attribute when you initialize your HDC model like variable = HDC():")
        print("nof_dimension (please enter the dimension of vector)")
        print("nof_class (please enter number of class)")
        print("{:-^40}".format("Usage"))
        print("a=HDC(dim,nof_class,nof_feature,level,PCA_projection)")
        print("a.train(x,y)")
        print("a.test(test_x)")
        print("a.result_to_csv('file name')")


if __name__ == "__main__":
    pass
