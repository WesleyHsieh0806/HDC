# -*- coding=utf-8 -*-
import time
import numpy as np
import pandas as pd
from math import floor
import sys
from multiprocessing import Pool
import multiprocessing as mul
# you can use HDC.help() to realize what parameters are necessary to be filled in
# self回傳object資訊 沒有self回傳class資訊
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
    def __init__(self, dim=10000, nof_class=0, nof_feature=0, level=21):
        ''' initialize some necessary attribute and data'''
        # 1.feature數量(how many IM vector?) -->not necessary
        # 2.vector dimension
        # 3.class數量(how many prototype vector)
        self.level = int(level)
        self.nof_feature = int(nof_feature)
        self.nof_dimension = int(dim)
        self.nof_class = int(nof_class)

    def train(self, x, y):
        ''' use train data x y to train prototype vector'''
        # x,y須為ndarray
        # 此步驟 需要創建IM(字典) CIM(字典) Prototype vector(字典)
        self.nof_feature = len(x[0])
        self.init_IM_vector()
        self.init_CIM_vector()
        self.init_prototype_vector()
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

        for CLASS in range(0, self.nof_class):
            # 這裡是train的最後了 把prototype vector再度變回1 -1
            self.Prototype_vector[CLASS][self.Prototype_vector[CLASS] > 0] = 1
            self.Prototype_vector[CLASS][self.Prototype_vector[CLASS] < 0] = -1

    def test(self, test_x):
        ''' return the predicted y array(class) '''
        # 首先要將test data經過同樣encoder 並產生query vector
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
        print(end-start)
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
        '''return the number of class(0~self.nof_class-1)which is the most similar to query_vector'''
        maximum = -100
        max_class = -1
        for Class in range(0, self.nof_class):
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
        for i in range(0, self.nof_class):
            self.Prototype_vector[i] = np.zeros(
                [1, self.nof_dimension]).astype(int)

    def encoder_query_vector(self, test_x, query_vector, data):
        ''' construct the query vector of each data, and construct the prediction result for each data in y_pred array'''

        for feature in range(1, self.nof_feature+1):
            # 因為maximum minimum使用的是train data的資料 要小心超過範圍
            if test_x[feature-1] > self.maximum[feature-1]:
                test_x[feature-1] = self.maximum[feature-1]
            elif test_x[feature-1] < self.minimum[feature-1]:
                test_x[feature-1] = self.minimum[feature-1]

            # 先看這個數值跟這個feature的minimum差多少  算出他的lev 藉此給他相對應的CIM vector
            lev = (test_x[feature-1] - self.minimum[feature-1]
                   )//((self.difference[feature-1])/self.level)

            query_vector += self.IM_vector['feature' +
                                           str(feature)] * self.CIM_vector[0 + lev]

            if feature == 1 and (self.nof_feature % 2) == 0:
                # 因為maximum minimum使用的是train data的資料 要小心超過範圍
                if test_x[feature] > self.maximum[feature]:
                    test_x[feature] = self.maximum[feature]
                elif test_x[feature] < self.minimum[feature]:
                    test_x[feature] = self.minimum[feature]

                # 當self.nof_feature有偶數個 需要補上E1*V1*E2*V2項
                LEV = (test_x[feature] - self.minimum[feature]
                       )//((self.difference[feature]) / (self.level))

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
            # data會是0~最後 字典的key會叫做'featurei'
            for feature in range(1, self.nof_feature + 1):
                # 先看這個數值跟這個feature的minimum差多少  算出他的lev 藉此給他相對應的CIM vector
                lev = (x[data, feature-1] - self.minimum[feature-1]
                       )//((self.difference[feature-1])/(self.level))

                # 每一筆data都會形成一個 Spatial vector S = IM1*CIM1 + IM2*CIM2 + ... 1 2 3代表feature號碼
                spatial_vector[data] += self.IM_vector['feature' +
                                                       str(feature)] * self.CIM_vector[0+lev]

                if feature == 1 and (self.nof_feature % 2) == 0:
                    # 若feature數量為偶數 需要再補上E1*V1*E2*V2這個向量

                    LEV = (x[data, feature] - self.minimum[feature]
                           )//((self.difference[feature]) / (self.level))

                    spatial_vector[data] += self.IM_vector['feature' +
                                                           str(feature)] * self.CIM_vector[0+lev] * self.IM_vector['feature' +
                                                                                                                   str(feature+1)] * self.CIM_vector[0+LEV]

            # 以y的數值 決定這是哪一個class (0~self.nof_class-1)
            whichclass = int(y[data])

            self.Prototype_vector[whichclass] += spatial_vector[data]

    def help():
        '''model usage instruction'''
        print("The necessary attribute when you initialize your HDC model like variable = HDC():")
        print("nof_dimension (please enter the dimension of vector)")
        print("nof_class (please enter number of class)")
        print("{:-^40}".format("Usage"))
        print("a=HDC(dim,nof_class,nof_feature,level)")
        print("a.train(x,y)")
        print("a.test(test_x)")
        print("a.result_to_csv('file name')")


if __name__ == "__main__":
    pass
