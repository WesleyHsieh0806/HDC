import numpy as np
import pandas as pd
import HDC_mulpc_ISOLET as HDC
import matplotlib.pyplot as plt
import time
import os


def main():
    ''' partition dataset (the last feature is label) '''

    def csv_to_XY(train_csv, test_csv):
        train_x, train_y, test_x, test_y = [], [], [], []
        '''為何要用utf-8-sig來讀取? 因為utf-8會把BOM當成是文件內容 就會使得string資料無法轉換成float
        而utf-8-sig則會把 BOM分開來讀 就不會有問題'''
        with open(train_csv, mode='r', encoding='utf-8-sig')as f:
            for line in f:
                train_x.append(
                    line.strip().split(',')[:-1])
                train_y.append(
                    line.strip().split(',')[-1])
        train_x = np.array(train_x).astype(float)
        train_y = np.array(train_y).astype(int).reshape(-1, 1)
        with open(test_csv, 'r', encoding='utf-8-sig')as f:
            for line in f:
                test_x.append(
                    line.strip().split(',')[:-1])
                test_y.append(
                    line.strip().split(',')[-1])
        test_x = np.array(test_x).astype(float)
        test_y = np.array(test_y).astype(int).reshape(-1, 1)

        return train_x, train_y, test_x, test_y

    train_csv_path = r'D:\電子書\專題\HDC\model implementation\HDC (for undergrduate)\Data\ISOLET\train.csv'
    test_csv_path = r'D:\電子書\專題\HDC\model implementation\HDC (for undergrduate)\Data\ISOLET\test.csv'

    train_x, train_y, test_x, test_y = csv_to_XY(train_csv_path, test_csv_path)

    ''' train '''

    ''' 目標 1.觀察dimension 100~10000對結果的影響 間隔為1000  100+x為當前dimension
            2.每個dimension level從2~21的結果
    '''
    x = 0
    ''' train_acc將acc分成不同level紀錄'''
    train_acc = {}
    # LEV_down代表最小level
    # LEV_up 代表最大切幾個level+1
    LEV_down = 2
    LEV_up = 22
    for level in range(LEV_down, LEV_up):
        # initialize
        train_acc[level] = []

    ''' dimension也要記錄下來 方便作圖'''
    # dim_down代表最小dimension
    # dim_up代表最大dimension+1
    # dimn_distance代表測試的dimension間隔 例如測試100 500 則distance為400
    dim_down = 100
    dim_up = 10001
    dim_distance = 1000
    dimension = []
    for i in range(dim_down-100, dim_up, dim_distance):
        if i == 0:
            dimension.append(100)
        else:
            dimension.append(i)

    ''' 紀錄level最大的時候 每個dimension要train+test多久'''
    train_time = []

    ''' use while loop to obtain multiple results '''
    while (dim_down+x) < dim_up:
        # contruct HDC model
        # dim class feature level
        for level in range(LEV_down, LEV_up):
            ISOLET = HDC.HDC(dim_down+x, 26, len(train_x[0]), level)

            start = time.time()
            # training and acquire prediction array
            ISOLET.train(train_x, train_y)
            ypred = ISOLET.test(test_x)

            end = time.time()

            # 若level為最大將train test 花的時間記錄下來
            if level == LEV_up-1:
                train_time.append(end-start)

            # print the accuracy
            acc = ISOLET.accuracy(y_pred=ypred, y_true=test_y)
            print("Dimension:{:^5} Level:{:^4} Accuracy:{:.8f}".format(
                dim_down+x, level, acc))

            # record the accuracy
            train_acc[level].append(acc)

        if x == 0:
            x += 900
        else:
            x += dim_distance

    '''將結果紀錄在csv 跟作圖'''
    # 先建立資料夾isolet_result
    if not os.path.isdir('./isolet_result'):
        os.mkdir('./isolet_result')
    # csv
    acc_result = pd.DataFrame.from_dict(train_acc)
    acc_result.index = dimension
    acc_result.to_csv('./isolet_result/isolet_result.csv')
    total_time = pd.DataFrame(
        np.array(train_time).reshape(1, -1), columns=dimension)
    total_time.to_csv('./isolet_result/isolet_time.csv')
    # 製圖
    for level in range(LEV_down, LEV_up):
        plt.plot(dimension, train_acc[level], 'bo-', label='level='+str(level))
        plt.legend(loc='best')

        plt.xlabel('dimension')
        plt.ylabel('Accuracy')
        plt.savefig('./isolet_result/Accuracy_level_'+str(level)+'.png')
        plt.close()
    plt.plot(dimension, train_time, 'bo-')

    plt.xlabel('dimension')
    plt.ylabel('time')
    plt.savefig('./isolet_result/dimension_time.png')
    plt.close()


if __name__ == '__main__':
    main()
