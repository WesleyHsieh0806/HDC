import HDC_mulpc_ISOLET as HDC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# HDC(dim,nof_class,nof_feature)
import time
# 這個檔案用來觀察dimension10000的時候 accuracy隨level的變化


def main():
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

    ''' #train '''

    # 觀察dimension 10000的時候每個level(2~21)的accuracy(每個level做5次) 記得每次要重設np.random.seed

    dimension = 5000
    accuracy = np.zeros([20, 5]).astype(float)

    for level in range(2, 22):
        np.random.seed(0)
        for times in range(5):
            # 執行5次的平均 # HDC(dim,nof_class,nof_feature)

            ISOLET = HDC.HDC(dimension, 26, len(train_x[0]), level)
            # train

            start = time.time()
            ISOLET.train(train_x, train_y)
            ypred = ISOLET.test(test_x)

            # print the accuracy
            acc = ISOLET.accuracy(y_pred=ypred, y_true=test_y)
            accuracy[level-2][times] = acc
            print("Train time:{:.2f} Dimension:{:^5} Level:{:^4} Accuracy:{:.8f}".format(time.time()-start,
                                                                                         dimension, level, acc))
    plt.plot(range(2, 22), np.average(accuracy, axis=1),
             'bo-', label='dimension 5000')
    plt.legend()
    plt.xlabel('level')
    plt.ylabel('Accuracy')
    plt.savefig('./isolet_result/Accuracy_between_level5000.png')
    plt.close()
    result = pd.DataFrame(np.average(accuracy, axis=1), index=[
                          i for i in range(2, 22)], columns=[dimension])
    result.to_csv('./isolet_result/Accuracy_between_level5000.csv')

    dimension = 2000
    accuracy = np.zeros([20, 5]).astype(float)

    for level in range(2, 22):
        np.random.seed(0)
        for times in range(5):
            # 執行5次的平均 # HDC(dim,nof_class,nof_feature)

            ISOLET = HDC.HDC(dimension, 26, len(train_x[0]), level)
            # train

            start = time.time()
            ISOLET.train(train_x, train_y)
            ypred = ISOLET.test(test_x)

            # print the accuracy
            acc = ISOLET.accuracy(y_pred=ypred, y_true=test_y)
            accuracy[level-2][times] = acc
            print("Train time:{:.2f} Dimension:{:^5} Level:{:^4} Accuracy:{:.8f}".format(time.time()-start,
                                                                                         dimension, level, acc))
    plt.plot(range(2, 22), np.average(accuracy, axis=1),
             'bo-', label='dimension 2000')
    plt.legend()
    plt.xlabel('level')
    plt.ylabel('Accuracy')
    plt.savefig('./isolet_result/Accuracy_between_level2000.png')
    plt.close()
    result = pd.DataFrame(np.average(accuracy, axis=1), index=[
                          i for i in range(2, 22)], columns=[dimension])
    result.to_csv('./isolet_result/Accuracy_between_level2000.csv')


if __name__ == "__main__":
    main()
