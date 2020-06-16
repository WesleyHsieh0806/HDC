import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import HDC_new_CIM as HDC


def csv_to_XY(train_csv_path, test_csv_path):
    ''' read the csv file and get the train/test data '''
    with open(train_csv_path, mode='r', encoding='utf-8-sig') as f:
        # Read training data
        print("Reading Data from {}...".format(train_csv_path))
        train_x = []
        train_y = []
        for line in f:
            line = line.strip().split(',')
            train_x.append(line[:-1])
            train_y.append(line[-1])
    # transform it into numpy array
    train_x = np.array(train_x).astype(np.float)
    train_y = np.array(train_y).astype(np.int).reshape(-1, 1)
    print("Size of Training Data:{}".format(len(train_x)))

    with open(test_csv_path, mode='r', encoding='utf-8-sig') as f:
        print("Reading Data from {}...".format(test_csv_path))
        test_x = []
        test_y = []
        for line in f:
            line = line.strip().split(',')
            test_x.append(line[:-1])
            test_y.append(line[-1])
    test_x = np.array(test_x).astype(np.float)
    test_y = np.array(test_y).astype(np.int).reshape(-1, 1)
    print("Size of Testing Data:{}".format(len(test_x)))

    return train_x, train_y, test_x, test_y


def partition_CIM_by_value(number_of_CIM, number_of_data, train_x):
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
        # feature_partition_point[0] = [number_of_CIM-1 points]
        feature_partition_point[feature] = []
        this_feature = train_x[:, feature]
        # sort the feature value: this_feature_index[i-1] means the index of the i-th smallest value in this_feature
        this_feature_index = np.argsort(this_feature)
        for i in range(1, number_of_CIM):
            # print(number_of_data//number_of_CIM*i-1)
            feature_partition_point[feature].append(
                this_feature[this_feature_index[number_of_data//number_of_CIM*i-1]])
    return feature_partition_point


def main():
    train_csv_path = r'../../HDC (for undergrduate)/Data/ISOLET/train.csv'
    test_csv_path = r'../../HDC (for undergrduate)/Data/ISOLET/test.csv'

    train_x, train_y, test_x, test_y = csv_to_XY(train_csv_path, test_csv_path)

    if not os.path.isdir('../new_CIM_ISOLET_result'):
        os.makedirs('../new_CIM_ISOLET_result')

    '''new CIM + PCA'''
    # parameters setup
    number_of_class = 26
    dimension = [2000, 5000, 10000]
    # use "accuracy" to record the results
    accuracy = dict()
    accuracy[2000] = np.zeros([20, 5]).astype(float)
    accuracy[5000] = np.zeros([20, 5]).astype(float)
    accuracy[10000] = np.zeros([20, 5]).astype(float)
    for dim in dimension:
        for level in range(2, 22):
            np.random.seed(0)
            for times in range(5):
                # 執行5次的平均 # HDC(dim,nof_class,nof_feature, level)

                ISOLET_new_CIM = HDC.HDC(dim, number_of_class,
                                         len(train_x[0]), level, PCA_projection=True)
                # train

                start = time.time()
                ISOLET_new_CIM.train(train_x, train_y)
                y_pred = ISOLET_new_CIM.test(test_x)

                # print the accuracy
                acc = ISOLET_new_CIM.accuracy(y_true=test_y, y_pred=y_pred)
                accuracy[dim][level-2][times] = acc
                print("Execution time:{:.2f} Secs| Dimension:{:^5} Level:{:^4} Accuracy:{:.8f}".format(time.time()-start,
                                                                                                       dim, level, acc))
    # write the result in csv files
    result = pd.DataFrame(np.average(accuracy[2000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[2000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level2000_newCIM_PCA.csv')

    result = pd.DataFrame(np.average(accuracy[5000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[5000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level5000_newCIM_PCA.csv')

    result = pd.DataFrame(np.average(accuracy[10000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[10000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level10000_newCIM_PCA.csv')
    ''' '''

    ''' new CIM no PCA '''
    # parameters setup
    number_of_class = 26
    dimension = [2000, 5000, 10000]
    # use "accuracy" to record the results
    accuracy = dict()
    accuracy[2000] = np.zeros([20, 5]).astype(float)
    accuracy[5000] = np.zeros([20, 5]).astype(float)
    accuracy[10000] = np.zeros([20, 5]).astype(float)
    for dim in dimension:
        for level in range(2, 22):
            np.random.seed(0)
            for times in range(5):
                # 執行5次的平均 # HDC(dim,nof_class,nof_feature, level)

                ISOLET_new_CIM = HDC.HDC(dim, number_of_class,
                                         len(train_x[0]), level, PCA_projection=False)
                # train

                start = time.time()
                ISOLET_new_CIM.train(train_x, train_y)
                y_pred = ISOLET_new_CIM.test(test_x)

                # print the accuracy
                acc = ISOLET_new_CIM.accuracy(y_true=test_y, y_pred=y_pred)
                accuracy[dim][level-2][times] = acc
                print("Execution time:{:.2f} Secs| Dimension:{:^5} Level:{:^4} Accuracy:{:.8f}".format(time.time()-start,
                                                                                                       dim, level, acc))
    # write the result in csv files
    result = pd.DataFrame(np.average(accuracy[2000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[2000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level2000_newCIM.csv')

    result = pd.DataFrame(np.average(accuracy[5000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[5000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level5000_newCIM.csv')

    result = pd.DataFrame(np.average(accuracy[10000], axis=1), index=[
                          lev for lev in range(2, 22)], columns=[10000])
    result.to_csv(
        '../new_CIM_ISOLET_result/Accuracy_between_level10000_newCIM.csv')
    # '''get the partition points of each feature'''
    # number_of_CIM = 10
    # partition_points = partition_CIM_by_value(
    #     number_of_CIM, len(train_x), train_x)
    ''' Draw the Value Distribution of Feature'''
    # for feature in range(0, 3):
    #     # the feature which we want to draw the distribution
    #     feature_distribution = train_x[:, feature]
    #     sns.distplot(feature_distribution, kde=True, hist=False,  color='g')

    #     plt.title("Feature{} Distribution".format(feature))
    #     plt.xlabel("Value of Feature")
    #     plt.ylabel("Density")
    #     for i in range(len(partition_points[feature])):
    #         if(i == len(partition_points[feature])-1):
    #             # draw the partition points(equal information)
    #             plt.axvline(x=partition_points[feature][i], linewidth=0.65,
    #                         color='r', label='Equal # of data')
    #             # draw the normal partition points(equal distance)
    #             plt.axvline(x=(np.min(feature_distribution)+(i+1)*(np.max(feature_distribution)-np.min(feature_distribution))/number_of_CIM), color='b', linewidth=0.65,
    #                         label='Equal Distance')
    #         else:
    #             plt.axvline(x=partition_points[feature][i], linewidth=0.65,
    #                         color='r')
    #             plt.axvline(x=(np.min(feature_distribution)+(i+1)*(np.max(feature_distribution)-np.min(feature_distribution))/number_of_CIM), color='b', linewidth=0.65,
    #                         )
    #     plt.legend()
    #     plt.savefig(
    #         '../new_CIM_ISOLET_result/{}}.png'.format(str(feature)))
    #     plt.close()

    ''' Draw the Value Distribution of Feature and the distribution of certain label'''
    # 目的是看看會不會partition point會把特定label
    # for feature in range(0, 3):
    #     # the feature which we want to draw the distribution
    #     feature_distribution = train_x[:, feature]
    #     plt.hist(feature_distribution,  color='g', label='All Data')
    #     for label in range(1, 2):
    #         label_distribution = train_x[:,
    #                                      feature].reshape(-1, 1)[train_y == label]
    #         plt.hist(label_distribution,
    #                  label='label '+str(label))

    #     plt.title("Feature{} Distribution".format(feature))
    #     plt.xlabel("Value of Feature")
    #     plt.ylabel("Amount")
    #     for i in range(len(partition_points[feature])):
    #         if(i == len(partition_points[feature])-1):
    #             # 為了強調最左邊跟最右邊的垂直線，將line width設為4
    #             # draw the partition points(equal information)
    #             plt.axvline(x=partition_points[feature][i], linewidth=4,
    #                         color='r', label='Equal # of data')
    #             # draw the normal partition points(equal distance)
    #             plt.axvline(x=(np.min(feature_distribution)+(i+1)*(np.max(feature_distribution)-np.min(feature_distribution))/number_of_CIM), color='b', linewidth=4,
    #                         label='Equal Distance')
    #         elif i == 0:
    #             # 為了強調最左邊跟最右邊的垂直線，將line width設為4
    #             # draw the partition points(equal information)
    #             plt.axvline(x=partition_points[feature][i], linewidth=4,
    #                         color='r')
    #             # draw the normal partition points(equal distance)
    #             plt.axvline(x=(np.min(feature_distribution)+(i+1)*(np.max(feature_distribution)-np.min(feature_distribution))/number_of_CIM), color='b', linewidth=4,
    #                         )
    #         else:
    #             plt.axvline(x=partition_points[feature][i], linewidth=0.5,
    #                         color='r')
    #             plt.axvline(x=(np.min(feature_distribution)+(i+1)*(np.max(feature_distribution)-np.min(feature_distribution))/number_of_CIM), color='b', linewidth=0.5,
    #                         )
    #     plt.legend()
    #     plt.savefig(
    #         '../new_CIM_ISOLET_result/label_distr_onfeature{}.png'.format(str(feature)))
    #     plt.close()


if __name__ == "__main__":
    main()
