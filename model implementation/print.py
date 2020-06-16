import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 這個檔案是把每一個level的data數量做成長條圖
with open('level.txt') as f:
    level_amount = [line.split() for line in f]


def createLabels(data):
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2.,
            height*1.,
            '%d' % int(height),
            ha="center",
            va="bottom",
        )


print(level_amount[418])
A = plt.bar('Level 1', int(level_amount[418][0]), color='r', width=0.4)

B = plt.bar('Level 2', int(level_amount[418][1]), color='y', width=0.4)
C = plt.bar('Level 3', int(level_amount[418][2]), color='g', width=0.4)
D = plt.bar('Level 4', int(level_amount[418][3]), color='b', width=0.4)
E = plt.bar('Level 5', int(level_amount[418][4]), color='k', width=0.4)
createLabels(A)
createLabels(B)
createLabels(C)
createLabels(D)
createLabels(E)
plt.show()
