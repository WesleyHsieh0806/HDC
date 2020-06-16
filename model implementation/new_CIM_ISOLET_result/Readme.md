# 中文版本:
* features_beforePCA 跟 feature_after_PCA:
    * 圖片代表著ISOLET中每個feature它們value的distribution
    * 如0.png代表feature0 的value的distribution


* 以下兩個file內容都是記錄CIM_level=5時的狀況:
    1. level.txt 每一行都代表一個feature 其中的數字代表每一個CIM區間有多少筆data
        * 例如1247 1247 1247 1247 1250 代表feature被切成5個區間 每個區間的data量有多少
之所以沒有完全相同是因為有時候你有很多point跟取到的切割點value相同，而value相同就會被分到同一區(如分割第一區 跟第二區的value 為1, 假如有些點本來想分成第二區 但是其value為1,就會被分到第一區)

    2. partition_points.txt/csv
紀錄每一個feature的切割點的value

* label_distribution 系列:
    * 某label在特定feature的分布情形
    可以看到在PCA之後 label1的資料幾乎都會集中在紅線(新CIM 切法)之間，這代表PCA成功使得相同label的data關係更靠近

# English ver:
In this directory, you can see all results obtained from applying a improved CIM partitioning method.
About the details, please go see the "final presentation" ppt


* feature_before_PCA or feature_after_PCA:
    * the value distribution of features in ISOLET dataset 
    * e.g. 0.png->the value distribution of feature 0


* two files below are constructed from setting CIM_level=5
    1.  level.txt : how many data are there in each CIM section
        * e.g.1247 1247 1247 1247 1250 --> the CIM is partitioned into five sections and the first section consists of 1247 samples
the reason why the number of samples in each section is different is that it is possible for many points to have the same value as the partitioning points.
e.g. the partitioning point for section 1 and 2 has value 1, if some points which were supposed to be in section 2 also have value 1, then they will be partitioned into section1.

    2.  partition_points.txt/csv: the value of partitioning points for each feature

* label_distribution:the data distribution of certain label on certain feature
    * you can see that the data mostly lie between the red lines after PCA, which means that PCA successfully improve the data (with same label) with closer relationship