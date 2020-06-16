# HDC 的實作 --python
* **HD_utils** 
  * 是參考學長怎麼寫平行化的部分  
  * reference version(about multiprocessing) from senior in Access IC lab
* **HDC.py** 
  * 最原始版本 沒有加入multiprocessing   
  * Oldedst version --without multiprocessing
* **HDC_mulpc.py** 
  * 加入multiprocessing的版本     
  * Version1 -- add multiprocessing in testing phases.
* **HDC_mulpl_ISOLET.py** 
  * 為了處理ISOLET DATASET，將class編號做改變的版本  
  * Version2 -- Modify some details such as number_of_class for ISOLET Dataset
* **ISOLET.py** 
  * 處理ISOLET DATASET測試 所用的main檔案
  * Run this .py file for experimenting with ISOLET Dataset.
* **result_between_level.py** 
  * 測試CIM level2~21的結果
  * plot the result of testing different CIM levels
* **main.py** 
  * 使用MNIST做測試時的main範例
  * Run this .py file for experimenting with MNIST Dataset.
* **new_CIM_ISOLET_result**
  * 測試新的CIM切法 以求更好的改善performance(dataset ISOLET)
  * Results from testing new CIM partitioning method.



檔案資料在google mail連接的dropbox可以找
是使用MNIST的資料