MNIST
    - MNIST是機器學習很常使用的小型手寫辨識數據集，每張圖片所對應到都是一個數字，總共有0~9十個數字
    - 28*28維的圖片

MNIST_CNN_4_feature.csv
    - 60000*4
    - MNIST原本60000筆的圖片經過CNN處理後剩下4維的資訊
    
MNIST_CNN_4_label.csv
    - 60000
    - 每筆資料所對應的label (class)，共有10種class
    
注意事項
    - 記得切training dataset和validation dataset
    - 剛開始在train HDC的時候，不用全部60000筆資料下去train，可以先拿個200筆試試看