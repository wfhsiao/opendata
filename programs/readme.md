<pre>
programs for helping classify RUMOR/NOT_RUMOR messages.

The parameters need to be tweaked:
cnn: 0, 1, 2, 3
lstm: 0, 1, 2, 3
vocabulary size: 1000, 2000, 3000
maximum sentence length: 100, 200, 250, 300, 320
embedding vector length: 64, 100, 128, 164, 192
dropout rate: .1, .2, .3, .4, .5
neuron number for
lstm layer1
lstm layer2
lstm layer3
neuron number for 
cnn layer1
cnn layer2
cnn layer3

After some trials I obtain the following network architecture which can beat traditional machine learning methods.

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 250, 100)          300000    
_________________________________________________________________
conv1d (Conv1D)              (None, 250, 32)           9632      
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 125, 32)           0         
_________________________________________________________________
spatial_dropout1d (SpatialDr (None, 125, 32)           0         
_________________________________________________________________
bidirectional (Bidirectional (None, 125, 100)          33200     
_________________________________________________________________
dropout (Dropout)            (None, 125, 100)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 125, 50)           25200     
_________________________________________________________________
dropout_1 (Dropout)          (None, 125, 50)           0         
_________________________________________________________________
flatten (Flatten)            (None, 6250)              0         
_________________________________________________________________
dense (Dense)                (None, 32)                200032    
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 568,097
Trainable params: 568,097
Non-trainable params: 0

正確率及f1值可達：
80.38954928182268 0.7751206645502221
88.79626159333262 0.5011329920412152
(第一行為正確率的平均值及標準差，第二行為f1值)請再自行四捨五入

五等份交叉驗證法重複執行十次(不同的亂數種子)，執行時間為828.74秒：
--- 828.7377300262451 seconds elasped for the whole program ---

詳細內容請查閱：good_cnnbilstm3000sl250es100dr04l1_50l2_25.txt
</pre>
