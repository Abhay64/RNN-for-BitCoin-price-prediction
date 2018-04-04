# RNN-for-BitCoin-price-prediction
Recurrent Neural Network (LSTM) by using TensorFlow and Keras in Python for BitCoin price prediction
# Prerequisites
* Python 3.0+
* ML Lib.(numpy, matplotlib, pandas, scikit learn)
* TensorFlow
* Keras
# What are RNNs and why we need that?
The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps (more on this later). Here is what a typical RNN looks like:
<p align="center"> 
<img src="https://cdn-images-1.medium.com/max/1600/0*x1vmPLhmSow0kzvK."></p>

The above diagram shows a RNN being unrolled (or unfolded) into a full network. By unrolling we simply mean that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network.
# RNN Extensions
Over the years researchers have developed more sophisticated types of RNNs to deal with some of the shortcomings of the vanilla RNN model.
**Bidirectional RNN** based on the idea that the output at time t may not only depend on the previous elements in the sequence, but also future elements. For example, to predict a missing word in a sequence you want to look at both the left and the right context. Bidirectional RNNs are quite simple. They are just two RNNs stacked on top of each other. The output is then computed based on the hidden state of both RNNs.
<p align="center">
  <img src=http://www.wildml.com/wp-content/uploads/2015/09/bidirectional-rnn.png></p>
  
**Deep (Bidirectional) RNN** similar to Bidirectional RNNs, only that we now have multiple layers per time step. In practice this gives us a higher learning capacity (but we also need a lot of training data).
<p align="center">
  <img src=http://www.wildml.com/wp-content/uploads/2015/09/Screen-Shot-2015-09-16-at-2.21.51-PM.png></p>
  
# LSTM Cell
Why LSTM ? In a traditional recurrent neural network, `during the gradient back-propagation phase, the gradient signal can end up being multiplied a large number of times (as many as the number of timesteps) by the weight matrix associated with the connections between the neurons of the recurrent hidden layer. This means that, the magnitude of weights in the transition matrix can have a strong impact on the learning process`.

If the weights in this matrix are small (or, more formally, if the leading eigenvalue of the weight matrix is smaller than 1.0), it can lead to a situation called vanishing gradients where the gradient signal gets so small that learning either becomes very slow or stops working altogether. It can also make more difficult the task of learning long-term dependencies in the data. Conversely, if the weights in this matrix are large (or, again, more formally, if the leading eigenvalue of the weight matrix is larger than 1.0), it can lead to a situation where the gradient signal is so large that it can cause learning to diverge. This is often referred to as exploding gradients.

LSTM networks are quite popular these days and we briefly talked about them above. LSTMs don’t have a fundamentally different architecture from RNNs, but they use a different function to compute the hidden state. The memory in LSTMs are called cells and you can think of them as black boxes that take as input the previous state h_{t-1} and current input x_t. Internally these cells decide what to keep in (and what to erase from) memory. They then combine the previous state, the current memory, and the input. It turns out that these types of units are very efficient at capturing long-term dependencies. The repeating module in an LSTM contains four interacting layers.
<p align="center"> 
<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png">
LSTM Cell</p>

# Implementing LSTM
**Importing Data**
```python
#Importing important libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Import the dataset and encode the date
df = pd.read_csv('bitcoin.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
# split data
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
Real_Price = group['Weighted_Price'].mean()
#process data
training_set=df_train.values
training_set=np.reshape(training_set, (len(training_set), 1))
```
**Scaling Data**
```python
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)
X_train= training_set[0:len(training_set)-1]
y_train= training_set[1:len(training_set)]
```
**reshape for keras**
```python
X_train= np.reshape(X_train, (len(X_train), 1, 1))
```
**RNN Layers**
```python
#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNNregressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
```
**To prevent Overfitting we can use DropOutLyaer** but it's a naive model so it's not really important.

**Making Prediction**
```python
# Making the predictions
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
```
**Output**
```python
# Visualising the results
plt.figure(figsize=(25,15), dpi=50, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=20)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
plt.xlabel('Time', fontsize=10)
plt.ylabel('BTC Price(USD)', fontsize=20)
plt.legend(loc=2, prop={'size': 25})
plt.show()
```
<p align="center">
  <img src=https://user-images.githubusercontent.com/26857440/38330299-d36ac12a-386d-11e8-99f1-7086b6aa7997.PNG></p>
  
# Reference
* [JonathanPhoon](https://www.kaggle.com/jphoon/bitcoin-time-series-prediction-with-lstm) Kaggle
* [WILDML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) Tutorial
* [StatsBot](https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f) Blog
* [Kimanalytics](https://github.com/kimanalytics/Recurrent-Neural-Network-to-Predict-Stock-Prices) for Code
