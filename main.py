
import os
import pandas as pd
import numpy as np
import tensorflow as tf

#TAKING DATA 
df = pd.read_csv(os.path.join('F:\\Toxic comment\\train.csv\\train.csv'))

#CONVERTING DATA INTO INTEGER FORMAT 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import TextVectorization

x = df['comment_text']
y = df[df.columns[8:]].values

MAx_Words = 200000 #number of words in the vocab
vectorizer = TextVectorization(max_tokens=MAx_Words,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(x.values)

vectorizer_text = vectorizer(x.values)

#CREATING DATASET AND BATCHING THE DATA
dataset = tf.data.Dataset.from_tensor_slices((vectorizer_text,y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

batch_x,batch_y =dataset.as_numpy_iterator().next()

#SPLITING THE DATA
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

#DEEP LEARNING LAYERS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding


model = Sequential()
model.add(Embedding(MAx_Words+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()

history = model.fit(train, epochs=5, validation_data=val)

from matplotlib import pyplot as plt

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()



input_text = vectorizer('i will kill you')
batch = test.as_numpy_iterator().next()
print(model.predict(np.array([input_text])))
res = model.predict(batch_x)
model.save('classify.h5')

input_text = vectorizer('')
batch = test.as_numpy_iterator().next()
print(model.predict(np.array([input_text])))
res = model.predict(np.expand_dims(input_text,0))

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    x_true, y_true = batch
    yhat = model.predict(x_true)

    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)
    print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')