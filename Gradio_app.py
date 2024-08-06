
import os
import pandas as pd
import numpy as np
import tensorflow as tf



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

import gradio as gr


model = tf.keras.models.load_model('Classify.h5')
input_str = vectorizer('hey i freaking hate you')
res = model.predict(np.expand_dims(input_str,0))
df.columns[8:-1]
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[8:]):
        text +='{}: {}\n'.format(col, results[0][idx]>0.5)

    return text
interface=gr.Interface(fn=score_comment,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment here'),
                         outputs='text')
interface.launch(share=True)