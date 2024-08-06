
from flask import Flask, render_template, send_file, request, url_for, redirect
import tensorflow as tf 
import pandas as pd
import os

from zmq import Message

app = Flask(__name__)

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

model = tf.keras.models.load_model('Classify.h5')
   
@app.route("/")
def Home():
    return render_template('index.html') 

@app.route('/output',methods=['POST'])
def output():

    Message=request.form['Message']

    vectorized_comment = vectorizer([Message])
    results = model.predict(vectorized_comment)
   
    return render_template('output.html',Comment=Message,Results=results)

if __name__ == '__main__':
    app.run( port='6060',debug=True)    
