from hotel import ytest,ytrain,vocab_len,max_len,sequence_matrix_train,xtest,tok
from hotel import clean_text,lb
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers import Embedding
from keras import regularizers
import tensorflow as tf
from keras.preprocessing import sequence
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from nltk.corpus import vader_lexicon
from nltk.corpus import stopwords


model = Sequential()  
# Embedding: to learn a dense representation of words or categorical data. Vocabulary length,output_dim,mask_zero to handle padding sequences
model.add(Embedding(vocab_len+1,500,input_length=max_len,mask_zero=True))     
#tanh squashes the input values between -1 to 1, it is zero-centered                                          
model.add(LSTM(16,activation='tanh'))                                                                                      # LSTM Layer                                                                                     
model.add(Dense(8,activation='relu',kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l2(0.001)))    # Hidden Layer
model.add(Dropout(0.5)) 
model.add(Dense(3,activation='softmax'))  


model.summary()
# compiling our model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#training the model
trained_model = model.fit(sequence_matrix_train,ytrain,epochs=40)

sequence_matrix_test = sequence.pad_sequences(tok.texts_to_sequences(xtest),maxlen=max_len)

print('Training_loss :',model.evaluate(sequence_matrix_train,ytrain))
print('Testing_loss :',model.evaluate(sequence_matrix_test,ytest))

# training
Y_pred = model.predict(sequence_matrix_test)
print(np.round(Y_pred,3))

Y_pred = [np.argmax(i) for i in Y_pred]


print(classification_report(ytest,Y_pred))
print(confusion_matrix(ytest,Y_pred))

# running the LSTM
def dl_predict(text):
    cleantext = clean_text(text)
    seq = tok.texts_to_sequences([cleantext])
    padded = sequence.pad_sequences(seq)

    pred = model.predict(padded)
    # Get the index of the maximum value in the prediction array
    predicted_index = np.argmax(pred, axis=1)[0]
    # Get the label name using the index
    result = lb.classes_[predicted_index]

    return result


text = 'Such a comfy place to stay with the loved one'

print('Prediction using DNN: {}'.format(dl_predict(text)))