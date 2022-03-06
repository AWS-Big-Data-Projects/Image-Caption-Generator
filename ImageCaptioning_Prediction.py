def prediction_function(img_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import keras
    import tensorflow as tf
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from keras.layers import concatenate, BatchNormalization, Input
    from keras.layers.merge import add
    from keras.utils import to_categorical, plot_model
    import io
    import boto3
    from smart_open import smart_open
    import string
    from keras.preprocessing.image import load_img, img_to_array
    from PIL import Image
    import numpy as np
    from numpy.testing import assert_allclose
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint
    import pickle
    from keras.applications.resnet50 import ResNet50
    from keras.optimizers import Adam
    from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
    from keras.models import Sequential, Model
    from keras.utils import np_utils
    from keras.preprocessing import image, sequence
    from gtts import gTTS
    import IPython.display as ipd
    import pickle

    s3_resource = boto3.resource('s3')

    wordtoix,ixtoword,max_length = pickle.loads(s3_resource.Bucket("projectdata27").Object("project.pkl").get()['Body'].read())

    bucket = s3_resource.Bucket('projectdata27')

    client = boto3.client('s3')
    client.download_file('projectdata27',
                     'model4.h5',
                     'model4.h5')
    # returns a compiled model
    # identical to the previous one
    model = load_model('model4.h5')

    client.download_file('projectdata27',
                     'modelR.h5',
                     'modelR.h5')
    modelR = load_model('modelR.h5')

    k='data/data/Images/'+img_path
    imag = bucket.Object(k)
    img_data = imag.get().get('Body').read()
    img=Image.open(io.BytesIO(img_data))
    img=img.resize((224,224))
    
    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = modelR.predict(im)

    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model3.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)

    print(final)

    language = 'en'
    myobj = gTTS(text=final, lang=language, slow=False)
    myobj.save("project.mp3")
    os.system("project.mp3")
    ipd.Audio("project.mp3", autoplay=True)




if __name__ == '__main__':
    i = input('Enter the image id')
    prediction_function(i)
    



