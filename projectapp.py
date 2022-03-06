def prediction_function():
    import streamlit as st
    import numpy as np
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
    import pickle
    import IPython.display as ipd

    st.title("Image upload")
    image_file = st.file_uploader("Upload Image", type=["jpg"])

    if image_file is not None:
        st.image(Image.open(image_file),width=200, height=200)

    if st.button("Generate Caption"):
        s3_resource = boto3.resource('s3')
        wordtoix,ixtoword,max_length = pickle.loads(s3_resource.Bucket("projectdata27").Object("project.pkl").get()['Body'].read())

        bucket = s3_resource.Bucket('projectdata27')
        client = boto3.client('s3')
        client.download_file('projectdata27',
                     'model4.h5',
                     'model4.h5')
        # returns a compiled model
        # identical to the previous one
        model = load_model('model4.h5',compile=False)
        client.download_file('projectdata27',
                     'modelR.h5',
                     'modelR.h5')
        modelR = load_model('modelR.h5',compile=False)
        img=Image.open(image_file)
        img=img.resize((224,224))
        im = img_to_array(img)
        im = np.expand_dims(im, axis=0)
        im = modelR.predict(im)
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([im,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
            final = in_text.split()
            final = final[1:-1]
            final = ' '.join(final)
        st.success('Description:  '.format(final))
        st.write(final)

        language = 'en'
        myobj = gTTS(text=final, lang=language, slow=False)
        myobj.save("project.mp3")
        audio_file = open('project.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')


if __name__ == '__main__':
    
    prediction_function()
    


