## Importing Required Libraries

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
from tensorflow.keras.applications.inception_v3 import InceptionV3
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


## Loading Text Data"""

token_path = 's3://projectdata27/data/data/captions.txt'
text = smart_open(token_path, 'r', encoding = 'utf-8').read()


## Preprocessing Text Data

descriptions = dict()
for line in text.split('\n'):
    # split line by white space
    tokens = line.split(',')
    
    # take the first token as image id, the rest as description
    image_id, image_desc = tokens[0], tokens[1:]
    
    # extract filename from image id
    image_id = image_id.split('.')[0]
    
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    if image_id not in descriptions.keys():
        descriptions[image_id] = list()
    descriptions[image_id].append(image_desc)

print(descriptions['3534548254_7bee952a0e'])


# prepare translation table for removing punctuation

table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)

del descriptions['']


t=[]
token_path = 's3://projectdata27/data/data/trainimages.txt'
train = smart_open(token_path, 'r', encoding = 'utf-8').read() 
for line in train.split('\n'):
    t.append(line[:-4])

t.remove('')


vocabulary = set()
for key in t:
    [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))

# Create a list of all the training captions
all_captions = []
for key, val in descriptions.items():
    if key in t:
        for cap in val:
            all_captions.append(cap)


# Consider only words which occur at least 10 times in the corpus

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ' % len(vocab))


#find the maximum length of a description in a dataset

max_length = max(len(des.split()) for des in all_captions) 
max_length

despc = dict() 
for key, des_list in descriptions.items():
    if key in t:
        despc[key] = list()
        for line in des_list:
            desc = 'startseq ' + line + ' endseq'
            despc[key].append(desc)


# word mapping to integers

ixtoword = {} 
wordtoix = {} 
  
ix = 1
for word in vocab: 
    wordtoix[word] = ix 
    ixtoword[ix] = word 
    ix += 1


# convert a dictionary of clean descriptions to a list of descriptions

def to_lines(descriptions):
 all_desc = list()
 for key in t:
  [all_desc.append(d) for d in descriptions[key]]
 return all_desc
# calculate the length of the description with the most words
def max_length(descriptions):
 lines = to_lines(descriptions)
 return max(len(d.split()) for d in lines)
# determine the maximum sequence length
max_length = max_length(despc)
print('Max Description Length: %d' % max_length)

s3 = boto3.resource('s3')

bucket = s3.Bucket('projectdata27')

temp = captions[10].split(",")
image = bucket.Object('data/data/Images/'+temp[0])
img_data = image.get().get('Body').read()
img=Image.open(io.BytesIO(img_data))
plt.imshow(img)

for ix in range(len(tokens[temp[0]])):
    print(tokens[temp[0]][ix])

modelR = load_model('modelR.h5')

train_path='s3://projectdata27/data/data/trainimages.txt'
x_train = smart_open(train_path, 'r', encoding = 'utf-8').read().split("\n")


x_train.remove('')


def preprocessing(img_path):
    k='data/data/Images/'+img_path
    imag = bucket.Object(k)
    img_data = imag.get().get('Body').read()
    img=Image.open(io.BytesIO(img_data))
    img=img.resize((224,224))
    
    
    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    return im

train_data = {}

for ix in x_train:
    img = preprocessing(ix)
    train_data[ix] = modelR.predict(img).reshape(2048)

train_data


# load glove vectors for embedding layer

vocab_size=1650
embeddings_index = {} 


g = smart_open('s3://projectdata27/glove.6B.200d.txt', 'r', encoding = 'utf-8').read()

for line in g.split("\n"): 
    values = line.split(" ") 
    word = values[0] 
    indices = np.asarray(values[1: ], dtype = 'float32') 
    embeddings_index[word] = indices

len(embeddings_index)

emb_dim= 200
emb_matrix = np.zeros((vocab_size, emb_dim)) 
for word, i in wordtoix.items(): 
    emb_vec = embeddings_index.get(word) 
    if emb_vec is not None: 
        emb_matrix[i] = emb_vec 
emb_matrix.shape

X1, X2, y = list(), list(), list() 
for key, des_list in despc.items():
    if key in t:
        pic = train_data[key + '.jpg']
        for cap in des_list:
            seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                # store 
                X1.append(pic) 
                X2.append(in_seq)
                y.append(out_seq)

X2 = np.array(X2) 
X1 = np.array(X1) 
y = np.array(y)


ip1 = Input(shape = (2048, )) 
fe1 = Dropout(0.2)(ip1) 
fe2 = Dense(256, activation = 'relu')(fe1) 
ip2 = Input(shape = (max_length, )) 
se1 = Embedding(vocab_size, emb_dim, mask_zero = True)(ip2) 
se2 = Dropout(0.2)(se1) 
se3 = LSTM(256)(se2) 
decoder1 = add([fe2, se3]) 
decoder2 = Dense(512, activation = 'relu')(decoder1) 
outputs = Dense(vocab_size, activation = 'softmax')(decoder2) 
model3 = Model(inputs = [ip1, ip2], outputs = outputs)
 
model3.layers[2].set_weights([emb_matrix]) 
model3.layers[2].trainable = False
model3.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])


# define the checkpoint

filepath = "model3.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model3.fit([X1,X2], y, epochs=50, batch_size=256, callbacks=callbacks_list)

def feature_extraction(img_path):
    k='data/data/Images/'+img_path
    imag = bucket.Object(k)
    img_data = imag.get().get('Body').read()
    img=Image.open(io.BytesIO(img_data))
    img=img.resize((224,224))
    
    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = modelR.predict(im)
    return im

def final_caption(photo):
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
    return final



## ResNet50

from IPython.core.display import display, HTML
display(HTML("""<a href="http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006">ResNet50 Architecture</a>"""))

modelR = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
modelR.summary()

R=load_model('modelR.h5')

modelR = load_model('modelR.h5')



## Progressive Loading

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n=0

embedding_dim=200
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(512, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model2 = Model(inputs=[inputs1, inputs2], outputs=outputs)

model2.layers[2].set_weights([emb_matrix])
model2.layers[2].trainable = False

model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

epochs = 10
number_pics_per_bath = 3
steps = len(despc)//number_pics_per_bath

for i in range(epochs):
    
    generator = data_generator(despc, train_data, wordtoix, max_length, number_pics_per_bath)
    
    
    model2.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
model2.save('model2.h5')


