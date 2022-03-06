## Image_captioning
# Image Caption Generator using Deep Learning
Image captioning is a research area of Artificial Intelligence (AI) that deals with image understanding and a language description for that image. Image understanding needs to detect and recognize objects. It also needs to understand scene type or location, object properties and their interactions. Generating well-formed sentences requires both syntactic and semantic understanding of the language. In this project, a framework is developed leveraging the capabilities of artificial neural networks to “caption an image based on its significant features”.The generation process of image semantics not only understands the objects or scene recognition in the image, but also has the ability to analyze their states, understand the relationship among them and generate a correct sentence. The model would make use of Convolution neural networks to read image data and Long Short Term Memory (LSTM) for learning sentences/captions for image. The Flickr8K dataset is used to train the model.   
  
## Problem Statement:  
The objective is to generate semantically and syntactically description about object or scene recognition in the image, and to understand the relationship among them.  
  
## Dataset:
1. **Flickr8k_Dataset**: Contains a total of 8092 images in JPG format with different shapes and sizes. Of which 6000 are used for training, 1000 for test and 1000 for development.
2. **Flickr8k_text** : Contains text files describing train_set ,test_set. Flickr8k.token.txt contains 5 captions for each image i.e. total 40460 captions.  
  
## Built With:  
Jupyter Interface on EMR  

## What is Image Captioning?  
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.  
![1](https://user-images.githubusercontent.com/63635084/105141690-89dce100-5b1f-11eb-8d64-103f5905aa5c.png)  
  
## Model Architecture:  
![WhatsApp Image 2021-01-20 at 1 34 45 PM](https://user-images.githubusercontent.com/63635084/105146486-1094bc80-5b26-11eb-9fbc-04128defd40d.jpeg)  
  
## Execution:  
  
The problem needs two models and seamless integration between them. The network can be viewed as a combination of encoder and decoder. Encoder would be a convolutional neural network(CNN). Image is processed by CNN layer and features are extracted. End of CNN layer is connected to a Long short-term memory(LSTM) networks, a special kind of Recurrent Neural Network(RNN). LSTM’s are capable of learning long term dependencies. The model is built using Keras, a deep learning library in python. Keras is a high-level library that is above Tensorflow. The API is very simple and makes use of Tensorflow backend.  

Transfer learning is used for CNN implementation. Transfer learning is a major topic in machine learning that involves storing knowledge from one model and applying to another problem. The reason we use pretrained network is because, CNN models are difficult to train from scratch and it could be very computationally expensive that it takes several hours on GPU. In the scientific community, it is very common to use pretrained model on larger dataset and then using the model as a feature extractor.  
  
The output of image model acts as input to language model. To understand the captions under the images a Recurrent Neural Network(RNN) is used to solve the problem. Long Short-term memory (LSTM), which is a variation of RNN is used. LSTM works better and has powerful update equation and backpropagation. LSTM is a language model and decoder trained on feature vectors. LSTM’s had phenomenal influence and success in different problems like language modelling, speech recognition, translation etc.  
  
LSTM picks part of image and maps to the appropriate word in the caption. An embedding layer is created to get a vector representation for each word in the caption. Then the output vector is given as input to LSTM for the model to learn the neighbouring words for each word. Then the LSTM output is converted to fixed dimension using dense layer. Now, the outputs from both Language Model and Image model are combined, and input the vector to LSTM. LSTM learns the different captions for that image in training phase. The LSTM output is converted to the size of vocabulary size using the dense layer and activate the model using activation method. In testing phase, LSTM predict the captions for the image. LSTM predicts next word for the given image with the partial caption available at that stage.  
  
## Network/Model:    
![1](https://user-images.githubusercontent.com/63635084/105346093-1b754d00-5c0b-11eb-92df-a0efcdc415b1.JPG)  
![cnn](https://user-images.githubusercontent.com/63635084/105346106-203a0100-5c0b-11eb-8d7d-3a1ea1679817.JPG)   
  
## Code:
[Click here for Code](https://github.com/nileshsingal/IMAGE_CAPTION_GENERATOR/blob/master/Image_Captioning.py)  
  
  
## Author:  
- [NileshSingal](https://github.com/nileshsingal)

