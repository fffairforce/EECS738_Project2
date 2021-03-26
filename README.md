# EECS738_Project2
text generator base on HMM

## Overview

HMM model is used to train on given text corpus

Forward-backward algorithm is used to identify probability matrices and generate text out of it

Viterbi-algorithm is used to make text prediction base on a sequential words, here I set it as two for simplification

## Dataset

- Shakespeare Plays : https://www.kaggle.com/kingburrito666/shakespeare-plays

- AllLines.txt - This file contains the text corpus for model building

## Running the Code

1. jupyter notebook is provided to have a quick glance on my process and results

2. I tried to play around with building .py project for whole repositery installation. It should be working if you clone the 
 
repository and run localy by executing textgenerator.py then follow the input instructions

## Method

1. data sorting
  
  Txt data is encoded by removing penctuations and spliting by space, In terms of downsizing the data sample, in this case I transfer 
  
all vocabulary to lower cases.  

2. initialize probability matrices
  
    A dictionary is created to store the initial probability of words given in the text corpus

3. Transition Probability Matrix
    
    This matrix stores transition probabities between the states. One Vocabulary is consider as a single state. The transition 
   
probability represents the probability of state transition from word A to word B.

4. Emission Probability Matrix

    This matrix stores emission probabities between the observations and states. The transition 
    
probability represents the probability of emission from observation i to state t.

5. Smoothing

    Note that zeros may occur a lot in the matrix since a lot of states are not connected, so a laplace smoothing technique is 
    
used to minimize the error

6. Text generation

    Forward-backward algoritm is used to find probabilities of each states, then base on the most promising states, a paragraph of 
    
text can be generated.

7. text prediction
    
    First, input of a sequence of 2 words is required to run through the process. Viterbi Algorithm is applied here to calculate 
    
the best path sentence which involve the input observations. Which help to predict the sentence.

## Results

1. Text Generator

![image](https://user-images.githubusercontent.com/42806161/112563582-a60f2f80-8da7-11eb-8e63-139086df0c13.png)

2. text predictor

![image](https://user-images.githubusercontent.com/42806161/112563631-b921ff80-8da7-11eb-8d21-6555a8b79d1c.png)

## Conclusions

I was debating on the algorithm within forward-baclward progess should I use a set initail probability matrix and fixed trans_Mat, 

emis_Mat or running a EM algorithm starting with randomnize those matrices.

## Reference

* Forward Backward Algorithm [https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm]

* Viterbi Algorithm [https://en.wikipedia.org/wiki/Viterbi_algorithm]
