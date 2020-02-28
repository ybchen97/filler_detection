# CS3244 Project

## Important details to note
**Project deadline:** Week 13, Friday, April 17th  
**Weekly meetings:** Friday 9.30am @ Cinnamon College

We are all busy, but let's all try to help each other out as much as possible when we do have the spare time :)

## Introduction
Verbal communications, from everyday conversations to public speakering, are often filled with filler words: meaningless word, phrase, or sound that marks hesitation in speech. How can speakers, who interested in ridding the usage of filler words, efficiently track the frequency of filler words used? Inspired by the trigger word detection model in Dr Andrew Ng's deep learning programming assignment, [Insert project name] is a filler word detection model that takes in an audio clip and outputs the frequency of filler words used.

## Projected timeline
### Week 7
- [x] Email Prof ryan
- [ ] Create Data Sets
- [ ] Reading and understanding notebook 

### Create Data Sets

**Recall.** To synthesize a single training example, you will:
1. Pick a random 10 second background audio clip
2. Randomly insert 0-4 audio clips of filler words into this 10sec clip
3. Randomly insert 0-2 audio clips of negative words into this 10sec clip
4. Because you had synthesized the word "activate" into the background clip, you know exactly when in the 10 second clip the "activate" makes its appearance

> What is the filler word?

**Basically**: we chose this as it has more syllables than other filler words.

> How long does the filler and negative words need to be?

~1s. The exact timing does not matter, as we take `len(audio_clip)`. These clips will then be randomly inserted into the background audio clip.

> How good should the quality be?

Ideally, there should be no background sound and the pronunciation of each syllable is clear.

> What kind of background sounds?

Preferably atypical of Singapore environment (city? gardens?quiet room?). Background sound should be snipped to 10s already (i'm concerned that snipping it during the training of model will be computationally costly).

### Reading and Understanding Notebook
To aid your understanding, I tried to explain the intuition of the neural network architecture. It's found below in the FAQ.

### Week 8
1. Data pre-processing
   - Splitting spectogram into discrete timestamps
2. Data synthesizing   
   - Overlaying positive, negative and background audio clips

### Week 9+ (KIV)
1. Model building & testing
2. Find resources
   - GPU
3. Reiterate minimum viable product (MVP)

## FAQ

> Please explain the architecture.

**CONV-1D:** Extracts low-level features (low-level: tone, frequency; high level: recognising syllables) and generates an output of smaller dimensions which speeds up the model.

**GRU:**
1. **Sequence Models:** Share features learned across different positions of the layers (thus the name "sequence")
2. **Bidirectional RNN:** For the sequence model to learn in both directions.
> “Teddy Roosevelt was a great President!”. The meaning of “Teddy” is determined by the word that comes after (ie “bears” or “Roosevelt”); the current iteration of sequence model only allows a neuron to learn from the words before it. To resolve this, we have bidirectional RNNs.
1. **GRU:** neural network to capture long range dependencies. 

> Q: I'm not too sure why we use GRU for trigger word detection model though

**Batch Norm:** Makes gradient descent quicker by normalising each hidden layer of each mini batch.

> Mini-batch is the middle ground between stochastic and batch gradient descent: split the training examples into m groups, and every step of gradient descent is trained over each of these m groups.

**ReLU:** Piecewise linear function (google to see its graph). This activation function that is computationally faster than sigmoid and tanh.

**Dropout:** Randomly "drops out" the output of each neuron, thus having a regularising effect ("don't put all your eggs in a basket" intuition).

**Dense Layer:** Fully connected neural network. Usually used at the deeper layers as the dimensions are much lower then, thus you can afford to use and exploit the "power" of the full neural network.

**Sigmoid:** Used at the end to achieve a more accurate binary output (ie is the trigger word detected or not?)


## Credits
![Andrew Ng's Trigger Word Detection Assignment](https://github.com/Kulbear/deep-learning-coursera/blob/master/Sequence%20Models/Trigger%20word%20detection%20-%20v1.ipynb)
