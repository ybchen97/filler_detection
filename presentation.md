# CS3244 Presentation to Prof Bryan
**Group Members:** Arushi Gupta, Chaitanya Baranwal, Chen Yuan Bo, Wang Zi Yun, Wong Jia Hwee, Law Ann Liat Larry

## Heuristics in Choosing Project Idea
1. **Feasibility:** Working product within 7 weeks.
2. **Novelty:** Fulfill criteria "Are you hungry enough for success?" and "Will Singapore (or maybe just Bryan) invest in you?)

## Introduction
Verbal communications, from everyday conversations to public speakering, are often filled with filler words: meaningless word, phrase, or sound that marks hesitation in speech. How can speakers, who interested in ridding the usage of filler words, efficiently track the frequency of filler words used? Inspired by the trigger word detection model in Dr Andrew Ng's deep learning programming assignment, **[Insert project name]** is a filler word detection model that takes in an audio clip and outputs the frequency of filler words used.

## Related Work
1. **Google Speech API:** speech-to-text. application (does not capture filler words). Link [here](https://cloud.google.com/speech-to-text)
2. **Lyrebird AI:** Removes filler words from audio recording. Link [here](https://www.coywolf.news/content/podcast-filler-word-detection-removal/)

**Our unique point:** Instead of removing, we highlight the frequency of the filler words.

## Tech Stack
1. Keras

## Broad Implementation Details
1. **Data Preprocessing:** Manual recording of filler word clips. Negative clips avaible in the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html): 65,000 one-second long utterances of 30 short words, by thousands of different people, contributed by members of the public through the AIY website. 
2. **Data Synthesis:** Overlaying of filler word, negative, and background audio clips and labelling of the created clips.
3. **Model:** Conv layer + GRU layer + dense layer. Architecture here

![Neural Network Architecture](/images/nn_architecture.png)

> We'll be working off the architecture from the assignment.

## Product Development Phases
1. **MVP:** Single filler word (`basically`) detection model.
2. **Phase 1:** Multiple filler word detection model.
3. **Phase 2:** Model that works on audio with normal speed (ideally Prof Bryan's webcast audio)

## Projected timeline
### Week 7
1. Create datasets
2. Reading and understanding notebook
3. Email Prof Bryan (done)
   - GPU
   - Feasibility
   - Time frame of plan

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

## Questions/Challenges
1. In the assignment, the model was trained on a GPU. Where do we get access to GPU?
2. Might not work with continuous stream of text. (possibly circumvent by slowing the audio down before inputting into the model?)

## Credits
[Andrew Ng's Trigger Word Detection Assignment](https://github.com/Kulbear/deep-learning-coursera/blob/master/Sequence%20Models/Trigger%20word%20detection%20-%20v1.ipynb)

TensorFlow and AIY teams for the negative word datasets
