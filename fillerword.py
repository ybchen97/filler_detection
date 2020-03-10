import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from tf_utils import *
# %matplotlib inline

###
# TODO: Change all positive data to .wav
###

POSITIVE_DIRECTORY = "./raw_data/positive_data/"
BACKGROUND_DIRECTORY = "./raw_data/background_data/"
NEGATIVES_DIRECTORY = "./raw_data/google_dataset/"
NEGATIVES_TRUNCATED_DIRECTORY = "./raw_data/google_dataset_truncated/"

# Load raw audio files for speech synthesis
def load_raw_audio(positiveDirectory, backgroundDirectory, negativesDirectory):
    BACKGROUND_DIRECTORY_IN_GOOGLE_DATASET = "_background_noise_/"
    positives = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(positiveDirectory):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(positiveDirectory + filename)
            positives.append(activate)
    for filename in os.listdir(backgroundDirectory):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(backgroundDirectory + filename)
            backgrounds.append(background)
    for directory in os.listdir(negativesDirectory):
        if os.path.isdir(os.path.join(negativesDirectory, directory)) and directory != BACKGROUND_DIRECTORY_IN_GOOGLE_DATASET: # Excludes background directory in google dataset
            for filename in os.listdir(negativesDirectory + directory):
                if filename.endswith("wav"):
                    negative = AudioSegment.from_wav(negativesDirectory + directory + "/" + filename)
                    negatives.append(negative)
    return positives, negatives, backgrounds

# Load audio segments using pydub 
# positives, negatives, backgrounds = load_raw_audio(POSITIVE_DIRECTORY, BACKGROUND_DIRECTORY, NEGATIVES_DIRECTORY)
positives, negatives, backgrounds = load_raw_audio(POSITIVE_DIRECTORY, BACKGROUND_DIRECTORY, NEGATIVES_TRUNCATED_DIRECTORY)

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
print("activate[0] len: " + str(len(positives[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(positives[1])))     # Different "activate" clips can have different lengths