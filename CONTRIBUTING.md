# Contributing workflow

## Setup
### Create Training Examples
Create them using `CreateExamples.ipynb`

### Training Model
1. Log in to google colab [here](https://colab.research.google.com/)
2. Upload `model.ipynb`. This file has a chunk of code that 1) syncs the data from the google shared drive and 2) pulls all files from github repo. 
> Why would I want to pull any file from the repo?
Useful files include `requirements.txt`, which is needed for the google colab notebook to download local dependencies.
3. To save the file in github, `File > Save a copy to Github`. 

## Data
Only store **data needed for the training** (ie exclude `.ipynb`, `.txt` etc) in the google shared drive; cloud has limited space.