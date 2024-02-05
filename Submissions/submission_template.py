# COMMENT UNWANTED SECTIONS
team_id = "27964" # Eigen_sharks

#### Change the file paths accordingly

# INPUT FILES
train_path = "/content/denoised_enrollments"
test_path = "/content/denoised_test"
trials_file_path = "/content/signle-channel-trials.txt" # IMPORTANT : "signle-channel-trials.txt" 

# OUTPUT FILES
# Path to cosine similarity file
cos_sim_file_path = "/content/drive/MyDrive/sp_cup_output/cos_sim.txt" 
# Path to cosine distance file
cos_dist_file_path = "/content/drive/MyDrive/sp_cup_output/cos_dist.txt"


#### IMPORTANT: For sp cup submission use cos_dist_file as the output file
# cosine_distance = 1 - cosine_similarity 
# You can generate both two files and check in case of an emergency
cos_sim_file = open(cos_sim_file_path,"a")
cos_dist_file = open(cos_dist_file_path,"a") 

#------------------------------------------------------------#

#Regarding submission
# FOR submission:
# Rename the team_id.txt as the file name to be submitted
# Then zip it 
# Change the name of the zip file to whatever you want. "First_sub.zip","Second_sub.zip" etc 

#------------------------------------------------------------#
# GET DENOISED AUDIO FILES 
import gdown

url = 'https://drive.google.com/uc?id=1S9lZmbaPyGohZ6INfTHhcmOxD9r4DQrR'
output = '/content/denoised_dataset.zip'

gdown.download(url, output, quiet=False)

# EXTRACT THE DENOISED AUDIO FILES CONTAINING AUDIO FILES

# TODO: Implement this below line in python
# !unzip -q "/content/denoised_dataset.zip" -d "/content/"

#------------------------------------------------------------#
import os
# Training 


# LOAD train files
train_files = os.listdir(train_path)

# store in a speaker files dictionary
speaker_files = {} # key: speaker, value: list of audio files

# For each speaker, store the train audio files in a list 
for file_name in train_files:
  [speaker, audio_f_name] = file_name.split("-")
  if speaker not in speaker_files:
    speaker_files[speaker] = []
    speaker_files[speaker].append(file_name)
  else:
    speaker_files[speaker].append(file_name)

# Count the number of files in the train set
speaker_count = 0
train_count_files = 0
for k, v in speaker_files.items():
  # print(k)
  speaker_count += 1
  train_count_files += len(v)

print("Speaker count", speaker_count)
print("Train count_files", train_count_files)
#------------------------------------------------------------#
# TODO : Implement your algorithm here to extract the embedding  - DELETE THE FOLLOWING
import numpy as np
from scipy.io import wavfile
import torch
from torchaudio.compliance import kaldi

from xvector_jtubespeech import XVector

# Function to extract the embedding from an audio file
def extract_embedding(
  model, # xvector model
  wav   # 16kHz mono
):
  # extract mfcc
  wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
  mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
  mfcc = mfcc.unsqueeze(0)

  # extract xvector
  xvector = model.vectorize(mfcc) # (1, 512)
  xvector = xvector.to("cpu").detach().numpy().copy()[0]

  return xvector

import torch
model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)

#------------------------------------------------------------#
# Extract embeddings corresponding to each speaker
speaker_names = list(speaker_files.keys())

# Store the embeddings in a dictionary
speaker_embeddings_dict = {} # key: speaker, value: list of embedding

for speaker in speaker_names:
  audio_path = train_path + "/" + speaker_files[speaker][-1]
  _, wav_file = wavfile.read(audio_path) # 16kHz mono

  speaker_embeddings_dict[speaker] = torch.tensor(extract_embedding(model, wav_file))

#------------------------------------------------------------#
# Load the trials file
trials_file = open(trials_file_path, "r")
trials = trials_file.readlines()
# trials

# function to get the trial information
def get_trial(index:int):
  trial_name = trials[index]
  trial_spk, trial_wav = trial_name.split("-")
  trial_wav = trial_wav.split("\n")
  trial_audio_name = trial_wav[0]
  trial_wav = trial_audio_name + ".wav"

  audio_path = test_path + "/" + trial_wav

  return (trial_name, trial_spk, trial_audio_name, trial_wav, audio_path)

#------------------------------------------------------------#
# CALCULATE THE COSINE SIMILAITY AND DISTANCE BETWEEN THE ENROLLMENT AND TEST FILES
# calculate_cosine_similarity
import torch
import torch.nn.functional as F

cosine_similarity_list = []
cosine_distance_list = []

# For each trail calculate the embedding for each
for i in range(len(trials)):
  # print(get_trial(i))

  trial_key, test_spk, test_audio_name, test_wav, test_audio_path = get_trial(i)

  # Calculate the embedding tensor for test_audio
  audio_path = train_path + "/" + speaker_files[speaker][-1]
  _, wav_file = wavfile.read(test_audio_path) # 16kHz mono
  test_audio_embedding = torch.tensor(extract_embedding(model, wav_file))

  # Test speaker embedding
  test_spk_embedding = speaker_embeddings_dict[test_spk]

  # Calculate cosine similarity
  cos_sim = F.cosine_similarity(test_spk_embedding, test_audio_embedding, dim=0).item()

  # Calculate cosine distance
  cos_dist = 1 - cos_sim
  # print(cos_sim, cos_dist)

  # Round values
  cos_sim_round = round(cos_sim, 7)
  cos_dist_round = round(cos_dist, 7)

  # Save to list
  cosine_similarity_list.append(cos_sim)
  cosine_distance_list.append(cos_dist)
  # break

  # Save to files
  cos_sim_line = test_spk + "\t" + test_audio_name + "\t" + str(cos_sim_round) + "\n"
  cos_dist_line = test_spk + "\t" + test_audio_name + "\t" + str(cos_dist_round) + "\n"
  # print(cos_sim_line)
  # print(cos_dist_line)

  cos_sim_file.write(cos_sim_line)
  cos_dist_file.write(cos_dist_line)

  if i % 1000 == 0:
    print(i)
  # if i == 3:
  #   break

cos_sim_file.close()
cos_dist_file.close()

#### If you get a file format error upon submission, if you can't figure it out, Please contact me : Sasika
#------------------------------------------------------------#