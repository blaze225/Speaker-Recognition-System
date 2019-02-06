
# Importing
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

# Global parameters
BASE_PATH="../spkrec_TIMIT/"
voiced_threshold = 0.05
num_mixtures= 32
em_iterations = 100
deltas=False
testing_samples_per_speaker=3

train_subdirectories = os.listdir(BASE_PATH+"train_wavdata")[1:]
test_subdirectories = os.listdir(BASE_PATH+"test_wavdata")[1:]

def delta_mfcc_features(mfcc):
    delta_mfcc=np.zeros(np.shape(mfcc))
    delta_delta_mfcc = np.zeros(np.shape(mfcc))
    for i in range(13):
        prev=max(i-1,0)
        nxt=min(i+1,12)
        delta_mfcc[:,i]=0.5*(mfcc[:,nxt] - mfcc[:,prev])

    for i in range(13):
        prev=max(i-1,0)
        nxt=min(i+1,12)
        delta_delta_mfcc[:,i]=0.5*(delta_mfcc[:,nxt] - delta_mfcc[:,prev])
    return np.concatenate((mfcc, delta_mfcc, delta_delta_mfcc),axis=1)

GMMs_train=[]

print "-> TRAINING FEATURE EXTRACTION AND GMM TRAINING "
for mdir in sorted(train_subdirectories):
    # print "--> speaker/directory:", mdir
    subdir=os.listdir(BASE_PATH+"train_wavdata/"+mdir)
    mfcc_train=[]
    for f in subdir:
        [Fs, x] = audioBasicIO.readAudioFile(BASE_PATH+"train_wavdata/"+mdir+"/"+f)
        
        # Apply Voiced/Unvoiced region filter based on mean energy
        energy = np.multiply(x,x)
        mean_energy = np.mean(energy)
        Threshold=voiced_threshold*mean_energy
        j=0
        voiced=[] # voiced signal
        for i in range(0,len(x),100):
            if sum(energy[i:i+100])/100 > Threshold:
                voiced[j:j+100]=x[i:i+100]
                j+=100
        if len(voiced)> 0.030*Fs:
            # Extracting MFCC features
            features= np.transpose(audioFeatureExtraction.stFeatureExtraction(voiced, Fs, 0.030*Fs, 0.015*Fs)[8:21])
            # Delta MFCC features
            if deltas:
                features = delta_mfcc_features(features)

            if len(mfcc_train)==0:
                mfcc_train=features
            else:
                mfcc_train=np.concatenate((mfcc_train,features),axis=0)
    # Train a GMM for each speaker
    gmm = GaussianMixture(n_components=num_mixtures,covariance_type='diag',max_iter=em_iterations).fit(mfcc_train)
    GMMs_train.append((mdir,gmm))

print "-> GMM TRAINING COMPLETE"

num_speakers = len(GMMs_train)

# Extract MFCC for testing data

MFCC_features_test=[]
print "-> FEATURE EXTRACTION FOR TESTING"
for mdir in sorted(test_subdirectories):
    # print "--> speaker/directory: ",mdir
    subdir=os.listdir(BASE_PATH+"test_wavdata/"+mdir)
    for f in subdir:
        [Fs, x] = audioBasicIO.readAudioFile(BASE_PATH+"test_wavdata/"+mdir+"/"+f)
        # Apply Voiced/Unvoiced region filter based on mean energy
        energy = np.multiply(x,x)
        mean_energy = np.mean(energy)
        Threshold=voiced_threshold*mean_energy
        j=0
        voiced=[]
        for i in range(0,len(x),100):
            if sum(energy[i:i+100])/100 > Threshold:
                voiced[j:j+100]=x[i:i+100]
                j+=100
        if len(voiced)> 0.030*Fs:
            features=np.transpose(audioFeatureExtraction.stFeatureExtraction(voiced, Fs, 0.030*Fs, 0.015*Fs)[8:21])
            if deltas:
                features = delta_mfcc_features(features)

            MFCC_features_test.append((mdir,features))

# Run GMMs on testing data
print "-> TESTING "
predicted_labels=[]
for test in MFCC_features_test:
    scores=[]
    gmms=[]
    for gmm in GMMs_train:
        scores.append(gmm[1].score(test[1]))
        gmms.append(gmm[0])
    max_index=scores.index(max(scores))
    predicted_labels.append((max_index,gmms[max_index],test[0]))

# Calculate score
y_true= sorted([i for i in range(num_speakers)]*testing_samples_per_speaker)
y_pred= [tup[0] for tup in predicted_labels]

score=[]
for t,p in zip(y_true,y_pred):
    if t==p:
        score.append(1)
    else:
        score.append(0)    

print "Accuracy:", sum(score)/float(testing_samples_per_speaker*num_speakers)


