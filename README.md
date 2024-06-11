
# Muscle Classification from sEMG Signals 

This repository contains the code and accompanying materials for my thesis project, which focuses on the classification of muscle activity using surface electromyography (sEMG) signals.

## Dataset
- The ENABL3S dataset can be downloaded from this link: https://figshare.com/articles/dataset/Benchmark_datasets_for_bilateral_lower_limb_neuromechanical_signals_from_wearable_sensors_during_unassisted_locomotion_in_able-bodied_individuals/5362627  (6GB Approx)


## Additional Files & Folders 
- Google Drive Link: https://drive.google.com/drive/folders/1zQPpjBJFwS2pnqS6kQK_Avy3FJhyjYbL?usp=sharing 
- pickled_dataframes: contains 10 pickled dataframes, each containing the cleaned EMG data of a subject. (7 columns, each is a muscle)
- pickled_detected_bursts: contains 7 pickled files that save the burst detection results of running the burst detection algorithms on the subject datatframes. this saves about 30mins of Preprocessing.
- tfrecords: contains the datasets to be used for the training. 


## Notebook
- `Preprocessing_and_7_Muscle_Identification.ipynb`: OG code: Data Loading, Visualisation, Preprocessing, TFRecords Creation, Model Training and Results Visualisation. 

[//]: # (## Hydra Script)
[//]: # (- `HYDRA_Script_7_Muscle_Identification.py`: Script to run the model on Hydra.)
[//]: # (- `submit.sh`: Submit script used to run the model's job on hydra &#40;super helpful as packages on hydra are tricky!&#41; )

## Notebook Structure 
- Generating dataframes from the EMG csv files 
   - Generating pickled dataframes 
   - Visualising the EMG signal 
   - Analysing the EMG signal
- Burst detection from EMG signal
   - Extracting bursts
   - Generating pickle burst detection files 
   - Loading pickle burst detection files
   - Visualising the detected bursts 
   - Analysing the detected bursts
- TFrecords creation
   - Separating the LOO subject
   - Extracting 1000ms burst windows
   - Creating the tfrecords
   - Training/Validation Split
   - Shapes Verification
   - Muscle bursts samples 
- Model Implementation: 
   - Parameters
   - Wandb initialisation 
   - Training
   - Confusion Matrices 
