
# Muscle Classification from sEMG Signals 

This repository contains the code and accompanying materials for my thesis project, which focuses on the classification of muscle activity using surface electromyography (sEMG) signals.

## Dataset
- The ENABL3S dataset can be downloaded from this link: https://figshare.com/articles/dataset/Benchmark_datasets_for_bilateral_lower_limb_neuromechanical_signals_from_wearable_sensors_during_unassisted_locomotion_in_able-bodied_individuals/5362627  (6GB Approx)


## Additional Files & Folders 
- Google Drive Link: https://drive.google.com/drive/folders/1zQPpjBJFwS2pnqS6kQK_Avy3FJhyjYbL?usp=sharing 
- pickled_dataframes: contains 10 pickled dataframes, each containing the cleaned EMG data of a subject. (7 columns, each is a muscle)
- pickled_detected_bursts: contains 7 pickled files that save the burst detection results of running the burst detection algorithms on the subject datatframes. this saves about 30mins of Preprocessing.
- tfrecords: contains the datasets to be used for the training. 


## Notebooks
- `Comparing_New_vs_Old_Datasets.ipynb`: Compares the new ENABLES dataset with tthe previous one collected by Esteban.
- `Preprocessing_and_Muscle_Classification_2_Muscles.ipynb`: *[Not Important]* Old File for Preprocessing steps and muscle classification model for classifying 2 muscles only. Was used in early stages to test the code and make it work.
- `Preprocessing_and_Muscle_Classification_7_Muscles.ipynb`: *[Interesting]* "stable" file where i save the good results and improvements.
- `Preprocessing_and_Muscle_Classification_14_Muscles.ipynb`: *[Not Important]* Old File for Preprocessing steps and muscle classification model for classifying 14 muscles (7 x 2 sides). Was used in early stages to test the model and it did horrible.
- `Testing_Preprocessing_and_Muscle_Classification_7_Muscles.ipynb`: *[Most Interesting]* Test file where i test the new features and models.  
- `Testing_New_Burst_Detections.ipynb`: Testing 3 algorithms for new burst detection from sEMG.
- `Testing_Pretraining.ipynb`: place holder file, nothing interesting yet.

## Notebooks Structure 
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
   - Extracting 300ms burst windows
   - Creating the tfrecords
   - Training/Validation Split
   - Shapes Verification
   - Muscle bursts samples 
- Model Implementation: 
   - Parameters
   - Wandb initialisation 
   - Training
   - Confusion Matrices 
