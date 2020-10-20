# Cognitive Computing HW2

 
###  :open_file_folder:  Folder Structure
- hw2.ipynb (result and report analysis)
- BuildFeatures.ipynb (Build offline database for extraction features)
- build_database.py (pyhton file version of BuildFeatures.ipynb)
- run_experiment.py (program to evaluate the MAP result)
- Features.py (sets of features extraction methods)
- utils.py
- result.txt (different setup of fusion features I have tried)
- Data
    - database (image folders cluster by its categories)
    - FeatureDatabase.pkl (dictionary of extracted images' features)


###  :memo:  Result


- HSV with AutoCorrelogram
- RGB with AutoCorrelogr
- Gray with PyramidHOG
- HSV with LocalColorHistogram
- Gray with GaborLocalHistogram
- Gray with SIFT Descriptor

the below features are the best fusion features  I have tried and reach **0.373821** on MAP.

### :rocket: Check up hw2.ipynb for more detailed result and analysis.
