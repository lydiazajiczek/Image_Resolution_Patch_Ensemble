# PatchEnsemble Binary Classifier
PatchEnsemble training and prediction framework for binary classification of degraded histopathology images from publicly available datasets

## Functions
* `PatchEnsemble.py`: instantiates the `PatchEnsemble` class. Edit this to change the network architecture.

* `train.py`: takes the following command line arguments:
	* `--dataset` - the base dataset to train on (accepted values 'BACH', 'BreaKHis4X', 'BreaKHis10X', 'BreaKHis20X', 'BreaKHis40X', 'PCam')
	* `--classifier_type` - type of classifier to use (accepted values 'patch' or 'whole_image', note that 'patch' is only compatible with 'PCam' dataset)
  * `--degradation` - (optional) the degradation applied to the dataset in the format e.g. '_degraded_0_13_to_0_12', if a value is not provided the model is trained on the baseline or original dataset
  * `--base_path` - (optional, but default is hardcoded in file) location of all data folders, default set in file

* `test.py`: takes the following command line arguments:
	* `--dataset` - the dataset to test on (same as `train.py`)
	* `--classifier_type` - type of classifier to use (same as `train.py`)
	* `--degradation` - (optional) the degradation applied to the dataset the model was trained on (same as `train.py`)
	* `--test-degradation` - (optional) the degraded test set to apply the baseline model to

* `params.py`: takes single argument `dataset`, returns `params` dictionary containing relevant parameters of imaging system used to capture dataset passed as argument

Two example files `sample_train.py` and `sample_test.py` are also provided to illustrate how to tune hyper parameters from the default values as well as set various flags of the testing and training functions `train_top`, `fine_tune` and `test` defined in the `PatchEnsemble` class. 

## Data
`PatchEnsemble` expects the histopathology datasets to be organized in the following folder hierarchy: (note that degraded images can be obtained using MATLAB code provided [here](https://github.com/lydiazajiczek/Image-Degradation)

./base_path/
* BACH/ located [here](https://iciar2018-challenge.grand-challenge.org/)
  * data/
    * train/
      * benign/
        * Images 
      * malignant/
        * Images
    * val/ (optional, same structure as /train/, otherwise pass `--validation_split` to `train.py`)
    * test/
      * test/
        * Images 
  * data_degraded_0_30_to_0_25/
    * same structure as /data/
  * ... 

* PCam/ located [here](https://www.kaggle.com/c/histopathologic-cancer-detection)
  * data/
    * train/
      * benign/
        * Images 
      * malignant/
        * Images
    * val/ (optional, same structure as /train/, otherwise pass `--validation_split` to `train.py`)
    * test/
      * test/
        * Images 
  * data_degraded_0_13_to_0_12/
    * same structure as /data/
  * ... 
      
* BreaKHis4X/ located [here](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
  * data/
    * train/
      * benign/
        * Images 
      * malignant/
        * Images
    * val/ (optional, same structure as /train/, otherwise pass `--validation_split` to `train.py`)
    * test/
      * benign
        * Images 
      * malignant
        * Images  
  * data_degraded_0_16_to_0_14/
    * same structure as /data/
  * ... 
  
* BreaKHis10X, same structure as /BreaKHis4X/

* BreaKHis20X, same structure as /BreaKHis4X/

* BreaKHis40X, same structure as /BreaKHis4X/

## Installation
Tested on Ubuntu 18.04.6 LTS with Python 3.6.9 using TensorFlow/Keras 2.3.0

To cite this dataset, please use the following:

[![DOI](https://zenodo.org/badge/481254878.svg)](https://zenodo.org/badge/latestdoi/481254878)
