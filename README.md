# Neural_Network_Charity_Analysis

# Background:
Alphabet Soup is a non-profit organization involved in raising and donating funds for environmental improvement efforts. The organization wants to ensure the funds are used effectively. Experience has shown not all funds donated to organizations with the intention of environmental improvements have been used wisely. Alphabet Soup is interested in classifying which organizations are worth donating to and which are high risk using a deep learning neural network that is capable of using all (most) of the available data in predicting whether applicants will be "successful" if funded by Alphabet Soup.  This model will be used to assist Alphabet Soup in determining which organizations receive future donations.

# Problem Statement: 
Create a binary classifier to predict whether applicants will be successful if funded by Alphabet Soup. The model developed must exceed a benchmark of 75% accuracy.

# Deliverables

1. Preprocessing Data for a Neural Network Model
2. Compile, Train, and Evaluate the Model
3. Optimize the Model
4. A Written Report on the Neural Network Model (README.md)

# Preporcessing the Data for a Neural Network (Deliverable 1)

Read in the charity_data.csv to a Pandas DataFrame:

## Data Dictionary:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

## The Target Variable:
* What variable is considered the target for your model?
    the target variable is IS_SUCCESSFUL
* Note: The target "IS_SUCCESSFUL" is split about 50/50 between 
    successful (18,261) and not successful (16,063) projects

## The Features Matrix:
   * What variable(s) are considered the feature(s) for your model?
        The following are all part of the features maxtrix (X)
        1. APPLICATION_TYPE
        2. AFFILIATION
        3. CLASSIFICATION
        4. USE_CASE
        5. ORGANIZATION
        6. STATUS
        7. INCOME_AMT
        8. SPECIAL_CONSIDERATIONS
        9. ASK_AMT




## Steps for Deliverable 1:
The following preprocessing steps have been performed:

* The EIN and NAME columns have been dropped
* The columns with more than 10 unique values have been grouped together
* The categorical variables have been encoded using one-hot encoding
* The preprocessed data is split into features and target arrays 
* The preprocessed data is split into training and testing datasets 
* The numerical values have been standardized using the StandardScaler() module 

# Compile, Train, and Evaluate the Model (Deliverable 2)
TensorFlow was used to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. 

The neural network model developed using Tensorflow Keras contains working code that performs the following steps:

* The number of layers, the number of neurons per layer, and activation function are defined 
* An output layer with an activation function is created 
* There is an output for the structure of the model 
* There is an output of the model’s loss and accuracy 
* The model's weights are saved every 5 epochs to a checkpoints folder
* The results are saved to an HDF5 file in the Trained_Models folder

## AlphabetSoupCharity.ipynb contains deliverables 1 and 2
### Inital Model: 
* was a sequential model
* with an input layer and two hidden layers using "relu" activation 
  - one having three times the number of neurons as the number of inputs
  - the other having about half the number of neurons as the number of inputs
* and an output layer using "sigmoid" activation
### Inital Model Training:
The Inial Model was trained using 100 epochs

### Inial Model Assessment:
The inital model acheived ~ 72.4% accuracy using data it was not trained on
(Loss: 0.557121753692627, Accuracy: 0.7248979806900024)


<br>

# Optimize the Model (Deliverable 3)
In order to exceed the benchmark of 75% model accuracy the model was optimized by exploring the following methods:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model
* Dropping more or fewer columns.
* Creating more bins for rare occurrences in columns.
* Increasing or decreasing the number of values for each bin.
  



* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.






