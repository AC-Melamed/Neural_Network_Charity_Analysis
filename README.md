# Neural_Network_Charity_Analysis

## Overview
This project includes four iterations on a machine learning model for predicting binary outcomes given a dataset of 34,000+ datapoints.  Each of these datapoints represents an organization which have recieved funding from a fictional corporation.  These models and their accompanying visualizations were created in Python 3 using pandas, scikit-learn, and tensorflow, among others.    

### Purpose
The ML algorithms designed for this project were trained and tested on the metadata contained within this dataset with the intention that they would be used to predict with 75%+ accuracy whether or not a given organization would be finanically successful if funded by the corporation.  


## Results

### Data Processing
- The target variable for this model and its iterations was "IS_SUCCESSFUL" i.e. the binary variable provided to indicate the financial success of the given organization.
- All other variables included in the provided metadata were considered feature variables for the model on account of not being strictly relevant to the primary question of predicted financial success.
- The columns labeled "EIN" and "NAME" included identification information which was not useable for the model, and were therefore dropped from the dataframe.  

### Compiling, Training, and Evaluating the Model
- The original version of the model was programmed with two hidden layers containing 80 and 30 nodes, respectively.  The first layer was programmed with a "relu" activation function, as was the second, and the output layer was programmed with a "sigmoid" activation function.  
![O0_eval](https://github.com/AC-Melamed/Neural_Network_Charity_Analysis/edit/main/Images/O0_eval.png)
- The original iteration failed to achieve a 75%+ accuracy evaluation, and so the first optimized iteration of the model was programmed with a constricted binning threshold for the "APPLICATION_TYPE" variable, which was adjusted from 156 to 1065.
![O1_eval](https://github.com/AC-Melamed/Neural_Network_Charity_Analysis/edit/main/Images/O1_eval.png)
- The first optimized iteration also failed to achieve a 75%+ accuracy evaluation, so the second optimized iteration expanded the binning threshold for the "APPLICATION_TYPE" variable from 156 to 16.
![O2_eval](https://github.com/AC-Melamed/Neural_Network_Charity_Analysis/edit/main/Images/O2_eval.png)
- The second optimized iteration also failed to achieve a 75%+ accuracy evaluation, so the third optimized iteration changed the first hidden layer to contain 90 nodes up from 80, and added a third hidden layer with 10 nodes and a "relu" activation function.  
![O3_eval](https://github.com/AC-Melamed/Neural_Network_Charity_Analysis/edit/main/Images/O3_eval.png)


## Summary
Ultimately, despite multiple attempts at optimization, it was not possible to achieve a 75%+ accuracy evaluation with the current data without drastically altering the current model framework.  The best result was acheived with the third optimized iteration, which produced a 73.236% accuracy rating, up 0.152% from the original model.  

### Recommendation for Alternative Model
After some additional exploration, including experimentation with increased epoch lengths for the model training period, it appears that the implementation of a random forest classifier would potentially increase the accuracy of the desired predictions for this dataset.  This is because, as was observed when the training period epochs were doubled or even tripled without any significant increase in accuracy, the risk of overfitting for this dataset seems low.  That makes it a possibly ideal fit for random forest classification, which takes advantage of multiple decision trees as a means of resolving issues with accurate classification.  
