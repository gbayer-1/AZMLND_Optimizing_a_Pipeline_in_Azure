# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The provided dataset contains data about a marketing campaign of a banking institution. We seek to predict whether a customer subscribes to the offered product colum "y").
The dataset is part of the Azure sample notebook data, the original source of this dataset is
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
The training data is read in, preprocessed and split into training and test data in the train.py script.
The preprocessing step includes:
 dropping NaN values, 
 using binary encoding on the columns "marital" (distinction between "married" =1 and combining "divorced", "single" and "unknown" = 0); "default", "housing", "loan" ("yes"=1 and "no"/"unknown"= 0) and "poutcome" ("succes"=1, "failure"/"nonexistent" = 0)
 using numeric encoding for dates
 and one-hot encoding for the other categorical axes ("job", "contact", "education")
The model is also specified in the train.py script. It is a LogisticRegression model with the hyperparameters C (regularizaion strength) and max_iter (maximum number of iterations).

The experiment uses Hyperdrive to tune the Hyperparameters of the Model, where the regularization strength is varied from 0.1 to 1.0 and the number of maximum iterations can be chosen from 100, 500 or 1000. The Sampler is a random parameter sampler. A BanditPolicy is used for early termination in the run. The primary metric used to determine the best model is the accuracy.

The experiment is run on a standard_d2_v2 cluster with a maximum of 4 nodes. Differents sets of hyperparameters are chosen randomly from the sample space and for each set a LogisticRegression model is trained using the training data and evaluated in regard of the primary metric using the test data.

The best model from the hyperdrive run is using the parameters C = ? and max_iter = ? and has an accuracy of XXX.

**What are the benefits of the parameter sampler you chose?**
I chose the RandomParameterSampler because it generates a given number of random sets of hyperparameters from the given sample space instead of using the entire grid (every possible combination of hyperparameters) to work through. This means a lot less computations to be done, therefore I get the result quicker. The resulting model has most of the time a similar quality (concerning the metrics) as one found using the whole grid search.

**What are the benefits of the early stopping policy you chose?**
I chose a BanditPolicy for early stopping with a slack factor of 0.1. This means that any model, that is more than 10% worse in regard to the primary metric than the current best model in the run, is terminated. That saves computation time, since any bad performing model is dropped freeing ressources for better models.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The best model found by AutoML was a VotingEnsemble with an accuracy of 0.9153262518968134. 

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
