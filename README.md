# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The provided dataset contains data about a marketing campaign of a banking institution. We seek to predict whether a customer subscribes to the offered product (colum "y").
The dataset is part of the Azure sample notebook data, the original source of this dataset is:

**[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014**

The best performing model was a VotingEnsemble found by the AutoML run with an accuracy of 0.91797.

## Scikit-learn Pipeline

The training data is read, preprocessed and split into training and test data in the train.py script.
The preprocessing step includes:
 - dropping NaN values, 
 - using binary encoding on the columns "marital" (distinction between "married" =1 and combining "divorced", "single" and "unknown" = 0); "default", "housing", "loan" ("yes"=1 and "no"/"unknown"= 0) and "poutcome" ("succes"=1, "failure"/"nonexistent" = 0)
 - using numeric encoding for dates
 - and one-hot encoding for the other categorical axes ("job", "contact", "education")

The model is also specified in the train.py script. It is a LogisticRegression model with the hyperparameters C (regularization strength) and max_iter (maximum number of iterations).

The experiment uses Hyperdrive to tune the Hyperparameters of the Model, where the regularization strength is varied from 0.1 to 1.0 and the number of maximum iterations can be chosen from 100, 500 or 1000. The Sampler is a random parameter sampler. A BanditPolicy is used for early termination in the run. The primary metric used to determine the best model is the accuracy.

The experiment is run on a standard_d2_v2 cluster with a maximum of 4 nodes. Differents sets of hyperparameters are chosen randomly from the sample space and for each set a LogisticRegression model is trained using the training data and evaluated in regard of the primary metric using the test data.

The best model from the hyperdrive run is using the parameters 'Regularization Strength:': 0.8037679473920307, 'Max iterations:': 1000, and has an accuracy of 0.91782
<img width="1532" alt="Screenshot 2021-10-07 162826" src="https://user-images.githubusercontent.com/92030321/136405310-4a110cb9-566b-4822-9637-5625ca7645f4.png">

**What are the benefits of the parameter sampler you chose?**

I chose the RandomParameterSampler because it generates a given number of random sets of hyperparameters from the given sample space instead of using the entire grid (every possible combination of hyperparameters) to work through. This means a lot less computations to be done, therefore I get the result quicker. The resulting model has most of the time a similar quality (concerning the metrics) as one found using the whole grid search.

**What are the benefits of the early stopping policy you chose?**

I chose a BanditPolicy for early stopping with a slack factor of 0.1. This means that any model, that is more than 10% worse in regard to the primary metric than the current best model in the run, is terminated. That saves computation time, since any bad performing model is dropped freeing ressources for better models.

## AutoML
The best model found by AutoML was a VotingEnsemble with an accuracy of 0.91797. 
<img width="866" alt="metrics_voting_ensemble" src="https://user-images.githubusercontent.com/92030321/136401742-984b8e8e-d75e-40e5-b44c-6d3efc6e9239.png">
The Ensemble consists of multiple XGBoostClassifiers, LightGBM Models and LogisticRegression with different weights.
<img width="302" alt="Screenshot 2021-10-07 161407" src="https://user-images.githubusercontent.com/92030321/136402570-710cb854-40be-4e5d-b7cf-03e5cb76ad32.png">


## Pipeline comparison
Both the Tuning of the Hyperparameters with Hyperdrive and the AutoML resulted in models with a similar accuracy with a difference of 0.00015. The VotingEnsemble had a slight advantage considering the accuracy, because it uses the predictions of multiple classifications models, including one LogisticRegression Model. 

## Future work
For future work the tuning of the LogisticRegression model can be improved. The best performing models in the hyperdrive run were all with high numbers of maximum iterations. For furture experiments the search space for the best hyperparameters can be expanded towards some higher numbers of iterations.



