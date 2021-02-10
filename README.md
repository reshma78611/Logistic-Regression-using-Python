# Logistic-Regression-using-Python

 In ML Algorithms we have :
1. Supervised Learning
2. Unsupervised Learning


In **Supervised Learning** we have :
1. Regression
2. Classification

     Firstly, we discuss about different types of *Regressions*.
      1. Linear Regression
          1. Simple Linear Regression
          2. Multi-linear Regression
      2. Logistic Regression
      3. Lasso Regression 
      4. Ridge Regression
     
     
 *In this repository we discuss about Logistic Regression*
 
 ## Logistic Regression:
          It works on same concept of Linear Regression but it is applicable when input X is continuous and the output Y to be predicted is descrete such as (yes,No),(Male,Female).
          Here X is independent variable and Y is dependent variable.
          Since output to be taken is descrete, we consider output in either 0 or 1. For example if Y is descrete (Yes,No) then it will taken as (1,0)
          Since we have taken input X as continuous, the output Y will also be in terms of continuous varaible to make it descrete we will pass it through a sigmoid function.
                                      sigmoid=(e^y)/(1+e^y)
          Sigmoid function will bring all the values in between [0,1], which are nothing but probabilities. Now from these probability values we will fix a threshold value like if P>=0.5 then '1', if P<0.5 then '0'. So that all the values of Y interms of 0 and 1
          
## Performance Metrics:
      Performance Metrics are those which help us in deciding whether model is good or not.
      1. Accuracy: 
                    Accuracy = (TP+TN)/(TP+FN+FP+TN)
              It tells us about from total observations how many are predicted correctly.
      2. Recall/TPR:
                      Recall= TP/(TP+FN)
              From all positive elements ,how many are actually predicted as positive.
      
      3. Specitivity/TNR:
                       Specitivity = TN/(TN+FP)
               From all negative elements how many are actually predicted as negative
      4. Precision:
                        Precision=TP/(TP+FP)
               From all predicted positives, how many are actual positives
      5. F1-score:
                        F1-score = 2*Recall*precision/(Recall+Precision)
               It is used to say from all observations how many are predicted incorrectly
 
 


## Data Used :
           claimants, bank - full
          
## Programming: 
          Python
 
 
## **The Codes regarding this Logistic Regression model with different business problems *Claimants ,Bank-full*  with their datasets are present in this Repository in detail**
          
