# Practical Machine Learning Project

## SYNOPSIS

From http://groupware.les.inf.puc-rio.br/har

I read that the data are collected from accelerometers on the belt, forearm, arm and dumbell of 6 male health participants (aged between 20-28 years, with little weight lifting experience).

Each participant were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

They were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

In this report, I will attempt to predict the type in which the exercise was done (the classe variable) based on the features extracted from the accelerometers.

## DATA PROCESSING

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

I will assume these 2 files are downloaded and placed in the working directory.

The working directory used here is set as below R code (ie "C:/predmachlearn" ).


```r
# set Global options
options(warn=-1)

library(knitr)
library(ggplot2)
library(lattice)
library(caret)
library(e1071)

opts_chunk$set(cache=TRUE)

# set working directory
setwd("C:/predmachlearn")
```

Extract and load the raw data into the respective dataframes.


```r
# read the raw data file and setting the NA strings accordingly
rawTrain <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA","","#DIV/0!") )
rawTest <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA","","#DIV/0!") )
```

A quick study on rawTrain revealed __19622 observations, consisting of 160 variables.__
Let us work to remove the data that are not important for use in predicting.

I will use __caret's nearZeroVar__ function to identify predictors that have one unique value or predictors that have both of the following characteristics: 

(1) they have very few unique values relative to the number of samples and 

(2) the ratio of the frequency of the most common value to the frequency of the second most common value is large.


```r
nzv_cols <- nearZeroVar(rawTrain[, -160])
if(length(nzv_cols) > 0) {
  # removing near zero variance predictors
  data_train <- rawTrain[, -nzv_cols]
  data_test  <- rawTest[, -nzv_cols]
}
```

Next, I remove columns whose values are all NAs.

Also, I will remove the first 5 columns as they are names and time-stamp information that will not be used.


```r
# remove first 5 columns as they are not used
data_train <- data_train[, -c(1:5)]
data_test <- data_test[, -c(1:5)]

# remove columns with all NAs
columns_all_NAs <- colSums(is.na(data_train)) > 0
data_train <- data_train[, !columns_all_NAs]
data_test <- data_test[, !columns_all_NAs]

# Show the remaining columns.
colnames(data_train)
```

```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

From 160 variables, I have reduced the variables to __53__ (excluding classe variable). 
I will use these 53 variables as predictors of the manner in which they did the exercise (classe variable).
  
## MODEL SELECTION, CROSS VALIDATION AND OUT OF SAMPLE ERROR

Let us recap the approach for Cross-Validation and what is Out of Sample Error:

__Cross Validation Approach:__

1) Use the training set

2) Split it into training/test sets

3) Build a model on the training set 

4) Evaluate on the test set

5) Repeat and average the estimated errors

__Out of Sample Error__ is the error rate you get on a new data set. Sometimes called generalization error.

__Model Selection__

From week 2 lecture, we read that Random Forests are highly accurate for this type of classification problem. 
And to answer the question of what I expect the out of sample error to be and to estimate the error appropriately with cross-validation, I will use the __caret package with Random Forest as my model and will do a 3 fold cross validation.__
(I understand that usually we will be doing 5 or 10 fold cross validation. But the running time for 5-fold on my PC is too long. Hence I adopted 3 folds cross validation)

First, I did a 70-30 split on the training data. I will be using the 70% to train the model and later use the 30% for testing.


```r
set.seed(123)

# Split training set into training/test sets
in_train <- createDataPartition(y=data_train$classe, p=0.7, list=FALSE)
training <- data_train[in_train, ]
testing <- data_train[-in_train, ]
```

Next, I train my Random Forest model using the caret package's Train function.
The cross validation is set in the trainControl function.


```r
# Train a random forest model using the training set 
modFit <- train(classe ~ ., data=training, method="rf", 
                trControl=trainControl(method="cv", number=3), # with 3 fold cross validation
                allowParallel=TRUE
                )
```


```r
print(modFit)
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 9158, 9159, 9157 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9917740  0.9895938  0.002605758  0.003296720
##   27    0.9964329  0.9954877  0.002605806  0.003296417
##   53    0.9933757  0.9916205  0.001485289  0.001879166
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.24%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    7 2647    4    0    0 0.0041384500
## C    0    3 2393    0    0 0.0012520868
## D    0    0   11 2241    0 0.0048845471
## E    0    0    0    6 2519 0.0023762376
```

Let us now test by using this to predict the 30% testing data we had segregated from the training dataset.


```r
pred <- predict(modFit, testing)

confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1136    4    0    0
##          C    0    0 1022    4    0
##          D    0    0    0  959    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9961   0.9948   1.0000
## Specificity            0.9993   0.9992   0.9992   1.0000   0.9998
## Pos Pred Value         0.9982   0.9965   0.9961   1.0000   0.9991
## Neg Pred Value         1.0000   0.9994   0.9992   0.9990   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1930   0.1737   0.1630   0.1839
## Detection Prevalence   0.2850   0.1937   0.1743   0.1630   0.1840
## Balanced Accuracy      0.9996   0.9983   0.9976   0.9974   0.9999
```

The accuracy of the model is 99.8%. The out of sample error is 0.2%.

With such a high accuracy and low out of sample error, I will not be running other models. But will be using this model on the 20 test cases ("pml-testing.csv").


```r
# using the model on the 20 test cases
submit20testcases <- predict(modFit,data_test)
submit20testcases
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


```r
# function for writing the output files
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

# write the predictons generated in the previsou step into files.
pml_write_files(submit20testcases)
```

## SUMMARY

As mentioned above. the accuracy of the Random Forest model is 99.8%. The out of sample error is 0.2%.

We also submitted the 20 test cases results and was graded all correct (ie 20/20).

This shows the Random Forest model we created is able to predict well.

Thanks for taking the time to grade my work.

