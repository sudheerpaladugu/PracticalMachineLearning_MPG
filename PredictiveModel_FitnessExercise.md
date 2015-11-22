---
title: "Predictive Model For Human Activity Recognition"
output: html_document
---

##Summary    
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.



##Analysis
###Data Processing  
The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

Downloading csv files for analysis   

```r
#Downloading data
if (!file.exists("./pml-training.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./pml-training.csv")
}
if (!file.exists("./pml-testing.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./pml-testing.csv")
}
```
Reading data from csv files  

```r
#Loading Training set
training <- read.csv("pml-training.csv", na.strings = c("NA", ""))
#Final Test set
final_test <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
```


```r
dim(training)
```

```
## [1] 19622   160
```
###Data Cleaning  
Removing unnecessary columns X, user_name, and cvtd_timestamp (factor instead of numeric) from data frame.  

```r
training <- training[, -grep("X|user_name|cvtd_timestamp", names(training))]
```

Training data frame has some near zero variance columns. Removing those columns from data frame since they will not contributed to the model.  


```r
suppressMessages(library(caret))
nearZero <- nearZeroVar(training)
```
(43) nearZero variance columns has been remove. Now we will remove NA columns as well from the data frame.  

```r
#Removing non zero variance elements from data frame
training <- training[,-nearZero]
#Removing NA data elements ('2' for columns) from data froma
nacols <- apply(training, 2, function(x) {sum(is.na(x))})
training <- training[, which(nacols == 0)]
dim(training)
```

```
## [1] 19622    56
```
Final data frame (training) has **19622**  rows and **56** columns. Which is less when compared to original data frame (when loaded from csv).  

###Test data preparation  
Creating training and test data sets from data frame, in which 80% of training and 20% of test data for further processing.  


```r
inTrain <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
training2 <- training[inTrain, ] 
test2 <- training[-inTrain, ]
```

##Model Creation

Creating a model on training2 and test2 data sets. Firt attempt w'll try fitting a single tree model with rpart.  

```r
library(rpart)
fit <- train(training2$classe ~ ., data = training2, method = "rpart")
fit
```

```
## CART 
## 
## 15699 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 15699, 15699, 15699, 15699, 15699, 15699, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03822875  0.5595343  0.43543616  0.04413576   0.06160444
##   0.05516244  0.4348688  0.23766931  0.07320464   0.11983049
##   0.11508678  0.3169855  0.04953064  0.04053424   0.06197974
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03822875.
```
This model accuracy is **55.95%**, which is low.  

####Random forests
We will create a model with Random forests to get better accuracy.  


```r
suppressMessages(library(randomForest))
tctrl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
fit2 <- train(training2$classe ~ ., data = training2, method = "rf", prof = TRUE, trControl = tctrl)
fit2
```

```
## Random Forest 
## 
## 15699 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 11775, 11774, 11774, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9959870  0.9949237  0.0014319049  0.0018115768
##   28    0.9984076  0.9979858  0.0008156424  0.0010316598
##   55    0.9980253  0.9975023  0.0004353998  0.0005507327
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```
Random forests provided a model with a high accurecy **99.84%**.

##Cross-Validation  
We will proceed with fit2 to predict the new values from test2 set created earlier for cross-validation.  

```r
pred <- predict(fit2, test2)
test2$predRight <- pred ==test2$classe
table(pred, test2$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1116    2    0    0    0
##    B    0  756    0    0    0
##    C    0    1  684    1    0
##    D    0    0    0  642    0
##    E    0    0    0    0  721
```
Predictions are not correct in all cases as expected. Caluclating accuracy:  

```r
postRes <- postResample(pred, test2$classe)
postRes
```

```
##  Accuracy     Kappa 
## 0.9989804 0.9987102
```
Prediction fitted for test set higher than training set (i.e 99.84%).  

###Sample error  
We calculate the expected out of sample error from test set (test2) that we created for cross-validation.  


```r
cfmtrx <- confusionMatrix(pred, test2$classe)
cfmtrx
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    2    0    0    0
##          B    0  756    0    0    0
##          C    0    1  684    1    0
##          D    0    0    0  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   1.0000   0.9984   1.0000
## Specificity            0.9993   1.0000   0.9994   1.0000   1.0000
## Pos Pred Value         0.9982   1.0000   0.9971   1.0000   1.0000
## Neg Pred Value         1.0000   0.9991   1.0000   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1744   0.1637   0.1838
## Detection Prevalence   0.2850   0.1927   0.1749   0.1637   0.1838
## Balanced Accuracy      0.9996   0.9980   0.9997   0.9992   1.0000
```

### Test set predictions


```r
testpred <- predict(fit2, final_test)#test data from csv file
test3 <- final_test
test3$classe <- testpred
#function to write answer to a file
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
         #file name
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

answers <- test3$classe
#writing answers to files
pml_write_files(answers)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


