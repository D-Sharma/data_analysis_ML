####################################################################
# Fraud Detection in Mobile Advertising (FDMA) on Final Train Set
####################################################################

# Get data location for datasets
path_train <- "data\\click-fraud-data_phua\\final_120_train_w_labels.csv"
path_valid <- "data\\click-fraud-data_phua\\final_120_validation_w_labels.csv"
path_test <- "data\\click-fraud-data_phua\\final_120_test_w_labels.csv"

# Load all datasets in memory
train <- read.csv(path_train, header=T, stringsAsFactors = F)
validation <- read.csv(path_valid, header=T, stringsAsFactors = F)
test <- read.csv(path_test, header=T, stringsAsFactors = F)

# View data
head(train)
summary(train)
str(train)

# Remove columns not required
remove_attributes <- c("partnerid")
idx <- which(names(train) %in% remove_attributes)
train <- train[, -idx]
idx <- which(names(test) %in% remove_attributes)
test <- test[,-idx]
idx <- which(names(validation) %in% remove_attributes)
validation <- validation[,-idx]


# Set the seed for randoam values
set.seed(12345)
err <- data.frame()

#Load libraries
library(pROC)
source("Errors.R")


########## Features subset based on Relative Influence #############
train_sub <- subset(train, select = c('status','std_per_hour_density', 'total_clicks', 'brand_Generic_percent', 'avg_distinct_referer', 
                                      'std_total_clicks', 'night_referer_percent', 'second_15_minute_percent', 'distinct_referer',
                                      'std_distinct_referer', 'morning_click_percent', 'std_spiky_iplong', 'avg_spiky_ReAgCnIpCi', 
                                      'night_avg_spiky_referer', 'avg_spiky_agent', 'avg_spiky_referer', 'avg_spiky_ReAgCn', 
                                      'night_avg_spiky_ReAgCnIpCi', 'afternoon_avg_spiky_ReAgCnIpCi', 'afternoon_avg_spiky_agent',
                                      'std_spiky_referer', 'cntr_id_percent', 'cntr_sg_percent', 'cntr_other_percent',
                                      'cntr_us_percent', 'cntr_th_percent', 'cntr_uk_percent', 'cntr_in_percent', 'cntr_ng_percent',
                                      'cntr_tr_percent', 'cntr_ru_percent'))

test_sub <- subset(test, select = c('status','std_per_hour_density', 'total_clicks', 'brand_Generic_percent', 'avg_distinct_referer', 
                                    'std_total_clicks', 'night_referer_percent', 'second_15_minute_percent', 'distinct_referer',
                                    'std_distinct_referer', 'morning_click_percent', 'std_spiky_iplong', 'avg_spiky_ReAgCnIpCi', 
                                    'night_avg_spiky_referer', 'avg_spiky_agent', 'avg_spiky_referer', 'avg_spiky_ReAgCn', 
                                    'night_avg_spiky_ReAgCnIpCi', 'afternoon_avg_spiky_ReAgCnIpCi', 'afternoon_avg_spiky_agent',
                                    'std_spiky_referer', 'cntr_id_percent', 'cntr_sg_percent', 'cntr_other_percent',
                                    'cntr_us_percent', 'cntr_th_percent', 'cntr_uk_percent', 'cntr_in_percent', 'cntr_ng_percent',
                                    'cntr_tr_percent', 'cntr_ru_percent'))


#----------------------------
# Genralized boosting method
#----------------------------

#In boosting, classifiers are constructed on weighted versions of the training set, which
#depend on previous classification results.
require(gbm)
Xtrain <- train_sub
model <- gbm(status~., 
             data=Xtrain, 
             shrinkage=0.001,
             interaction.depth=5, 
             distribution="bernoulli",
             n.trees=5000,
             n.minobsinnode=5)

# Distribution and Shrinkage:

#Distribution is the loss function. For most classification problems bernoulli distribution is used.
#Shrinkage is lambda (or learning rate). Performance is best when lambda is as small as possible (decreasing the learning rate to prevent overfitting)


# Purpose of 'minobsinnode' - minimum observationin each node:
# At each step of the GBM algorithm, a new decision tree is constructed. 
# The question when growing a decision tree is ''when to stop?''. 
# The furthest you can go is to split each node until there is only 1 observation in each terminal node. 
# This would correspond to n.minobsinnode=1. 
# Alternatively, the splitting of nodes can cease when a certain number of observations are in each nod (minobsinnode=5)
# The default minobsinnode for the R GBM package is 10.

#What is the best value to use? 
#It depends on the data set and whether you are doing classification or regression. 
#Since each trees prediction is taken as the average of the dependent variable of all inputs in the terminal node, a value of 1 probably wont work so well for regression but may be suitable for classification.

#Higher values mean smaller trees so make the algorithm run faster and use less memory.
#Generally, results are not very sensitive to this parameter and given the stochastic nature of GBM performance it might actually be difficult to determine exactly what value is 'the best'. 
#The interaction depth, shrinkage and number of trees will all be much more significant in general.

summary(model)

# Model Performance: Estimating the optimal number of iterations
gbm.perf(model)
#Using OOB method... 3674

# Predictions using gbm
train_pred <- predict(model, newdata=Xtrain, n.trees=5000, type='response')
train_pred <- round(train_pred)
err_train <- calculateError(train_pred, Xtrain$status, "GBM - Train")
#          actual
#predict    0    1  Sum
#   0   3009   22 3031
#   1      0   50   50
#   Sum 3009   72 3081

test_pred <- predict(model, newdata=test_sub, n.trees=5000, type="response")
test_pred <- round(test_pred)
err <- rbind(err, calculateError(test_pred, test_sub$status, "GBM")) 
#        actual
#predict    0    1  Sum
#    0   2904   57 2961
#    1     14   25   39
#    Sum 2918   82 3000

#-------------------------------------------------------------------------------
# One Class SVM Classification (training the model only on the normal behavior)
#-------------------------------------------------------------------------------

#
# One-Class Support Vector Machine : One class support vector machine implements a binary classifier where the the training data consists of examples of only one class (normal data). 
# The model attempts to separate the collection of training data from the origin using maximum margin. 
# By default, a radial basis kernel is used.


library(e1071)
library(dplyr)

new_train <- filter(train_sub, status==0)
table(new_train$status)

trainpredictors <- new_train[,2:31]
trainLabels <- new_train[,1]

testpredictors <- test_sub[,2:31]
testLabels <- test[,1]

set.seed(12345)
svm.model<-svm(x=trainpredictors, y=trainLabels,
               type='one-classification',
               scale=TRUE,
               nu=0.10,
               gamma=0.01)

summary(svm.model)

#Interpretation of nu
#The parameter nu is an upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors relative to the total number of training examples. 
#For example, if you set it to 0.05 you are guaranteed to find at most 5% of your training examples being misclassified (at the cost of a small margin, though) and at least 5% of your training examples being support vectors.

# Predictions 
svm.predtrain <- predict(svm.model, trainpredictors, type="response")
svm.predtrain <- as.numeric(svm.predtrain=="FALSE")
#err_train_svm <- calculateError(svm.predtrain, trainLabels, "One-class SVM")

svm.predtest <- predict(svm.model, testpredictors, type="response")
svm.predtest <- as.numeric(svm.predtest=="FALSE")
err <- rbind(err, calculateError(svm.predtest, testLabels, "One-class SVM"))


# --------------------
# Rule based system
# --------------------
library(C50)
# Create a rule based model learning rules from training data using decision trees.
treeModel <- C5.0(factor(status) ~ ., 
                  data = train_sub, 
                  rules = F)

# Predict classification for test data
yhat <- predict(treeModel,
                newdata=test_sub[,-which(names(test_sub) %in% c("partnerid", "status"))],
                type="class")
err <- rbind(err, calculateError(as.numeric(as.character(yhat)), test_sub[,c("status")], "Rule based - Decision trees"))
print(err)

#---------------
# Random forest
#---------------
library(randomForest)

rf_ntree <- 15
rf_mtry <- 5
rf_importance <- TRUE

# fit and predict click fraud
rf_fit <- randomForest(factor(status, levels = c(0,1))~., 
                          data=train_sub, 
                          ntree=rf_ntree, 
                          mtry=rf_mtry, 
                          importance=rf_importance)

# Predict click fraud
yhat <- predict(rf_fit, test_sub[, -which(names(test_sub) == "status")])
err <- rbind(err, calculateError(as.numeric(as.character(yhat)), test_sub[,c("status")], "Random forest"))
print(err)


#-------------------
# Bayesian Networks
#-------------------

#http://stackoverflow.com/questions/24367141/naive-bayes-imbalanced-test-dataset
#http://stats.stackexchange.com/questions/19787/naive-bayes-classifier-for-unequal-groups

library(e1071)
nb_model <- naiveBayes(factor(status)~., data = train_sub, laplace = 1)
summary(nb_model)

# Set a threshold value to predict classification
yhat <- predict(nb_model, newdata=test_sub[,-1],  
                class="class",
                threshold = 0.035)

err <- rbind(err, calculateError(as.numeric(as.character(yhat)), test_sub[,c("status")], "Naive Bayes"))
print(err)

# http://people.orie.cornell.edu/davidr/or678/r2winbugs/MultiNormalBug.pdf

#---------------------------
# Replicator Neural Network
#---------------------------
# http://r.789695.n4.nabble.com/Issues-with-nnet-default-for-regression-classification-td3060505.html
# https://beckmw.wordpress.com/2013/11/14/visualizing-neural-networks-in-r-update/
library(nnet)
set.seed(12345)
nn_fit <- nnet(train_sub[, -which(names(train_sub)=="status")], 
               train_sub[,c("status")],
               size = 3, 
               maxit = 150,
               decay = 5e-3,
               entropy = T, 
               skip = T,
               rang = 0.001 ) # c(-0.25, 0.25)  0.001

yhat <- predict(nn_fit, test_sub[,-which(names(test_sub)=="status")], 
                type = "class")

err <- rbind(err, calculateError(as.numeric(yhat), test_sub[,c("status")], "NNET"))
print(err)

#       actual
#predict    0    1  Sum
#    0   2865   60 2925
#    1     53   22   75
#Sum 2918   82 3000

#-------------------------------------------
# Multilayer Perceptron Neural Networks in R
#-------------------------------------------
'''
The multi-layer perceptron (MLP) is a simple feed-forward neural network with an
input layer, several hidden layers and one output layer. It means that information can
only flow forward from the input units to the hidden layer and then to the output
unit(s). 
'''
#http://stackoverflow.com/questions/16228954/how-to-use-mlp-multilayer-perceptron-in-r
#http://ape.iict.ch/teaching/MLBD2014/MLBD_Labo/MLBD_2014_lab3/
#http://cran.r-project.org/web/packages/monmlp/monmlp.pdf

library(monmlp)
#?monmlp
#?monmlp.predict

# Fit the model and compute the predictions
set.seed(12345)
#train predictors and labels converted into matrix
x <- train_sub[, -which(names(train_sub)=="status")]
x <- data.matrix(x)
y <- train_sub[,c("status")]
y <- data.matrix(y)

#test predictors and labels converted into matrix
test_x <- test_sub[, -which(names(test_sub)=="status")]
test_x <- data.matrix(test_x)

model <- monmlp.fit(x, y, 
                    hidden1=3, 
                    n.ensemble=15, 
                    monotone=1, 
                    bag=TRUE)
yhat <- monmlp.predict(x = test_x, weights = model) #there is no class option in monmlp.predict
map <- as.data.frame(cbind(yhat, test_sub[,c("status")])) 
library(plyr)
map <- rename(map, c("V1" = "yhat", "V2" = "y"))
map$predhat <- 1
map$predhat[map$yhat <= 0.35] <- 0 #tested 0.35 and 0.5 threshold value to split the values into 0 and 1
head(map,10)
err <- rbind(err, calculateError(map$predhat, test_sub[,c("status")], "MONMLP"))
print(err)

#  name recall_prec precision_perc fbmeasure_perc roc_perc mcc_perc
#MONMLP      30.488         44.643         32.552   64.713   35.449 (threshold = 0.5)
#MONMLP      54.878         42.453         51.843   76.394   46.621 (threshold = 0.35)

# Compute the AUC
install.packages("ROCR")
library(ROCR)
plot(performance( prediction(yhat, test_sub[,c("status")]), "tpr","fpr"))
performance(prediction(yhat, test_sub[,c("status")]), "auc")@y.values[[1]]

