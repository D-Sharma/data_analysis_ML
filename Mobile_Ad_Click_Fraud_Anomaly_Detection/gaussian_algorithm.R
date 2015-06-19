# Get mean for each feature x, by applying function mean to each column
u <- apply(train[, -which(names(train) == "status")], 2, mean)

# Get variance for each feature x, by applying function var to each column
variance <- apply(train[, -which(names(train) == "status")], 2, var)

gaussian_px <- function(xi, ui, vari){
        # Get probability of x using gaussian distribution function 
        # with feature vector x, u (mean) and variance var
        # 
        # Args:
        #       xi - A vector of values of each feature
        #       ui - A vector of mean for each feature
        #       vari - A vector of variance for each feature
        #
        # Returns:
        #       probablity of x
        
        fn <- (1/sqrt(2*pi*vari)) * exp(-1*((xi-ui)^2/(2*vari)))
        px <- prod(fn)
        return(px)
}

n_train <- dim(train)[1]
pred_train <- rep(0.0, n_train)

# Make predictions for train data
for(i in 1:n_train){
        # Coerce row to feature vector
        df <- data.frame(t(train[i,-which(names(train) == "status")]))
        pred_train[i] <- gaussian_px(df, u, variance)       
}

# Issue to resolve:
# 1. Scaling issue
# 2. Probablity for each instance of train or test is very small, close to 0.
# The gaussian distribution function On train dataset predicts minimum probability 2.532086e-320 
# and maximum probablity of 7.838386e-128

e <- 3.0e-321

# Number of observations in the test dataset
n_test <- dim(test)[1] 
pred <- rep(0.0, n_test)

# Make predictions for test data
for(i in 1:n_test){
        df <- data.frame(t(test[i,-which(names(test) == "status")]))
        pred[i] <- gaussian_px(df, u, variance)       
}

#http://stats.stackexchange.com/questions/62069/anomaly-detection-with-dummy-features-and-other-discrete-categorical-features

# Anomaly detection codes for gaussian distribution
# https://github.com/nelsonmanohar/machinelearning/blob/master/R_SCRIPTS/anomaly_detection.R
# http://www.holehouse.org/mlclass/15_Anomaly_Detection.html
# https://nelsonmanohar.files.wordpress.com/2015/04/screenshot-04202015-105944-pm.png
# https://bitbucket.org/nelsonmanohar/machinelearning/src/ad13b5e7538f6a7c5b973f71f8757d6dce70428e/KAGGLE_CLICKTHRU_40/anomaly_detection.R?at=master


# Skweness:
# https://rpubs.com/chrisbrunsdon/skewness

x <- apply(train_sub[,-1], 2, log10)
x <- apply(train_sub[,-1], 2, minmaxnor)

pairs(x[,1:5])
hist(x[,3])

# min max nomalization
minmaxnor<-function(x){
        data_nor = (x-min(x))/(max(x)-min(x))
        return(data_nor)
}




shapiro.test(train_sub[train_sub$status == 0,2])
shapiro.test(train_sub[,3])

hist(log10(train_sub[train_sub$status == 0,3]))
hist(train_sub[train_sub$status == 0,3])
summary(train_sub[train_sub$status == 0,3])

# Unusual large values are 407839, 105696, 105271, 104744
sort(train_sub[train_sub$status == 0,3], 
      decreasing = T)

hist(train_sub[train_sub$total_clicks < 100 ,3])
hist(scale(train_sub[train_sub$status == 0,3]))



hist(minmaxnor(train_sub[train_sub$status == 0,3]))

library(moments)
skewness(train_sub[train_sub$status == 0,3])
#install.packages("moments")
