
calculateError <- function(yhat, target, name){
        # This method creates a confusion matrix and 
        # calculates recall, precision, and fbmeasure.
        #
        # Args:
        #       yhat - Predicted output values
        #       target - Actual output values
        
        # Classes: Not Fraud  0 (True Negative)
        #          Fraud      1 (True Positive)
        result <- table(predict= yhat, actual=target)
        print(addmargins(result))
        
        # Formula: recall = True Positive /(True Positive + False Negative) 
        recall <- result[2,2]/(result[2,2] + result[1,2])
        # Convert to percentage rounded upto 3 digits
        recall_prec <- recall*100
        recall_prec <- round(recall_prec, 3) 
        
        # Formula: precision = True Positive/(True Positive + False Positive)
        precision <- result[2,2]/(result[2,2] + result[2,1]) 
        # Convert to percentage rounded upto 3 digits
        precision_perc <- precision*100
        precision_perc <- round(precision_perc, 3)
        
        # Formula: fbmeasure = (1 + B2) * ( (precision * recall)/ ((b2 * precision) + recall)) 
        beta <- 2
        fbmeasure <- (1+beta^2) * ((precision * recall)/ ((beta^2 * precision) + recall))  
        # Convert to percentage rounded upto 3 digits
        fbmeasure_perc <- fbmeasure*100
        fbmeasure_perc <- round(fbmeasure_perc, 3)
        
        roc_perc <- round(get_roc(yhat = yhat, target = target) * 100, 3)
        mcc_perc <- round( get_mcc(result) * 100, 3)
        return(data.frame(cbind(name, recall_prec, precision_perc, fbmeasure_perc, roc_perc, mcc_perc)))
}

get_roc <- function(yhat, target){
        ROC <- roc( response = target, predictor = yhat)
        plot(ROC, print.auc=TRUE)
        return(ROC$auc)
}

get_mcc <- function(conf_matrix){
        # Calculates matthews correlation coefficient error
        
        #The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications. 
        #It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. 
        #The MCC is in essence a correlation coefficient between the observed and predicted binary classifications; it returns a value between -1 and +1
        
        # Get total number of observations in the dataset
        N = sum(conf_matrix)
        
        # Formula: S = (True Positive + False Negative)/ Total number of observations
        S = ( conf_matrix[2,2] + conf_matrix[1,2] ) / N
        
        # Formula: P = (True Positive + False Positive)/ N
        P = ( conf_matrix[2,2] + conf_matrix[2,1] ) /  N
        
        # Formula: MCC = (True Positive/ N-S * P)/sqrt(P*S*( 1 - S)*( 1 - P )) 
        MCC = (conf_matrix[2,2] / N - S * P)/sqrt(P*S*( 1 - S)*( 1 - P ))
        
        return(MCC)
}
