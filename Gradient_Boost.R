#  PRACTICAL BUSINESS ANALYTICS
#  MEACHINE LEARNING & VISULISATIONS
#
# Practical Business Analytics
# Dept. of Computer Science
# University of Surrey
# GUILDFORD
# Surrey GU2 7XH
#
# R Script For Course Work

#  clears all objects in "global environment"
rm(list=ls())

# ************************************************
# Global Environment variables
# - i.e. available to all functions
# Good practice to place "constants" in named variables
# I use UPPERCASE to identify these in my code

DATASET_FILENAME  <- "weatherAUS.csv"          # Name of input dataset file
OUTPUT_FIELD      <- "RainTomorrow"   # Field name of the output class to predict

OUTLIER_CONF      <- 0.95                 # Confidence p-value for outlier detection

TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
TYPE_IGNORE       <- "IGNORE"             # field is not encoded

DISCRETE_BINS     <- 6                    # Number of empty bins to determine discrete
MAX_LITERALS      <- 55                   # Maximum number of 1-hot ecoding new fields

# ************************************************
# Define and then load the libraries used in this project

# Library from CRAN     Version
# ************************************************
# pacman	               0.5.1
# outliers	             0.14
# corrplot	             0.84
# MASS	                 7.3.53
# formattable 	         0.2.0.1
# stats                  4.0.3
# PerformanceAnalytics   2.0.4

MYLIBRARIES<-c("outliers",
               "corrplot",
               "MASS",
               "formattable",
               "stats",
               "PerformanceAnalytics",
               "VIM",
               "caret",
               "ROSE",
               "dplyr",
               "e1071",
               "xgboost",
               "pROC")


# User Defined Functions
# ************************************************
# To manually set a field type
# This will store $name=field name, $type=field type
manualTypes <- data.frame()

# NPREPROCESSING_removePunctuation()
NPREPROCESSING_removePunctuation<-function(fieldName){
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}

# ************************************************
# NreadDataset() :
NreadDataset<-function(csvFilename){
  
  dataset<-read.csv(csvFilename,encoding="UTF-8",stringsAsFactors = FALSE)
  
  # The field names "confuse" some of the library algorithms
  # As they do not like spaces, punctuation, etc.
  names(dataset)<-NPREPROCESSING_removePunctuation(names(dataset))
  
  print(paste("CSV dataset",csvFilename,"has been read. Records=",nrow(dataset)))
  return(dataset)
}

# ************************************************
# NPREPROCESSING_initialFieldType() :
NPREPROCESSING_initialFieldType<-function(dataset){
  
  field_types<-vector()
  for(field in 1:(ncol(dataset))){
    
    entry<-which(manualTypes$name==names(dataset)[field])
    if (length(entry)>0){
      field_types[field]<-manualTypes$type[entry]
      next
    }
    
    if (is.numeric(dataset[,field])) {
      field_types[field]<-TYPE_NUMERIC
    }
    else {
      field_types[field]<-TYPE_SYMBOLIC
    }
  }
  return(field_types)
}

# Min-max scaling function
min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

identify_outliers <- function(data, column_name) {
  # Extract the specified column
  column_data <- data[[column_name]]
  
  # Calculate IQR, Lower fence, and Upper fence
  IQR_value <- IQR(column_data)
  Lower_fence <- quantile(column_data, 0.25) - (IQR_value * 3)
  Upper_fence <- quantile(column_data, 0.75) + (IQR_value * 3)
  
  # Print the result
  cat(sprintf("%s outliers are values < %.2f or > %.2f\n", column_name, Lower_fence, Upper_fence))
}

#Function to replace the cap value of a column with the non outlier value
max_value <- function(data, variable, top) {
  return(ifelse(data[[variable]] > top, top, data[[variable]]))
}

# ************************************************
# main() :
# main entry point to execute analytics
main<-function(){
  
  print("Inside main function")
  
  print(DATASET_FILENAME)
  
  Rain<-NreadDataset(DATASET_FILENAME)
  #The file contains 10 years of weather data, out of which we just take 5000 observations.
  
  # ************************************************
  # Randomly sample 50000 rows from the dataframe
  
  sampled_df <- Rain[sample(nrow(Rain), 50000), ]
  write.csv(sampled_df, "weatherAUS_sampled.csv", row.names = FALSE)
  
  # ************************************************
  # Display the sampled dataframe
  
  head(sampled_df)
  
  # ************************************************
  # Brief summary of sampled data set using summary() to understand what 
  #type of data we are working with
  
  summary(sampled_df)
  
  # ************************************************
  # Check if Target Variable has any NULL values
  
  sum(is.na(sampled_df$RainTomorrow))
  
  # ************************************************
  #Since we have null values in the target variable,we are going to drop those
  #rows as we cant train the model with these
  
  data_clean <- sampled_df[!is.na(sampled_df$RainTomorrow),]
  
  # ************************************************
  #The new dataset has 0 nulls in the clean dataset
  
  sum(is.na(data_clean$RainTomorrow))
  
  # ************************************************
  # Check if is there is any imbalance in the target variable.
  
  print(table(data_clean$RainTomorrow))
  
  # It is clearly evident that there is huge imbalance in the target
  #Variable. So we need to work on reducing it
  
  # ************************************************
  #It is observed that datatype of Date column is string, we will encode it into datetime format.
  
  data_clean$Date <- as.Date(data_clean$Date, format = "%Y-%m-%d")
  
  data_clean$Day   <- sin(2 * pi * as.numeric(format(data_clean$Date, "%w")) / 7)
  data_clean$Month <- sin(2 * pi * as.numeric(format(data_clean$Date, "%m")) / 12)
  data_clean$Year  <- sin(2 * pi * as.numeric(format(data_clean$Date, "%Y")) / max(as.numeric(format(data_clean$Date, "%Y")), na.rm = TRUE))
  data_clean <- data_clean[, -which(names(data_clean) == "Date")]
  
  # ************************************************
  #Determining field types of each field
  
  field_types<-NPREPROCESSING_initialFieldType(data_clean)
  
  print(field_types)
  
  numeric_fields<-names(data_clean)[field_types=="NUMERIC"]
  symbolic_fields<-names(data_clean)[field_types=="SYMBOLIC"]
  
  number_of_numeric<-length(numeric_fields)
  number_of_symbolic<-length(symbolic_fields)
  
  print(paste("NUMERIC FIELDS=",number_of_numeric))
  print(numeric_fields)
  print(paste("SYMBOLIC FIELDS=",number_of_symbolic))
  print(symbolic_fields)
  
  # ************************************************
  # Count null values in numerical and symbolic fields
  
  print(colSums(is.na(data_clean[, symbolic_fields])))
  print(colSums(is.na(data_clean[, numeric_fields])))
  
  # ************************************************
  # Exclude rows with NULL values in the column RainToday
  
  table(data_clean$RainToday,exclude =NULL )
  data_clean <- data_clean[!is.na(sampled_df$RainToday),]
  
  # ************************************************
  # Replace missing values in NumericFields with the median
  
  for (col in names(data_clean[numeric_fields])) {
    # Calculate the median of the column
    col_median <- median(data_clean[[col]], na.rm = TRUE)
    
    data_clean[[col]][is.na(data_clean[[col]])] <- col_median
  }
  
  # ************************************************
  # Replace missing values in CategoricalFields using Mode Imputation
  
  # Initialize an empty list to store the modes
  modes_list <- list()
  
  for (col in names(data_clean[symbolic_fields])) {
    # Find the mode for the current column
    mode_value <- names(sort(table(data_clean[[col]]), decreasing = TRUE)[1])
    modes_list[[col]] <- mode_value
  }
  # Print the list of modes
  print(modes_list)
  
  # Loop through each column in the modes list
  for (col in names(modes_list)) {
    # Replace NA values with the mode in the current column
    data_clean[[col]] <- ifelse(is.na(data_clean[[col]]), modes_list[[col]], data_clean[[col]])
  }
  
  #check for nulls after mode imputation
  print(colSums(is.na(data_clean[, symbolic_fields])))
  
  # Test if any ordinals are outliers and replace with mean values
  
  identify_outliers(data_clean,"Rainfall")
  #For Rainfall column - the min and max  are 0.0 and 371.0. Therefore, the out-liers are values > 2.4
  
  identify_outliers(data_clean,"Evaporation")
  #For Evaporation column - the min and max  are 0.0 and 43.0. Therefore, the out-liers are values > 9
  
  identify_outliers(data_clean,"WindSpeed9am")
  #For WindSpeed9am column - the min and max  are 0.0 and63.0. Therefore, the out-liers are values > 55
  
  identify_outliers(data_clean,"WindSpeed3pm")
  #For Windspeed3pm column - the min and max  are 0.0 and 76.0. Therefore, the out-liers are values > 57
  
  
  
  #we are using top coding to cap  the maximum value and remove outliers from these  variables.
  
  data_clean$Rainfall <- max_value(data_clean, 'Rainfall', 2.4)
  
  data_clean$Evaporation <- max_value(data_clean, 'Evaporation', 9)
  
  data_clean$WindSpeed9am <- max_value(data_clean, 'WindSpeed9am', 55)
  
  data_clean$WindSpeed3pm <- max_value(data_clean, 'WindSpeed3pm', 57)
  
  # Q8: Process the catagorical (symbolic/discrete) fields using 1-hot-encoding
  #catagoricalReadyforML<-NPREPROCESSING_categorical(dataset=Rain,field_types=field_types1)
  
  # Binary encoding the RainToday and RainTommorow variables to 0 and 1
  
  data_clean$RainToday <- ifelse(data_clean$RainToday == "Yes", 1, 0)
  
  data_clean$RainTomorrow <- ifelse(data_clean$RainTomorrow == "Yes", 1, 0)
  
  sum(is.na(data_clean$RainTomorrow))
  
  # converting the data type of all categorical variables to factors for encoding
  
  print(sapply(data_clean, class))
  
  data_clean$Location<-as.factor(data_clean$Location)
  
  data_clean$WindGustDir<-as.factor(data_clean$WindGustDir)
  
  data_clean$WindDir9am<-as.factor(data_clean$WindDir9am)
  
  data_clean$WindDir3pm<-as.factor(data_clean$WindDir3pm)
  
  ###finding count of all the extra columns while encoding
  unique(data_clean$WindDir3pm)
  #WindDir3pm will have 16  produce extra columns
  
  unique(data_clean$WindDir9am)
  #WindDir9am will have 16 columns as well
  
  length(unique(data_clean$WindGustDir))
  #WindGustDir will have 16 columns as well
  
  unique(data_clean$Location)
  # Location will have 49 columns
  
  # Perform one-hot encoding for the categorcial column
  dummy <- dummyVars(" ~ .", data=data_clean)
  newdata <- data.frame(predict(dummy, newdata = data_clean))
  
  sum(is.na(newdata$RainTomorrow))
  
  # Apply min-max scaling function to all numeric columns
  scaled_data <- apply(newdata, 2, min_max_scale)
  
  #Converting the matrix back to a dataframe
  newdata <- as.data.frame(scaled_data)
  
  sum(is.na(newdata$RainTomorrow))
  
  # Split the data into training and testing sets
  set.seed(42)
  training_records <- createDataPartition(newdata$RainTomorrow, p = 0.7, list = FALSE)
  trainData <- newdata[training_records, ]
  testData <- newdata[-training_records, ]
  
  # We will Over Sample the data as there is huge imbalance in the target variable.
  oversampled_data <- ROSE(RainTomorrow ~ ., data = trainData, seed = 123, N = 2 * nrow(trainData))$data
  
  #Modeling
  #We will start with Gradient Boosting Machine Algorithm
  
  # Define the feature columns
  feature_columns <- setdiff(names(oversampled_data), "RainTomorrow")
  
  # Create a matrix of features for training and testing
  X_train <- as.matrix(oversampled_data[, feature_columns])
  y_train <- oversampled_data$RainTomorrow
  
  X_test <- as.matrix(testData[, feature_columns])
  y_test <- testData$RainTomorrow
  
  GBM_Model <- xgboost(data = X_train, label = y_train, nrounds = 100, objective = "binary:logistic")
  
  # Print summary of the model
  summary(GBM_Model)
  
  # Make predictions on the test set
  predictions_GBM <- predict(GBM_Model, newdata = X_test)
  
  # Convert predicted probabilities to binary predictions (0 or 1)
  binary_predictions_GBM <- ifelse(predictions_GBM > 0.5, 1, 0)
  
  # Display the confusion matrix
  confusion_matrix_GBM <- table(binary_predictions_GBM, y_test)
  print(confusion_matrix_GBM)
  
  # Calculate accuracy
  accuracy_GBM <- sum(diag(confusion_matrix_GBM)) / sum(confusion_matrix_GBM)
  cat("Accuracy for GBM:", accuracy_GBM, "\n")
  
  # Calculate precision and recall
  precision_GBM <- confusion_matrix_GBM[2, 2] / sum(confusion_matrix_GBM[, 2])
  recall_GBM <- confusion_matrix_GBM[2, 2] / sum(confusion_matrix_GBM[2, ])
  cat("Precision for GBM:", precision_GBM, "\n")
  cat("Recall for GBM:", recall_GBM, "\n")
  
  #ROC curve for GBM model
  
  roc_curve_GBM <- roc(y_test, predictions_GBM)
  plot(roc_curve_GBM, main = "ROC Curve", col = "blue", lwd = 2)
  auc_value_GBM <- auc(roc_curve_GBM)
  legend("bottomright", legend = paste("AUC =", round(auc_value_GBM, 3)), col = "blue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  title("Receiver Operating Characteristic (ROC) Curve for GBM")
  xlab("False Positive Rate (1 - Specificity)")
  ylab("True Positive Rate (Sensitivity)")
  
  # Now we will try to hypertune the parameters to increase the accuracy
  
  # Define the training control
  ctrl_GBM <- trainControl(method = "cv",  # Cross-validation
                       number = 5,      # Number of folds
                       verboseIter = TRUE)
  
  hyper_grid_GBM <- expand.grid(
    nrounds = c(50),
    max_depth = c(3),
    eta = c(0.01),
    gamma = c(0),
    colsample_bytree = c(0.6),
    min_child_weight = c(1),
    subsample = c(0.6)             
  )
  
  y_train <- as.factor(y_train)
  
  # Perform grid search and training
  GBM_Model_Tuned <- train(
    x = X_train,
    y = y_train,
    method = "xgbTree",
    trControl = ctrl_GBM,
    tuneGrid = hyper_grid_GBM
  )
  
  # Print the model
  print(GBM_Model_Tuned)
  
  # Make predictions on the test set
  predictions_GBM_Tuned <- as.numeric(predict(GBM_Model_Tuned, newdata = X_test))
  
  # Display the confusion matrix
  confusion_matrix_GBM_Tuned <- table(predictions_GBM_Tuned , y_test)
  print("Confusion Matrix for Tuned GBM:")
  print(confusion_matrix_GBM_Tuned)
  
  # Calculate accuracy
  accuracy_GBM_Tuned <- sum(diag(confusion_matrix_GBM_Tuned)) / sum(confusion_matrix_GBM_Tuned)
  cat("Accuracy for Tuned GBM:", accuracy_GBM_Tuned, "\n")
  
  # Calculate precision and recall
  precision_GBM_Tuned <- confusion_matrix_GBM_Tuned[2, 2] / sum(confusion_matrix_GBM_Tuned[, 2])
  recall_GBM_Tuned <- confusion_matrix_GBM_Tuned[2, 2] / sum(confusion_matrix_GBM_Tuned[2, ])
  cat("Precision for Tuned GBM:", precision_GBM_Tuned, "\n")
  cat("Recall for Tuned GBM:", recall_GBM_Tuned, "\n")
  
  # ROC Curve for Tuned GBM
  roc_curve_GBM_Tuned <- roc(y_test, predictions_GBM_Tuned)
  plot(roc_curve_GBM_Tuned, main = "ROC Curve", col = "blue", lwd = 2)
  auc_value_GBM_Tuned <- auc(roc_curve_GBM_Tuned)
  legend("bottomright", legend = paste("AUC =", round(auc_value_GBM_Tuned, 3)), col = "blue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  title("Receiver Operating Characteristic (ROC) Curve for Tuned GBM")
  xlab("False Positive Rate (1 - Specificity)")
  ylab("True Positive Rate (Sensitivity)")
  
  # End of Gradient Boosting Machine Modeling
  
  # Now we will start with Logistic Regression
  
  # Fit binary logistic regression model
  logistic_model <- glm(RainTomorrow ~ ., data = oversampled_data, family = binomial)
  
  # Print summary of the model
  summary(logistic_model)
  
  # Predictions on the training data
  predictions_LR <- predict(logistic_model,newdata = testData, type = "response")
  
  # Convert predicted probabilities to binary predictions (0 or 1)
  predicted_classes_LR <- ifelse(predictions_LR > 0.5, 1, 0)
  
  # Create a confusion matrix
  confusion_matrix_LR <- table(Actual = testData$RainTomorrow, Predicted = predicted_classes_LR)
  
  # Print the confusion matrix
  print(confusion_matrix_LR)
  
  # Calculate accuracy
  accuracy_LR <- sum(diag(confusion_matrix_LR)) / sum(confusion_matrix_LR)
  cat("Accuracy for LR:", accuracy_LR, "\n")
  
  # Calculate precision and recall
  precision_LR <- confusion_matrix_LR[2, 2] / sum(confusion_matrix_LR[, 2])
  recall_LR <- confusion_matrix_LR[2, 2] / sum(confusion_matrix_LR[2, ])
  cat("Precision for LR:", precision_LR, "\n")
  cat("Recall for LR:", recall_LR, "\n")
  
  # ROC curve for Logistic Regression
  roc_curve_LR <- roc(y_test, predictions_LR)
  plot(roc_curve_LR, main = "ROC Curve", col = "blue", lwd = 2)
  auc_value_LR <- auc(roc_curve_LR)
  legend("bottomright", legend = paste("AUC =", round(auc_value_LR, 3)), col = "blue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  title("Receiver Operating Characteristic (ROC) Curve for LR")
  xlab("False Positive Rate (1 - Specificity)")
  ylab("True Positive Rate (Sensitivity)")
  
  # Logistic Regression with Hyper Parameter Tuning
  # Define the control parameters for caret's train function
  ctrl_LR <- trainControl(method = "cv",
                       number = 5)
  
  # Create a grid of hyperparameters to search over
  hyper_grid_LR <- expand.grid(alpha = seq(0, 1, by = 1), lambda = seq(0.01, 1, by = 0.5))
  
  oversampled_data$RainTomorrow <- as.factor(oversampled_data$RainTomorrow)
  
  # Create the logistic regression model
  logistic_model_tuned <- train(
    RainTomorrow ~ .,
    data = oversampled_data,
    method = "glmnet",
    trControl = ctrl_LR,
    tuneGrid = hyper_grid_LR,
    family = "binomial"
  )
  
  # Print the model and its hyperparameters
  print(logistic_model_tuned)
  
  # Make predictions using the best model
  predictions_LR_Tuned <- predict(logistic_model_tuned, newdata = testData, type = "raw")
  
  # Create a confusion matrix
  confusion_matrix_LR_Tuned <- table(Actual = testData$RainTomorrow, Predicted = predictions_LR_Tuned)
  
  # Print the confusion matrix
  print("Confusion Matrix for Tuned LR")
  print(confusion_matrix_LR_Tuned)
  
  # Calculate accuracy
  accuracy_LR_Tuned <- sum(diag(confusion_matrix_LR_Tuned)) / sum(confusion_matrix_LR_Tuned)
  cat("Accuracy for Tuned LR:", accuracy_LR_Tuned, "\n")
  
  # Calculate precision and recall
  precision_LR_Tuned <- confusion_matrix_LR_Tuned[2, 2] / sum(confusion_matrix_LR_Tuned[, 2])
  recall_LR_Tuned <- confusion_matrix_LR_Tuned[2, 2] / sum(confusion_matrix_LR_Tuned[2, ])
  cat("Precision for Tuned LR:", precision_LR_Tuned, "\n")
  cat("Recall for Tuned LR:", recall_LR_Tuned, "\n")
  
  predictions_LR_Tuned <- as.numeric(predictions_LR_Tuned)
  
  # ROC curve for Tuned LR
  roc_curve_LR_Tuned <- roc(y_test, predictions_LR_Tuned)
  plot(roc_curve_LR_Tuned, main = "ROC Curve", col = "blue", lwd = 2)
  auc_value_LR_Tuned <- auc(roc_curve_LR_Tuned)
  legend("bottomright", legend = paste("AUC =", round(auc_value_LR_Tuned, 3)), col = "blue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  title("Receiver Operating Characteristic (ROC) Curve for LR")
  xlab("False Positive Rate (1 - Specificity)")
  ylab("True Positive Rate (Sensitivity)")
  
  # End of Logistic Regression Modeling
  
}
# ************************************************
# This is where R starts execution

# clears the console area
cat("\014")

# Loads the libraries
library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)

#This [optionally] sets working directory
#setwd("")

set.seed(123)

print("WELCOME TO Course Work File")

# ************************************************
main()

print("end")