
#Research Question :  Rainfall Prediction and Influence Analysis: A Comparative Study of Machine Learning Techniques on Imbalanced Datasets.

#Dataset Information:
#This dataset contains about 10 years of daily weather observations from numerous Australian weather stations.

#RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No?

#Kaggle-Link :https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data?select=weatherAUS.csv

#Data Dictonary:

#"Date"          "Location"      "MinTemp"       "MaxTemp"       "Rainfall"      "Evaporation"
# "Sunshine"      "WindGustDir"   "WindGustSpeed" "WindDir9am"    "WindDir3pm"    "WindSpeed9am"
# "WindSpeed3pm"  "Humidity9am"   "Humidity3pm"   "Pressure9am"   "Pressure3pm"   "Cloud9am"
# "Cloud3pm"      "Temp9am"       "Temp3pm"       "RainToday"     "RainTomorrow"


#In this code we have included the Common EDA part of the data set and individual models of each person.
#As everyone has used different libraries and functions so To avoide any errors during compilation
#we are attaching Our individual Code files with full code and eda so that code can run without any issues

#Dataset Information:
#Predict if there is rain by running classification methods on  objective variable RainTomorrow.

#LOADING THE LIBRARIES

library(GGally)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(ggcorrplot)
library(data.table)
library(caret)
library(gplots)
library(corrplot)
library(ROSE)
library(dplyr)
library(pROC)

# Set a seed for reproducibility
set.seed(123)

#  clears all objects in "global environment"
rm(list=ls())


#########User Defined Funtions


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

# Min-max scaling function
min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}



#
############ DATA LOADING
#
# file preview
full_df <- read.csv("weatherAUS.csv", header = TRUE)
#The file contains 10 years of weather data, out of which we just take 5000 observations.


# Randomly sample 10 % rows from the dataframe due to memory constraints
sampled_df <- full_df[sample(nrow(full_df), 14546), ]
write.csv(sampled_df, "weatherAUS_sampled.csv", row.names = FALSE)


#############Data EXPLORATION



# Display the sampled dataframe
head(sampled_df)


# first look at the data set using summary() to understand what type of data we are working with
summary(sampled_df)

################### EDA of Target Variable
sum(is.na(sampled_df$RainTomorrow))

#Since we have  103 null values in the target variable,we are going to drop those rows as we cant train the model with these
data_clean <- sampled_df[!is.na(sampled_df$RainTomorrow),]

sum(is.na(data_clean$RainTomorrow))
#The new dataset has 0 nulls in the clean dataset

#
print(table(data_clean$RainTomorrow))

# As we can see there is am imbalance of the target variable RainTommorow which we will have to work around

###############EDA of Categorical Variables

# Use sapply to get the data type of each column
print(sapply(data_clean, class))

# Identify categorical columns
categorical_columns <- sapply(data_clean, function(x) is.factor(x) || is.character(x))
print(categorical_columns)

# Count null values in categorical columns
print(colSums(is.na(data_clean[, categorical_columns])))


# Exploring each variable
table(data_clean$Location)

table(data_clean$RainToday,exclude =NULL )

#Identify Numeric Columns
numeric_columns <- sapply(data_clean, function(x) is.numeric(x) || is.integer(x))
print(numeric_columns)

# Count null values in numeric columns
print(colSums(is.na(data_clean[, numeric_columns])))
#
############ VISUALIZING THE DATA & OUTLIER DETECTION
#


summary(data_clean[,numeric_columns])

#From the summary function we can presume that the columns-
#Evaporation, WindSpeed9am,Rainfall, WindSpeed3pm  might have outliers.

# Create boxplots for all columns
par(mfrow = c(2, 2))  # Set up a 2x2 grid for the plots

boxplot(data_clean$Evaporation, main = "Boxplot of Evaporation", ylab = "Evaporation", col = "lightgreen", border = "black")


boxplot(data_clean$WindSpeed9am, main = "Boxplot of WindSpeed9am", ylab = "WindSpeed9am", col = "lightgreen", border = "black")


boxplot(data_clean$Rainfall, main = "Boxplot of Rainfall", ylab = "Rainfall", col = "lightgreen", border = "black")


boxplot(data_clean$WindSpeed3pm, main = "Boxplot of WindSpeed3pm", ylab = "WindSpeed3pm", col = "lightgreen", border = "black")

# The boxplot confirms the existence of outliers

#####VISUALSING THE DISTRIBUTION OF DATA

# Plot histograms for numerical columns
par(mfrow = c(4, 4),mar = c(4, 4, 2, 1))  # Set up a 4x4 grid for the plots
for (col in names(data_clean)) {
  if (is.numeric(data_clean[[col]])) {
    hist(data_clean[[col]], main = paste("Histogram of", col), col = "pink", border = "black",xlab = col)
  }
}



###########Feature Engineering


# Replacing the null values with the median of the column
library(dplyr)



print(colSums(is.na(data_clean[, categorical_columns])))


numeric_columns <- sapply(data_clean, function(x) is.numeric(x) || is.integer(x))
# Loop through columns
for (col in names(data_clean[numeric_columns])) {
  # Calculate the median of the column
  col_median <- median(data_clean[[col]], na.rm = TRUE)

  # Replace missing values with the median
  data_clean[[col]][is.na(data_clean[[col]])] <- col_median
}


# Identify categorical columns
categorical_columns <- sapply(data_clean, function(x) is.factor(x) || is.character(x))
print(categorical_columns)

#mode imputation for categorical

# Initialize an empty list to store the modes
modes_list <- list()

for (col in names(data_clean[categorical_columns])) {
  # Find the mode for the current column
  mode_value <- names(sort(table(data_clean[[col]]), decreasing = TRUE)[1])
  modes_list[[col]] <- mode_value
}
# Print the list of modes
print(modes_list)

#
# Loop through each column in the modes list
for (col in names(modes_list)) {
  # Replace NA values with the mode in the current column
  data_clean[[col]] <- ifelse(is.na(data_clean[[col]]), modes_list[[col]], data_clean[[col]])
}

#check for nulls after mode imputation
print(colSums(is.na(data_clean[, categorical_columns])))

# Plotting a heatmap to see the co-relation between variables

numerical_matrix <- cor(data_clean[,numeric_columns])

### Create a correlation heatmap

# Reset the plotting parameters
par(mfrow = c(1, 1))  # Set the layout to a single plot


corrplot(numerical_matrix, method = "color", type = "upper", order = "hclust", tl.cex = 0.7,number.cex = 0.6,tl.srt = 45   )

# From the above plot we can see that the pairs of  following variables are highly correlated;

#(MinTemp,Temp3pm)
#(MinTemp,Temp9am)
#(MaxTemp,Temp9am)
#(MaxTemp,Temp3pm)
#(WindGustSpeed,WindSpeed3pm)
#(Pressure9am,Pressure3pm)
#(Temp9am,Temp3pm)

#We might need to drop some of these columns if the model is failing to learn due to multi co-linearity

#### Outlier removal


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

# Checking the max value of these columns
summary(data_clean)

# Binary encoding the RainToday and RainTommorow variables to 0 and 1

data_clean$RainToday <- ifelse(data_clean$RainToday == "Yes", 1, 0)

data_clean$RainTomorrow <- ifelse(data_clean$RainTomorrow == "Yes", 1, 0)

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
#WindDir9am will have 16 columns

length(unique(data_clean$WindGustDir))
#WindGustDir will have 16 columns as well

unique(data_clean$Location)
# Location will have 49 columns if encoded

# Perform one-hot encoding for the categorical columns
dummy <- dummyVars(" ~ .", data=data_clean[,-c(1, 2)])

df_loc_dat <- data.frame(predict(dummy, newdata = data_clean))


#The dataframe df_loc_dat has   pure numerical columns , encoded categorical variables without dates and location

#so that we can pass all numerical columns so any model that requires numerical data

# Apply min-max scaling function to all numeric columns


#Converting the matrix back to a dataframe
df_loc_dat <- as.data.frame(apply(df_loc_dat, 2, min_max_scale))


final_data <- cbind(df_loc_dat, data_clean[, c("Date", "Location")])

#It is observed that datatype of Date column is string,we will encode it into datetime format.

final_data$Date <- as.Date(final_data$Date, format="%Y-%m-%d")
#creating a copy of final dataframe to create new version for modelling


df_dat_enc <- copy(final_data)
#This version of the data will have dates encoded as days,months and years as separate columns

# Extract month, year, and day into separate version of the dataset
df_dat_enc$Month <- as.integer(month(df_dat_enc$Date))
df_dat_enc$Year <- as.integer(year(df_dat_enc$Date))
df_dat_enc$Day <- as.integer(day(df_dat_enc$Date))

#Dropping the date column
df_dat_enc <- df_dat_enc[, -which(names(df_dat_enc) == "Date")]


######Akhil (6828141) Modelling ########

#Declaring Dependent and independent variable

X <- df_dat_enc[, !(names(df_dat_enc) %in% c('RainTomorrow'))]

y <- df_dat_enc$RainTomorrow

#Spliting data into separate training and test set
indices <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[indices, ]
X_test <- X[-indices, ]
y_train <- y[indices]
y_test <- y[-indices]

# Fiting logistic regression model
logreg_model <- glm(y_train ~ ., data = X_train, family = binomial)

summary(logreg_model)

# Predictions on the test set
predictions <- predict(logreg_model, newdata = X_test, type = "response")

# Evaluating the model and calculating accuracy

predicted_labels <- ifelse(predictions > 0.5, 1, 0)
accuracy <- mean(predicted_labels == y_test)
cat("Accuracy on the test set:", accuracy, "\n")

# Creating a confusion matrix
conf_matrix <- confusionMatrix(data = factor(predicted_labels, levels = c(0, 1)),
                               reference = factor(y_test, levels = c(0, 1)))
print(conf_matrix)

# Function to create a heatmap from a confusion matrix

heatmap_confusion_matrix <- function(conf_matrix, title) {
  df <- as.data.frame(as.table(conf_matrix))

  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "lightblue", high = "lightcoral") +
    theme_minimal() +
    labs(title = title, x = "Actual", y = "Prediction") +
    geom_text(aes(label = Freq), vjust = 1.5) +
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "bold"),
          plot.title = element_text(size = 16, face = "bold"))
}

# heatmap for logistic regression
heatmap_confusion_matrix(conf_matrix, "Confusion Matrix (Logistic Regression)")


# Calculating performance metrices- precision, recall, and F1-score

precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]

# Printing the results

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1_score, "\n")

#Roc curve
roc_curve <- pROC::roc(y_test, predictions)

# Ploting ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# Adding AUC (area under curve) to the plot

auc_value <- pROC::auc(roc_curve)
legend("bottomright", legend = paste("AUC =", round(auc_value, 2)), col = "blue", lwd = 2)

# Adding the diagonal line for reference
abline(a = 0, b = 1, lty = 2, col = "gray")

# Printing the ROC curve
print(roc_curve)



# Naive bayes model without sampling

library(e1071)


naive_bayes_model <- naiveBayes(as.factor(y_train) ~ ., data = X_train)

# Predictions on the test set
predictions_nb <- predict(naive_bayes_model, newdata = X_test, type = "raw")

# Evaluating the model-finding accuracy

predicted_labels_nb <- ifelse(predictions_nb[, 2] > 0.5, 1, 0)
accuracy_nb <- mean(predicted_labels_nb == y_test)
cat("Accuracy on the test set (Naive Bayes):", accuracy_nb, "\n")

# Creating  a confusion matrix

conf_matrix_nb <- confusionMatrix(data = factor(predicted_labels_nb, levels = c(0, 1)),
                                  reference = factor(y_test, levels = c(0, 1)))
print(conf_matrix_nb)

#creating heatmap for Naive Bayes

heatmap_confusion_matrix(conf_matrix_nb, "Confusion Matrix (Naive Bayes)")


# Calculating Performance metrices like precision, recall, and F1-score

precision_nb <- conf_matrix_nb$byClass["Precision"]
recall_nb <- conf_matrix_nb$byClass["Recall"]
f1_score_nb <- conf_matrix_nb$byClass["F1"]

# Printing the results

cat("Precision (Naive Bayes):", precision_nb, "\n")
cat("Recall (Naive Bayes):", recall_nb, "\n")
cat("F1-Score (Naive Bayes):", f1_score_nb, "\n")

#ROC curve
roc_curve_nb <- pROC::roc(y_test, predictions_nb[, 2])

# Ploting ROC curve

plot(roc_curve_nb, main = "ROC Curve (Naive Bayes)", col = "green", lwd = 2)

# Adding AUC to the plot

auc_value_nb <- pROC::auc(roc_curve_nb)
legend("bottomright", legend = paste("AUC (Naive Bayes) =", round(auc_value_nb, 2)), col = "green", lwd = 2)

# Adding the diagonal line for reference
abline(a = 0, b = 1, lty = 2, col = "gray")

# Printing the ROC Curve
print(roc_curve_nb)

library(data.table)
library(e1071)
library(ROCR)


# Function to calculate classification metrics
calculate_classification_metrics <- function(true_labels, predicted_labels) {
  confusion_matrix <- confusionMatrix(predicted_labels, true_labels)
  precision <- confusion_matrix$byClass['Pos Pred Value']
  recall <- confusion_matrix$byClass['Sensitivity']
  f1_score <- 2 * (precision * recall) / (precision + recall)

  tp <- confusion_matrix$table[2, 2]
  tn <- confusion_matrix$table[1, 1]
  fp <- confusion_matrix$table[1, 2]
  fn <- confusion_matrix$table[2, 1]

  cat("True Positives:", tp, "\n")
  cat("True Negatives:", tn, "\n")
  cat("False Positives:", fp, "\n")
  cat("False Negatives:", fn, "\n")

  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")

  return(list(
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn
  ))
}

# Function to plot confusion matrix
plot_confusion_matrix <- function(conf_matrix, title) {
  # Plot the confusion matrix
  image(conf_matrix, main = title, col = c("red", "Blue"),
        xlab = "Predicted", ylab = "Actual")
}


##############Naima Rashid Modelling ##############################################

# Part 1: KNN with Date and Location
df_knn <- df_dat_enc

# Feature Engineering
X_features <- df_knn[, !names(df_knn) %in% c("RainTomorrow")]
df_knn$RainTomorrow <- as.factor(df_knn$RainTomorrow)

# Split the data into training and testing sets
set.seed(42)
splitIndex <- createDataPartition(df_knn$RainTomorrow, p = 0.8, list = FALSE)
X_train <- X_features[splitIndex, ]
X_test <- X_features[-splitIndex, ]
y_train <- df_knn$RainTomorrow[splitIndex]
y_test <- df_knn$RainTomorrow[-splitIndex]

# Initialize the KNN classifier with default parameters
knn_model_date_loc <- knn(train = X_train, test = X_test, cl = y_train, k = 5)

# Calculate accuracy
accuracy_date_loc <- sum(knn_model_date_loc == y_test) / length(y_test)
cat("Accuracy with default parameters (Date and Location):", accuracy_date_loc, "\n")

# Confusion Matrix
confusion_matrix_date_loc <- confusionMatrix(knn_model_date_loc, y_test)

# Extract Precision, Recall, and F1 Score
precision_date_loc <- confusion_matrix_date_loc$byClass['Pos Pred Value']
recall_date_loc <- confusion_matrix_date_loc$byClass['Sensitivity']
f1_score_date_loc <- 2 * (precision_date_loc * recall_date_loc) / (precision_date_loc + recall_date_loc)

cat("Precision (Date and Location):", precision_date_loc, "\n")
cat("Recall (Date and Location):", recall_date_loc, "\n")
cat("F1 Score (Date and Location):", f1_score_date_loc, "\n")

# Calculate AUC-ROC
auc_roc_date_loc <- calculate_auc_roc(knn_model_date_loc, as.numeric(y_test))
cat("AUC-ROC (Date and Location):", auc_roc_date_loc, "\n")


# Part 2: KNN without Date and Location - Basic Model
df_knn_basic <- df_loc_dat

# Feature Engineering
X_features_basic <- df_knn_basic[, !names(df_knn_basic) %in% c("RainTomorrow")]
df_knn_basic$RainTomorrow <- as.factor(df_knn_basic$RainTomorrow)

# Split the data into training and testing sets
set.seed(42)
splitIndex_basic <- createDataPartition(df_knn_basic$RainTomorrow, p = 0.8, list = FALSE)
X_train_basic <- X_features_basic[splitIndex_basic, ]
X_test_basic <- X_features_basic[-splitIndex_basic, ]
y_train_basic <- df_knn_basic$RainTomorrow[splitIndex_basic]
y_test_basic <- df_knn_basic$RainTomorrow[-splitIndex_basic]

# Initialize the KNN classifier with default parameters
knn_model_basic <- knn(train = X_train_basic, test = X_test_basic, cl = y_train_basic, k = 5)

# Calculate accuracy
accuracy_basic <- sum(knn_model_basic == y_test_basic) / length(y_test_basic)
cat("Accuracy with default parameters (Basic Model):", accuracy_basic, "\n")

# Confusion Matrix
confusion_matrix_basic <- confusionMatrix(knn_model_basic, y_test_basic)

# Extract Precision, Recall, and F1 Score
precision_basic <- confusion_matrix_basic$byClass['Pos Pred Value']
recall_basic <- confusion_matrix_basic$byClass['Sensitivity']
f1_score_basic <- 2 * (precision_basic * recall_basic) / (precision_basic + recall_basic)

cat("Precision (Basic Model):", precision_basic, "\n")
cat("Recall (Basic Model):", recall_basic, "\n")
cat("F1 Score (Basic Model):", f1_score_basic, "\n")

# Calculate AUC-ROC
auc_roc_basic <- calculate_auc_roc(knn_model_basic, as.numeric(y_test_basic))
cat("AUC-ROC (Basic Model):", auc_roc_basic, "\n")

# Part 3: Hyperparameter Tuning
param_grid_hyperparam <- expand.grid(k = c(3, 5, 7, 9, 11))
ctrl_hyperparam <- trainControl(method = "cv", number = 5)
knn_tune_hyperparam <- train(x = X_train, y = y_train, method = "knn", trControl = ctrl_hyperparam, tuneGrid = param_grid_hyperparam)

# Get the best parameters and the best accuracy
best_params_hyperparam <- knn_tune_hyperparam$bestTune
best_accuracy_hyperparam <- knn_tune_hyperparam$results$Accuracy[which.max(knn_tune_hyperparam$results$Accuracy)]
best_k_hyperparam <- best_params_hyperparam$k

cat("Best Accuracy after Hyperparameter Tuning:", best_accuracy_hyperparam, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_model_hyperparam <- knn(train = X_train, test = X_test, cl = y_train, k = best_k_hyperparam)

# Calculate the accuracy of the KNN model with the best parameters
accuracy_hyperparam <- sum(knn_model_hyperparam == y_test) / length(y_test)
cat("Accuracy with the best 'k' parameter after Hyperparameter Tuning:", accuracy_hyperparam, "\n")

# Confusion Matrix after Hyperparameter Tuning
confusion_matrix_hyperparam <- confusionMatrix(knn_model_hyperparam, y_test)

# Extract Precision, Recall, and F1 Score after Hyperparameter Tuning
precision_hyperparam <- confusion_matrix_hyperparam$byClass['Pos Pred Value']
recall_hyperparam <- confusion_matrix_hyperparam$byClass['Sensitivity']
f1_score_hyperparam <- 2 * (precision_hyperparam * recall_hyperparam) / (precision_hyperparam + recall_hyperparam)

cat("Precision after Hyperparameter Tuning:", precision_hyperparam, "\n")
cat("Recall after Hyperparameter Tuning:", recall_hyperparam, "\n")
cat("F1 Score after Hyperparameter Tuning:", f1_score_hyperparam, "\n")

# Calculate AUC-ROC after Hyperparameter Tuning
auc_roc_hyperparam <- calculate_auc_roc(knn_model_hyperparam, as.numeric(y_test))
cat("AUC-ROC after Hyperparameter Tuning:", auc_roc_hyperparam, "\n")



# Part 4: KNN without Date and Location - Log Transform - Hyperparameter Tuning - All Features
df_knn_all <- df_loc_dat
df_knn_all$constant <- 1e-6
skewed_vars_all <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')

# Applying log transformation to skewed variables
df_knn_all[paste0(skewed_vars_all, '_log')] <- lapply(df_knn_all[skewed_vars_all], function(x) log(x + df_knn_all$constant))

# Selecting features and target variable
X_all <- df_knn_all[, !names(df_knn_all) %in% c("RainTomorrow")]
y_all <- df_knn_all$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_all <- createDataPartition(y_all, p = 0.8, list = FALSE)
X_train_all <- X_all[splitIndex_all, ]
X_test_all <- X_all[-splitIndex_all, ]
y_train_all <- y_all[splitIndex_all]
y_test_all <- y_all[-splitIndex_all]

# Standardizing the features
scaler_all <- preProcess(X_train_all, method = c("center", "scale"))
X_train_all <- predict(scaler_all, X_train_all)
X_test_all <- predict(scaler_all, X_test_all)

# Hyperparameter Tuning
param_grid_all <- expand.grid(k = c(1, 3, 5, 7, 9))
ctrl_all <- trainControl(method = "cv", number = 5)

# Use train function for hyperparameter tuning
knn_tune_all <- train(
  x = X_train_all,
  y = y_train_all,
  method = "knn",
  trControl = ctrl_all,
  tuneGrid = param_grid_all
)

# Get the best parameters and the best accuracy
best_params_all <- knn_tune_all$bestTune
best_accuracy_all <- knn_tune_all$results$Accuracy[which.max(knn_tune_all$results$Accuracy)]
best_k_all <- best_params_all$k

cat("Best k (All Features):", best_k_all, "\n")
cat("Best Accuracy (All Features):", best_accuracy_all, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_all_model <- knn(train = X_train_all, test = X_test_all, cl = y_train_all, k = best_k_all)

# Making predictions with the model using all features
y_pred_all <- as.factor(knn_all_model)

# Calculating accuracy
accuracy_all <- sum(y_pred_all == y_test_all) / length(y_test_all)
cat("Accuracy with all features (Hyperparameter Tuning):", accuracy_all, "\n")

# Check unique levels in y_pred_all and y_test_all
unique_levels_pred <- levels(y_pred_all)
unique_levels_test <- levels(y_test_all)

cat("Unique levels in y_pred_all:", unique_levels_pred, "\n")
cat("Unique levels in y_test_all:", unique_levels_test, "\n")


# Set levels of y_test_all based on unique levels in y_pred_all
y_test_all <- factor(y_test_all, levels = unique_levels_pred)

# Now calculate the confusion matrix
confusion_matrix_all <- confusionMatrix(y_pred_all, y_test_all)

# Extract Precision, Recall, and F1 Score
precision_all <- confusion_matrix_all$byClass['Pos Pred Value']
recall_all <- confusion_matrix_all$byClass['Sensitivity']
f1_score_all <- 2 * (precision_all * recall_all) / (precision_all + recall_all)

cat("Precision after Hyperparameter Tuning - All Features:", precision_all, "\n")
cat("Recall after Hyperparameter Tuning - All Features:", recall_all, "\n")
cat("F1 Score after Hyperparameter Tuning - All Features:", f1_score_all, "\n")

# Calculate AUC-ROC
auc_roc_all <- calculate_auc_roc(y_pred_all, as.numeric(y_test_all))
cat("AUC-ROC after Hyperparameter Tuning - All Features:", auc_roc_all, "\n")

# Part 5: KNN without Date and Location - Log transform - Hyperparameter Tuning - Selected Features
df_knn_selected <- df_loc_dat
df_knn_selected$constant <- 1e-6
skewed_vars_selected <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')

# Applying log transformation to skewed variables
df_knn_selected[paste0(skewed_vars_selected, '_log')] <- lapply(df_knn_selected[skewed_vars_selected], function(x) log(x + df_knn_selected$constant))

# Selecting features and target variable
selected_vars <- c('MinTemp', 'MaxTemp', 'Rainfall_log', 'Evaporation_log', 'Sunshine_log', 'WindGustSpeed_log', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm')
X_selected <- df_knn_selected[, selected_vars]
y_selected <- df_knn_selected$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_selected <- createDataPartition(y_selected, p = 0.8, list = FALSE)
X_train_selected <- X_selected[splitIndex_selected, ]
X_test_selected <- X_selected[-splitIndex_selected, ]
y_train_selected <- y_selected[splitIndex_selected]
y_test_selected <- y_selected[-splitIndex_selected]

# Standardizing the features
scaler_selected <- preProcess(X_train_selected, method = c("center", "scale"))
X_train_selected <- predict(scaler_selected, X_train_selected)
X_test_selected <- predict(scaler_selected, X_test_selected)

# Convert outcome variable to factor
y_train_selected <- as.factor(y_train_selected)

# Hyperparameter Tuning - Selected Features
param_grid_selected <- expand.grid(k = c(1, 3, 5, 7, 9))
ctrl_selected <- trainControl(method = "cv", number = 5)
knn_tune_selected <- train(x = X_train_selected, y = y_train_selected, method = "knn", trControl = ctrl_selected, tuneGrid = param_grid_selected)


# Get the best parameters and the best accuracy
best_params_selected <- knn_tune_selected$bestTune
best_accuracy_selected <- knn_tune_selected$results$Accuracy[which.max(knn_tune_selected$results$Accuracy)]
best_k_selected <- best_params_selected$k

cat("Results after Hyperparameter Tuning - Selected Features:\n")
cat("Best k:", best_k_selected, "\n")
cat("Best Accuracy:", best_accuracy_selected, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_selected_model <- knn(train = X_train_selected, test = X_test_selected, cl = y_train_selected, k = best_k_selected)

# Making predictions with the best model
y_pred_selected <- as.factor(knn_selected_model)
y_test_selected <- factor(y_test_selected, levels = unique_levels_pred)

# Confusion Matrix after Hyperparameter Tuning - Selected Features
confusion_matrix_selected <- confusionMatrix(y_pred_selected, y_test_selected)

# Extract Precision, Recall, and F1 Score after Hyperparameter Tuning - Selected Features
precision_selected <- confusion_matrix_selected$byClass['Pos Pred Value']
recall_selected <- confusion_matrix_selected$byClass['Sensitivity']
f1_score_selected <- 2 * (precision_selected * recall_selected) / (precision_selected + recall_selected)

cat("Precision after Hyperparameter Tuning - Selected Features:", precision_selected, "\n")
cat("Recall after Hyperparameter Tuning - Selected Features:", recall_selected, "\n")
cat("F1 Score after Hyperparameter Tuning - Selected Features:", f1_score_selected, "\n")

# Calculate AUC-ROC
auc_roc_selected <- calculate_auc_roc(y_pred_selected, as.numeric(y_test_selected))
cat("AUC-ROC after Hyperparameter Tuning - Selected Features:", auc_roc_selected, "\n")



# Part 6: KNN with SMOTE Balanced Dataset
# Creating a new dataframe with selected features
df_selected <- df_knn_selected[, c(selected_vars, "RainTomorrow")]

# Perform SMOTE oversampling on the dataset
df_SMOTE_balanced <- ovun.sample(RainTomorrow ~ . , data = df_selected, seed = 1)$data

# Selecting features and target variable
X_balanced <- df_SMOTE_balanced[, selected_vars]
y_balanced <- df_SMOTE_balanced$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_balanced <- createDataPartition(y_balanced, p = 0.8, list = FALSE)
X_train_balanced <- X_balanced[splitIndex_balanced, ]
X_test_balanced <- X_balanced[-splitIndex_balanced, ]
y_train_balanced <- y_balanced[splitIndex_balanced]
y_test_balanced <- y_balanced[-splitIndex_balanced]

# Initialize the KNN classifier with default parameters
knn_model_balanced <- knn(train = X_train_balanced, test = X_test_balanced, cl = y_train_balanced, k = 5)

# Calculate the accuracy of the KNN model on the balanced dataset
accuracy_balanced <- sum(knn_model_balanced == y_test_balanced) / length(y_test_balanced)
cat("Accuracy on the balanced dataset:", accuracy_balanced, "\n")

y_test_balanced <- factor(y_test_balanced, levels = levels(as.factor(knn_model_balanced)))

# Confusion Matrix on the balanced dataset
confusion_matrix_balanced <- confusionMatrix(knn_model_balanced, y_test_balanced)

# Extract Precision, Recall, and F1 Score on the balanced dataset
precision_balanced <- confusion_matrix_balanced$byClass['Pos Pred Value']
recall_balanced <- confusion_matrix_balanced$byClass['Sensitivity']
f1_score_balanced <- 2 * (precision_balanced * recall_balanced) / (precision_balanced + recall_balanced)

cat("Precision on the balanced dataset:", precision_balanced, "\n")
cat("Recall on the balanced dataset:", recall_balanced, "\n")
cat("F1 Score on the balanced dataset:", f1_score_balanced, "\n")

# Calculate AUC-ROC on the balanced dataset
auc_roc_balanced <- calculate_auc_roc(knn_model_balanced, as.numeric(y_test_balanced))
cat("AUC-ROC on the balanced dataset:", auc_roc_balanced, "\n")

############################# LOGISTIC REGRESSION ###################################################

# Part 1 - Logistic Regression without date and location
df_part1 <- df_loc_dat
df_part1$RainTomorrow <- as.factor(df_part1$RainTomorrow)

set.seed(42)
split_index_part1 <- createDataPartition(df_part1$RainTomorrow, p = 0.8, list = FALSE)
X_train_part1 <- df_part1[split_index_part1, !(names(df_part1) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part1 <- df_part1[-split_index_part1, !(names(df_part1) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part1 <- df_part1$RainTomorrow[split_index_part1]
y_test_part1 <- df_part1$RainTomorrow[-split_index_part1]

# Train the logistic regression model
logistic_model_part1 <- glmnet(as.matrix(X_train_part1), as.factor(y_train_part1), family = "binomial", alpha = 1, lambda = 0)

# Predict on the test set
predicted_probabilities_part1 <- predict(logistic_model_part1, newx = as.matrix(X_test_part1), type = "response")
predicted_labels_part1 <- as.factor(ifelse(predicted_probabilities_part1 > 0.5, 1, 0))

accuracy_part1 <- sum(predicted_labels_part1 == y_test_part1) / length(y_test_part1)
cat("Accuracy with default parameters:", accuracy_part1, "\n")

metrics_part1 <- calculate_classification_metrics(y_test_part1, predicted_labels_part1)

# Calculate AUC-ROC for the logistic regression model in Part 1
roc_curve_part1 <- multiclass.roc(as.numeric(y_test_part1), as.numeric(predicted_probabilities_part1))
auc_roc_part1 <- auc(roc_curve_part1)
cat("AUC-ROC with default parameters:", auc_roc_part1, "\n")

# Part 2 - Logistic Regression without date and location with log transformation
df_part2 <- df_loc_dat
skewed_vars_part2 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part2[paste0(skewed_vars_part2, '_log')] <- lapply(df_part2[skewed_vars_part2], function(x) log(x + 1))
df_part2$RainTomorrow <- as.factor(df_part2$RainTomorrow)

set.seed(42)
split_index_part2 <- createDataPartition(df_part2$RainTomorrow, p = 0.8, list = FALSE)
X_train_part2 <- df_part2[split_index_part2, !(names(df_part2) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part2 <- df_part2[-split_index_part2, !(names(df_part2) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part2 <- df_part2$RainTomorrow[split_index_part2]
y_test_part2 <- df_part2$RainTomorrow[-split_index_part2]

# Train the logistic regression model
logistic_model_part2 <- glmnet(as.matrix(X_train_part2), as.factor(y_train_part2), family = "binomial", alpha = 1, lambda = 0)

# Apply log transformation to skewed variables in the test set
X_test_part2[paste0(skewed_vars_part2, '_log')] <- lapply(X_test_part2[skewed_vars_part2], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part2 <- predict(logistic_model_part2, newx = as.matrix(X_test_part2), type = "response")
predicted_labels_part2 <- as.factor(ifelse(predicted_probabilities_part2 > 0.5, 1, 0))

accuracy_part2 <- sum(predicted_labels_part2 == y_test_part2) / length(y_test_part2)
cat("Accuracy with log-transformed:", accuracy_part2, "\n")

metrics_part2 <- calculate_classification_metrics(y_test_part2, predicted_labels_part2)


# Calculate AUC-ROC for the logistic regression model in Part 2
roc_curve_part2 <- multiclass.roc(as.numeric(y_test_part2), as.numeric(predicted_probabilities_part2))
auc_roc_part2 <- auc(roc_curve_part2)
cat("AUC-ROC with default parameters:", auc_roc_part2, "\n")


# Part 3 - Logistic Regression without date and location with log transformation and hyperparameter tuning
df_part3 <- df_loc_dat

# Apply log transformation to skewed variables
skewed_vars_part3 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part3[paste0(skewed_vars_part3, '_log')] <- lapply(df_part3[skewed_vars_part3], function(x) log(x + 1))

df_part3$RainTomorrow <- as.factor(df_part3$RainTomorrow)

# Define a grid of alpha and lambda values for hyperparameter tuning
param_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),  # alpha values from 0 to 1
  lambda = seq(0.0001, 1, length = 10)  # lambda values from a small positive value to 1
)


# Splitting the data into features and target variable
X_part3 <- df_part3[, !(names(df_part3) %in% c("RainTomorrow", "Date", "Location"))]
y_part3 <- df_part3$RainTomorrow

# Splitting the data into training and testing sets
set.seed(42)
split_index_part3 <- createDataPartition(y_part3, p = 0.8, list = FALSE)
X_train_part3 <- X_part3[split_index_part3, ]
X_test_part3 <- X_part3[-split_index_part3, ]
y_train_part3 <- y_part3[split_index_part3]
y_test_part3 <- y_part3[-split_index_part3]

# Train the logistic regression model with hyperparameter tuning
ctrl_part3 <- trainControl(method = "cv", number = 5)
logistic_tune_part3 <- train(
  x = X_train_part3,
  y = y_train_part3,
  method = "glmnet",
  trControl = ctrl_part3,
  tuneGrid = param_grid,
  family = "binomial"
)

# Getting the best hyperparameters
best_params_part3 <- logistic_tune_part3$bestTune
best_alpha_part3 <- best_params_part3$alpha
best_lambda_part3 <- best_params_part3$lambda

# Train the logistic regression model with the best hyperparameters
final_logistic_model_part3 <- glmnet(as.matrix(X_train_part3), as.factor(y_train_part3), family = "binomial", alpha = best_alpha_part3, lambda = best_lambda_part3)

# Apply log transformation to skewed variables in the test set
X_test_part3[paste0(skewed_vars_part3, '_log')] <- lapply(X_test_part3[skewed_vars_part3], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part3 <- predict(final_logistic_model_part3, newx = as.matrix(X_test_part3), type = "response")
predicted_labels_part3 <- as.factor(ifelse(predicted_probabilities_part3 > 0.5, 1, 0))

# Evaluate the model
accuracy_hyperparam_part3 <- sum(predicted_labels_part3 == y_test_part3) / length(y_test_part3)
cat("Accuracy on the test set:", accuracy_hyperparam_part3, "\n")

# Calculate additional metrics
metrics_part3 <- calculate_classification_metrics(y_test_part3, predicted_labels_part3)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part3 <- multiclass.roc(y_test_part3, as.numeric(predicted_probabilities_part3))
auc_roc_part3 <- auc(roc_curve_part3)
cat("AUC-ROC with default parameters:", auc_roc_part3, "\n")

# Part 4 - Logistic Regression without date and location with log transformation, hyperparameter tuning, and SMOTE balancing
df_part4 <- df_loc_dat

# Apply log transformation to skewed variables
skewed_vars_part4 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part4[paste0(skewed_vars_part4, '_log')] <- lapply(df_part4[skewed_vars_part4], function(x) log(x + 1))

df_part4$RainTomorrow <- as.factor(df_part4$RainTomorrow)

# Splitting the data into features and target variable
X_part4 <- df_part4[, !(names(df_part4) %in% c("RainTomorrow", "Date", "Location"))]
y_part4 <- df_part4$RainTomorrow

# Splitting the data into training and testing sets
set.seed(42)
split_index_part4 <- createDataPartition(y_part4, p = 0.8, list = FALSE)
X_train_part4 <- X_part4[split_index_part4, ]
X_test_part4 <- X_part4[-split_index_part4, ]
y_train_part4 <- y_part4[split_index_part4]
y_test_part4 <- y_part4[-split_index_part4]

# Add SMOTE to balance the training set
df_SMOTE_balanced_part4 <- ovun.sample(RainTomorrow ~ . , data = df_part4, seed = 1)$data
X_train_part4 <- df_SMOTE_balanced_part4[, !(names(df_SMOTE_balanced_part4) %in% c("RainTomorrow"))]
y_train_part4 <- df_SMOTE_balanced_part4$RainTomorrow

# Train the logistic regression model with hyperparameter tuning
ctrl_part4 <- trainControl(method = "cv", number = 5)
logistic_tune_part4 <- train(
  x = X_train_part4,
  y = y_train_part4,
  method = "glmnet",
  trControl = ctrl_part4,
  tuneGrid = param_grid,
  family = "binomial"
)

# Getting the best hyperparameters
best_params_part4 <- logistic_tune_part4$bestTune
best_alpha_part4 <- best_params_part4$alpha
best_lambda_part4 <- best_params_part4$lambda

# Train the logistic regression model with the best hyperparameters
final_logistic_model_part4 <- glmnet(as.matrix(X_train_part4), as.factor(y_train_part4), family = "binomial", alpha = best_alpha_part4, lambda = best_lambda_part4)

# Apply log transformation to skewed variables in the test set
X_test_part4[paste0(skewed_vars_part4, '_log')] <- lapply(X_test_part4[skewed_vars_part4], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part4 <- predict(final_logistic_model_part4, newx = as.matrix(X_test_part4), type = "response")
predicted_labels_part4 <- as.factor(ifelse(predicted_probabilities_part4 > 0.5, 1, 0))

# Evaluate the model
accuracy_hyperparam_part4 <- sum(predicted_labels_part4 == y_test_part4) / length(y_test_part4)
cat("Accuracy on the test set:", accuracy_hyperparam_part4, "\n")

# Calculate additional metrics
metrics_part4 <- calculate_classification_metrics(y_test_part4, predicted_labels_part4)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part4 <- multiclass.roc(as.numeric(y_test_part4), as.numeric(predicted_probabilities_part4))
auc_roc_part4 <- auc(roc_curve_part4)
cat("AUC-ROC with default parameters:", auc_roc_part4, "\n")


# Part 5 - Logistic Regression without date and location with SMOTE balancing
df_part5 <- df_loc_dat
df_part5$RainTomorrow <- as.factor(df_part5$RainTomorrow)

set.seed(42)
split_index_part5 <- createDataPartition(df_part5$RainTomorrow, p = 0.8, list = FALSE)
X_train_part5 <- df_part5[split_index_part5, !(names(df_part5) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part5 <- df_part5[-split_index_part5, !(names(df_part5) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part5 <- df_part5$RainTomorrow[split_index_part5]
y_test_part5 <- df_part5$RainTomorrow[-split_index_part5]

# Add SMOTE to balance the training set
df_SMOTE_balanced_part5 <- ovun.sample(RainTomorrow ~ . , data = df_part5, seed = 1)$data
X_train_part5 <- df_SMOTE_balanced_part5[, !(names(df_SMOTE_balanced_part5) %in% c("RainTomorrow"))]
y_train_part5 <- as.factor(df_SMOTE_balanced_part5$RainTomorrow)

# Apply SMOTE to the test set to match the number of variables
df_SMOTE_test_part5 <- ovun.sample(RainTomorrow ~ . , data = df_part5[-split_index_part5, ], seed = 1)$data
X_test_part5 <- df_SMOTE_test_part5[, !(names(df_SMOTE_test_part5) %in% c("RainTomorrow"))]
y_test_part5 <- as.factor(df_SMOTE_test_part5$RainTomorrow)

# Train the logistic regression model
logistic_model_part5 <- glmnet(as.matrix(X_train_part5), y_train_part5, family = "binomial", alpha = 1, lambda = 0)

# Predict on the test set
predicted_probabilities_part5 <- predict(logistic_model_part5, newx = as.matrix(X_test_part5), type = "response")
predicted_labels_part5 <- as.factor(ifelse(predicted_probabilities_part5 > 0.5, 1, 0))

accuracy_part5 <- sum(predicted_labels_part5 == y_test_part5) / length(y_test_part5)
cat("Accuracy with default parameters:", accuracy_part5, "\n")

metrics_part5 <- calculate_classification_metrics(y_test_part5, predicted_labels_part5)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part5 <- multiclass.roc(as.numeric(y_test_part5), as.numeric(predicted_probabilities_part5))
auc_roc_part5 <- auc(roc_curve_part5)
cat("AUC-ROC with default parameters:", auc_roc_part5, "\n")








#### Ratan Rana Paleka Sheeba 6826777.  #####


library(rpart)
library(rpart.plot)
library(glmnet)

#Dropping the Date column from the final data

final_data_new <- final_data %>% select(-Date)

#Checking the number of na values in the new dataset
total_NA <- sum(is.na(final_data_new))
print(total_NA)

#Splitting the features into training and test sets
indexSet <- sample(2,nrow(final_data_new), replace = T, prob = c(0.8,0.2))
trained<- final_data_new[indexSet == 1,]
tested <- final_data_new[indexSet == 2,]


#Implementing Logistic Regression algorithm
logreg <- glm(RainTomorrow ~ ., data = trained, family = 'binomial' )
summary(logreg)

# Predicting on the test data
predicted_probs <- predict(logreg, newdata = tested, type = "response")
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Implementing Confusion Matrix
conf_matrix <- table(tested$RainTomorrow, predicted_classes)


# Evaluating the performance of Logistic Regression using necessary metrics.
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_score <- 2 * ((precision * recall) / (precision + recall))

# Implementing ROC Curve and AUC.

roc_obj_lr <- roc(tested$RainTomorrow, predicted_probs)
auc_score_lr <- auc(roc_obj_lr)

# Printing the metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("AUC Score:", auc_score_lr, "\n")

# Plotting ROC curve

plot(roc_obj_lr, main = "ROC Curve")
abline(a = 0, b = 1, col = "red")

#--------------------------------------

#Implementing Decision Tree algorithm

# Ensure the target variable 'RainTomorrow' is a factor
df_dat_enc$RainTomorrow <- as.factor(df_dat_enc$RainTomorrow)

# Splitting the dataset into training and testing sets

splitIndex <- createDataPartition(df_dat_enc$RainTomorrow, p = .70, list = FALSE, times = 1)
trainData <- df_dat_enc[splitIndex,]
testData <- df_dat_enc[-splitIndex,]


# Building Decision Tree
treeModel <- rpart(RainTomorrow ~ ., data = trained, method = "class")

# Visualizing the tree
rpart.plot(treeModel, main="Decision Tree", extra=100)

# Predictions on test set
predictions <- predict(treeModel, testData, type = "class")

# Model evaluation
# Create the confusion matrix
conf_matrix <- confusionMatrix(predictions, testData$RainTomorrow)

# Extracting metrics from confusion matrix
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Sensitivity']
f1_score <- 2 * ((precision * recall) / (precision + recall))


# ROC and AUC calculation
roc_obj_dt <- roc(tested$RainTomorrow, predicted_probs)
auc_value_dt <- auc(roc_obj_dt)


# Printing the metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("AUC:", auc_value_dt, "\n")

# Plotting ROC curve

plot(roc_obj_dt, main = "ROC Curve")
abline(a = 0, b = 1, col = "red")









#######Sandeep-6829480 Modeling######
library(xgboost)

#User Defined Functions
# ************************************************
# To manually set a field type
# This will store $name=field name, $type=field type
DATASET_FILENAME  <- "weatherAUS.csv"
TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
TYPE_IGNORE       <- "IGNORE"             # field is not encoded
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




##############Govind Konnanat (ID-6797843) #####################


#####LOADING THE LIBRARIES####
library(devtools)
library(GGally)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(ggcorrplot)
library(data.table)
library(caret)
library(gplots)
library(corrplot)
library(pROC)
library(broom)
library(tune)
library(tidymodels)
library(glmnet)
library(yardstick)
library(ROSE)
library(keras)
library(rsample)
library(tensorflow)
library(conflicted)
# Set a seed for reproducibility


# Extract month, year, and day into separate version of the dataset
df_dat_enc$Month <- as.integer(data.table::month(df_dat_enc$Date))
df_dat_enc$Year <- as.integer(data.table::year(df_dat_enc$Date))
df_dat_enc$Day <- as.integer(format(df_dat_enc$Date, "%d"))

#Dropping the data column
df_dat_enc <- df_dat_enc[, -which(names(df_dat_enc) == "Date")]



# Create an results data frame for comparsion of all models
results_df <- data.frame(
  Model = character(),
  Dataset = character(),
  Accuracy = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  AUC=numeric(),
  F1_Score=numeric(),
  stringsAsFactors = FALSE
)
colnames(results_df) <- c("Model", "Dataset", "Accuracy", "Precision", "Recall","AUC","F1-Score")

##### User Defined Functions for Modelling


# Function to create training and testing sets
create_train_test_split <- function(data, target_column, train_percentage = 0.6) {


  #Reference : https://www.statology.org/createdatapartition-in-r/

  # Use  createDataPartition from rsample for class imbalance
  split_data <- initial_split(data, prop = train_percentage)

  train_data <- training(split_data)
  test_data <- testing(split_data)

  # Return the training and testing sets
  return(list(train_data = train_data, test_data = test_data))
}



# Function to plot a confusion matrix visually
plot_confusion_matrix <- function(conf_matrix, title) {
  conf_matrix_df <- as.table(as.table(conf_matrix))

  ggplot(data = data.frame(conf_matrix_df), aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), vjust = 1) +  # Add frequency count as text
    scale_fill_gradient(low = "white", high = "orange", na.value = NA) +
    theme_minimal() +
    labs(title = title,
         x = "Actual",
         y = "Predicted",
         fill = "Frequency") +
    guides(fill = guide_legend(title = "Frequency"))
}

#Function to encode date
encode_date <- function(data, col, max_val) {
  data[[paste0(col, '_sin')]] <- sin(2 * pi * data[[col]] / max_val)
  data[[paste0(col, '_cos')]] <- cos(2 * pi * data[[col]] / max_val)
  return(data)
}

#Function to print metrics by class
calculate_metrics_byclass <- function(confusion_matrix) {
  TP <- confusion_matrix$table[2, 2]
  FP <- confusion_matrix$table[1, 2]
  FN <- confusion_matrix$table[2, 1]
  TN <- confusion_matrix$table[1, 1]

  precision_pos <- TP / (TP + FP)
  recall_pos <- TP / (TP + FN)
  f1_score_pos <- 2 * precision_pos * recall_pos / (precision_pos + recall_pos)

  precision_neg <- TN / (TN + FN)
  recall_neg <- TN / (TN + FP)
  f1_score_neg <- 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

  metrics_byClass <- data.frame(
    Class = c("Positive (1)", "Negative (0)"),
    Precision = c(precision_pos, precision_neg),
    Recall = c(recall_pos, recall_neg),
    F1_Score = c(f1_score_pos, f1_score_neg)
  )

  return(metrics_byClass)
}


######### Model-1(Logistic Regression) ######


#####Splitting into train and test

#First I will be using the dataset without date and location and will further add features and compare performance

# Define the target variable and features
target <- 'RainTomorrow'
features <- setdiff(names(df_loc_dat), target)

resultset <- create_train_test_split(df_loc_dat, target)

# Access the training and testing sets
df_loc_dat_train_data <- resultset$train_data
df_loc_dat_test_data <- resultset$test_data

table(df_loc_dat_train_data$RainTomorrow)
#this deals with some of class imbalance as it samples data based on the RainTommorow variable for the train  and testset

##########Model Training##########

####Dataset-1 (Without Location and Date)####

# Train the logistic regression model
df_loc_dat_model <- glm(as.formula(paste(target, '~', paste(features, collapse = '+'))),
                        data = df_loc_dat_train_data, family = 'binomial')

#predicting on the test data
test_predictions <- predict(df_loc_dat_model, newdata = df_loc_dat_test_data, type = "response")

#squishing the  prediction values to 0 and 1
testpredictedClass <- ifelse(test_predictions > 0.5, 1, 0)

# Generating a classification report to measure performance
print(confusionMatrix(as.factor(testpredictedClass), as.factor(df_loc_dat_test_data[[target]])))


TestconfusionMatrix<-confusionMatrix(as.factor(testpredictedClass), as.factor(df_loc_dat_test_data[[target]]))
#plotting the matrix visually
plot_confusion_matrix(TestconfusionMatrix,"Confusion Matrix Without Location and Date")

metrics_byClass_Loc_Date<- calculate_metrics_byclass(TestconfusionMatrix)

print(metrics_byClass_Loc_Date)
#As we can see here the model for this dataset gives good accuracy on when it's not raining but
# not when is going to rain so we will keep this in mind when going forward with other models and datasets.

# peep Accuracy
TestconfusionMatrix$overall['Accuracy']

# peep sensitivity (recall)
TestconfusionMatrix$byClass['Sensitivity']

#The model without the location and the dates gives us the accuracy of .0834,

# The p-value is extremely small(<2.2e-16), which is close to zero. This is strong evidence to reject the null hypothesis


#Checking for over-fitting on the training data

#predicting on the training data
train_predictions <- predict(df_loc_dat_model, newdata = df_loc_dat_train_data, type = "response")

trainpredictedClass <- ifelse(train_predictions > 0.5, 1, 0)


# Generating a classification report on train
TrainconfusionMatrix<-confusionMatrix(as.factor(trainpredictedClass), as.factor(df_loc_dat_train_data[[target]]))

TrainconfusionMatrix$overall['Accuracy']
#The training accuracy score is 0.8454 while the test accuracy to be 0.0834,
#These two values are quite similar, so we can conclude there is no overfitting.



# Fitting a ROC curve
test_roc_curve <- roc(df_loc_dat_test_data$RainTomorrow, test_predictions)

# Plot the ROC curve
plot(test_roc_curve, main = "ROC Curve of Test without Location & Date", col = "black", lwd = 5)

# Add  value of AUC to the plot
df_loc_dat_auc_value <- auc(test_roc_curve)
legend("bottomright", legend = paste("AUC =", round(df_loc_dat_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.85 indicates that this model can classify between the positive and negative classes pretty well

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("Logistic Regression", "Dataset without Location & Date", TestconfusionMatrix$overall['Accuracy'],
    TestconfusionMatrix$byClass['Precision'],TestconfusionMatrix$byClass['Sensitivity'],df_loc_dat_auc_value,TestconfusionMatrix$byClass['F1'])
)

# Print the data frame
print(results_df)


####Dataset-2 (Date Encoded)####
#I am going to use the dataset with location and date to see if it adds to the performance of the model


#One hot encoding the location variable which should increase 49 addition columns
dummy <- dummyVars(" ~ Location", data=df_dat_enc)

#creating a location encoded dataframe
loc_enc <- data.frame(predict(dummy, newdata = df_dat_enc))

#joining them together
df_dat_enc<-cbind(df_dat_enc, loc_enc)

#dropping the location column from the original dataframe
df_dat_enc <- subset(df_dat_enc, select = -Location)

####Splitting the  test and train for 2nd dataset

resultset <- create_train_test_split(df_dat_enc, target)

# Access the training and testing sets
df_dat_enc_train_data <- resultset$train_data
df_dat_enc_test_data <- resultset$test_data


##Model Training
features <- setdiff(names(df_dat_enc), target)

# Train the logistic regression model
df_dat_enc_model <- glm(as.formula(paste(target, '~', paste(features, collapse = '+'))),
                        data = df_dat_enc_train_data, family = 'binomial')

#predicting on the test data
test_predictions <- predict(df_dat_enc_model, newdata = df_dat_enc_test_data, type = "response")

#squishing the  prediction values to 0 and 1
testpredictedClass <- ifelse(test_predictions > 0.5, 1, 0)

# Generating a classification report to measure performance
print(confusionMatrix(as.factor(testpredictedClass), as.factor(df_dat_enc_test_data[[target]])))

TestconfusionMatrix<-confusionMatrix(as.factor(testpredictedClass), as.factor(df_dat_enc_test_data[[target]]))
#The model with the location encoded and the dates gives us  the accuracy of 0.8407 which is slightly better than the previous model

plot_confusion_matrix(TestconfusionMatrix,"Confusion Matrix With Location and Date")

metrics_byClass_dat_enc <- calculate_metrics_byclass(TestconfusionMatrix)

print(metrics_byClass_dat_enc)
#As we can see here , the precision is almost 93 % for the negative class but,
#the model captures around 70% of the actual instances of rain which has increased after adding location and date


# Fitting a ROC curve
test_roc_curve <- roc(df_dat_enc_test_data$RainTomorrow, test_predictions)

# Plot the ROC curve
plot(test_roc_curve, main = "ROC Curve of Test with Location & Date", col = "black", lwd = 5)

# Add  value of AUC to the plot
df_dat_enc_auc_value <- auc(test_roc_curve)
legend("bottomright", legend = paste("AUC =", round(df_dat_enc_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.856 is really close to the previous model

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("Logistic Regression", "Dataset with  Encoded Location & Date", TestconfusionMatrix$overall['Accuracy'],
    TestconfusionMatrix$byClass['Precision'],TestconfusionMatrix$byClass['Sensitivity'],df_dat_enc_auc_value,TestconfusionMatrix$byClass['F1'])
)

# Print the data frame
colnames(results_df) <- c("Model", "Dataset", "Accuracy", "Precision", "Recall","AUC","F1-Score")
print(results_df)

####Identifying the top features which contributes to the prediction####

tidy_df <- tidy(df_dat_enc_model)

top_features <- tidy_df[order(abs(tidy_df$estimate), decreasing = TRUE),]

print(top_features)

top_15_features <- head(top_features, 15)

#plot the top features
ggplot(top_15_features, aes(x = reorder(term, estimate), y = estimate, fill = estimate > 0)) +
  geom_bar(stat = "identity", position = "identity", color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("pink", "purple")) +
  labs(title = "Top  15 Features by Coefficient",
       x = "Coefficient Estimate",
       y = "Feature") +
  theme_minimal()

#since Sunshine had 55 % null values  and , with coefficent estimate high(-1.63) & p value is low, Along with Cloud3pm
#there is a chance of bias so we will dropping it when tuning.

#Update: The columns were dropped and the model were trained but this led to an accuracy drop of 2%
#so it did help with unseen data

######Hyper Parameter Tuning for logistic regression####

#Reference=https://www.datacamp.com/tutorial/logistic-regression-R

# model with penalty and hyperparameters
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet")

# Define the grid search for the hyperparameters
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 1, penalty = 5))

# Define the workflow for the model
log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_formula(RainTomorrow ~ .)

df_dat_enc_train_data$RainTomorrow <- as.factor(df_dat_enc_train_data$RainTomorrow)

# Define the resampling method for the grid search
folds <- vfold_cv(df_dat_enc_train_data, v = 10)


# using gridsearchCV for fine-tuning
log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

#selecting the best model based on AUC
select_best(log_reg_tuned, metric = "accuracy")

#Running the model with the hyper parameters

df_dat_enc_tuned_model <- logistic_reg(penalty = 0.000000001, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(RainTomorrow~., data = df_dat_enc_train_data)

# Evaluate the tuned model
tuned_testpredictedClass <- predict(df_dat_enc_tuned_model,
                                    new_data = df_dat_enc_test_data,
                                    type = "class")

tuned_testpredicted <- predict(df_dat_enc_tuned_model,
                               new_data = df_dat_enc_test_data,
                               type = "prob")

#converting the dataframe to a numeric vector
tuned_testpredictedClass <- tuned_testpredictedClass$.pred_class

confusionMatrix(as.factor(tuned_testpredictedClass), as.factor(df_dat_enc_test_data[[target]]))

TunedconfusionMatrix<-confusionMatrix(as.factor(tuned_testpredictedClass), as.factor(df_dat_enc_test_data[[target]]))
#The model with the tuning gives us  the accuracy which is slightly better than the previous model

plot_confusion_matrix(TunedconfusionMatrix,"Confusion Matrix With Hyperparameter Tuning")


confusionMatrix(as.factor(tuned_testpredictedClass), as.factor(df_dat_enc_test_data[[target]]),positive='1')

metrics_byClass_log_tuned<- calculate_metrics_byclass(TunedconfusionMatrix)
print(metrics_byClass_log_tuned)

#These metrics indicate there is a slight improvement after hyper parameter tuning

# Fitting a ROC curve
tuned_roc_curve <- roc(df_dat_enc_test_data$RainTomorrow, tuned_testpredicted$.pred_1)

# Plot the ROC curve
plot(tuned_roc_curve, main = "ROC Curve of Test with Tuning", col = "black", lwd = 5)

# Add  value of AUC to the plot
df_dat_enc_tuned_auc_value <- auc(tuned_roc_curve)
legend("bottomright", legend = paste("AUC =", round(df_dat_enc_tuned_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.86 is really close to the previous model

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("Logistic Regression", "Dataset After Hyper-parameter Tuning", TunedconfusionMatrix$overall['Accuracy'],
    TunedconfusionMatrix$byClass['Precision'],TunedconfusionMatrix$byClass['Sensitivity'],df_dat_enc_tuned_auc_value,TunedconfusionMatrix$byClass['F1'])
)

# Print the data frame
colnames(results_df) <- c("Model", "Dataset", "Accuracy", "Precision", "Recall","AUC","F-1 Score")
print(results_df)


### Train the logistic regression model using cross-validation

ctrl <- trainControl(method = "cv", number = 10)

conflicts()
conflicts_prefer(caret::train)


log_reg_cv <- train(
  RainTomorrow ~ .,
  data = df_dat_enc_train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl
)

# Print the cross-validated results
print(log_reg_cv)

conflicts_prefer(tensorflow::train)
#The accuracy after 10-fold cross validation was 0.8471 so , it does not significantly increase the performance of the model
#the number was increased to 20 which led to the drop of accuracy indicating overfitting

############# Model - 2 (Artificial Neural Networks) ############

#Getting the data ready for a NN
# We cannot use the same dataset that we used on logistic regression model as it is due to the high dimensionality it takes too long to train,


# so I will be doing dimensionality reduction by label encoding the categorical variables.

cat_columns <- c("Location", "WindGustDir","WindDir9am","WindDir3pm")

df_ANN <- data_clean[, !names(data_clean) %in% cat_columns]
# Label encode each categorical column
for (col in cat_columns) {
  df_ANN[[paste0(col, "_encoded")]] <- as.numeric(factor(data_clean[[col]]))
}

#Scaling the data
df_ANN[, -which(names(df_ANN) == "Date")] <- as.data.frame(apply(df_ANN[, -which(names(df_ANN) == "Date")] , 2, min_max_scale))

#The months and days  should be a cyclic continuous feature so that the model can understand the jump from 12 to 1  month and 30 to 1 for days.
#So I will transforming them using sin and cosine to get the amplitude and phase of the dates.

df_ANN$Date <- as.Date(df_ANN$Date, format="%d-%m-%Y")
# Extract month, year, and day into separate version of the dataset
df_ANN$Month <- as.integer(data.table::month(df_ANN$Date))
df_ANN$Year <- as.integer(data.table::year(df_ANN$Date))
df_ANN$Day <- as.integer(format(df_ANN$Date,"%d"))


df_ANN <- encode_date(df_ANN, 'Month', 12)

df_ANN <-encode_date(df_ANN, 'Day', 31)

#dropping the columns to reduce redundancy
df_ANN <-subset(df_ANN,select=-Day)
df_ANN <- subset(df_ANN, select = -Month)
df_ANN <- subset(df_ANN, select = -Date)

#Scaling the year column alone

df_ANN$Year <- min_max_scale(df_ANN$Year)

#Now we have 27 pure numerical columns which is be optimized for the Neural Network

##### Splitting the Neural Network and Dealing with Class Imbalance #####

resultset <- create_train_test_split(df_ANN, target)
#I have decided to go with a 60-40 split so that it represent enough minority class in the test set

# Access the training and testing sets
df_ANN_train_data <- resultset$train_data
df_ANN_test_data <- resultset$test_data

table(df_ANN_train_data$RainTomorrow)
table(df_ANN_test_data$RainTomorrow)

#applying the SMOTE method to deal with the class imbalance

#synthetic samples created by SMOTE are only for the train data in order to,
#prevent data leakage and providing a more accurate evaluation of the model's performance.


df_SMOTE_balanced<- ovun.sample(RainTomorrow ~ . , data = df_ANN_train_data, seed=1)$data
df_SMOTE_oversample <- ovun.sample(RainTomorrow ~ . , data = df_ANN_train_data, method = "over", N = 12663)$data
df_SMOTE_undersample <- ovun.sample(RainTomorrow ~ . , data = df_ANN_train_data, method = "under", N = 5000)$data

#see the number of majority and minority class after each sampling techniques

table(df_SMOTE_balanced$RainTomorrow)
table(df_SMOTE_oversample$RainTomorrow)
table(df_SMOTE_undersample$RainTomorrow)

# We have used SMOTE on df_SMOTE_balanced  to both oversampling of the minority class and undersampling of the majority class to create a balanced dataset.
# On the df_SMOTE_oversample we have oversampled the minority dataset by adding 2000 generated values for the minority class.
# On the df_SMOTE_undersample we have undersampled the majority dataset by removing values for the majority class.
# Since over sampling can add noise/overfitting, and undersampling can cause important data to be lost or uderfitting , we will test out the model on all the datasets.

#####Building the Neural Network####


#neuralnet library was used at first but the training was too long,so I use keras now

# Split the data into features and target variable
ANN_Model_X <- df_SMOTE_balanced[, -which(names(df_SMOTE_balanced) == "RainTomorrow")]
ANN_Model_y <- df_SMOTE_balanced$RainTomorrow



# Build the neural network model
ANN_model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",kernel_initializer = "uniform", input_shape = c(26)) %>%
  layer_dense(units = 32, kernel_initializer= "uniform",activation = "relu") %>%
  layer_dense(units = 16, kernel_initializer= "uniform",activation = "relu") %>%
  layer_dense(units = 8, kernel_initializer = "uniform",activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
ANN_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0009,clipnorm = 1.0),
  metrics = c("accuracy")
)
#View the shape of the neural network
summary(ANN_model)


ANN_Model_X_mat<-as.matrix(ANN_Model_X)

options(tensorflow.eager_mode = TRUE)

##### Fitting the NN on datasets ####

history <- ANN_model %>% fit(
  ANN_Model_X_mat, ANN_Model_y,
  epochs = 50,
  validation_split = 0.3,  # Use a portion of the data for validation
)


ANN_model_test_X<-df_ANN_test_data[, -which(names(df_ANN_test_data) == "RainTomorrow")]
ANN_model_test_Y<- df_ANN_test_data$RainTomorrow

# Convert test_data to a matrix
ANN_model_test_X <- as.matrix(ANN_model_test_X)

evaluation <- ANN_model %>% evaluate(
  x = ANN_model_test_X,
  y = ANN_model_test_Y
)
# The model seems to be doing well on the test data with accuracy of 0.8324 ,
#we will now test it on all the other datasets.


ANN_Model_Oversampled_X <- df_SMOTE_oversample[, -which(names(df_SMOTE_oversample) == "RainTomorrow")]
ANN_Model_Oversampled_y <- df_SMOTE_oversample$RainTomorrow

ANN_Model_Oversampled_X_mat<-as.matrix(ANN_Model_Oversampled_X)

options(tensorflow.eager_mode = TRUE)

# Train the model
history <- ANN_model %>% fit(
  ANN_Model_Oversampled_X_mat, ANN_Model_Oversampled_y,
  epochs = 20,
  validation_split = 0.3,  # Use a portion of the data for validation
)

#Evaluating the model again after fitting the oversampled data

evaluation <- ANN_model %>% evaluate(
  x = ANN_model_test_X,
  y = ANN_model_test_Y
)

#As it is observed here, the the accuracy drops a little bit on the testset ,so  we can conclude oversampling doesn't help much

ANN_Model_Undersampled_X <- df_SMOTE_undersample[, -which(names(df_SMOTE_undersample) == "RainTomorrow")]
ANN_Model_Undersampled_y <- df_SMOTE_undersample$RainTomorrow

ANN_Model_Undersampled_X<-as.matrix(ANN_Model_Undersampled_X)

# Train the model
history <- ANN_model %>% fit(
  ANN_Model_Undersampled_X, ANN_Model_Undersampled_y,
  epochs = 20,
  validation_split = 0.2,  # Use a portion of the data for validation
)

#Evaluating the model again after fitting the oversampled data

evaluation <- ANN_model %>% evaluate(
  x = ANN_model_test_X,
  y = ANN_model_test_Y
)

#With the undersampled data, the training accuracy increases but the test accuracy remains the same

#running the ANN without SMOTE
ANN_NoSmote_X <- df_ANN_train_data[, -which(names(df_ANN_train_data) == "RainTomorrow")]
ANN_NoSmote_y <- df_ANN_train_data$RainTomorrow

ANN_NoSmote_X<-as.matrix(ANN_NoSmote_X)

# Train the model
history <- ANN_model %>% fit(
  ANN_NoSmote_X, ANN_NoSmote_y,
  epochs = 50,
  validation_split = 0.4,  # Use a portion of the data for validation
)

# In this model we can see that the validation loss and loss in starting to diverge which is cause for overfitting,
# but the validation accuracy is much higher than other models

#Evaluating the  noSMOTE model

evaluation <- ANN_model %>% evaluate(
  x = ANN_model_test_X,
  y = ANN_model_test_Y
)

# Once again, the test accuracy is 0.8308 which is almost the same as models with, resampling technquies

#### Metrics Evaluation for NN ####

#fitting the best dataset (Balanced with SMOTE)
history <- ANN_model %>% fit(
  ANN_Model_X_mat, ANN_Model_y,
  epochs = 50,
  validation_split = 0.3,  # Use a portion of the data for validation
)

ANN_predictions <- ANN_model %>% predict(ANN_model_test_X)
ANN_binary_predictions <- ifelse(ANN_predictions > 0.5, 1, 0)

confusionMatrix(as.factor(ANN_binary_predictions), as.factor(df_ANN_test_data[[target]]))

ANN_ConfusionMatrix<-confusionMatrix(as.factor(ANN_binary_predictions), as.factor(df_ANN_test_data[[target]]))


plot_confusion_matrix(ANN_ConfusionMatrix,"Confusion Matrix of ANN")

metrics_byClass_ANN <- calculate_metrics_byclass(ANN_ConfusionMatrix)
print(metrics_byClass_ANN)

#As we can see here using the SMOTE method  for ANN significantly helps us predict the rainy days (58%): non rainy(88%).

# Rather then if we use the non SMOTE dataset which gives us better overall accuracy but more bias towards no rainy days(50%:90%)

# Fitting a ROC curve
ANN_roc_curve <- roc(df_ANN_test_data$RainTomorrow, ANN_binary_predictions)

# Plot the ROC curve
plot(ANN_roc_curve, main = "ROC Curve of ANN", col = "black", lwd = 5)

# Add  value of AUC to the plot
ANN_auc_value <- auc(ANN_roc_curve)
legend("bottomright", legend = paste("AUC =", round(ANN_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.74 is worse compared to logistic regression

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("ANN", "Dataset after class-balancing", ANN_ConfusionMatrix$overall['Accuracy'],
    ANN_ConfusionMatrix$byClass['Precision'],ANN_ConfusionMatrix$byClass['Sensitivity'],ANN_auc_value,ANN_ConfusionMatrix$byClass['F1'])
)

####Hyper-parameter Tuning for ANN####

ANN_tuned_model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",kernel_initializer = "uniform", input_shape = c(26)) %>%
  layer_dense(units = 16, kernel_initializer= "uniform",activation = "relu") %>%
  layer_dense(units = 2, kernel_initializer= "uniform",activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#I have reduced the dense layer, adjusted the learning rate and added a batch size parameter to fit the data better


# Compile the model
ANN_tuned_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0001,clipnorm = 1.0),
  metrics = c("accuracy")
)
#View the shape of the neural network
summary(ANN_tuned_model)


##### Fitting the NN on datasets ####

tuned_history <- ANN_tuned_model %>% fit(
  ANN_Model_X_mat, ANN_Model_y,
  epochs = 100,
  batch_size=64,
  validation_split = 0.2,  # Use a portion of the data for validation
)

evaluation <- ANN_tuned_model %>% evaluate(
  x = ANN_model_test_X,
  y = ANN_model_test_Y
)

ANN_predictions <- ANN_tuned_model %>% predict(ANN_model_test_X)
ANN_binary_predictions <- ifelse(ANN_predictions > 0.5, 1, 0)

confusionMatrix(as.factor(ANN_binary_predictions), as.factor(df_ANN_test_data[[target]]))

ANN_tuned_ConfusionMatrix<-confusionMatrix(as.factor(ANN_binary_predictions), as.factor(df_ANN_test_data[[target]]))


plot_confusion_matrix(ANN_tuned_ConfusionMatrix,"Confusion Matrix of  Tuned ANN")

metrics_byClass_Tuned_ANN<- calculate_metrics_byclass(ANN_tuned_ConfusionMatrix)

print(metrics_byClass_Tuned_ANN)


#Hyper parameter tuning helps us predict the rainy days (63%): non rainy(86%) better .

# Fitting a ROC curve
ANN_roc_curve <- roc(df_ANN_test_data$RainTomorrow, ANN_binary_predictions)

# Plot the ROC curve
plot(ANN_roc_curve, main = "ROC Curve of ANN", col = "black", lwd = 5)

# Add  value of AUC to the plot
ANN_auc_value <- auc(ANN_roc_curve)
legend("bottomright", legend = paste("AUC =", round(ANN_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.75 is improved as well as precision after ANN's epochs are increased

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("ANN", "Dataset After Hyper-parameter Tuning", ANN_tuned_ConfusionMatrix$overall['Accuracy'],
    ANN_tuned_ConfusionMatrix$byClass['Precision'],ANN_tuned_ConfusionMatrix$byClass['Sensitivity'],ANN_auc_value,ANN_tuned_ConfusionMatrix$byClass['F1'])
)

####Trying sampling in the logistic regression model ####

df_SMOTE_balanced$RainTomorrow <- as.factor(df_SMOTE_balanced$RainTomorrow)

sampled_log_reg_model <- logistic_reg(penalty = 0.000000001, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(RainTomorrow~., data = df_SMOTE_balanced)


sampled_log_reg_predictedClass <- predict(sampled_log_reg_model,
                                          new_data = df_ANN_test_data,
                                          type = "class")

sampled_log_reg_testpredicted <- predict(sampled_log_reg_model,
                                         new_data = df_ANN_test_data,
                                         type = "prob")

#converting the dataframe to a numeric vector
sampled_log_reg_predictedClass <- sampled_log_reg_predictedClass$.pred_class

confusionMatrix(as.factor(sampled_log_reg_predictedClass), as.factor(df_ANN_test_data[[target]]))

sampled_log_reg_confusionMatrix<-confusionMatrix(as.factor(sampled_log_reg_predictedClass), as.factor(df_ANN_test_data[[target]]))

metrics_byClass_sampled_log_reg<- calculate_metrics_byclass(sampled_log_reg_confusionMatrix)

print(metrics_byClass_sampled_log_reg)

# Fitting a ROC curve
sampled_log_reg_roc_curve <- roc(df_ANN_test_data$RainTomorrow, sampled_log_reg_testpredicted$.pred_1)

# Plot the ROC curve
plot(sampled_log_reg_roc_curve, main = "ROC Curve of Sampled Logistic Regression", col = "black", lwd = 5)

# Add  value of AUC to the plot
sampled_log_reg_auc_value <- auc(sampled_log_reg_roc_curve)
legend("bottomright", legend = paste("AUC =", round(sampled_log_reg_auc_value, 2)), col = "black", lty = 1, cex = 0.9)

#an AUC of 0.86 was obtained with the sampled dataframe which is the same as unsampled

#add all the metrics  of this model to the  results dataframe

results_df <- rbind(
  results_df,
  c("Logistic Regressiom", "Dataset After Sampling", sampled_log_reg_confusionMatrix$overall['Accuracy'],
    sampled_log_reg_confusionMatrix$byClass['Precision'],sampled_log_reg_confusionMatrix$byClass['Sensitivity'],sampled_log_reg_auc_value,sampled_log_reg_confusionMatrix$byClass['F1'])
)


#This marks the end of my modelling, we will be comparing and evaluating the results

##################################Conclusion##############################################

##### Generating final dataframe for comparison of all the models and datasets

metrics_byClass_sampled_log_reg<-metrics_byClass_sampled_log_reg  %>%
  mutate(Model = " Logistic Regression with Sampling")
metrics_byClass_Tuned_ANN <- metrics_byClass_Tuned_ANN %>%
  mutate(Model = "Tuned ANN")
metrics_byClass_Loc_Date <- metrics_byClass_Loc_Date %>%
  mutate(Model = "Logistic Reg without Location & Date")
metrics_byClass_dat_enc <- metrics_byClass_dat_enc %>%
  mutate(Model = "Logistic Reg with Date Encoded")
metrics_byClass_log_tuned<- metrics_byClass_log_tuned %>%
  mutate(Model = "Logistic Regression With Hyperparameter Tuning")
metrics_byClass_ANN <- metrics_byClass_ANN %>%
  mutate(Model = "ANN with Sampling Techniques")

#Combine them into a single data frame
metrics_byClass_Final <- bind_rows(
  metrics_byClass_Loc_Date,
  metrics_byClass_dat_enc,
  metrics_byClass_log_tuned,
  metrics_byClass_ANN,
  metrics_byClass_Tuned_ANN,
  metrics_byClass_sampled_log_reg
)
print(results_df)

#In this dataframe , we can see that logistic regression gives the best overall accuracy,recall& F1
# but cannot conclude it is the better as quickly as we have an imbalanced dataset , we need to evaluate
#how good the model is at predicting rainy days (ie:minority class),so i have complied a dataframe of
#all the metrics for the models by class and which gives best score for both positive and negative class.

#It is evident each model with each dataset/model outperforms the other in some or the other metric so we will
#diving more deeper into our specific use-case: ie predicting when there is rainfall

# Print the combined data frame
print(metrics_byClass_Final)

#In this notebook,I have tried to increase the performance of the positive class, as prioritize  identifying
#rain rather predicting when does it not rain.

# From this we can infer that , Neural network has better precision, F1 score, for positive class,
# due to the sampling techniques we performed in it, sampling increases the precision of the models.The same is observed in logistic regression
#with the precision of the postive class higher than any other model.
#the logistic regression model without location and date gives better recall and it would the ideal model as we want to maximize True Positives.




