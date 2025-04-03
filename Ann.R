#Group 12
#Students Names: Govind Konnanat, Naima Rashid, Akhil Bhardwaj, Venkat Sandeep Imandi, Ratan Rana Paleka Sheeba

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
 

#Predict if there is rain by running classification methods on  objective variable RainTomorrow.

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
set.seed(123)

#  clears all objects in "global environment"
rm(list=ls())


#########User Defined Funtions for EDA ####


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
############ DATA LOADING ####
#
# file preview 
full_df <- read.csv("weatherAUS.csv", header = TRUE)
#The file contains 10 years of weather data, out of which we just take 5000 observations.


# Randomly sample 10 % rows from the dataframe due to memory constraints
sampled_df <- full_df[sample(nrow(full_df), 14546), ]
write.csv(sampled_df, "weatherAUS_sampled.csv", row.names = FALSE)


#############Data EXPLORATION #######



# Display the sampled dataframe
head(sampled_df)

names(sampled_df)
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

###############EDA of Categorical Variables #####

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



###########Feature Engineering ####


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

#### Outlier removal ####


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

#######Creating multiple dataframes for training

final_data <- cbind(df_loc_dat, data_clean[, c("Date", "Location")])

#It is observed that datatype of Date column is string,we will encode it into datetime format.

final_data$Date <- as.Date(final_data$Date, format="%d-%m-%Y")
#creating a copy of final dataframe to create new version for modelling


df_dat_enc <- copy(final_data)
#This version of the data will have dates encoded as days,months and years as separate columns

# Extract month, year, and day into separate version of the dataset
df_dat_enc$Month <- as.integer(data.table::month(df_dat_enc$Date))
df_dat_enc$Year <- as.integer(data.table::year(df_dat_enc$Date))
df_dat_enc$Day <- as.integer(format(df_dat_enc$Date, "%d"))

#Dropping the data column
df_dat_enc <- df_dat_enc[, -which(names(df_dat_enc) == "Date")]


##############Govind Konnanat (ID-6797843) ##################### 


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
#with the precision of the postivie class higher than any other model.
#the logistic regression model without location and date gives better recall and it would the ideal model as we want to maximize True Positives.


#################### End of Modelling ################# 
