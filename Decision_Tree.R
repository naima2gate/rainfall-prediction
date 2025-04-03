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
library(rpart)
library(rpart.plot)
library(pROC)
library(dplyr)
library(glmnet)
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

final_data$Date <- as.Date(final_data$Date, format="%d-%m-%Y")
#creating a copy of final dataframe to create new version for modelling


df_dat_enc <- copy(final_data)
#This version of the data will have dates encoded as days,months and years as separate columns

# Extract month, year, and day into separate version of the dataset
df_dat_enc$Month <- as.integer(month(df_dat_enc$Date))
df_dat_enc$Year <- as.integer(year(df_dat_enc$Date))
df_dat_enc$Day <- as.integer(day(df_dat_enc$Date))

#Dropping the data column
df_dat_enc <- df_dat_enc[, -which(names(df_dat_enc) == "Date")]

#_

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


