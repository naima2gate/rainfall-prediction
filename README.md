# **Rainfall Prediction and Influence Analysis Using Machine Learning**  

## **Overview**  
This project aims to predict rainfall using advanced machine learning techniques and analyze key meteorological factors that influence precipitation patterns. Leveraging a dataset spanning ten years of daily weather observations from various Australian weather stations, this study systematically evaluates multiple classification models. A core focus is addressing class imbalance while optimizing predictive accuracy, recall, and overall model performance.  

## **Dataset Description**  
- **Source**: [Kaggle - Australian Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data?select=weatherAUS.csv)  
- **Size**: 145,460 observations (after preprocessing)  
- **Target Variable**: `RainTomorrow` – A binary classification variable (1: rain occurred, 0: no rain)  
- **Features**:  
  - **Temporal Data**: Date, location of weather station  
  - **Temperature Metrics**: Minimum, maximum, and hourly temperature readings  
  - **Precipitation Metrics**: Rainfall, evaporation  
  - **Wind Characteristics**: Wind gust direction and speed, wind direction/speed at different times  
  - **Humidity and Pressure**: Humidity percentages, atmospheric pressure at different times  
  - **Cloud Cover**: Cloud density observations at key times  
  - **Binary Indicators**: `RainToday` (if rainfall >1mm)  

## **Exploratory Data Analysis (EDA)**  
A thorough data exploration phase was conducted to assess feature distributions, identify outliers, handle missing values, and understand correlations among meteorological variables.  
- **Distribution Analysis**: Examined numerical and categorical feature distributions  
- **Class Imbalance Evaluation**: Assessed disproportion in the `RainTomorrow` variable  
- **Correlation Matrix**: Identified key interdependencies between temperature, humidity, wind speed, and precipitation  
- **Outlier Detection**: Boxplots and histogram analysis to cap extreme values  
- **Missing Data Treatment**: Visualization of missing values to inform imputation strategies  

## **Data Preprocessing**  
Comprehensive preprocessing steps were undertaken to enhance model efficiency and reliability:  
- **Handling Missing Values**:  
  - Removed entries with missing target values (`RainTomorrow`)  
  - Imputed numerical variables using median values  
  - Imputed categorical variables using mode (most frequent category)  
- **Outlier Treatment**:  
  - Applied upper and lower capping to variables with extreme distributions (Rainfall, Evaporation, WindSpeed)  
- **Feature Engineering**:  
  - **Encoding**:  
    - Binary transformation for `RainToday` and `RainTomorrow`  
    - One-hot encoding for categorical variables (location, wind direction)  
  - **Scaling**:  
    - Min-max normalization applied to continuous variables  
  - **Date-Based Features**: Extracted month, seasonality patterns  
- **Class Imbalance Handling**:  
  - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples and balance positive/negative classes  

## **Machine Learning Models**  
A diverse range of machine learning algorithms were implemented, each evaluated based on classification performance metrics (accuracy, precision, recall, F1-score, and AUC):  
1. **Logistic Regression** – A baseline probabilistic model for binary classification  
2. **K-Nearest Neighbors (KNN)** – Distance-based classification method emphasizing local patterns  
3. **Decision Tree** – Hierarchical decision-based model to capture complex patterns  
4. **Naive Bayes** – A probabilistic approach assuming feature independence  
5. **Gradient Boosting Machine (GBM)** – An ensemble method using boosting for enhanced accuracy  
6. **Artificial Neural Networks (ANN)** – A deep learning-based approach capturing non-linear dependencies  

## **Hyperparameter Optimization**  
Each model underwent hyperparameter tuning to maximize predictive performance:  
- **Grid Search & Randomized Search** – For optimizing Decision Trees, KNN, and Naive Bayes  
- **Learning Rate & Regularization Adjustment** – For ANN and GBM  
- **Cross-Validation** – Applied stratified K-fold validation to ensure robust generalization  

## **Model Evaluation & Key Findings**  
- **Logistic Regression**: Established a strong baseline, demonstrating robust performance across precision and recall metrics.  
- **Artificial Neural Networks (ANN)**: Achieved high precision, effectively capturing non-linear relationships.  
- **K-Nearest Neighbors (KNN)**: Notable for high recall, favoring sensitivity over specificity.  
- **Feature Importance Analysis**:  
  - **Most Influential Predictors**:  
    - **Atmospheric Pressure** – Strong inverse correlation with rainfall probability  
    - **Temperature** – Affects evaporation and precipitation likelihood  
    - **Humidity** – Direct correlation with precipitation patterns  
    - **Wind Speed** – Influence on moisture movement and cloud formations  
  - **Least Impactful Predictors**: Variables such as `WindGustDir` exhibited minimal influence on predictions.  

## **Conclusions**  
- **Comparative Model Performance**: Logistic Regression served as a reliable baseline, while ANN and GBM demonstrated superior predictive accuracy.  
- **Impact of Class Imbalance**: Raw data exhibited significant class imbalance, which was mitigated using SMOTE, improving recall without overfitting.  
- **Feature Significance**: Pressure, temperature, humidity, and wind speed were identified as the most critical meteorological factors for rainfall prediction.  
- **Future Enhancements**:  
  - Incorporating deep learning architectures (e.g., LSTMs for sequential dependencies).  
  - Leveraging ensemble methods for further refinement.  
  - Integrating external meteorological datasets for enhanced forecasting capabilities.  
