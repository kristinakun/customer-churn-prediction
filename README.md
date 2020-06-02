<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100" align="right"/>


#   Project - Customer Churn Prediction with Machine Learning

KRISTINA KUNCEVICIUTE

*Data Part Time Barcelona Dic 2019*


## Content
- [Project Description](#project)
- [Datasets](#datasets)
- [Notebooks](#notebooks)
- [Workflow](#workflow)
- [Libraries](#libraries)
- [Insights and Results](#results)
- [Recommendations](#recommendations)

<a name="project"></a>

## Project Description

- **Problem** - it’s cheaper to retain a customer than to acquire a new one but it’s too time-consuming and too expensive to focus on each individual customer. 

- **Solution** - by predicting in advance which customers will be leaving after the first month (and what are the features that influence them to do so), you could reduce customer retention efforts by focusing on segmented customers and distributing your marketing budget towards the most important features.

<a name="datasets"></a>

## Datasets

- raw_encrypted.csv - raw dataset, data was encrypted due to confidentiality reasons to not enclose specific business information
    - entry level data of the customers of the subscription business
    - data was collected during 2017-2019
    - data consists of 26799 entries (one row per customer)
- population_region.csv - additional information about the population per region
- income_region.csv - additional information about the average income per region
- clean_data.csv - clean dataset, with correct data types, additional features, no missing data, ready for encoding
- encoded_data.csv - encoded dataset, all categorical features transformed to numerical, ready for Machine Learning algorithms

<a name="notebooks"></a>

## Notebooks

- 1_EDA - Exploratory Data Analysis
- 2_Data_Preprocessing_and_Correlation - Data encoding and correlation check
- 3_Models - Main notebook, Machine Learning algorithms
- 4_Dimensionality_Reduction - Dimensionality Reduction Methods

<a name="workflow"></a>

## Workflow

1. Data Cleaning and Encryption
2. Exploratory Data Analysis
3. Data Preprocessing
    - Label Encoder
    - One Hot Encoder
4. Model Selection
    - Model Testing
        - Tested 10 models:
            - Random Forest Classifier
            - Extreme Gradient Boosting Classifier
            - Extra Trees Classifier
            - Light Gradient Boosting Machine Classifier
            - Logistic Regression 
            - Support Vector Machine
            - K-Nearest Neighbors Classifier
            - Decision Tree Classifier
            - Linear Support Vector Machine
            - Multi-Layer Perceptron Classifier
        - Identifying important features with Random Forest Classifier
        - Testing models with the short and long lists of the most important features
        - Running models with default parameters and comparing to results after hyperparameter tuning
        - Selecting 3 top-performing models for the in-depth hyperparameter tuning and scaler and balancing technique testing:
            - Random Forest Classifier
            - Support Vector Machine
            - K-Nearest Neighbors Classifier        
    - Model Tuning
        - Feature Scaling (Standard Scaler, MinMax Scaler, Normalizer)
        - Data Balancing (Undersampling: Near Miss, Instance Hardness Threshold. Oversampling: Synthetic Minority Oversampling (SMOTE))
        - Hyperparameter tuning
5. Conclusions and Recommendations

<a name="libraries"></a>

## Libraries

- Pandas
- NumPy
- Seaborn
- Matplotlib
- category_encoders
- Scikit-learn
- imbalanced-learn
- LightGBM
- XGBoost
- umap-learn
 
<a name="results"></a>

## Insights and Results

1. **Exploratory Data Analysis**
    - Our target feature (churned after 1st month) is not balanced, but the difference between the classes is not big: 54% of customers are churning after the first month and 46% staying longer than one month.
    - There is a big difference between the two genders. We have a lower number of customers with gender '1' and also it tends to churn more compared to the other class.
    - Channels 'a' and 'f' have a much higher number of churned customers compared to the rest. Channel 'e' has much higher retention compared to the rest.
    - Customers that don't subscribe to the newsletter tend to churn less.
    - The majority of the customer get their box delivered home. 
    - 95% of customers have signed up for their first box with a promotion.
    - Customers who had more touchpoints, tend to churn less.
    - There is no big difference in churn in any month.
    - **Main insights:**
        - Features as sex, master channel, newsletter, use pick up point, and is lead could have the biggest impact on churn. 
        - Data has very low variability (95% of the customers entering with the promotion), which might make it difficult to find different patterns of customers who churn and who don't.
        - There is no strong correlation to our target feature.
 2. **Machine Learning Models**
     - **From Random Forest Classifier we understood that newsletter subscription,  channel 'F' and price are the three most important features:**
         - From the exploratory data analysis, we saw that those that don't subscribe to the newsletter tend to cancel less. It could be a signal that the email marketing strategy is too abusing and making customers cancel, rather than keeping them engaged.
         - Channel 'F' has a much higher number of churned users compared to the rest. It could be that a different marketing strategy is being used to acquire new customers through this channel that is making customers churn a lot more compared to the rest. The company should keep an eye on this channel and maybe adjust their strategy, unless the risk is being calculated and other metrics (such as maybe a low cost) is still keeping the channel profitable.
         - Registration payment usually influences customer behavior a lot. Lower the price a lot and you will have a lot of new customers but they will cancel a lot more, increase the price and customers will stay longer but you might have very little new ones entering. The idea here is to find the silver lining between the two and see what works best for the company.
    - **Identified 3 top-performing models to be used in the model tuning step:**
        - Random Forest Classifier
        - Support Vector Machine
        - K-Nearest Neighbors Classifier
    - **Data Preprocessing (Scaling and Balancing) Conclusions:**
        - We have been checking if any Scaling or Balancing techniques could improve our results and we saw that no balancing technique is necessary for our data (not improving predictor results), regarding the scaler, K-Nearest Neighbors Classifier tends to work a bit better when using the Normalizer, Random Forest doesn't need any data preprocessing and Support Vector Machine works slightly better with the Standard Scaler.
    - **Hyperparameter tuning final conclusions**
        - After testing multiple models, we discovered that Random Forest Classifier performs best. We have F1 score 69%, it is also predicting the highest number of TP while having the lowest number of FN. We don't need to scale nor to balance our data. It's best to use the long list of the most important features.    
3. **Dimensionality Reduction**:
    - Using PCA and UMAP we can see that there are relations between the data but it's not based on our target feature. This analysis is just confirming the previous result that the decision to cancel most of the times is not influenced by any feature, a random factor is very big.
 
<a name="recommendations"></a> 

## Recommendations
 
 - **More behavioral data**: collect more data about customer behavior, interactions with the customer service, add survey data, there might be some hidden features that are actually the decision-makers.
- **Diverse marketing strategy**: use different pricing strategies in different channels. Open new acquisition channels to be able to identify different customer patterns.
- **With more data try to predict if the customer is going to cancel after the 3rd month**: track customer performance during 2-3 months, try to predict if he or she is going to churn in the later stage of his/hers lifetime. 