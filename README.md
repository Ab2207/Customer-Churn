# Customer Churn Predictor

Steps involved in predicting customer churn:

1. Using sqlalchemy library in Python, connected to PostgreSQL server and loaded the data from the database. 
2. Data pre-processing and feature engineering was carried out using Pandas and Numpy. Checked for missing values, outliers, data distribution etc.
3. Missing data and outliers were handled by determining the distribution of data and replaced with mean & median accordingly. 
4. Seaborn and Matplotlib were used to conduct Exploratory Data Analysis (EDA) and extract knowledge from the data.
5. For Feature Selection, Pearson Correlation for numerical variables and Chi-Square test for categorical variables were conducted. The columns with high correlation were removed.
6. For numerical columns, the cut-off correlation was decided at .60 and for categorical variable a p-value of < 0.05.
7. Additionally, dummy variables have been created for the remaining categorical columns so that their input should be considered by the Machine Learning model. 
8. During EDA, it was observed that the data was imbalanced with high number of Male customers when comapred to Female. To handle the data imbalance, used SMOTE resampling technique which integrated additional rows to ensure balance in data. 
9. Post train-test split of 70-30, applied Logistic Regression to predict if the customer will Churn or not. The model yielded testing accuracy of 82.6% and a training accuracy of 83.1%. A Confusion Matrix was plot to visualize the number of TypeI and TypeII errors. 
10. Finally, the model was saved to a pickle file and a user input was given to test the model. The model predicted the user input with high accuracy. 

P.S. I'm currently working on deploying this model to cloud using MS Azure. 
