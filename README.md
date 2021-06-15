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
11. Used Flask as a web framework to integrate the back-end Machine Learning model with HTML as front-end. 
12. Deployed the main.py file on Google Cloud Plarform and it can be accessed here - https://customer-churn-predictor.ew.r.appspot.com

# Directions to use Web Application

While using the web application, for below variables please enter 0 for "No" and 1 for "Yes"

Payment COD, Payment Cash on Delivery, Payment Credit Card, Payment Debit Card, Payment E Wallet, Payment UPI, Prefered Order Cat_ Grocery, PreferedOrderCat_Laptop & Accessory,
PreferedOrderCat_Mobile, PreferedOrderCat_Mobile Phone, PreferedOrderCat_Others, MaritalStatus_Married, MaritalStatus_Single, Complain_1

For example: If the customer's preferred payment method is Credit Card, the value of "Payment Credit Card" should be "1" while other payment variables should be "0". Similarly, the Preferred Order Category, Marital Status and Complain. 

Do try it out and feel free to drop in any suggestions to make it better. 

P.S. The front-end UI is not that great as I do not specialise in front-end development. Feel free to fork it and make your own additions and changes. 

