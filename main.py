import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def loadPage():
	return render_template('home.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == "POST":
            
        inputQuery1 = request.form['query1']
        inputQuery2 = request.form['query2']
        inputQuery3 = request.form['query3']
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = request.form['query7']
        inputQuery8 = request.form['query8']
        inputQuery9 = request.form['query9']
        inputQuery10 = request.form['query10']
        inputQuery11 = request.form['query11']
        inputQuery12 = request.form['query12']
        inputQuery13 = request.form['query13']
        inputQuery14 = request.form['query14']       
        inputQuery15 = request.form['query15']
        inputQuery16 = request.form['query16']
        inputQuery17 = request.form['query17']
        inputQuery18 = request.form['query18']
        inputQuery19 = request.form['query19']
        inputQuery20 = request.form['query20']
        inputQuery21 = request.form['query21']
        inputQuery22 = request.form['query22']
        inputQuery23 = request.form['query23']
        inputQuery24 = request.form['query24']
    
    
        
        
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                 inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14, 
                 inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19, inputQuery20, inputQuery21, 
                 inputQuery22, inputQuery23, inputQuery24]]
        
        new_df = pd.DataFrame(data, columns = ['Tenure', 'WareDist', 'AppHours', 'No_of_devices', 'SatScore',
       'NumberOfAddress', 'AmountHike', 'OrderCount', 'LastOrder',
       'CashbackAmount', 'Payment_COD', 'Payment_Cash on Delivery',
       'Payment_Credit Card', 'Payment_Debit Card', 'Payment_E wallet',
       'Payment_UPI', 'PreferedOrderCat_Grocery',
       'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile',
       'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'Complain_1'])
        
        #df_2 = pd.concat([df_1, new_df], ignore_index = True) 
        # Group the tenure in bins of 12 months
        #labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        
        #df_2['tenure_group'] = pd.cut(df_2.Tenure, range(1, 80, 12), right=False, labels=labels)
        #drop column customerID and tenure
        #df_2.drop(columns= ['Tenure'], axis=1, inplace=True)   
        
        
        
        
        '''new_df__dummies = pd.get_dummies(new_df[['Tenure', 'WareDist', 'Payment', 'AppHours', 'No_of_devices',
           'PreferedOrderCat', 'SatScore', 'MaritalStatus', 'NumberOfAddress',
           'Complain', 'AmountHike', 'OrderCount', 'LastOrder', 'CashbackAmount']])'''
        
        
        #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
            
        
        single = model.predict(new_df)
        probablity = model.predict_proba(new_df)
        
        if single==1:
            o1 = "This customer is likely to be churned!!"
            o2 = "Confidence: {}%".format(round(probablity[0][1]*100),2)
        else:
            o1 = "This customer is likely to continue!!"
            o2 = "Confidence: {}%".format(round(probablity[0][0]*100),2)
            
        return render_template('home.html', output1=o1, output2=o2, 
                               query1 = request.form['query1'], 
                               query2 = request.form['query2'],
                               query3 = request.form['query3'],
                               query4 = request.form['query4'],
                               query5 = request.form['query5'], 
                               query6 = request.form['query6'], 
                               query7 = request.form['query7'], 
                               query8 = request.form['query8'], 
                               query9 = request.form['query9'], 
                               query10 = request.form['query10'], 
                               query11 = request.form['query11'], 
                               query12 = request.form['query12'], 
                               query13 = request.form['query13'], 
                               query14 = request.form['query14'],                              
                               query15 = request.form['query15'], 
                               query16 = request.form['query16'],
                               query17 = request.form['query17'],
                               query18 = request.form['query18'],
                               query19 = request.form['query19'], 
                               query20 = request.form['query20'], 
                               query21 = request.form['query21'], 
                               query22 = request.form['query22'], 
                               query23 = request.form['query23'], 
                               query24 = request.form['query24']
                                )
    else:
        return render_template('home.html')
                       



if __name__=="__main__":
    app.run(debug=True)