"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from django.http import HttpResponse 
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
names = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
dataset = pd.read_csv(url, names=names)
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
array = dataset.values
X = array[:,6:11]
X = X.astype('int')
Y = array[:,12]
Y = Y.astype('int')
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))

def home(request):
    """Renders the home page."""
    if request.method == 'POST':
        applicantIncome = int(request.POST["ApplicantIncome"])
        coapplicantIncome = float(request.POST["CoapplicantIncome"])
        loanAmount = int(request.POST["LoanAmount"])
        loan_Amount_Term = int(request.POST["Loan_Amount_Term"])
        credit_History = int(request.POST["Credit_History"])
        property_Area = request.POST["Property_Area"]
        data = np.array([[applicantIncome,coapplicantIncome,loanAmount,loan_Amount_Term,credit_History]])
        yesNo = model.predict(data)
        return render(request,'app/index.html',{'desicion':yesNo})
    elif request.method == 'GET':
        assert isinstance(request, HttpRequest)
        return render(
            request,
            'app/index.html',
            {
                'title':'Home Page',
                'year':datetime.now().year,
            }
        )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )

