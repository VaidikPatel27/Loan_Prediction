from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('/workspaces/Loan_Prediction/pickle_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_sales():
    
    acc_bal = int(request.form.get('AccountBalance'))
    duration = int(request.form.get('duration'))
    prev_pay = int(request.form.get('PreviousPayment'))
    purpose = int(request.form.get('Purpose'))
    cred_amt = int(int(request.form.get('CreditAmount'))/90)
    savings_stocks = int(int(request.form.get('SavingsStocks'))/100)
    employement_duration = int(request.form.get('EmploymentDuration'))
    installment = int(request.form.get('Installment'))
    sex = int(request.form.get('Sex'))
    guarantors = int(request.form.get('Guarantors'))
    address_duration = int(request.form.get('DurationAdress'))
    asset = int(request.form.get('Asset'))
    age = int(request.form.get('Age'))
    curr_bank_credit = int(request.form.get('CreditsAtThisBank'))
    all_bank_credit = int(request.form.get('CurrentCredits'))
    apt_type = int(request.form.get('ApartmentType'))
    occupation = int(request.form.get('Occupation'))
    dependents = int(request.form.get('Dependents'))
    telephone = int(request.form.get('Telephone'))
    foreign_worker = int(request.form.get('ForeignWorker'))
    
    input = [acc_bal,
         duration,
         prev_pay,
         purpose,
         cred_amt,
         savings_stocks,
         employement_duration,
         installment,
         sex,
         guarantors,
         address_duration,
         asset,
         age,
         all_bank_credit,
         apt_type,
         curr_bank_credit,
         occupation,
         dependents,
         telephone,
         foreign_worker]
    
    prediction = model.predict(np.array(input).reshape(1,20))
    
    if prediction[0] == 1:
        outcome = "Eligible for Loan"
    elif prediction[0] == 0:
        outcome = 'Not Eligible for Loan'
        
    return render_template('index.html',result=outcome)

if __name__ == '__main__':
    app.run(debug=True)