from flask import Flask,render_template,request
from model import load_model, predict_credit_classification

app = Flask(__name__, static_url_path='/static')

model = load_model()

@app.route('/',methods=['GET', 'POST'])
def home_page():
    
    chak=""
    duration=""
    history=""
    purpose=""
    credit_amount=""
    balance=""
    employment=""
    install_rate=""
    marital_status=""
    co_applicant=""
    present_residence=""
    real_estate=""
    age=""
    other_installment=""
    residence=""
    no_of_credits=""
    job=""
    no_of_dependents=""
    phone=""
    foreign=""
    
    if request.method == 'POST':
       
        chak = request.form['CHK_ACCT']
        duration = request.form['Duration']
        history = request.form['History']
        purpose = request.form['Purpose_of_credit']
        credit_amount = request.form['Credit_Amount']
        balance = request.form['Balance_in_Savings_A_C']
        employment = request.form['Employment']
        install_rate = request.form['Install_rate']
        marital_status = request.form['Marital_status']
        co_applicant = request.form['Co_applicant']
        present_residence = request.form['Present_Resident']
        real_estate = request.form['Real_Estate']
        age = request.form['Age']
        other_installment = request.form['Other_installment']
        residence = request.form['Residence']
        no_of_credits = request.form['Num_Credits']
        job = request.form['Job']
        no_of_dependents = request.form['No_dependents']
        phone = request.form['Phone']
        foreign = request.form['Foreign']
        
        data = {
            'CHK_ACCT': chak,
            'Duration': duration,
            'History': history,
            'Purpose_of_credit': purpose,
            'Credit_Amount': credit_amount,
            'Balance_in_Savings_A_C': balance,
            'Employment': employment,
            'Install_rate': install_rate,
            'Marital_status': marital_status,
            'Co_applicant': co_applicant,
            'Present_Resident': present_residence,
            'Real_Estate': real_estate,
            'Age': age,
            'Other_installment': other_installment,
            'Residence': residence,
            'Num_Credits': no_of_credits,
            'Job': job,
            'No_dependents': no_of_dependents,
            'Phone': phone,
            'Foreign': foreign
        }
        
        prediction = predict_credit_classification(model, [list(data.values())])
    

        return render_template('index.html', prediction=prediction,chak=chak,duration=duration,history=history,credit_amount=credit_amount,balance=balance,employment=employment,install_rate=install_rate,marital_status=marital_status,co_applicant=co_applicant,present_residence=present_residence,real_estate=real_estate,other_installment=other_installment,no_of_credits=no_of_credits,job=job,no_of_dependents=no_of_dependents,phone=phone,purpose=purpose)
    
    
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=False)