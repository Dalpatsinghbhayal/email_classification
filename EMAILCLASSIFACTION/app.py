from flask import Flask, render_template, url_for, request
import joblib


model = joblib.load(r'C:\Users\dalpa\upflair\EMAILCLASSIFACTION\BNB_project.lb')
countvectorizer = joblib.load(r'C:\Users\dalpa\upflair\EMAILCLASSIFACTION\countvectorizer.lb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        email_message = request.form['email_message']
        email = [email_message]
        transformed_email = countvectorizer.transform(email)
        prediction = model.predict(transformed_email)[0]
        
        label = "Ham" if prediction == 0 else "Spam"

        with open('email.txt', 'a') as file:
            file.write(f"{label}\t{email_message}\n")
        
        return label  

if __name__== "__main__":
    app.run(debug=True)









































# from flask import Flask ,render_template ,url_for,request# type: ignore
# import joblib
# model = joblib.load('BNB_project.lb')
# countvectorizer =  joblib.load('countvectorizer.lb')

# app = Flask(__name__)


# @app.route('/')
# def home():
#   return render_template('index.html')

# @app.route('/predict',methods=['GET','POST'])
# def predict():
#   if request.method == 'POST':
#     email_message = str(request.form['email_message'])

#     email = [email_message]

#     transformed_email = countvectorizer.transform(email)

#     print(transformed_email.shape)
    
#     prediction = str(model.predict(transformed_email)[0])
#     print(prediction)



#     label = "Ham" if prediction == 0 else "Spam"

#     with open('email.txt', 'a') as file:
#       file.write(f"{label}\t{email_message}\n")
        
#     return label  


#     # label = 

#     # with open('email.txt',a) as file:
#     #   file.write(f"{label}\t{email_message}\n")

#     # dt = {'0':'ham','1':'spam'}

#     # return dt[prediction]



# if __name__== "__main__":
#   app.run(debug=True)



