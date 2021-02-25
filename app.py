import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle                                           # runs along with sklearn only.Hence, sklearn should be installed

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))            #Loading the model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    exp = request.form.get('experience')
    test_score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')
    # or we could have directly done this through list comprehension, int_features = [int(x) for x in request.form.values()]

    # convert all the features into int
    int_features = [int(exp),int(test_score),int(interview_score)]
    
    final_features = [np.array(int_features)]       # converting the list into a list of array, [array([x,y,z])]
    prediction = model.predict(final_features)      # prediction->array of predicted salaries with only 1 element

    output = round(prediction[0], 2)                # Rounding off the predicted salary

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

 
if __name__ == "__main__":                          # Understand the meaning of this 
    app.run(debug=True)
