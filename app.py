from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__, static_url_path='/static')

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        A = [float(x) for x in request.form.values()]
        model_probability = model.predict([A])
        print(A)
        print(model_probability)
        #Directly use model_probability as input, error handling below
        prediction_text = get_result_text(int(model_probability[0]))
    
    except ValueError:  #handle value errors (eg., Non-numeric values, empty fields)
        return render_template('index.html', result='Invalid input. Please check the data and retry')
    except (IndexError, TypeError):  #handle array or data type issues from model prediction
        return render_template('index.html', result='An unexpected error happened while processing your prediction. Contact the administrator')
    except Exception as e:
        return render_template('index.html', result=f'An unexpected error occured : {e}')  #General Exception handling


    return render_template('index.html', result=prediction_text)

def get_result_text(prediction_number):
    if prediction_number == 1:
        return "Underweight"
    elif prediction_number == 2:
        return "Normal Weight"
    elif prediction_number == 3:
        return "Overweight"
    elif prediction_number == 4:
        return "Obesity"
    else:
        return "Invalid Prediction Result"  # Handle cases where the model returns a value outside 1-4


if __name__ == "__main__":
    app.run(debug=True)