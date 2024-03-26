from flask import Flask, render_template, request
import numpy as np
import pickle

# model = pickle.load(open("rf_model.pkl", "rb"))
model = pickle.load(open("models\lg_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    state = int(request.form["state"])
    district = int(request.form["district"])
    variety = int(request.form["variety"])
    min_price = int(request.form["min_price"])
    max_price = int(request.form["max_price"])

    data = ([[state, district, variety, min_price, max_price]])
    prediction = model.predict(data)  # Extracting the prediction from the array
    print(np.round(prediction,2))

    return render_template("predict.html", prediction = np.round(prediction,2))
    
    # return render_template("predict.html", prediction="No data submitted.")

if __name__ == "__main__":
    app.run(debug=True)
