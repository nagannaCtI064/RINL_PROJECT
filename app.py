from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np

app = Flask(__name__)


with open("scaling.pkl", "rb") as f:
    scale = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        
        input_data = [float(data[field]) for field in data]
        
        final_input = []
        for x in input_data:
            final_input.append(scale.transform(x))
        final_input=np.array(final_input).reshape(1,-1)
        output = model.predict(final_input)[0]
        return render_template("home.html", output=output)
        
    except Exception as e:
        return render_template("home.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
