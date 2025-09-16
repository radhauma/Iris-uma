from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('iris_classifier_model.pkl')

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # extract the data from the request
    data = request.get_json(force=True)
    features = data.get("features")

    # check input validation
    if not features or len(features) != 4:
        return jsonify({"error": "Please provide 4 numeric features"}), 400

    # convert features into numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # make prediction
    prediction = model.predict(features)[0]

    # map prediction to class names
    classes = ['setosa', 'versicolor', 'virginica']
    pred_class = classes[prediction]

    # prepare response
    result = {"class": pred_class}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)