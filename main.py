import joblib
import pickle
from flask import Flask, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


modelo = joblib.load(
    "/mnt/c/Users/rodol/OneDrive/Documentos/projeto/predict_heart_risk/Heart_Desease_Prediction_Machine/Modelling/random_forest.plk"
)
random_forest = joblib.load("random_forest.plk")
loaded_objects = joblib.load("preprocessing_objects.pkl")
ondehotenconder_heart = loaded_objects["ondehotenconder_heart"]
lebel_encoder_sex = loaded_objects["lebel_encoder_sex"]
lebel_encoder_cp = loaded_objects["lebel_encoder_cp"]
lebel_encoder_restecg = loaded_objects["lebel_encoder_restecg"]
lebel_encoder_exang = loaded_objects["lebel_encoder_exang"]
scaler = loaded_objects["scaler"]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


unit = ColumnTransformer(
    [
        (
            "normalize",
            scaler.fit_transform(features[:, [0, 3, 5]], features[:, [0, 3, 5]]),
        ),
        (
            "lebel_encoder_sex",
            lebel_encoder_sex.fit_transform(features[:, 1]),
            features[:, 1],
        ),
        (
            "lebel_encoder_cp",
            lebel_encoder_cp.fit_transform(features[:, 2]),
            features[:, 2],
        ),
        (
            "lebel_encoder_restecg",
            lebel_encoder_restecg.fit_transform(features[:, 4]),
            features[:, 4],
        ),
        (
            "lebel_encoder_exang",
            lebel_encoder_exang.fit_transform(features[:, 6]),
            features[:, 6],
        ),
        ("onehotencoder", ondehotenconder_heart.fit_transform(features), features),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
