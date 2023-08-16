import joblib
from flask import Flask, render_template, request


modelo = joblib.load('/mnt/c/Users/rodol/OneDrive/Documentos/projeto/predict_heart_risk/Heart_Desease_Prediction_Machine/Modelling/random_forest.plk')
app