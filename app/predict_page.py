import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('predictormodel.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]
le = data["le"]
