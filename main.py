
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template

model = tf.keras.models.load_model('filesuse/project_model1.h5')
print(model.summary())
#y_pred = model.predict(scaler.transform([x_test]))