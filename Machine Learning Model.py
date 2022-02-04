import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings("ignore")

clean_df = pd.read_csv("clean_Master.csv")

X = clean_df.drop('Label', axis = 1)
y = clean_df['Label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc_model = RandomForestClassifier()
rfc_model.fit(x_train,y_train)

pickle.dump(rfc_model,open('model.pkl','wb'))