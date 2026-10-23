#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import streamlit as st
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from PIL import Image
import streamlit as st


#importing dataset
df = pd.read_csv("accident_prediction.csv")


#remove time
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M")
df["hour"] = df["Time"].dt.hour
df["Minute"] = df["Time"].dt.minute

#drop time column and create new dataset
new_df = df.copy()
new_df = df.drop("Time", axis=1)


#encoding target column
le = LabelEncoder()
new_df["Accident"] = le.fit_transform(new_df["Accident"])

print(new_df["Accident"].value_counts())


#balance dataset if its has unbalanced num of accidents
x = new_df.drop(["Accident"], axis=1)
y = new_df["Accident"]


#test train split data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


#random sampler
random = RandomOverSampler(random_state=42)

x_resampled, y_resampled = random.fit_resample(
    x_train,
    y_train
)

#Immediately after resampling, convert x_resampled back to a DataFrame:

x_resampled = pd.DataFrame(
    x_resampled,
    columns=x_train.columns
)


#find missing values
print(new_df.isnull().sum())


#encode categorical column
trf1 = ColumnTransformer(
    transformers=[
        ("imputer_Day",
         SimpleImputer(strategy="most_frequent"),
         ["Day"]),

        ("imputer_Age",
         SimpleImputer(strategy="mean"),
         ["Age"]),

        ("imputer_Speed",
         SimpleImputer(strategy="mean"),
         ["Speed"]),

        ("imputer_RoadCondition",
         SimpleImputer(strategy="most_frequent"),
         ["RoadCondition"]),

        ("imputer_Traffic",
         SimpleImputer(strategy="most_frequent"),
         ["Traffic"]),

        ("imputer_Alcohol",
         SimpleImputer(strategy="most_frequent"),
         ["Alcohol"]),

        ("imputer_Weather",
         SimpleImputer(strategy="most_frequent"),
         ["Weather"]),

        ("imputer_hour",
         SimpleImputer(strategy="mean"),
         ["hour"]),

        ("imputer_Minute",
         SimpleImputer(strategy="mean"),
         ["Minute"])
    ],
    remainder="drop"
)


x_train_trf = trf1.fit_transform(x_resampled)

x_test_trf = trf1.transform(x_test)

print(trf1.get_feature_names_out())
# print(x_test_trf)

#label encoding output column
trf2=ColumnTransformer(
    transformers=[
        ("label encoding accident",LabelEncoder,["Accident"])
    ],
    remainder="passthrough"
)
# x_train_trf1 = trf1.fit_transform(x_train)

# print(trf1.get_feature_names_out())
# print(x_train_trf1.shape)
# print(new_df["Accident"])

#encode categorical column
cat_index=[0,3,4,5,6]
trf2=ColumnTransformer(
    transformers=[
      ("cat_col_indexes",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),cat_index)
      

],
remainder="passthrough"
)
x1 = trf1.fit_transform(x_resampled)

print(trf1.get_feature_names_out())
print(x1.shape)
#scaling all data
# tf4=ColumnTransformer(
# transformers=[
#     ("scaling",StandardScaler,["Age","Speed"])

# ],
   
# remainder="passthrough"
# )chi will not work on scaled values bec it works only on positive values

# FEATURE SELECTION USING CHI-SQUARED STATISTIC
# ============================================================
# chi2 measures relevance between each categorical feature and the target
from sklearn.feature_selection import chi2,SelectKBest
trf3=SelectKBest(chi2,k="all")

#Calculate the chi-square score for every feature, then keep the best 20 features.

##Random forest classifier
#helps in majority tree prediction
#             Random Forest
#                  │
#       ┌──────────┼──────────┐
 #      ↓          ↓          ↓
    # Tree 1     Tree 2     Tree 3
    #    ↓          ↓          ↓
    #   Yes         No         Yes
    #Then the forest uses majority voting:
trf4=RandomForestClassifier()

#===========================================================================
#ab pipe line mn sary steps ko enter kro or is x test and x train in fit kro
#===========================================================================
pipe = Pipeline([
    ('trf1', trf1),   # impute missing values
    ('trf2', trf2),   # one-hot encode categorical columns
    ('trf3', trf3),   # select best 50 features via chi2
    ('trf4', trf4),   # random forest classifier
])
pipe.fit(x_resampled,y_resampled)#use only for traning not testing)

#prediction
y_pred=pipe.predict(x_test)


#accuracy,classification report confussion matrix
#===============================================
#evaluted model
#===============================================

acc=accuracy_score(y_test,y_pred)
class_rep=classification_report(y_test,y_pred)
con=confusion_matrix(y_test,y_pred)
print(class_rep)

#============================================
#save file with pickle
# import pickle as pkl
# pkl.dump(open("my pipeline code.py","wb"))

# ============================================================
# LOAD PIPELINE FOR PREDICTIONS (E.G. FOR A WEBSITE / NEW INPUT)
# ============================================================
# pipe=pkl.loads(open("my pipeline code.py","rb"))
def predict_accident(Day, Age,Speed,Weather,RoadCondition,Traffic,Alcohol, hour,Minute):
    features = pd.DataFrame({
    "Day": [Day],
    "Age": [Age],
    "Speed": [Speed],
    "Weather": [Weather],
    "RoadCondition": [RoadCondition],
    "Traffic": [Traffic],
    "Alcohol": [Alcohol],
    "hour": [hour],
    "Minute": [Minute]
})
    return pipe.predict(features)



prediction= predict_accident(
    Day="Thursday",
    Age=35,
    Speed=60,
    Weather="Clear",
    RoadCondition="Dry",
    Traffic="High",
    Alcohol="No",
    hour=12,
    Minute=30
)


print("Predicted Accident:", prediction)
prediction_map={1:"Accident",0:"not accident"}
if prediction ==1:
    print("accisent yes")
else:
    print("accident no")

#web app===================================


st.title("Road Accident prediction pipeline ")


# Image on top
st.image("accident.png", use_container_width=True)

# tips====================================================
accident_precautions = [
    "Reduce your speed immediately.",
    "Maintain a safe following distance.",
    "Avoid sudden braking or sharp turns.",
    "Do not use your phone while driving.",
    "Wear a seat belt and ensure all passengers do the same.",
    "Follow traffic signals and road signs.",
    "Avoid driving under the influence of alcohol.",
    "Stay extra alert in poor weather or road conditions."
]

not_accident_precautions = [
    "Continue maintaining a safe speed.",
    "Keep a safe distance from other vehicles.",
    "Follow traffic rules and road signs.",
    "Stay focused and avoid using your phone while driving.",
    "Always wear a seat belt.",
    "Check mirrors and blind spots before changing lanes.",
    "Keep your vehicle properly maintained.",
    "Stay alert to pedestrians and other road users."
]
Day = st.selectbox(
    "Day",
    options=[
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
)

Age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

Speed = st.number_input(
    "Speed",
    min_value=0,
    max_value=200,
    value=50
)

Weather = st.selectbox(
    "Weather",
    options=["Clear", "Rain", "Fog"]
)

RoadCondition = st.selectbox(
    "Road Condition",
    options=["Dry", "Wet", "Icy"]
)

Traffic = st.selectbox(
    "Traffic",
    options=["Low", "Medium", "High"]
)

Alcohol = st.selectbox(
    "Alcohol",
    options=["Yes", "No"]
)

hour = st.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

Minute = st.number_input(
    "Minute",
    min_value=0,
    max_value=59,
    value=30
)
# button==================================================
if st.button("prediction"):
    prediction= predict_accident(
    Day="Thursday",
    Age=35,
    Speed=60,
    Weather="Clear",
    RoadCondition="Dry",
    Traffic="High",
    Alcohol="No",
    hour=12,
    Minute=30
)


print("Predicted Accident:", prediction)
prediction_map={1:"Accident",0:"not accident"}
if prediction ==1:
    st.success("accisent yes")
    st.write(accident_precautions)
else:
    st.success("accident no")
    st.write(not_accident_precautions)
