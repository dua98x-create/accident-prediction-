
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
