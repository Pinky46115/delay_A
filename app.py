import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Deliery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features (assuming the original X had these columns)
# You might need to adjust these based on the actual columns in your X_train
delivery_distance = st.slider('Delivery Distance (km)', 0.0, 50.0, 20.0)
traffic_congestion = st.selectbox('Traffic Congestion', [1, 2, 3, 4, 5])
weather_condition = st.selectbox('Weather Condition', [1, 2, 3, 4, 5])
delivery_slot = st.selectbox('Delivery Slot', [1, 2, 3])
preparation_time = st.slider('Preparation Time (minutes)', 0, 60, 30)
vehicle_type = st.selectbox('Vehicle Type', [1, 2, 3, 4]) # Assuming numerical encoding
customer_loyalty_score = st.slider('Customer Loyalty Score', 0, 10, 5)
time_of_day = st.selectbox('Time of Day', [1, 2, 3, 4]) # Assuming numerical encoding
order_value = st.slider('Order Value ($)', 10.0, 500.0, 50.0)
num_items = st.slider('Number of Items', 1, 20, 5)
restaurant_rating = st.slider('Restaurant Rating', 1.0, 5.0, 3.5)

# Create a DataFrame from inputs
input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                            delivery_slot, preparation_time, vehicle_type,
                            customer_loyalty_score, time_of_day, order_value,
                            num_items, restaurant_rating]],
                           columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                    'Delivery_Slot', 'Preparation_Time', 'Vehicle_Type',
                                    'Customer_Loyalty_Score', 'Time_of_Day', 'Order_Value',
                                    'Num_Items', 'Restaurant_Rating'])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.error(f'Prediction: Likely Delayed (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Prediction: On Time (Probability: {1 - prediction_proba[0]:.2f})')

st.write('---')
st.write('**Note:** This is a demo application. The accuracy of predictions depends on the training data and model performance.')
