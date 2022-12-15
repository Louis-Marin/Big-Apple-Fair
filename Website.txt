import streamlit as st
import datetime
import requests

st.set_page_config(
            page_title="Quick reference", 	
            page_icon="🚕",
            layout="centered", 	
            initial_sidebar_state="auto") 	

date = st.date_input('Give us the date')

time = st.time_input('Give us the time')

pickup_longitude = st.number_input('Give us the Pickup Longitude')

pickup_latitude = st.number_input('Give us the Pickup Latitude')

dropoff_longitude = st.number_input('Give us the dropoff longitude')

dropoff_latitude = st.number_input('Give us the dropoff latitude')

passenger_count = st.slider('Passenger count', 1,6,2)

pickup_datetime = f"{date} {time}"

params = {
    'pickup_datetime' : pickup_datetime,
    'pickup_longitude' : pickup_longitude,
    'pickup_latitude' : pickup_latitude,
    'dropoff_longitude' : dropoff_longitude,
    'dropoff_latitude' : dropoff_latitude,
    'passenger_count' : passenger_count
}

url = 'http://fake-taxi-42069.herokuapp.com/'

response = requests.get(url, params = params).json()

prediction = response['prediction']

if st.button('I Want the fare !'):
    st.write(f'your estimated fare is {round(prediction,2)}')
