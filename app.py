import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import joblib

st.title(":blue[Laptop] Price Predictor APP :sunglasses:" )
st.image('https://amtradez.com/cdn/shop/articles/top-10-best-laptop-brands-in-dubai-uae.jpg?v=1689257129')

data = pd.read_csv('featured_data.csv')

features = list(data.columns)


selections = {}
for ele in features[1:]:
    if ele not in ['Processor', 'Operating System', 'SSD', 'Display', 'Warranty', 'Company',
         'Generation', 'HDD', 'RAM TYPE', "RAM In GB", 'Storage', 'Touch Display']:
        continue
    option = st.selectbox(
        f'Select {ele} from the options',
        data[ele].unique())
    
    selections[ele] = option
    st.write(':red[You selected:]', option)


selected_features = ['Processor', 'Operating System', 'SSD', 'Display', 'Warranty', 'Generation', 
                     'HDD', 'RAM In GB', 'RAM TYPE', 'Storage', 'Touch Display']


# Create a new dictionary with only the selected features in the specified order
selected_data = {feature: selections.get(feature, '') for feature in selected_features}

# Convert the dictionary to a DataFrame with one row
df = pd.DataFrame([selected_data])

st.subheader('The Selected Features are below:')
st.dataframe(df)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

le_processor = joblib.load('le_processor.pkl')
le_os = joblib.load('le_os.pkl')
le_gen = joblib.load('le_gen.pkl')
le_ram_type = joblib.load('le_ram_type.pkl')

df['Processor'] = le_processor.transform(df['Processor'])
df['Operating System'] = le_os.transform(df['Operating System'])
df['Generation'] = le_gen.transform(df['Generation'])
df['RAM TYPE'] = le_ram_type.transform(df['RAM TYPE'])

val = model.predict(df.iloc[[0]])

st.subheader('The Predicted Price is below:')
st.write(':red[The Predicted Price of the Laptop is:]', val[0])