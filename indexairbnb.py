#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install joblib')


# In[6]:


# Imports

import streamlit as st
import pandas as pd
import pickle
import joblib

model = joblib.load('model.joblib')

data = {
    'm_interactions':[10],
    'contact_channel_first': [2],
    'total_reviews': [11.0],
    'guest_user_stage_first': [False]
}
df_test = pd.DataFrame(data)
y_pred = model.predict(df_test)
st.set_page_config(page_title = str(y_pred) 
                    ,page_icon = ":musical_note:" 
                    )



"""
# pandas options
pd.set_option('display.max_colwidth', None)

# setting the basic configuration of the web app. This is shown in the Tab
st.set_page_config(page_title = "#🎶Song Recommendations" 
                    ,page_icon = ":musical_note:" 
                    )

# read the data 
url = 'https://github.com/vkoul/data/raw/main/misc/spotify_data.zip'
df = pd.read_csv(url)

# add a column
df['song_url'] = 'https://open.spotify.com/track/' + df['id']


# Opening intro text
st.write("# 🎶Song Recommender✨")
st.write("#### Choose the parameters of the song:")

# Sliders for the values
# dance
dance_value = st.slider('Set the minimum level of dancebility?', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# acousticness
acoustic_value = st.slider('Set the minimum level of acousticness?', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# energy
energy_value = st.slider('Set the minimum level of energy?', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# instrumentalness
instrumental_value = st.slider('Set the minimum level of instrumentalness?', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# popularity
popularity_value = st.slider('Set the minimum level of popularity?', min_value=1.0, max_value=100.0, value=5.0, step=1.0)

# year
year_value = st.slider('Set the minimum level of year?', min_value=1921, max_value=2020, value=2019, step=1)

# filtering the data based on above criteria
df_filtered = df.query('danceability >= @dance_value & acousticness >= @acoustic_value & energy >= @energy_value &  instrumentalness >= @instrumental_value & popularity >= @popularity_value & year >= @year_value')

# Songs available in new 
total_unique_songs = df_filtered['name'].nunique()
st.write("Unique songs in the dataset mataching the above criteria: " + str(total_unique_songs))

# get the song list
song_name_list = df_filtered['name'].unique().tolist()

# Convert the names as a drop-down
option = st.selectbox(
    '**Which is your fav song?**',
    song_name_list)

st.write('**You selected:**', option)


# pass through the prediction
selection = df.query("name == @option ").select_dtypes(['int', 'float']).drop(columns = ['year', 'explicit', 'mode'])

#st.write(selection)

# load the scaler to scale the data 
scaler = pickle.load(open('scaler_pickle.sav','rb'))

# transform the data for for predictions
selection_scaled = scaler.transform(selection)

# st.write(selection_scaled)


# load the nearest neighbour model
nn_model = pickle.load(open('nn_model_prediction.sav','rb'))


# Generate the neigbors
neighbor_list = nn_model.kneighbors(selection_scaled,  return_distance=False)[:,1:].tolist()[0] # Skip the first value 

# st.write("The songs shorlisted are:", neighbor_list)


# Final list of predictions
neighbors = df.loc[neighbor_list,['name', 'song_url']]

# cleaning operations
neighbors = neighbors.reset_index(drop= True).rename(columns = {'name': 'Song', 'song_url': 'Spotify Link'})

# st.write(neighbors, unsafe_allow_html = True)

# use this to make table more clean https://github.com/softhints/Pandas-Tutorials/blob/master/styling/create-clickable-link-pandas-dataframe-jupyterlab.ipynb
# show the table
# st.dataframe(neighbors)
"""


# In[ ]:




