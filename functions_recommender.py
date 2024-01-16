# 1 create function to seach for song:
import pandas as pd
import numpy as np
import requests
from typing import Optional
import spotipy
from spotipy import Spotify


# import files
songs_w_clusters = pd.read_csv('songs_w_clusters.csv')
song_database_transformed = pd.read_csv('song_database_transformed.csv')


#spotify authentification (1)
# config file for spotify app
import requests
import sys

from config import *

#spotify authentication (2)
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials


#Initialize SpotiPy with user credentias #
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_ID,
                                                           client_secret=Client_Secret))



# function a

from typing import Optional


# app functions

def search_song_limit(title: str, artist: str = None, sp: spotipy.Spotify = None, limit:int = 5) -> Optional[list[str]]:

    '''
    This function uses Spotify's API to find song IDs based on song title and artist.
    inputs:
     title (str): The title of the song.
     artist (str): The artist of the song.
     limit (int): The maximum number of songs to return. Default is 5.
     
    outputs:
     Optional[List[str]]: The IDs of the top songs matching the title and artist, otherwise None.
    '''
    if sp is None:
       raise ValueError("Please check Spotify credentials/object.")

    query = f'track:{title}'
    if artist is not None and artist != '':
        query += f' artist:{artist}'

    results = sp.search(q=query, type='track', limit=limit)
    items = results['tracks']['items']

    if len(items) > 0:
       song_ids = [song['id'] for song in items]
    else:
       song_ids = []
    return song_ids






# function b

import spotipy
from typing import Optional, List, Union

def get_track_info_limit(track_ids: Union[List[str], pd.Series, pd.DataFrame], sp: spotipy.Spotify = None) -> pd.DataFrame:
    '''
    This function uses Spotify's API to find track details based on track IDs.
    inputs:
     track_ids (Union[List[str], pd.Series, pd.DataFrame]): A list, series or dataframe of track IDs.
     sp (spotipy.Spotify): The Spotify object.
     
    outputs:
     pd.DataFrame: A dataframe with track IDs, track names, and artist names.
    '''
    if isinstance(track_ids, pd.DataFrame):
       track_ids = track_ids.tolist()
    elif isinstance(track_ids, pd.Series):
       track_ids = track_ids.values.tolist()
    elif isinstance(track_ids, list):
       pass
    else:
       raise TypeError("Invalid input type. Expected a list, series or dataframe.")
    
    track_info = []
    for track_id in track_ids:
       track = sp.track(track_id)
       track_info.append({
           'track_id': track_id,
           'track_name': track['name'],
           'artist_name': track['artists'][0]['name']
       })
    
    df = pd.DataFrame(track_info)
    if not df.empty: # Check if DataFrame is not empty before printing
        print(df[['track_name','artist_name']])
    else:
        print('error 001: could not find song')
        return None
    return df




# function c
def select_display_row(df, index):
    correct_user_song_id = df['track_id'].iloc[index]
    return correct_user_song_id




# function d
def user_song_audio_features(correct_user_song_id, sp: spotipy.Spotify = None):
    audio_features_columns = ['danceability', 'energy', 'acousticness', 'key', 'valence']
    user_song_audio_features = sp.audio_features(correct_user_song_id)
    user_song_audio_features_df = pd.DataFrame(user_song_audio_features) 
    user_song_audio_features_df = user_song_audio_features_df[audio_features_columns]
    user_song_audio_features_result = user_song_audio_features_df
    return user_song_audio_features_result



# (child) function e
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
import pickle

def get_song_cluster(iso_filename: str='isomap.pickle', user_song_features: pd.DataFrame=user_song_audio_features, song_database_transformed: pd.DataFrame=song_database_transformed, songs_w_clusters:pd.DataFrame=songs_w_clusters) -> int:
    """
    function to predict the cluster of a song(s) in the given dataframe based on the selected features.
    
    input:
    scaler_filename: str: name of file of scaler.
    model_filename: str: name fo file of trained KMeans model used to predict the clusters.
    song_df (pd.DataFrame): dataframe containing the songs and their features.
    selected_features (list): List of feature names to select from the dataframe.
    
    output:
    int: with predicted cluster for each song.
    """    
    with open(iso_filename, 'rb') as iso_model_f:  # Load the model
        model_iso = pickle.load(iso_model_f)
    print('model loaded...')

    user_song_features_transformed = model_iso.transform(user_song_features)
    user_song_features_transformed = pd.DataFrame(user_song_features_transformed, columns=["ISO_1","ISO_2"])

    song_database_transformed = song_database_transformed.copy()
    song_database_transformed = song_database_transformed[['ISO_1', 'ISO_2']]
    song_database_transformed = song_database_transformed.reset_index(drop=True)

    d = distance_matrix(user_song_features_transformed, song_database_transformed)
    print('dipping in the magic sauce...')
    
    closest_song_index = (np.argmin(d))
    song_cluster = songs_w_clusters['cluster'].iloc[closest_song_index]
    
    return song_cluster




# (child) function f
def check_hotness(songs_w_clusters:pd.DataFrame, song_id:str):
    # Check if the song is in the 'id' column where 'hotness' is 'yes'
    if song_id in songs_w_clusters[songs_w_clusters['hotness'] == 'yes']['id'].values:
       song_hotness = 'yes'
       return song_hotness
    # If not found, check if the value is in the 'id' column where 'hotness' is 'no'
    elif song_id in songs_w_clusters[songs_w_clusters['hotness'] == 'no']['id'].values:
       song_hotness = 'no'
       return song_hotness
    else:
       song_hotness = 'no'
    return song_hotness    



# function g
def get_cluster_sample(df:pd.DataFrame, song_cluster:int, song_hotness: str) -> pd.DataFrame:
    '''
    '''
    #filter dataframe with correct cluster and song hotness
    filtered_df = df.loc[(df['cluster'] == song_cluster) & (songs_w_clusters['hotness'] == song_hotness)]
    #generate random songs 
    random_songs_from_cluster = filtered_df[['track_name','artist_name']].sample(n=5)
    return random_songs_from_cluster





# PARENT SONG RECOMMENDER FUNCTION:

def song_recommender() -> pd.DataFrame:
    '''
    This function provides song recommendations based on the user's input. 
    It repeatedly asks the user to enter a song name and artist name, then uses these inputs to find similar songs. 
    The user can choose to receive more recommendations or stop receiving them.
    
    The function works as follows:
    welcomes the user and asks for a song name and artist name -> searches for the song on Spotify and displays the top 5 results
    -> asks the user to select the correct song from the displayed results -> retrieves the audio features of the selected song and determines its cluster
    -> checks if the song is a hot song and gets a sample of songs from the same cluster ->  displays the sample of songs to the user.
    -> asks the user if they want more recommendations. If the user answers 'yes', the process starts again. 
    If the user answers not 'yes', the function stops and returns the last set of recommendations.
    
    output:
       cluster_sample_test: dataframe containing the sample of songs from the same cluster as the user's selected song. 
    '''
    cluster_sample_test = pd.DataFrame()
    while True:
       
        # welcome and user input (1): add track name and optionally artist
        print('this app will provide song recommendations based on your own personal music tastes. Please fill in the form...')
        title = input("Enter song name: ")
        artist = input("Enter artist name or press enter to skip: ")

        print(f"Searching the dark web for {title}...")
        
        #fn a: get the IDs from the top 5 results from spotify
        if artist == '' or artist is None:
            print('Opps, it seems like the artist has hard name to remember... Don\'t worry, we will find your song!')
            user_song_id = search_song_limit(title=title, artist=None, sp=sp, limit=5)
        else:
            user_song_id = search_song_limit(title, artist, sp) 

        #fn b: get the artist and track names & display them
        user_search_results = get_track_info_limit(user_song_id, sp)
        if user_search_results is None: # Check if DataFrame is empty
            break
            return
        #user input (2): select correct song from results
        song_index_input = int(input("please select the correct song using values from 0 to 4"))
        
        #fn c: get selected song's ID
        correct_user_song_id = select_display_row(user_search_results, song_index_input)
        
        #fn d: get selected song's audio features
        user_song_audio_features_result = user_song_audio_features(correct_user_song_id, sp)
        
        #fn e: get selected song's cluster
        song_cluster = get_song_cluster('isomap.pickle', user_song_audio_features_result, song_database_transformed, songs_w_clusters)
        
        #fn f: check if is hot_song
        song_hotness = check_hotness(songs_w_clusters, correct_user_song_id)
    
        #fn g: get sample songs from cluster after filtering df based on song cluster and song hotness
        cluster_sample_test = get_cluster_sample(songs_w_clusters, song_cluster, song_hotness)
        display(cluster_sample_test)
    
        #user input (3): should the recommendations continue?
        
        continue_prompt = input("Thanks for using our song recommender, would you like more recommendations? (please answer: yes or no)").lower()
        if continue_prompt != 'yes':
            print('thank you!')

            break
        
    return cluster_sample_test

