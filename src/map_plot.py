import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from tqdm import tqdm
import folium
from folium.features import DivIcon

def coords_vessel(df,number):
    '''
    returns a list of coords for a single vessel
    
    :param df: the data
    :type df: dataframe
    
    :param number: the number of the vessel you want to get the coords
    :type number: int
    
    :return: a list of tuple with the coords (lat,lon)
    '''
    vessel_number = df[df.mmsi==df.mmsi.unique()[number]]
    latitudes = np.array(vessel_number['lat'])
    longitudes = np.array(vessel_number['lon'])
    coords = [(lat,lon) for lat,lon in zip(latitudes,longitudes)]
    
    return coords

def coords_vessel_0_1(df,number,is_fishing_value):
    '''
    returns a list of coords for a single vessel and given the condition is fishing or not
    
    :param df: the data
    :type df: dataframe
    
    :param number: the number of the vessel you want to get the coords
    :type number: int
    
    :param is_fishing_value: 0 or 1 depending on the condition you want to filter
    :type is_fishing_value: int  
    
    :return: a list of tuple with the coords (lat,lon)
    '''
    valor = df['mmsi'].unique()[number]
    vessel_number = df[(df['is_fishing']==is_fishing_value) & (df['mmsi']==valor)]
    latitudes = np.array(vessel_number['lat'])
    longitudes = np.array(vessel_number['lon'])
    coords = [(lat,lon) for lat,lon in zip(latitudes,longitudes)]
    
    return coords

def folium_polyline(coords):
    '''
    given a list of coords, plot a polyline folium map
    
    :param coords: the list of coordenates to plot
    :type coords: list of tuples  
    
    :return: a folium map with the coords plotted using polyline style
    '''
    temp_map = folium.Map()
    folium.PolyLine(coords, 
                    color='green', 
                    weight=2, 
                    opacity=0.8, 
                    dash_array='2,7').add_to(temp_map)
    folium.FitBounds(coords).add_to(temp_map)
    return temp_map

def folium_markers_polyline(coords):
    '''
    given a list of coords, plot a folium map using markers and polyline
    
    :param coords: the list of coordenates to plot
    :type coords: list of tuples  
    
    :return: a folium map with the coords plotted using markers and polyline
    '''
    temp_map = folium.Map()
    
    folium.PolyLine(coords, 
                    color='blue', 
                    weight=2, 
                    opacity=0.8, 
                    dash_array='2,7').add_to(temp_map)
    
    for i in coords:
        folium.CircleMarker(i, 
                            radius=1, 
                            color='green', 
                            fill_color='green', 
                            fill_opacity=1, 
                            popup=None).add_to(temp_map)
        
    
    
    folium.FitBounds(coords).add_to(temp_map)

    return temp_map

def folium_markers_real(df,mmsi):
    '''
    given a list of coords and a mmsi number, plot a marker folium map only for the real data
    
    :param coords: the list of coordenates to plot
    :type coords: list of tuples  
    
    :return: a folium map with the coords plotted using markers and polylines
    '''
    temp_map = folium.Map()
    
    latitudes = np.array(df[df.mmsi==mmsi]['lat'])
    longitudes = np.array(df[df.mmsi==mmsi]['lon'])
    coords = [(lat,lon) for lat,lon in zip(latitudes,longitudes)]
    
    folium.PolyLine(coords, 
                    color='blue', 
                    weight=2, 
                    opacity=0.8, 
                    dash_array='2,7').add_to(temp_map)
    
    for index, row in df[df.mmsi==mmsi].iterrows():
        folium.CircleMarker([row['lat'],row['lon']], 
                            radius=1, 
                            color=row['color_is_fishing'], 
                            fill_color=row['color_is_fishing'], 
                            fill_opacity=1, 
                            popup=None).add_to(temp_map)

    folium.FitBounds(coords).add_to(temp_map)
    
    return temp_map

def folium_markers_pred(df,mmsi):
    '''
    given a list of coords and a mmsi number, plot a marker folium map only for the predicted data
    
    :param coords: the list of coordenates to plot
    :type coords: list of tuples  
    
    :return: a folium map with the coords plotted using markers and polylines
    '''
    
    temp_map = folium.Map()
    
    latitudes = np.array(df[df.mmsi==mmsi]['lat'])
    longitudes = np.array(df[df.mmsi==mmsi]['lon'])
    coords = [(lat,lon) for lat,lon in zip(latitudes,longitudes)]
    
    folium.PolyLine(coords, 
                    color='blue', 
                    weight=2, 
                    opacity=0.8, 
                    dash_array='2,7').add_to(temp_map)
    
    for index, row in df[df.mmsi==mmsi].iterrows():
        folium.CircleMarker([row['lat'],row['lon']], 
                            radius=1, 
                            color=row['color_prediction'], 
                            fill_color=row['color_prediction'], 
                            fill_opacity=1, 
                            popup=None).add_to(temp_map)
        
    folium.FitBounds(coords).add_to(temp_map)
    
    return temp_map