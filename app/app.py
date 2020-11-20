#=================IMPORTS=================

from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
import numpy as np
import folium
import pickle
from geopy import distance

preds_df=pd.read_pickle('..\\data\\pickled_files\\predictions.p')
display_df=preds_df[['magnitude', 'depth', 'latitude', 'longitude', 'utc_datetime', "state"]]

#add a manual index column due to json/tabulator quirks.
display_df['index']=display_df.index
display_json=display_df.to_json(orient='records', date_format='iso')
app = Flask(__name__)


#==============================FUNCTIONS================================================

def map_quake(index, df):
    quake=df.loc[index, :]
    act_lat=quake['latitude']
    act_long=quake['longitude']
    pred_lat=quake['pred_lat']
    pred_long=quake['pred_long']
    m = folium.Map(location=[(act_lat + pred_lat)/2, (act_long + pred_long)/2], zoom_start=7, width=800)
    actual = folium.Circle(radius=10_000,
                           location=[act_lat, act_long],
                           color='black',
                           fill=True,
                           fill_color='orange',
                           fill_opacity=1,
                           popup=folium.Popup("Earthquake Location", max_width=200))
    pred = folium.Circle(radius=10_000,
                         location=[pred_lat, pred_long],
                         color='black',
                         fill=True,
                         fill_color='green',
                         fill_opacity=1,
                         popup=folium.Popup(f'Predicted Location \n Error: {int(round(quake["error"],0))} km', max_width=100)
                         )
    line= folium.PolyLine([(act_lat, act_long), (pred_lat, pred_long)],
                          color="black",
                          dash_array="4",
                          weight=1)
    line.add_to(m)
    actual.add_to(m)
    pred.add_to(m)
    return m
    

#==================================FLASK PAGE=============================================

@app.route('/')
def index():
    return render_template('index.html', 
        tabulator_table=display_json)

@app.route("/predict")
def predict():
    row=int(request.args['row'])
    predict_map=map_quake(row, preds_df)
    map_html=predict_map._repr_html_()
    return render_template('predict.html',
        tabulator_table=display_json,
        predict_display=map_html)


if __name__=='__main__':
    app.run(debug=True)