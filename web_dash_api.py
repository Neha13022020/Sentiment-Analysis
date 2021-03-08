# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:38:16 2021

@author: Neha
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
#from tqdm import tqdm
#from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import pickle
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

# https://thingsolver.com/dash-by-plotly/
def pie_char():
    labels = np.sort(balanced_reviews['overall'].unique())
    df_cl_label = balanced_reviews['overall'].value_counts().to_frame().sort_index()
    value_list = df_cl_label['overall'].tolist()

    trace = go.Pie(labels=labels,
                   values=value_list,
                   marker=dict(colors=['rgb(42,60,142)', 'rgb(165,12,12)']))
    data = [trace]
    return data

    
    
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess(sentance):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    return sentance.strip()


def vectorization(sent):
    sent_vec = np.zeros(50) 
    cnt_words = 0 
    for word in sent.split(): 
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    return sent_vec

def load_model():
    global pickle_model
    global w2v_model
    global w2v_words
    global scrappedReviews
    global balanced_reviews
    
    w2v_model = Word2Vec.load("word2vec.model")
    w2v_words = list(w2v_model.wv.vocab)
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    balanced_reviews = pd.read_csv('C:/Users/Neha/Desktop/forsk/balanced_review.csv')
    balanced_reviews = balanced_reviews[balanced_reviews['overall'] != 3]
    balanced_reviews['overall'] = np.where(balanced_reviews['overall'] > 3, 1, 0 )
    
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    

def check_review(reviewText):
    reviewText = preprocess(reviewText)
    reviewText = vectorization(reviewText)
    reviewText = reviewText.reshape(1,-1)
    return pickle_model.predict(reviewText)



def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),
                    
                    html.Div(className='mat-card', 
                             style={"display": "block", "margin": "15px"},
                             children=[html.H4(children='Sentiment Pie Chart'),
                                       dcc.Graph(figure=pie_fig)]
                             ),
                    
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px'}
                    )],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'Enter your text..', style = {'height': '150px'}),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-4", id = 'button1', style = {'width': '100px'}),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout


@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    print("Data Type  = ", str(type(n_clicks)))
    print("Value      = ", str(n_clicks))    

    print("Data Type  = ", str(type(value)))
    print("Value      = ", str(value))  
    if (n_clicks > 0):
        result_list = check_review(value)
        
        if (result_list[0] == 0 ):
            return dbc.Alert("Negative", color="danger")
        elif (result_list[0] == 1 ):
            return dbc.Alert("Positive", color="success")
        else:
            return dbc.Alert("Unknown", color="dark")
    else:
        return ""
        
    
@app.callback(
    Output('result1', 'children'),
    [
    Input('button1', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    print("Data Type  = ", str(type(n_clicks)))
    print("Value      = ", str(n_clicks))    

    print("Data Type  = ", str(type(textarea)))
    print("Value      = ", str(textarea))  
    
    if (n_clicks > 0):
        result_list = check_review(textarea)
        
        if (result_list[0] == 0 ):
            return dbc.Alert("Negative", color="danger")
        elif (result_list[0] == 1 ):
            return dbc.Alert("Positive", color="success")
        else:
            return dbc.Alert("Unknown", color="dark")
    else:
        return ""



def main():
    global app
    global project_name
    global pie_fig
    load_model()
    
    data = pie_char()
    layout = go.Layout(title='For Balanced Reviews')
    pie_fig = go.Figure(data=data, layout=layout)
    
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
if __name__ == '__main__':
    main()