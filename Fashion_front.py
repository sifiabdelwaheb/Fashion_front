import dash
import datetime
import logging
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import base64
from PIL import Image
from dash.exceptions import PreventUpdate
import urllib
import cv2
from urllib.request import urlopen
import io
from scipy import misc
import imageio
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
PLOTLY_LOGO = "http://dslv9ilpbe7p1.cloudfront.net/8V5JLcdly3Zl_55kbC6H2g_store_banner_image.jpeg"
search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),

    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, style={
                            'height': '70px', 'borderRadius': '120px', 'width': '70px'})),
                    dbc.Col(dbc.NavbarBrand(
                        " Fashion image", className="ml-4")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),

    ],
    color="primary",
    dark=True,
)

badges = html.Span(
    [
        dbc.Badge("or choose with", pill=True,
                  color="light", className="mr-1"),

    ], style={'marginLeft': '30px', 'fontSize': '22px', 'marginRight': '30px'}
)
app.layout = html.Div(
    [
        navbar,

        html.Div([
            html.Div('Predict Fashion image with CNN ', style={
                 'color': '#151313', 'fontSize': 20, "fontWeight": 1000, 'marginBottom': 12}),
            dbc.Row([

                html.Div(dbc.Input(id='input-box', type='text', placeholder="Enter your Image Url",
                                   style={'color': 'blue', 'fontSize': '14px',  'width': '110%',})),
                dbc.Button("Predict", color="primary",
                           className="mr-1", id='button',
                           style={'borderRaduis': '30px', 'width': '14%',
                                  'color': 'beige', 'fontSize': 16,
                                  
                                  'marginLeft': '90px',
                                  'height': '40px',
                                  "fontWeight": 700}),
                badges,

                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Upload Image ',

                    ]),
                    style={
                        'width': '120%',
                        'height': '40px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'padding': '3px',
                        'marginRight': '40px',
                        'cursor': 'pointer'
                    },

                    # Allow multiple files to be uploaded
                    multiple=True
                ),

                dbc.Button("Predict&", color="primary",
                           className="mr-1", id='show-secret',
                           style={'borderRaduis': '30px', 'width': '14%',
                                  'color': 'beige', 'fontSize': 16,
                                 
                                  'marginLeft': '90px',
                                  'height': '40px',
                                  "fontWeight": 700}),


            ], style={"width": '100%', 'height': '100px', 'marginTop': '30px'}),

            dbc.Row([
                html.Div(id='output-container-button'),
                html.Div(id='output-image-upload'), ], style={"width": '80%'}),
            html.Div(id='body-div')


        ], style={'marginTop': 40, 'marginLeft': "10%"})

    ])

# METHOD #1: OpenCV, NumPy, and urllib


def numbers_to_class(argument):
    switcher = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return switcher.get(argument)


def url_to_image(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.COLOR_BGR2HSV)
    return image


def predict_image(value):
    model = tf.keras.models.load_model('firstmodel.h5')

    val = "{}".format(value)
    ii = url_to_image(val)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([150, 150, 150])
    gray_images = cv2.cvtColor(ii, cv2.COLOR_BGR2HSV)
    gray_image = cv2.inRange(ii, lower_blue, upper_blue)
    width = int(28)
    height = int(28)
    dim = (width, height)
    resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)
    resized = resized.astype('float32') / 255
    resized = resized.reshape(1, 28, 28, 1)
    pred = model.predict(resized)

    pred = np.argmax(pred, axis=1)

    return pred


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
)
def update_output(n_clicks, value):

    if (value == None or value == "unknown" or value == ""):
        return ''
    else:
        val = "{}".format(value)

        predict = predict_image(value)
        pred = predict[0]
        pred = numbers_to_class(pred)
        
        return html.Div([
            
            dbc.Card([
                 dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(html.Div(html.Img(src="{}".format(val), style={
                            'height': '240px', 'width': '300px'}))),
                        dbc.Col(html.H1(f" {pred}")),
                    ],
                    align="center",
                    
                  
                )),

               
            ], style={'boxShadow': '0 8px 8px 0 rgba(0,0,0,0.2)', "width": '100%'}),


            html.Hr()], style={"width": "70%", "minWidth": "1000px","marginLeft":20})


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output1(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output(component_id='body-div', component_property='children'),
    [Input(component_id='show-secret', component_property='n_clicks')]

)
def update_output3(n_clicks):
    model = tf.keras.models.load_model('firstmodel.h5')
    ii = cv2.imread("test11.png")
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([150, 150, 150])
    gray_images = cv2.cvtColor(ii, cv2.COLOR_BGR2HSV)

    gray_image = cv2.inRange(ii, lower_blue, upper_blue)

    width = int(28)
    height = int(28)
    dim = (width, height)
    resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)
    resized = resized / 255.0
    image = resized.reshape(1, 28, 28, 1)
    predictions = model.predict(image)
    pred = np.argmax(predictions, axis=1)
    pred = pred[0]
    pred = numbers_to_class(pred)
    
    if n_clicks is None:
        raise PreventUpdate
    else:
        return html.H1(f" {pred}")


def parse_contents(contents, filename, date):
    value = contents.split(',')
    value = value[1]
    image = base64.b64decode("{}".format(value))
    img = Image.open(io.BytesIO(image))
    img.save("test11.png", 'png')
    return html.Div([
          
           
        dbc.Card([
           
            dbc.CardBody(
                 dbc.Row(
                    [
                    html.Div(html.Img(src=contents, style={
                            'height': '250px', 'width': '300px','marginRight':'220px'})),
                 
                 html.H1(id='body-div')],
                 align="center",
                  
                )),
                 
               
            
        ], style={'boxShadow': '0 8px 8px 0 rgba(0,0,0,0.2)', "width": '100%'})
    ], style={"width": "70%", "minWidth": "1000px","marginLeft":20})


if __name__ == '__main__':
    app.run_server(debug=True)
