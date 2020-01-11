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

import io
from scipy import misc
import imageio
import numpy as np
logger = logging.getLogger(__name__)
img = cv2.imread("https://jpeg.org/images/jpegxl-logo.png")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
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
first_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P("This card has some text content, but not much else"),
            dbc.Button("Go somewhere", color="primary"),
        ]
    )
)


second_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This card also has some text content and not much else, but "
                "it is twice as wide as the first card."
            ),
            dbc.Button("Go somewhere", color="primary"),
        ]
    )
)


cards = dbc.Row([dbc.Col(first_card, width=4), dbc.Col(second_card, width=6)])
navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand(
                        " Fashion image", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),

    ],
    color="dark",
    dark=True,
)
second_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This card also has some text content and not much else, but "
                "it is twice as wide as the first card."
            ),
            dbc.Button("Go somewhere", color="primary"),
        ]
    )
)
badges = html.Span(
    [
        dbc.Badge("or chose with", pill=True,
                  color="light", className="mr-1"),

    ], style={'marginLeft': '30px', 'fontSize': '22px', 'marginRight': '30px'}
)
app.layout = html.Div(
    [
        navbar,

        html.Div([
            html.Div('Predict Fashion image with CNN ', style={
                 'color': 'blue', 'fontSize': 20, "fontWeight": 1000, 'marginBottom': 12}),
            dbc.Row([

                html.Div(dbc.Input(id='input-box', type='text', placeholder="Enter your Image Url",
                                   style={'color': 'blue', 'fontSize': '14px', 'marginBottom': 12, 'width': '110%', 'marginRight': '30px', })),
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
                dbc.Button("Predict", color="primary",
                           className="mr-1", id='button',
                           style={'borderRaduis': '30px', 'width': '14%',
                                  'color': 'beige', 'fontSize': 18,
                                  'marginBottom': 12,
                                  'marginLeft': '120px',
                                  'height': '40px',
                                  "fontWeight": 700}),
                dbc.Button("Predict", color="primary",
                           className="mr-1", id='show-secret',
                           style={'borderRaduis': '30px', 'width': '14%',
                                  'color': 'beige', 'fontSize': 18,
                                  'marginBottom': 12,
                                  'marginLeft': '120px',
                                  'height': '40px',
                                  "fontWeight": 700}),


            ], style={"width": '100%', 'height': '100px', 'marginTop': '30px'}),

            dbc.Row([
                html.Div(id='output-container-button'),
                html.Div(id='output-image-upload'), ], style={"width": '80%'}),
                html.Div(id='body-div')


        ], style={'marginTop': 40, 'marginLeft': "10%"})

    ])


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
)
def update_output(n_clicks, value):
    ii = cv2.imread("test11.png")
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='Greys')
    width = int(28)
    height = int(28)
    dim = (width, height)
    # resize image
    resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)
    resized = resized / 255.0
    image = resized.reshape(1, 28, 28, 1)
    if (value == None):
        return ''
    else:
        return html.Div([
            # HTML images accept base64 encoded strings in the same format
            html.Img(src="{}".format(value)),
            html.Hr()])


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
def update_output(n_clicks):
    ii = cv2.imread("test11.png")
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([112, 122, 120])
    gray_images = cv2.cvtColor(ii, cv2.COLOR_BGR2HSV)

    gray_image = cv2.inRange(gray_images, lower_blue, upper_blue)
    print(gray_image.shape)
    plt.imshow(gray_image, cmap='Greys')
    width = int(28)
    height = int(28)
    dim = (width, height)
    # resize image
    resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)
    resized = resized / 255.0
    image = resized.reshape(1, 28, 28, 1)
    if n_clicks is None:
        raise PreventUpdate
    else:
        return f" {image.shape}"


def parse_contents(contents, filename, date):
    value = contents.split(',')
    value = value[1]
    image = base64.b64decode("{}".format(value))
    img = Image.open(io.BytesIO(image))
    img.save("test11.png", 'png')
    return html.Div([
        dbc.Card([
            dbc.CardHeader(filename, className="card-title"),
            dbc.CardBody(
                [html.Img(src=contents),
                 html.H5("Image Class:"),
                 html.Div(id='body-div')]
            ),
        ], style={'boxShadow': '0 8px 8px 0 rgba(0,0,0,0.2)', "width": '160%'})
    ], style={'marginLeft': "330px"})


if __name__ == '__main__':
    app.run_server(debug=True)


