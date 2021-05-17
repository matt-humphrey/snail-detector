import os
import shutil
from PIL import Image
from pykrige.ok import OrdinaryKriging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import math
from statistics import mean
from GPSPhoto import gpsphoto
import plotly.graph_objects as go
import cloudinary.api
import torch
from flask import Flask, render_template, request, redirect, jsonify, url_for

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLOv5 Model for Object Detection
model = torch.hub.load('matt-humphrey/yolov5', 'yolov5s', classes=1, pretrained=False)  # , force_reload=True)
checkpoint = torch.load('models/s.1440.pt')['model']
model.load_state_dict(checkpoint.state_dict())
model = model.autoshape()

# Google Static Maps API Key
gmaps_api_key = "XXX" 

# url variable store url
url = "https://maps.googleapis.com/maps/api/staticmap?"

# Information to access Cloudinary account if processing images via the cloud
cloud_api_key = "359879132351349"
api_secret = "XXX"
cloud_name = "mhumphrey"

results = cloudinary.api.resources(type="upload", max_results=30,
                                   cloud_name=cloud_name, api_key=cloud_api_key, api_secret=api_secret)
cloud_imgs = []
for result in results['resources']:
    cloud_imgs.append(result['url'])


# Define function to truncate length of longitude and latitude values
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


# Remove existing unnecessary files from static and process folder
def clean():
    path = os.listdir(dir_sta)
    lst_ex = ['pytorch.png', 'style.css', 'script.js']
    lst = []
    for f in range(len(path)):
        if path[f] not in lst_ex:
            lst.append(path[f])
    for f in lst:
        os.remove(dir_sta + f)

    path = os.listdir(dir_pro)
    for f in range(len(path)):
        os.remove(dir_pro + path[f])


# Get predictions for input images
def get_prediction():
    imgs = []
    x = len(os.listdir(dir_sta)) - 3
    for f in range(len(os.listdir(dir_det))):
        img_path = os.listdir(dir_det)[0]
        new_path = dir_pro + 'result' + str(x) + '.jpg'
        os.rename(dir_det + img_path, new_path)
        img = Image.open(new_path)
        imgs.append(img)  # batched list of images
        x += 1

    # # Images obtained from Cloudinary
    # img_urls = []
    # count = 0
    # for url in cloud_imgs:
    #     name = dir_pro + str(count) + ".jpg"
    #     # torch.hub.download_url_to_file(url, name)  # Uncomment if downloading newly uploaded files
    #     img_urls.append(gpsphoto.getGPSData(name))
    #     count += 1
    # imgs = img_urls

# Inference
    results = model(imgs, size=1600)  # includes NMS
    return results


def mapping():
    # Determine GPS Coordinates of each image processed
    path = dir_pro
    d = {}
    la, lo = [], []
    count = 0

    for f in os.listdir(path):
        if f.endswith('jpg'):
            data = gpsphoto.getGPSData(path + f)
            if "Latitude" in data:
                d_lat, d_long = truncate(data['Latitude'], 6), truncate(data['Longitude'], 6)
                d[count] = [d_lat, d_long]
                la.append(d_lat)
                lo.append(d_long)
            else:
                d[count] = [None, None]
            count += 1

    with open("output.txt", "r") as f:
        data = [line.rstrip() for line in f]
    data = [x.split(":") for x in data]

    ii = []
    for k in d:
        ii.append(k)

    for n in range(len(os.listdir(path))):
        d[ii[n]].append(int(data[n][1]))

    lat, lon, snail_count = [], [], []

    marker = ""
    for k in d:
        if d[k][0] is not None:
            lat.append(d[k][0])
            lon.append(d[k][1])
            snail_count.append(d[k][2])
            marker += '&markers=size:mid|'
            if d[k][2] < 5:
                marker += "color:green|label:L|"
            elif d[k][2] < 10:
                marker += "color:yellow|label:M|"
            elif d[k][2] < 15:
                marker += "color:orange|label:H|"
            else:
                marker += "color:red|label:!|"
            marker += f"{d[k][0]},{d[k][1]}"

    # center defines the center of the map,
    # equidistant from all edges of the map.
    center = f"{mean([min(la), max(la)])},{mean([min(lo), max(lo)])}"

    # size
    size = "640x640"  # 640x640 is the max resolution for a free account

    # zoom defines the zoom
    # level of the map
    zoom = 18  # will probably be between 16-18 for a paddock depending on the scale

    # get url for static map image: one regular image and one with markers denoting GPS locations
    plain_map_url = (url + "center=" + center + "&zoom=" + str(zoom) + "&size=" +
                     size + "&maptype=" + "satellite" + "&key=" + gmaps_api_key)

    markers_map_url = (url + "center=" + center + "&zoom=" + str(zoom) + "&size=" +
                       size + "&maptype=" + "satellite" + marker + "&key=" + gmaps_api_key)

    n = str(len(os.listdir(dir_map)))
    # torch.hub.download_url_to_file(plain_map_url, dir_map + "plain" + n + ".png")
    # torch.hub.download_url_to_file(markers_map_url, dir_map + "markers" + n + ".png")

    size = len(data)
    lat = np.array(lat[:size])
    lon = np.array(lon[:size])
    z = np.array(snail_count[:size])

    # Generate a regular grid with X° longitude and Y° latitude steps:
    grid_lat = np.linspace(np.min(lat), np.max(lat), size)
    grid_lon = np.linspace(np.min(lon), np.max(lon), size)

    # Define the variogram models available using pykrige
    vm = ["gaussian", "exponential", "spherical", "linear", "power", "hole-effect"]

    # Hole-Effect
    OK = OrdinaryKriging(lon, lat, z, variogram_model=vm[5], verbose=False, enable_plotting=False)
    z6, ss6 = OK.execute("grid", grid_lon, grid_lat)

    # Create Pandas Dataframe to visualise data
    b = np.concatenate(([lat], [lon], [z]), axis=0)
    df = pd.DataFrame(b.T, columns='Lat Long Snails'.split())
    df = df.pivot_table(values='Snails', index='Lat', columns='Long')

    # Display plots/graphs

    # fig, axis = plt.subplots(1, 1, figsize=(15, 10), dpi=100)

    # Display basic grid map showing snail counts related to the corresponding lat and lon
    ax = sns.heatmap(df, cmap='viridis', annot=True, cbar=True)
    ax.invert_yaxis()
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # ax.set(yticks=[])
    # ax.set(xticks=[])

    # plt.savefig("heatmap3.png")
    plt.show()

    fig3 = go.Figure(data=go.Contour(z=z6))
    fig3.show()


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        return redirect(url_for('map'))
    model_selection_options = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    model_dict = {model_name: None for model_name in model_selection_options}
    return render_template('index.html', model_selection_options=model_selection_options)


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/map')
def map():
    return render_template('export.html')


# @app.route('/predict')
# def predict():
#     # if len(os.listdir(dir_det)) < 1:
#     #     for f in range(4):
#     #         shutil.copy(dir + "data/" + str(f) + '.jpg', dir_det)
#     # clean()  # remove unnecessary existing files from static and process folders
#     results = get_prediction()
#     results.display(save_txt=True)
#     results.save(dir_sta)  # save as results1.jpg, results2.jpg... etc.
#     # mapping()
#
#     return render_template('result.html')  # CHANGE TO SHOW HEATMAP AND PHOTO RESULTS!!!


if len(os.listdir(dir_det)) < 1:
    for f in range(5):
        shutil.copy(dir + "data/" + str(f) + '.jpg', dir_det)
clean()  # remove unnecessary existing files from static and process folders
results = get_prediction()
results.display(save_txt=True)
results.save(dir_sta)  # save as results1.jpg, results2.jpg... etc.
# mapping()
