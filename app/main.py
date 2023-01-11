from flask import Flask
from vgg_utils_withsave import *
from vgg_scratch import *
from tensorflow.keras.models import Model
from firebase_admin import credentials, storage
from io import BytesIO
import time
import os
import mtcnn
import firebase_admin
import json

# Create the ShareServiceClient object which will be used to create a container client
app = Flask(__name__)
cred = credentials.Certificate("fyptest-5e73d-firebase-adminsdk-8zrex-99bb1b9dcc.json")
default_app = firebase_admin.initialize_app(cred, {'storageBucket': 'fyptest-5e73d.appspot.com'})

vgg_descriptor = None
detector = None


def initialize_model():
    global vgg_descriptor
    global detector
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()


@app.route('/')
def hello_world():
    return 'Hello, World!'


# Go through all directories and files and list their paths
@app.route('/list')
def list_files():
    blobs = storage.bucket().list_blobs()
    blob_list = []
    for blob in blobs:
        blob_list.append(blob.name)
    return json.dumps(blob_list, indent=4)

@app.route('/pickle')
def list_pickle():
    blobs = storage.bucket().list_blobs()
    npy_dict = dict()
    blob_list = []
    for blob in blobs:
        if blob.name.endswith('.npy'):
            content = blob.download_as_string()
            filesize = len(content).to_bytes(4, 'big')
            blob_list.append(content)
            npy_dict[blob.name] = np.load(BytesIO(content)).tolist()
    return json.dumps(npy_dict, indent=4)


@app.route('/write')
def write_txt_file():
    file_path = "requirements.txt"
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    return 'Success'


# @app.route('/verify')
# def predict():
#     try:
#         input_img_path = os.path.join(input_path, "input.jpg")
#         input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
#         if input_embedding is None:
#             raise Exception("No face detected in input image")
#
#         all_distance = {}
#         for persons in os.listdir(verified_path):
#             # print(persons)
#             person_distance = []
#             images = []
#             for image in os.listdir(os.path.join(verified_path, persons)):
#                 full_img_path = os.path.join(verified_path, persons, image)
#                 if full_img_path[-3:] == "jpg":
#                     images.append(full_img_path)
#                 # Get embeddings
#             embeddings = get_embeddings(images, detector, vgg_descriptor)
#             if embeddings is None:
#                 print("No faces detected")
#                 continue
#             # Check if the input face is a match for the known face
#             # print("input_embedding", input_embedding)
#             for embedding in embeddings:
#                 score = is_match(embedding, input_embedding)
#                 person_distance.append(score)
#             # Calculate the average distance for each person
#             all_distance[persons] = np.mean(person_distance)
#         top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
#         return top_ten
#     except Exception as e:
#         return str(e)


if __name__ == "__main__":
    initialize_model()
    app.run(debug=True, host="0.0.0.0", port=80)
