import datetime
from os.path import isdir

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
cred = credentials.Certificate(str(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))
default_app = firebase_admin.initialize_app(cred, {'storageBucket': str(os.environ.get('BUCKET_NAME'))})

vgg_descriptor = None
detector = None

mnt_dir = os.environ.get('MNT_DIR', 'mnt')
input_path = os.path.join(mnt_dir, "application-data", "input_faces")
verified_path = os.path.join(mnt_dir, "application-data", "verified_faces")
filename = 'test-file'


def initialize_model():
    global vgg_descriptor
    global detector
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    path = os.path.join(mnt_dir, path)
    html = '<html><body><h1>Files</h1>\n'
    if path == mnt_dir:
        try:
            write_file(mnt_dir, filename)
            html += '<p>File written to {}</p>\n'.format(path)
        except Exception as e:
            return str(e)

    if isdir(path):
        return json.dumps(os.listdir(path))
    else:
        try:
            html += read_file(path)
        except Exception as e:
            return str(e)

    html += '</body></html>'
    return html


def write_file(mnt_dir, filename):
    """Write files to a directory with date created"""
    date = datetime.datetime.utcnow()
    file_date = '{dt:%a}-{dt:%b}-{dt:%d}-{dt:%H}:{dt:%M}-{dt:%Y}'.format(dt=date)
    with open(f'{mnt_dir}/{filename}-{file_date}.txt', 'a') as f:
        f.write(f'This test file was created on {date}.')


def read_file(full_path):
    """Read files and return contents"""
    with open(full_path, 'r') as reader:
        return reader.read()


# Go through all directories and files and list their paths
# @app.route('/list')
# def list_files():
#     blobs = storage.bucket().list_blobs()
#     blob_list = []
#     for blob in blobs:
#         blob_list.append(blob.name)
#     return json.dumps(blob_list, indent=4)
#
#
# @app.route('/pickle')
# def list_pickle():
#     blobs = storage.bucket().list_blobs()
#     npy_dict = dict()
#     blob_list = []
#     for blob in blobs:
#         if blob.name.endswith('.npy'):
#             content = blob.download_as_string()
#             blob_list.append(content)
#             npy_dict[blob.name] = np.load(BytesIO(content)).tolist()
#     return json.dumps(npy_dict, indent=4)


# @app.route('/write')
# def write_txt_file():
#     file_path = "requirements.txt"
#     bucket = storage.bucket()
#     # Create a blob with the file path (destination blob name)
#     blob = bucket.blob("test_folder/" + file_path)
#     # Upload the file to the destination path using the source file name
#     blob.upload_from_filename(file_path)
#     return 'Success'
#
#
# @app.route('/writemnt')
# def write_mnt_file():
#     file_path = os.path.join(mnt_dir, filename + '.txt')
#     with open(file_path, 'w') as f:
#         f.write('test')
#     return 'Success'
#
#
# @app.route('/download')
# def download():
#     bucket = storage.bucket()
#     blob = bucket.blob(input_path)
#     blob.download_to_filename("input.jpg")
#     # Get file stat
#     stat = os.stat("input.jpg")
#     # Delete the file
#     os.remove("input.jpg")
#     return str(stat.st_size)
#
#
# @app.route('/dir')
# def mount_dir():
#     return str(os.listdir(mnt_dir))

# @app.route('upload/<filepath>', methods=['POST'])
# def upload(filepath):
#     s

@app.route('/verify/<filepath>', methods=['GET'])
def predict(filepath):
    try:
        input_img_path = os.path.join(input_path, filepath)
        input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
        if input_embedding is None:
            raise Exception("No face detected in input image")

        all_distance = {}
        for persons in os.listdir(verified_path):
            # print(persons)
            person_distance = []
            images = []
            for image in os.listdir(os.path.join(verified_path, persons)):
                full_img_path = os.path.join(verified_path, persons, image)
                if full_img_path[-3:] == "jpg":
                    images.append(full_img_path)
                # Get embeddings
            embeddings = get_embeddings(images, detector, vgg_descriptor)
            if embeddings is None:
                print("No faces detected")
                continue
            # Check if the input face is a match for the known face
            # print("input_embedding", input_embedding)
            for embedding in embeddings:
                score = is_match(embedding, input_embedding)
                person_distance.append(score)
            # Calculate the average distance for each person
            all_distance[persons] = np.mean(person_distance)
        top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
        # convert top_ten[0][1] to float
        verified = "False"
        if float(top_ten[0][1]) < 0.4:
            verified = "True"

        return {"message": "Verification Success", "content": top_ten, "verified": verified}, 200
    except Exception as e:
        return {"message": str(e)}, 400


@app.route('/upload/<filepath>', methods=['POST'])
def upload(filepath):
    # Try to get embeddings of the uploaded image
    try:
        input_img_path = os.path.join(input_path, filepath)
        input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
        if input_embedding is None:
            raise Exception("No face detected in input image")
        return {"message": "Success"}, 200

    except Exception as e:
        return {"message": str(e)}, 400


if __name__ == "__main__":
    initialize_model()
    # print working directory
    print(os.getcwd())
    app.run(debug=True, host="0.0.0.0", port=80, use_reloader=False)
