import datetime
import logging
from os.path import isdir

from flask import Flask, request, jsonify
from vgg_utils_withsave import *
from vgg_scratch import *
from tensorflow.keras.models import Model
from firebase_admin import credentials, storage, firestore
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
db = firestore.client(default_app)
bucket = storage.bucket(name=str(os.environ.get('BUCKET_NAME')), app=default_app)
logging.getLogger().setLevel(logging.INFO)

vgg_descriptor = None
detector = None

mnt_dir = os.environ.get('MNT_DIR', 'mnt')
input_path = os.path.join(mnt_dir, "application-data", "input_faces")
verified_path = os.path.join(mnt_dir, "application-data", "verified_faces")
# filename = 'test-file'


def initialize_model():
    global vgg_descriptor
    global detector
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()


# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def index(path):
#     path = os.path.join(mnt_dir, path)
#     html = '<html><body><h1>Files</h1>\n'
#     if path == mnt_dir:
#         try:
#             write_file(mnt_dir, filename)
#             html += '<p>File written to {}</p>\n'.format(path)
#         except Exception as e:
#             return str(e)
#
#     if isdir(path):
#         return json.dumps(os.listdir(path))
#     else:
#         try:
#             html += read_file(path)
#         except Exception as e:
#             return str(e)
#
#     html += '</body></html>'
#     return html


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
        # dnnFaceDetector = dlib.cnn_face_detection_model_v1("./app/weights/mmod_human_face_detector.dat")
        # mmod_embedding = get_embedding_mmod(input_img_path, dnnFaceDetector, vgg_descriptor)
        # print(mmod_embedding)
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
                logging.debug("No faces detected")
                continue
            # Check if the input face is a match for the known face
            # print("input_embedding", input_embedding)
            for index, embedding in enumerate(embeddings):
                score = is_match(images[index], embedding, input_embedding)
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


@app.route('/verifyfromdb', methods=['GET'])
def predict_from_db():
    # Get request headers

    # received_date_time = datetime.datetime.now()
    # print(request.get_json(force=True))
    # header = dict(request.headers)
    # print(header)
    # print(type(request.headers), request.headers)
    # header['received_date_time'] = str(received_date_time)
    # header = json.dumps(header)
    # print(type(header), header)
    # db.collection(u'requests').add(request.get_json(force=True))
    
    try:
        input_url = db.collection(u'input_faces').document(u'input').get().to_dict()['image_url']
        logging.info(input_url)
        input_embedding = get_embedding_from_url(input_url, detector, vgg_descriptor)
        if input_embedding is None:
            raise Exception("No face detected in input image")

        verified_faces_ref = db.collection(u'verified_faces')
        verified_faces = verified_faces_ref.stream()

        all_distance = {}
        for person in verified_faces:
            person_name = person.id
            person_distance = []
            person_faces_ref = verified_faces_ref.document(person_name).collection(u'faces')
            person_faces = person_faces_ref.stream()
            for image in person_faces:
                raw_embedding = np.array(image.to_dict()['raw_embedding'])
                score = is_match(image.to_dict()['image_name'], raw_embedding, input_embedding)
                person_distance.append(score)
            all_distance[person_name] = np.mean(person_distance)
        top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
        verified = "False"
        if float(top_ten[0][1]) < 0.4:
            verified = "True"

        return {"message": "Verification Success", "content": top_ten, "verified": verified}, 200
    except Exception as e:
        return {"message": str(e)}, 400


# @app.route('/upload/<filepath>', methods=['POST'])
# def upload(filepath):
#     # Try to get embeddings of the uploaded image
#     try:
#         input_img_path = os.path.join(input_path, filepath)
#         input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
#         if input_embedding is None:
#             raise Exception("No face detected in input image")
#         return {"message": "Success"}, 200
#
#     except Exception as e:
#         return {"message": str(e)}, 400


@app.route('/uploadtodb/<filename>', methods=['POST'])
def upload_to_db(filename):
    try:
        # Image name format = (Lastname_Firstname_Datetime).jpg
        filename_fragment = filename.split('_')
        person_name = filename_fragment[0] + '_' + filename_fragment[1]
        upload_dict = db.collection(u'upload_faces').document(u'upload').get().to_dict()
        # logging.debug(upload_dict)
        upload_url = upload_dict['image_url']
        logging.info(upload_url)
        upload_embedding = get_embedding_from_url(upload_url, detector, vgg_descriptor)
        # if upload_embedding is None:
        #     raise Exception("No face detected in uploaded image")
        image_name_without_extension = filename.split('.')[0]

        # Move image to a correct location (correct bucket)
        source_blob = bucket.blob(f"application-data/upload_faces/{filename}")
        destination_blob_name = f"application-data/verified_faces/{person_name}/{filename}"
        copied_blob = bucket.copy_blob(source_blob, bucket, destination_blob_name)
        logging.info(f"Blob {source_blob.name} moved to {destination_blob_name}")
        # Delete source blob
        bucket.delete_blob(source_blob.name)

        # Upload to firestore
        db.collection(u'verified_faces').document(person_name).collection(u'faces')\
            .document(image_name_without_extension).set({'image_name': filename, 'image_url': upload_url, 'raw_embedding': upload_embedding.tolist()})
        return {"message": "Upload success", "image_name": filename, 'image_url': copied_blob.public_url, "person_name": person_name}
    except Exception as e:
        return {"message": str(e)}, 400

if __name__ == "__main__":
    initialize_model()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 80)), use_reloader=False)
