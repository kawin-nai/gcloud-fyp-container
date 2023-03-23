from flask import Flask, request

from vgg_utils_withsave import *
from firebase_admin import credentials, storage, firestore
from resnet_scratch import *
import os
import mtcnn
import firebase_admin

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


def initialize_model():
    global vgg_descriptor
    global detector
    detector = mtcnn.MTCNN()
    vgg_descriptor = RESNET50(input_shape=(224, 224, 3))


@app.route('/')
def index():
    return 'Welcome to the face recognition API!'


def calculate_person_distance(person, input_embedding, verified_faces_ref):
    person_name = person.id
    person_dict = person.to_dict()
    person_role = person_dict['role']
    person_distance = []
    person_faces_ref = verified_faces_ref.document(person_name).collection(u'faces')
    person_faces = person_faces_ref.stream()
    representative_url = None
    if 'representative_embedding' in person_dict:
        representative_embedding = person_dict['representative_embedding']
        representative_embedding = np.array(representative_embedding)
        score = is_match("representative", representative_embedding, input_embedding)
        person_distance.append(score)
        representative_url = person_dict['representative_url']
    else:
        for idx, image in enumerate(person_faces):
            image_dict = image.to_dict()
            if idx == 0:
                representative_url = image_dict['image_url']
            # Get embeddings
            resnet_embedding = np.array(image_dict['resnet_embedding'])
            score = is_match(image_dict['image_name'], resnet_embedding, input_embedding)
            person_distance.append(score)

    return {
        'person_name': person_name,
        'role': person_role,
        'distance': np.mean(person_distance),
        'face_url': representative_url
    }


def is_verified(top_ten):
    return "True" if float(top_ten[0]['distance']) < 0.5 else "False"


@app.route('/api/verify', methods=['POST'])
def predict_from_post():
    try:
        # print(request)
        args = request.args
        camera_choice = args.get('camera')
        print(camera_choice)
        file = request.files['image']
        # logging.info(input_url)
        input_embedding = get_embedding_from_post(file, detector, vgg_descriptor, version=2, camera=camera_choice)
        if input_embedding is None:
            raise Exception("No face detected in input image")

        verified_faces_ref = db.collection(u'verified_faces')
        verified_faces = verified_faces_ref.stream()

        all_distance = []
        for person in verified_faces:
            person_object = calculate_person_distance(person, input_embedding, verified_faces_ref)
            all_distance.append(person_object)
        # print(all_distance)
        top_ten = sorted(all_distance, key=lambda x: x['distance'])[:10]

        return {"message": "Verification Success", "content": top_ten, "verified": is_verified(top_ten)}, 200
    except Exception as e:
        return {"message": str(e)}, 400


@app.route('/api/register/<filename>', methods=['POST'])
def upload_to_post(filename):
    try:
        args = request.args
        camera_choice = args.get('camera')
        role = args.get('role')
        file = request.files['image']
        filestream = file.stream
        # Image name format = (Lastname_Firstname_Datetime).jpg
        filename_fragment = filename.split('_')
        upload_embedding = get_embedding_from_post(file, detector, vgg_descriptor, version=2, camera=camera_choice)

        # Concatenate every fragment except the last one
        person_name = '_'.join(filename_fragment[:-1])
        image_name_without_extension = filename.split('.')[0]

        filestream.seek(0)
        # Uplaod image to cloud storage
        destination_blob = bucket.blob(f"application-data/verified_faces/{person_name}/{filename}")
        destination_blob.content_type = 'image/jpeg'
        destination_blob.upload_from_file(filestream)

        # Upload to firestore
        verified_ref = db.collection(u'verified_faces').document(person_name)
        # if this document doesn't exist
        if not verified_ref.get().exists:
            verified_ref.set({'name': person_name,
                              'role': role,
                              'embedding_count': 1,
                              'representative_embedding': upload_embedding.tolist(),
                              'representative_url': destination_blob.public_url})
        else:
            # Get current representative embedding
            representative_embedding = np.array(verified_ref.get().to_dict()['representative_embedding'])
            # Get current embedding count
            embedding_count = verified_ref.get().to_dict()['embedding_count']
            # Calculate new representative embedding
            new_representative_embedding = representative_embedding + (upload_embedding - representative_embedding) / (embedding_count + 1)
            # Update representative embedding and embedding count
            verified_ref.update({'representative_embedding': new_representative_embedding.tolist(),
                                 'embedding_count': embedding_count + 1,
                                 'representative_url': destination_blob.public_url})
        verified_ref.collection(u'faces') \
            .document(image_name_without_extension).set(
            {'image_name': filename,
             'image_url': destination_blob.public_url,
             'resnet_embedding': upload_embedding.tolist()})

        return {"message": "Upload success",
                "image_name": filename,
                'image_url': destination_blob.public_url,
                "person_name": person_name}, 200
    except Exception as e:
        return {"message": str(e)}, 400


if __name__ == "__main__":
    initialize_model()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 80)), use_reloader=False)
