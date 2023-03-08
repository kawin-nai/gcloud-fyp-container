from flask import Flask

from senet_scratch import SENET50
from vgg_utils_withsave import *
from vgg_scratch import *
from firebase_admin import credentials, storage, firestore
from resnet_scratch import *
import os
import mtcnn
import firebase_admin

# Create the ShareServiceClient object which will be used to create a container client
app = Flask(__name__)
cred = credentials.Certificate(str(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))
default_app = firebase_admin.initialize_app(cred, {'storageBucket': str(os.environ.get('BUCKET_NAME'))})
db = firestore.client(default_app)
bucket = storage.bucket(name=str(os.environ.get('BUCKET_NAME')), app=default_app)
logging.getLogger().setLevel(logging.INFO)

vgg_descriptor = None
vgg_descriptor_senet = None
detector = None

mnt_dir = os.environ.get('MNT_DIR', 'mnt')
input_path = os.path.join(mnt_dir, "application-data", "input_faces")
verified_path = os.path.join(mnt_dir, "application-data", "verified_faces")


def initialize_model():
    global vgg_descriptor
    global vgg_descriptor_senet
    global detector
    # model = define_model()
    # vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    vgg_descriptor = RESNET50(input_shape=(224, 224, 3))
    # vgg_descriptor = Model(inputs=resnet.layers[0].input, outputs=resnet.layers[-1].output)
    # vgg_descriptor.summary()

    model = SENET50(input_shape=(224, 224, 3))
    vgg_descriptor_senet = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


@app.route('/verify/<filepath>', methods=['GET'])
def predict(filepath):
    try:
        input_img_path = os.path.join(input_path, filepath)
        input_embedding = get_embedding(input_img_path, detector, vgg_descriptor, save_to_file=False)
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
            embeddings = get_embeddings(images, detector, vgg_descriptor, read_from_file=False, save_to_file=False)
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

        # Convert float to string
        for i in range(len(top_ten)):
            top_ten[i] = list(top_ten[i])
            top_ten[i][1] = str(top_ten[i][1])
            top_ten[i] = tuple(top_ten[i])

        return {"message": "Verification Success", "content": top_ten, "verified": verified}, 200
    except Exception as e:
        return {"message": str(e)}, 400


@app.route('/verifyfromdb', methods=['GET'])
def predict_from_db():
    try:
        input_url = db.collection(u'input_faces').document(u'input').get().to_dict()['image_url']
        logging.info(input_url)
        input_embedding = get_embedding_from_url(input_url, detector, vgg_descriptor, version=2)
        input_embedding_senet = get_embedding_from_url(input_url, detector, vgg_descriptor_senet, version=2)
        if input_embedding is None:
            raise Exception("No face detected in input image")
        # Save input embedding to local json file
        # np.save(os.path.join(input_path, "input_embedding_v2.npy"), input_embedding)


        verified_faces_ref = db.collection(u'verified_faces')
        verified_faces = verified_faces_ref.stream()

        all_distance = []
        for person in verified_faces:
            person_name = person.id
            person_distance = []
            person_distance_senet = []
            person_faces_ref = verified_faces_ref.document(person_name).collection(u'faces')
            person_faces = person_faces_ref.stream()
            for image in person_faces:
                # raw_embedding = np.array(image.to_dict()['raw_embedding'])
                resnet_embedding = np.array(image.to_dict()['resnet_embedding'])
                senet_embedding = np.array(image.to_dict()['senet_embedding'])
                score = is_match(image.to_dict()['image_name'], resnet_embedding, input_embedding)
                score_senet = is_match(image.to_dict()['image_name'], senet_embedding, input_embedding_senet)
                person_distance.append(score)
                person_distance_senet.append(score_senet)

            # Calculate the average distance for each person
            person_object = dict()
            person_object['person_name'] = person_name
            person_object['distance'] = np.mean(person_distance)
            person_object['distance_senet'] = np.mean(person_distance_senet)
            all_distance.append(person_object)
        top_ten = sorted(all_distance, key=lambda x: x['distance'])[:10]

        verified = "False"
        if float(top_ten[0]['distance']) < 0.5:
            verified = "True"

        return {"message": "Verification Success", "content": top_ten, "verified": verified}, 200
    except Exception as e:
        return {"message": str(e)}, 400


@app.route('/uploadtodb/<filename>', methods=['POST'])
def upload_to_db(filename):
    try:
        # Image name format = (Lastname_Firstname_Datetime).jpg
        filename_fragment = filename.split('_')

        # Concatenate every fragment except the last one
        person_name = '_'.join(filename_fragment[:-1])

        upload_dict = db.collection(u'upload_faces').document(u'upload').get().to_dict()
        # logging.debug(upload_dict)
        upload_url = upload_dict['image_url']
        logging.info(upload_url)
        upload_embedding = get_embedding_from_url(upload_url, detector, vgg_descriptor, version=2)
        senet_embedding = get_embedding_from_url(upload_url, detector, vgg_descriptor_senet, version=2)

        image_name_without_extension = filename.split('.')[0]

        # Move image to a correct location (correct bucket)
        source_blob = bucket.blob(f"application-data/upload_faces/{filename}")
        destination_blob_name = f"application-data/verified_faces/{person_name}/{filename}"
        copied_blob = bucket.copy_blob(source_blob, bucket, destination_blob_name)
        logging.info(f"Blob {source_blob.name} moved to {destination_blob_name}")
        # Delete source blob
        bucket.delete_blob(source_blob.name)

        # Upload to firestore
        verified_ref = db.collection(u'verified_faces').document(person_name)
        verified_ref.set({'name': person_name, 'role': 'student'})
        verified_ref.collection(u'faces') \
            .document(image_name_without_extension).set(
            # {'image_name': filename, 'image_url': copied_blob.public_url, 'raw_embedding': upload_embedding.tolist()})
            {'image_name': filename, 'image_url': copied_blob.public_url, 'resnet_embedding': upload_embedding.tolist(), 'senet_embedding': senet_embedding.tolist()})
        return {"message": "Upload success", "image_name": filename, 'image_url': copied_blob.public_url,
                "person_name": person_name}
    except Exception as e:
        return {"message": str(e)}, 400


if __name__ == "__main__":
    initialize_model()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 80)), use_reloader=False)
