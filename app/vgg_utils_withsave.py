import logging
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def extract_face_from_url(url, detector, required_size=(224, 224)):
    try:
        # img = np.array(Image.open(urllib.request.urlopen(url)))
        img = Image.open(urllib.request.urlopen(url))
    except Exception as e:
        raise e
    # Rotate image
    img = np.array(img)
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    faces = detector.detect_faces(rotated_img)
    if not faces:
        raise Exception("No face detected in extract_face")
    # extract details of the largest face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face_boundary = img[y1:y2, x1:x2]
    # resize pixels to the model size
    face_image = Image.fromarray(face_boundary)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    return face_array


def extract_face(img_path, detector, required_size=(224, 224)):
    try:
        img = plt.imread(img_path)
    except Exception as e:
        raise e
    faces = detector.detect_faces(img)
    if not faces:
        raise Exception("No face detected in extract_face")
    # extract details of the largest face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face_boundary = img[y1:y2, x1:x2]
    # resize pixels to the model size
    face_image = Image.fromarray(face_boundary)
    # plt.imshow(face_image)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    return face_array


def get_embeddings(filenames, detector, model, read_from_file=True, save_to_file=True):
    try:
        # extract the largest face in each filename
        # convert into an array of samples
        embeddings = []
        for file in filenames:
            if read_from_file:
                try:
                    with open(file[:-4] + "_embedding.npy", "rb") as f:
                        file_embedding = np.load(f, allow_pickle=True)
                        embeddings.append(file_embedding)
                except FileNotFoundError:
                    face_embedding = get_embedding(file, detector, model)
                    if face_embedding is not None:
                        embeddings.append(face_embedding)
                        if save_to_file:
                            save_embedding(face_embedding, file[:-4] + "_embedding.npy")
                    else:
                        logging.exception("Get embedding function return None", file)
                        raise Exception("Get embedding function return None")
            else:
                face_embedding = get_embedding(file, detector, model)
                if face_embedding is not None:
                    embeddings.append(face_embedding)
                    if save_to_file:
                        save_embedding(face_embedding, file[:-4] + "_embedding.npy")
                else:
                    logging.exception("Get embedding function return None", file)
                    raise Exception("Get embedding function return None")
        return embeddings
    except Exception as e:
        logging.exception(e)
        return None


def get_embedding(filename, detector, model, save_to_file=True):
    # extract largest face in each filename
    face = [extract_face(filename, detector)]
    # convert into an array of samples
    try:
        sample = np.asarray(face, 'float32')
        # prepare the face for the model, e.g. center pixels
        # samples = preprocess_input(samples, version=2)
        sample = preprocess_input(sample, data_format='channels_last')
        # create a vggface model
        # model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        # perform prediction
        yhat = model.predict(sample)
        if save_to_file:
            save_embedding(yhat[0], filename[:-4] + "_embedding.npy")
        return yhat[0]
    except Exception as e:
        print(e)
        return None


def get_embedding_from_url(url, detector, model):
    try:
        # extract largest face in each filename
        face = [extract_face_from_url(url, detector)]
        # convert into an array of samples
        sample = np.asarray(face, 'float32')
        # prepare the face for the model, e.g. center pixels
        # samples = preprocess_input(samples, version=2)
        sample = preprocess_input(sample, data_format='channels_last')
        # perform prediction
        yhat = model.predict(sample)
        return yhat[0]
    except Exception as e:
        raise e


def is_match(image_name, known_embedding, candidate_embedding, thresh=0.3):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        logging.debug('>face is a Match (%.3f <= %.3f) %s' % (score, thresh, image_name))
        logging.debug(candidate_embedding)
    else:
        logging.debug('>face is NOT a Match (%.3f > %.3f) %s' % (score, thresh, image_name))
    return score


def save_embedding(embeddings, filename):
    with open(filename, "wb") as f:
        np.save(f, embeddings)
