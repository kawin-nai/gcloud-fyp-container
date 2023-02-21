import urllib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from firebase_admin import credentials, storage


# def extract_faces(img_path, detector, required_size=(224, 224)):
#     img = plt.imread(img_path)
#     faces = detector.detect_faces(img)
#     print(img_path, len(faces))
#     if not faces:
#         return None
#     # Extract the list of faces in a photograph
#     face_images = []
#     for face in faces:
#         print(face)
#         # extract the bounding box from the requested face
#         x1, y1, width, height = face['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#
#         # extract the face
#         face_boundary = img[y1:y2, x1:x2]
#         plt.imshow(face_boundary)
#         # resize pixels to the model size
#         face_image = Image.fromarray(face_boundary)
#         face_image = face_image.resize(required_size)
#         face_array = np.asarray(face_image)
#         face_images.append(face_array)
#     return face_images

def extract_face_from_url(url, detector, required_size=(224, 224)):
    try:
        img = np.array(Image.open(urllib.request.urlopen(url)))
    except Exception as e:
        print("extract_face Exception", e)
        return None
    faces = detector.detect_faces(np.array(img))
    if not faces:
        print ("No face detected in extract_face")
        return None
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
        print("extract_face Exception", e)
        return None
    faces = detector.detect_faces(img)
    if not faces:
        print ("No face detected in extract_face")
        return None
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


# def draw_faces(faces):
#     # draw each face separately
#     for i in range(len(faces)):
#         plt.subplot(1, len(faces), i + 1)
#         plt.axis('off')
#         plt.imshow(faces[i])
#     plt.show()


# def preprocess_input(x, data_format=None, version=1):
#     x_temp = np.copy(x)
#     if data_format is None:
#         data_format = K.image_data_format()
#     assert data_format in {'channels_last', 'channels_first'}
#
#     if version == 1:
#         if data_format == 'channels_first':
#             x_temp = x_temp[:, ::-1, ...]
#             x_temp[:, 0, :, :] -= 93.5940
#             x_temp[:, 1, :, :] -= 104.7624
#             x_temp[:, 2, :, :] -= 129.1863
#         else:
#             x_temp = x_temp[..., ::-1]
#             x_temp[..., 0] -= 93.5940
#             x_temp[..., 1] -= 104.7624
#             x_temp[..., 2] -= 129.1863
#
#     elif version == 2:
#         if data_format == 'channels_first':
#             x_temp = x_temp[:, ::-1, ...]
#             x_temp[:, 0, :, :] -= 91.4953
#             x_temp[:, 1, :, :] -= 103.8827
#             x_temp[:, 2, :, :] -= 131.0912
#         else:
#             x_temp = x_temp[..., ::-1]
#             x_temp[..., 0] -= 91.4953
#             x_temp[..., 1] -= 103.8827
#             x_temp[..., 2] -= 131.0912
#     else:
#         raise NotImplementedError
#
#     return x_temp


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
                        print("Get embedding function return None", file)
                        raise Exception("Get embedding function return None")
            else:
                face_embedding = get_embedding(file, detector, model)
                if face_embedding is not None:
                    embeddings.append(face_embedding)
                    if save_to_file:
                        save_embedding(face_embedding, file[:-4] + "_embedding.npy")
                else:
                    print("Get embedding function return None", file)
                    raise Exception("Get embedding function return None")
        return embeddings
    except Exception as e:
        print(e)
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


def is_match(known_embedding, candidate_embedding, thresh=0.4):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return score


def save_embedding(embeddings, filename):
    with open(filename, "wb") as f:
        np.save(f, embeddings)
