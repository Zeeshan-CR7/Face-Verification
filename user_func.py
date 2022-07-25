from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

def create_box(image):
    faces = detector.detect_faces(image)
    bounding_box = faces[0]['box']
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255), 
                  2)
    return image

def extract_face(image, resize=(224, 224)):
    pixcels = np.asarray(image)
    detector = MTCNN()
    faces = detector.detect_faces(pixcels)
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1+width, y1+height
    face_boundary = pixcels[y1:y2, x1:x2]
    face_image = cv2.resize(face_boundary, resize)
    return face_image

def get_embeddings(faces):
    face = np.asarray(faces, 'float32')
    face = preprocess_input(face, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model.predict(face)

def get_similarity(faces):
    embeddings = get_embeddings(faces)
    score = cosine(embeddings[0], embeddings[1])
    distance = euclidean(embeddings[0], embeddings[1])
    return(score, distance)
