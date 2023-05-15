import face_recognition
# import cv2
import pickle
# from sklearn.neighbors import KNeighborsClassifier
import os
# import math
from PIL import Image, ImageDraw
import re
import numpy as np


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

# def load_image_file(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


def main():

    model_file = "model/knn_model_trained.clf"
    classifier = None
    if os.path.isfile(model_file):
        with open(model_file, 'rb') as f:
            classifier = pickle.load(f)
    else:
        raise Exception(f"Invalid classifier path: {model_file}. Check if the path is correct or pretained model is present.")
    
    for image_file in os.listdir("dataset/test"):
        full_file_path = os.path.join("dataset/test", image_file)

        print(f"Looking for faces in {image_file}")

        predictions = predict(full_file_path, knn_clf=classifier)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print(f"- Found {name} at ({left}, {top})")

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("dataset/test", image_file), predictions)

if __name__ == "__main__":
    main()