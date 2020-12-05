import streamlit as st

# import the necessary packages for image recognition
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib

# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Image Classification")
st.sidebar.subheader("Input")
models_list = ["VGG16", "VGG19", "Inception", "Xception", "ResNet"]
network = st.sidebar.selectbox("Select the Model", models_list)

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "Inception": InceptionV3,
    "Xception": Xception,  # TensorFlow ONLY
    "ResNet": ResNet50,
}

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)
# component for toggling code
show_code = st.sidebar.checkbox("Show Code")

if uploaded_file:
    bytes_data = uploaded_file.read()

    # initialize the input image shape (224x224 pixels) along with
    # the pre-processing function (this might need to be changed
    # based on which model we use to classify our image)
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    # if we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image pre-processing function
    if network in ("Inception", "Xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    Network = MODELS[network]
    model = Network(weights="imagenet")

    # load the input image using PIL image utilities while ensuring
    # the image is resized to `inputShape`, the required input dimensions
    # for the ImageNet pre-trained network
    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through the network
    image = np.expand_dims(image, axis=0)
    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess(image)

    preds = model.predict(image)
    predictions = imagenet_utils.decode_predictions(preds)
    imagenetID, label, prob = predictions[0][0]

    st.image(bytes_data, caption=[f"{label} {prob*100:.2f}"])
    st.subheader(f"Top Predictions from {network}")
    st.dataframe(
        pd.DataFrame(
            predictions[0], columns=["Network", "Classification", "Confidence"]
        )
    )

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = "https://raw.githubusercontent.com/nithishr/streamlit-ml-demo/main/" + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if show_code:
    st.code(get_file_content_as_string("ml_frontend.py"))
