import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('model.hdf5')
  return model
model = load_model()
st.title("Skin Cancer Detection")
file = st.file_uploader("Please upload an image file", type=["jpg"])


def import_and_predict(image_data, model):
    
        size = (32,32)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    string = "This image most likely is: " + class_names[np.argmax(predictions)]
    st.success(string)