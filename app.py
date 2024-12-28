import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model


# Streamlit app title
st.title("Watch Brand Recognition")

# Upload Image
uploaded_file = st.file_uploader("Upload an image of a watch", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    model = load_model('trained_model/brand_recognition_model.h5')  # Replace with your model path
    class_labels = ['Casio_watch', 'Omega_watch', 'Rolex_watch', 'Seiko_watch', 'Tag_Heuer_watch']  # Update with your class labels
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image for prediction
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the result
    st.write(f"Predicted Brand: {predicted_class_label}")
