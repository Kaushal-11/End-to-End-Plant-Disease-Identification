import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("trained_model.keras")

def predict_disease(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 

    predictions = model.predict(input_arr)
    index = np.argmax(predictions)  

    return index

st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌿",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-header {
        background-color: #e0f7e0; 
        padding: 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header .description {
        flex: 1;
    }
    .main-header h1 {
        font-size: 3em; /* Increased font size */
        color: #333333;
    }
    .main-header p {
        font-size: 1.5em; /* Increased font size */
        color: #666666;
    }
    .tabs-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        font-size: 1.5em; /* Increased font size for tab bar */
    }
    .tabs-container .tab {
        margin: 0 20px; /* Increased spacing between tabs */
        padding: 10px 20px;
        border: none;
        background-color: #f0f0f0;
        color: #333333;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .tabs-container .tab:hover {
        background-color: #28a745;
        color: white;
    }
    .description-section {
        background-color: white;
        padding: 50px;
        text-align: left;
        color: black;
        font-size: 1.5em; /* Increased font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["Home", "About", "Disease Recognition"])

with tabs[0]:
    
    st.markdown(
        """
        <div class="main-header">
            <div class="description">
                <h1>Plant Disease Identification Tool</h1>
                <p>Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
            ## Welcome to the Plant Disease Recognition System! 🌿🔍
            
            ### How It Works
            - **Upload Image:** Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
            - **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
            - **Results:** View the results and recommendations for further action.
            
            ### Why Choose Us?
            - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
            - **User-Friendly:** Simple and intuitive interface for seamless user experience.
            - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
            
            ### Get Started
            Click on the *Disease Recognition* tab to upload an image and experience the power of our Plant Disease Recognition System!
            
            ### About Us
            Learn more about the project, model then click on the *About* tab.
        """,
    )

with tabs[1]:
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this [Kaggle repository](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                
                #### About CNN Model
                Convolutional Neural Network (CNN) is a type of deep learning algorithm that is widely used for image classification tasks. In this project, a CNN model has been trained on the provided dataset to recognize plant diseases from images. The model architecture consists of convolutional layers, pooling layers, and fully connected layers, enabling it to learn intricate patterns and features from the input images.
                
                The CNN model has been trained using TensorFlow and Keras, popular libraries for building deep learning models. By leveraging the power of CNNs, this system achieves high accuracy in identifying plant diseases, contributing to efficient crop management and disease prevention.
                """)

with tabs[2]:
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image"):
        if test_image:
            st.image(test_image)
        else:
            st.warning("Please upload an image first.")

    if st.button("Predict"):
        if test_image:
            with st.spinner("Predicting..."):
                result_index = predict_disease(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image first.")