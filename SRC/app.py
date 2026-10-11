# Imporiting Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
import google.generativeai as genai

#client = OpenAI(api_key=st.secrets["api_keys"]["openai"])
# Loading the Model and saving to cache
@st.cache(allow_output_mutation=True)
def load_model(path):
    
    # Xception Model
    xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])


    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model


# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model(r"C:\Users\annaj\Downloads\DIP\Backup\model.h5")

# Title and Description
st.title('Plant Diesease Detection')
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])

# If there is a uploaded file, start making prediction
if uploaded_file != None:
    
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    i = 0
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(Image.fromarray(
        np.array(image)).resize((700, 400), Image.Resampling.LANCZOS)), width=None)
    my_bar.progress(i + 40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 30)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Removing progress bar and text after prediction done
    my_bar.progress(i + 30)
    progress.empty()
    i = 0
    my_bar.empty()
    # Show the results
    # Show the results
    st.write(f"The plant {result['status']} with {result['prediction']} prediction.")
    
    # client = OpenAI(api_key="Your_API_KEY_here")  # Replace with your actual API key or use st.secrets

    #genai.configure(api_key="Your_API_Key_Here")  # Replace with your actual API key or use st.secrets

    # Load model
    model = genai.GenerativeModel("gemini-2.5-pro")

    # Streamlit section
    if result['status'] == "is Healthy":
        st.markdown("### The plant is healthy, no treatment needed!")
    else:
        # Configure Gemini API key (SECURITY WARNING: Use st.secrets instead of hardcoding)
        try:
            genai.configure(api_key="Your_API_Key_Here")  # Replace with your actual API key or use st.secrets
        except (KeyError, FileNotFoundError):
            st.error("ERROR: Gemini API key not found. Please add it to your Streamlit secrets.")
            st.stop() # Stop the app if the key is not found

        # Load the correct model
        model = genai.GenerativeModel("gemini-2.5-pro")

        # Define the prompt based on the plant's health status
        if "Healthy" in result['status']:
            st.success("The plant appears to be healthy!")
            st.markdown("### The plant is healthy, no treatment needed!")
            # st.markdown("### AI-Based Tips for Healthy Plants (via Google Gemini)")
            prompt = f"The plant is healthy with {result['status']} prediction . Provide 2-3 lines of useful tips for farmers to maintain its health and ensure a good yield."
        else:
            #st.warning(f"The plant appears to have: {result['status']}")
            st.markdown("### AI-Based Suggestion (via Google Gemini)")
            prompt = f"In 2-3 lines, what fertilizer should be used and what should be avoided for treating {result['status']} disease with {result['prediction']}in plants? Also, provide some useful tips for farmers."
            # with st.spinner("Generating AI suggestions..."):
            #     try:
            #         response = model.generate_content(prompt)
            #         if response.text:
            #             st.markdown(response.text)
            #         else:
            #             st.warning("Gemini API returned an empty response.")
            #     except Exception as e:
            #         st.error("Failed to fetch response from Gemini API.")
            #         st.error(str(e))
        # Generate content from Gemini
        with st.spinner("Generating AI suggestions..."):
            try:
                response = model.generate_content(prompt)
                if response.text:
                    st.markdown(response.text)
                else:
                    st.warning("Gemini API returned an empty response.")
            except Exception as e:
                st.error("Failed to fetch response from Gemini API.")
                st.error(str(e))


