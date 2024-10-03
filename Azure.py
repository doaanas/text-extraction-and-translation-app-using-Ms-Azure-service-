import streamlit as st
import os
from PIL import Image , ImageDraw
from dotenv import load_dotenv
import requests, json
from matplotlib import pyplot as plt
import time
import io

# Azure imports
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient
from azure.ai.translation.text import *
from azure.ai.translation.text.models import InputTextItem
import azure.cognitiveservices.speech as speechsdk


from IPython.display import display, Audio

# Load environment variables
load_dotenv()

# Azure credentials
ai_endpoint = "https://final1.cognitiveservices.azure.com/"
ai_key = "8de58bdd42e54f7499c7c28a7b9e2622"
cog_key = "dd4c8df8d8ee4176878a260c7ccca826"
cog_region = "eastus"

# Initialize clients
cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))
credential = TranslatorCredential(cog_key, cog_region)
client = TextTranslationClient(credential)
# Streamlit UI
st.title("Image Text Extractor and Translator")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Initialize session state for extracted text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "translated_text" not in st.session_state:
    st.session_state.translated_text  = ""


def translate(text, target_language):
    # Use the Azure AI Translator to translate the text
    input_text_elements = [InputTextItem(text=text)]
    translation_response = client.translate(content=input_text_elements, to=[target_language])
    translation = translation_response[0].translations[0].text if translation_response else None

    # Return the translation
    return translation
    
def speak(text):
      # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
      speech_config = speechsdk.SpeechConfig(subscription='2b9659352e704d1fbaab2fc10de5e226', region='eastus')
      audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

     # The neural multilingual voice can speak different languages based on the input text.
      speech_config.speech_synthesis_voice_name='en-US-AvaMultilingualNeural'

      speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


      speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
      stream = speechsdk.AudioDataStream(speech_synthesis_result)
      stream.save_to_wav_file("speech.wav")

      if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
         print("Speech synthesized for text [{}]".format(text))
         display(Audio(filename="speech.wav", autoplay=True))
      elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        if cancellation_details is not None:
           print("Speech synthesis canceled: {}".format(cancellation_details.reason))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Extract Text"):
        # Extract text from the image
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open("temp_image.png", "rb") as f:
            image_data = f.read()

        result = cv_client.analyze(image_data=image_data, visual_features=[VisualFeatures.READ])
        
        if result.read is not None:
            # Store extracted text in session state
            st.session_state.extracted_text = "\n".join([line.text for block in result.read.blocks for line in block.lines])
            st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
              

    # Language selection and translation if extracted text is available
    if st.session_state.extracted_text:
        target_language = st.selectbox("Select target language", options=["fr", "es", "de", "zh-Hans",'en','ar'])

        if st.button("Translate"):
            try:
                # Use the extracted text from session state
                st.session_state.translated_text  = translate(st.session_state.extracted_text, target_language)
                st.success(f"Translated text: {st.session_state.translated_text }")

            except Exception as e:
                st.error(f"Error: {e}")
        
        # button for listening to translation
      #  if st.session_state.translated_text:
       #    if st.button("Listen to translation"):
         #      speak(st.session_state.translated_text )
    
        
                

  
      

            
