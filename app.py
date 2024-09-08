# required libraries
import streamlit as st
import tempfile
from pathlib import Path
import google.generativeai as genai
import os

from api_key import api_key

genai.configure(api_key=api_key)

# Function to upload image to Gemini
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file as: {file.uri}")
    return file 

# Model setting
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Default safety settings
  
)

# App Front-end

st.set_page_config(page_title = "MedImage Diagnosio" , page_icon = ":robot:" )

st.image("Medim.png" , width = 230)

st.title("MedImage Diagnoso 🩺")

st.subheader("Transforming Medical Images into Precise Diagnoses")
uploaded_file = st.file_uploader("Upload the medical image for diagnosis and analysis", type = ["png", "jpg", "jpeg"])
submit_button = st.button("Diagnose and Analysis")


if submit_button:
     # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name  # Get the file path
        
        st.image(uploaded_file, caption="Uploaded Image for Analysis", use_column_width=True)
        # Upload the image to Gemini using the file path
        uploaded_image = upload_to_gemini(temp_file_path, mime_type="image/jpeg")

        # Chat session for interacting with the model
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        uploaded_image,
                        "What is going on in this particular image?\n",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "The image shows a medical condition that needs to be analyzed...",
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        """As a highly skilled medical practitioner specializing in image analysis, 
                        you are tasked with examining medical images for a renowned hospital. 
                        Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

                        Your responsibilities include:
                        1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
                        2. Findings Report: Document all observed anomalies or signs of disease in a structured format.
                        3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps.
                        4. Treatment Suggestions: If appropriate, recommend possible treatment options.

                        Important Notes:
                        1. Scope of Response: Only respond if the image pertains to human health issues.
                        2. Image Clarity: If the image quality impedes clear analysis, note that certain aspects cannot be determined.
                        3. Disclaimer: Accompany your analysis with: "Consult with a doctor before making any decisions."
                        """,
                    ],
                },
            ]
        )

        # Sending image and request to the model
        response = chat_session.send_message("Please analyze the image.")
        # Displaying response in the app
        st.write(response.text)

        os.remove(temp_file_path)
else:
    st.error("Please upload an image before submitting.")
        