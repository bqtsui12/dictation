import os
import streamlit as st
from transformers import pipeline

# Cache the Whisper model loading
@st.cache_resource
def load_whisper_model():
    model_id = "bqtsio/whisper-medium-med"  # Update with your model ID
    return pipeline("automatic-speech-recognition", model=model_id, device_map="auto")

# Cache the LLM loading
@st.cache_resource
def load_llm_model():
    return pipeline(model="Qwen/Qwen2.5-1.5B-Instruct", device_map='auto')

# Function to process audio and generate a report
def process_audio(filepath):
    if filepath is None:
        return "No audio file provided. Please upload an audio file or record from the microphone."
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"

    try:
        # Load cached Whisper model
        pipe = load_whisper_model()

        # Transcribe the audio using Whisper model
        output = pipe(filepath, max_new_tokens=256, chunk_length_s=30, batch_size=8)
        transcription = output["text"]

        # Load cached LLM model
        llm = load_llm_model()

        # Generate formatted report using LLM
        prompt = f"""
        Your task is to help a radiologist format the following text delimited by triple quotes into a radiology report with consistent style.
        Write the report with the following sections and format:
        Study: 
        Technique: 
        Comparison: \n\n 
        Findings: \n\n 
        Impression: 

        If the text does not contain information for that section, then do not include that section in the report.
        Correct spelling mistakes.

        \"\"\"{transcription}\"\"\"
        
        Formatted Report:
        """
        
        generated_text = llm(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
        
        # Extract the formatted report from generated text
        formatted_report = generated_text.split("Formatted Report:")[1].strip()
        
        return formatted_report

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI for uploading audio and generating reports
st.title("Radiology Report Generator")

# Upload or record audio from microphone
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Process audio and display report
    st.write("Processing audio...")
    report = process_audio("temp_audio.wav")
    
    st.subheader("Generated Radiology Report:")
    st.write(report)
