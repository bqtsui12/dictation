import os
import streamlit as st
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Cache the Whisper model loading
@st.cache_resource
def load_whisper_model():
    model_id = "bqtsio/whisper-medium-med"  # Update with your model ID
    return pipeline("automatic-speech-recognition", model=model_id, device_map="auto")

# Cache the LLM loading
@st.cache_resource
def load_llm_model():
    return pipeline(model="Qwen/Qwen2.5-1.5B-Instruct", device_map='auto')

# Define an audio processor class for capturing and processing microphone input
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = b""
    
    def recv(self, frame):
        self.audio_data += frame.to_ndarray().tobytes()
        return frame

def process_audio(audio_data):
    try:
        # Save captured audio data to a temporary file
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)

        # Load cached Whisper model
        pipe = load_whisper_model()

        # Transcribe the audio using Whisper model
        output = pipe(temp_audio_path)
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

# Streamlit UI for capturing audio from microphone and generating reports
st.title("Radiology Report Generator")

# WebRTC streamer for capturing audio from microphone
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.audio_receiver:
    # Get audio data from microphone input
    audio_processor = webrtc_ctx.audio_processor
    if audio_processor and len(audio_processor.audio_data) > 0:
        st.write("Processing audio...")
        
        # Process and generate radiology report from captured audio data
        report = process_audio(audio_processor.audio_data)
        
        st.subheader("Generated Radiology Report:")
        st.write(report)
else:
    st.warning("Please enable your microphone to start dictating.")
