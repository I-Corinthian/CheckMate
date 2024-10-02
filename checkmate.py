import streamlit as st
import whisper
from googlesearch import search
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI
import os
import pytesseract
from PIL import Image
from googletrans import Translator

import key
os.environ["OPENAI_API_KEY"] = key.OPENAI_API_KEY


model = whisper.load_model("base")

translator = Translator()

def transcribe_video(video_path: str):
    result = model.transcribe(video_path)
    return result['text'], result['language']

def extract_claims_from_text(text: str):
    llm = OpenAI(temperature=0.0)
    
    prompt_template = """
    You are an expert at extracting factual claims from text. 
    Given the following video transcript, extract key factual claims:

    Video Transcript:
    {transcript}

    Extracted Claims:
    """
    
    prompt = PromptTemplate(input_variables=["transcript"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    claims = llm_chain.run(transcript=text)
    return claims.split("\n")

def google_fact_check(claim: str):
    query = f"Is {claim} true?"
    search_results = list(search(query, num_results=3))  # Convert generator to list
    return search_results

def get_final_verdict(claim: str, search_results: list):
    llm = OpenAI(temperature=0.0)
    print(claim)
    print(search_results)
    prompt = f"""
    You are an expert fact-checker. Evaluate the following claim based on the provided search results. 
    If the claim is explicitly supported by the search results, indicate it is true. 
    If it is explicitly denied, indicate it is false. 
    If the results are ambiguous or insufficient to support or deny the claim, indicate uncertainty.

    Claim: "{claim}"

    Search Results:
    {search_results}

    Provide a clear verdict: True, False, or Uncertain.
    """
    
    verdict = llm(prompt)
    return verdict.strip()

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

st.title("Enhanced Fact-Checker App")

languages = ['English', 'Hindi', 'Spanish', 'French', 'German']
selected_language = st.selectbox("Select language for transcription", languages)

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
uploaded_image = st.file_uploader("Upload an image file containing text", type=["jpg", "jpeg", "png"])

if uploaded_video is not None:
    with open(os.path.join("uploaded_video.mp4"), "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.success("Video uploaded successfully!")
    st.info("Transcribing the video, please wait...")
    transcript, detected_lang = transcribe_video("uploaded_video.mp4")
    st.subheader(f"Transcript (Language Detected: {detected_lang}):")
    st.write(transcript)

    st.info("Extracting claims from the transcript...")
    claims = extract_claims_from_text(transcript)
    st.subheader("Extracted Claims:")
    st.write(claims)

    st.info("Fact-checking the claims...")
    for claim in claims:
        if claim.strip():
            st.write(f"Fact-checking claim: **{claim}**")
            results = google_fact_check(claim)
            st.write(f"Search results for claim '{claim}':")
            for result in results:
                st.write(result)
            verdict = get_final_verdict(claim, results)
            st.subheader(f"Final Verdict: {verdict}")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.success("Image uploaded successfully!")

    st.info("Extracting text from the image, please wait...")
    extracted_text = extract_text_from_image(image)
    st.subheader("Extracted Text from Image:")
    st.write(extracted_text)

    if selected_language != 'English':
        extracted_text = translator.translate(extracted_text, dest='en').text
        st.subheader("Translated Text:")
        st.write(extracted_text)

    st.info("Extracting claims from the text...")
    claims = extract_claims_from_text(extracted_text)
    st.subheader("Extracted Claims:")
    st.write(claims)

    st.info("Fact-checking the claims...")
    for claim in claims:
        if claim.strip(): 
            st.write(f"Fact-checking claim: **{claim}**")
            results = google_fact_check(claim)
            st.write(f"Search results for claim '{claim}':")
            for result in results:
                st.write(result)
            verdict = get_final_verdict(claim, results)
            st.subheader(f"Final Verdict: {verdict}")

st.success("Fact-checking completed!")
