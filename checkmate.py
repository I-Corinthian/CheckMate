import streamlit as st
import whisper
from googlesearch import search
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentExecutor
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
    try:
        result = model.transcribe(video_path)
        return result['text'], result['language']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "", ""

llm = OpenAI(temperature=0.0)

def extract_claims_tool(text: str):
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

def google_fact_check_tool(claim: str):
    try:
        query = f"Is {claim} true?"
        search_results = list(search(query, num_results=3))  
        return search_results
    except Exception as e:
        st.error(f"Error during Google Search: {e}")
        return []

tools = [
    Tool(
        name="Extract Claims",
        func=extract_claims_tool,
        description="Extract key factual claims from text"
    ),
    Tool(
        name="Google Fact-Check",
        func=google_fact_check_tool,
        description="Perform a Google search to verify claims"
    )
]

agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

st.title("Fact-Checker App")

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
    
    if transcript:
        st.subheader(f"Transcript (Language Detected: {detected_lang}):")
        st.write(transcript)

        st.info("Extracting and fact-checking claims using the agent...")
        
        try:
            claims = agent.run(f"Extract key factual claims from the following transcript:\n{transcript}")
            st.subheader("Extracted Claims:")
            st.write(claims)

            for claim in claims.split("\n"):
                if claim.strip():
                    st.write(f"Fact-checking claim: **{claim}**")
                    results = agent.run(f"Perform a Google search to fact-check the following claim: {claim}")
                    st.subheader(f"Fact-check Results for '{claim}':")
                    st.write(results)

        except Exception as e:
            st.error(f"Error during agent execution: {e}")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.success("Image uploaded successfully!")

    st.info("Extracting text from the image, please wait...")
    extracted_text = extract_text_from_image(image)
    
    if extracted_text:
        st.subheader("Extracted Text from Image:")
        st.write(extracted_text)

        if selected_language != 'English':
            extracted_text = translator.translate(extracted_text, dest='en').text
            st.subheader("Translated Text:")
            st.write(extracted_text)

        st.info("Extracting and fact-checking claims using the agent...")
        
        try:
            claims = agent.run(f"Extract key factual claims from the following text:\n{extracted_text}")
            st.subheader("Extracted Claims:")
            st.write(claims)

            for claim in claims.split("\n"):
                if claim.strip(): 
                    st.write(f"Fact-checking claim: **{claim}**")
                    results = agent.run(f"Perform a Google search to fact-check the following claim: {claim}")
                    st.subheader(f"Fact-check Results for '{claim}':")
                    st.write(results)

        except Exception as e:
            st.error(f"Error during agent execution: {e}")

st.success("Fact-checking completed!")
