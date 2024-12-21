import streamlit as st
from google.cloud import aiplatform
import pdfplumber
import google.generativeai as genai
import os
from PIL import Image

from dotenv import load_dotenv

# # Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to get a summary from Gemini-pro API
def summarize_text(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Please summarize the following text:\n\n{text}"])  
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Function to question text using Gemini-pro API
def question_text(text, question):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Please answer the following question based on the provided text:\n\nText: {text}\n\nQuestion: {question}"])  
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
def main():
    st.title("Talk to your PDF using LLM")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)

        # Limit the text displayed to 500 characters
        display_text = text[:500] + ('...' if len(text) > 500 else '')
        
        # Display the extracted text
        st.subheader("Extracted Text")
        st.text_area("Text from PDF", display_text, height=100)

        # Get a summary
        if st.button("Get Summary"):
            summary = summarize_text(text)
            st.subheader("Summary")
            st.write(summary)

        # Ask a question
        question = st.text_input("Enter your question about the text")
        if st.button("Get Answer"):
            if question:
                answer = question_text(text, question)
                st.subheader("Answer")
                st.write(answer)
            else:
                st.warning("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()