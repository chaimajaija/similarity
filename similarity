import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
# Define function to upload Excel files and process data
def process_excel_files():
    st.sidebar.header("Upload Excel Files")
    uploaded_usine = st.sidebar.file_uploader("Upload Extraction USINE Excel File", type="xlsx")
    uploaded_jira = st.sidebar.file_uploader("Upload Extraction JIRA Excel File", type="xlsx")
 
    if uploaded_usine is not None and uploaded_jira is not None:
        # Read Excel files into pandas DataFrames
        extraction_usine_df = pd.read_excel(uploaded_usine, engine='openpyxl')
        extraction_jira_df = pd.read_excel(uploaded_jira, engine='openpyxl')
 
        return extraction_usine_df, extraction_jira_df
    else:
        return None, None
 
# Define function to calculate similarities and display results
def calculate_and_display_similarities(extraction_usine_df, extraction_jira_df):
    if extraction_usine_df is not None and extraction_jira_df is not None:
        # Load pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
 
        # Generate embeddings for the data in extraction_usine_df and extraction_jira_df
        embeddings_usine = model.encode(extraction_usine_df['Décrire votre idée\n'].astype(str))
        embeddings_jira = model.encode(extraction_jira_df['Sujet'].astype(str))
 
        # Calculate cosine similarity between embeddings
        similarity_matrix = cosine_similarity(embeddings_usine, embeddings_jira)
 
        st.header("Similarities Results")
 
        # Print and display similarities greater than or equal to 0.7
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] >= 0.7:
                    st.write("Similarity between sentence {} of extraction_usine_df and sentence {} of extraction_jira_df: {:.2f}".format(i+1, j+1, similarity_matrix[i, j]))
    else:
        st.warning("Please upload both Extraction USINE and Extraction JIRA Excel files.")
 
# Main Streamlit app
def main():
    st.title("Excel Similarity Analyzer")
 
    # Sidebar
    st.sidebar.title("Options")
    extraction_usine_df, extraction_jira_df = process_excel_files()
 
    # Calculate and display similarities
    calculate_and_display_similarities(extraction_usine_df, extraction_jira_df)
 
if __name__ == "__main__":
    main()
