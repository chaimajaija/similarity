import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl
from openpyxl.styles import Font
import base64

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

# Define function to calculate similarities and save results to Excel
def calculate_and_save_similarity(extraction_usine_df, extraction_jira_df):
    if extraction_usine_df is not None and extraction_jira_df is not None:
        # Load pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for the data in extraction_usine_df and extraction_jira_df
        embeddings_usine = model.encode(extraction_usine_df['Décrire votre idée\n'].astype(str))
        embeddings_jira = model.encode(extraction_jira_df['Sujet'].astype(str))

        # Calculate cosine similarity between embeddings
        similarity_matrix = cosine_similarity(embeddings_usine, embeddings_jira)

        # Create a new workbook and select the active sheet
        wb = openpyxl.Workbook()
        ws = wb.active

        # Write the header row
        ws.append(['Extraction USINE', 'Extraction JIRA', 'Code JIRA', 'Similarity Score'])
        header_font = Font(bold=True)
        for col in ws[1]:
            col.font = header_font

        # Write the similarity results
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] >= 0.7:
                    ws.append([extraction_usine_df.iloc[i, 14], extraction_jira_df.iloc[j, 2], extraction_jira_df.iloc[j, 1], similarity_matrix[i, j]])

        # Save the workbook to an Excel file with .xlsx extension
        wb.save('similarities.xlsx')

# Main Streamlit app
def main():
    st.title("Excel Similarity Analyzer")

    # Sidebar
    st.sidebar.title("Options")
    extraction_usine_df, extraction_jira_df = process_excel_files()

    if extraction_usine_df is not None and extraction_jira_df is not None:
        # Calculate similarity matrix and save to Excel
        if st.button("Calculate Similarity and Download"):
            calculate_and_save_similarity(extraction_usine_df, extraction_jira_df)
            st.success("Similarity matrix saved as similarities.xlsx. Click the link below to download.")
            st.markdown(get_binary_file_downloader_html('similarities.xlsx', 'Similarity_Matrix'), unsafe_allow_html=True)

# Helper function to generate a link for downloading files
def get_binary_file_downloader_html(file_path, file_label="File"):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}.xlsx">Click here to download {file_label}</a>'
    return href

if __name__ == "__main__":
    main()
