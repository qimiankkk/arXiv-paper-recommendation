import streamlit as st
import requests
import os

def download_pdf(arxiv_id, data_dir="data/temp_pdfs"):
    """
    Downloads the PDF from ArXiv if it doesn't exist locally.
    Returns the full path to the downloaded file.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    pdf_path = os.path.join(data_dir, f"{arxiv_id}.pdf")
    
    # Only download if we don't have it already
    if not os.path.exists(pdf_path):
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download PDF from ArXiv: {e}")
            return None
            
    return pdf_path
