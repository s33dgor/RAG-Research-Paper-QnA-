import streamlit as st
import nbimporter
import pandas as pd
import re
import os
from helper_functions import (
    load_pdfs, 
    setup_qa_chain, 
    generate_citation, 
    PaperMetadata,
    SectionBasedTextSplitter,
    llm
)
from langchain.document_loaders import PyPDFLoader
from typing import Tuple
import re

def extract_arxiv_id(file_path: str) -> str:
    """
    Extract the arXiv ID from the file path and construct the arXiv PDF URL.
    """
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    arxiv_url = f"https://arxiv.org/pdf/{filename}.pdf"
    return arxiv_url

def summarize_text(text: str, llm_model) -> str:
    """
    Summarize the given text using the provided LLM model.
    Returns a summary of approximately 100 words.
    """
    prompt = f"""Please summarize the following text in about 100 words while maintaining the key technical details:
    
    {text}"""
    return llm_model(prompt)

# Set page configuration
st.set_page_config(
    page_title="Research Paper Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #FF3333;
            color: white;
        }
        .upload-section {
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #ccc;
            margin-bottom: 2rem;
            text-align: center;
        }
        .citation-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .source-text {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin: 0.5rem 0;
        }
        h1 {
            color: #FF4B4B;
            font-size: 2.5rem;
            margin-bottom: 2rem;
        }
        h2 {
            color: #333333;
            font-size: 1.8rem;
            margin: 1.5rem 0;
        }
        .stProgress > div > div > div > div {
            background-color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'current_file_path' not in st.session_state:
    st.session_state.current_file_path = None

def save_uploaded_file(uploaded_file, save_dir):
    """
    Save the uploaded file to the specified directory and return the file path
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Main app layout
st.title("📚 Research Paper Analysis Hub")

# Sidebar for configuration and information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
        This tool helps you analyze research papers and generate citations.
        
        **Features:**
        - Upload PDF research papers
        - Ask questions about the content
        - Generate citations in multiple formats
        - Extract key insights
        
        **How to use:**
        1. Upload your research paper
        2. Enter your research question
        3. Choose citation format
        4. Get instant analysis
    """)
    
    st.header("🔧 Settings")
    citation_format = st.selectbox(
        "Citation Format",
        ("APA", "MLA", "Chicago", "BibTeX", "RIS"),
        help="Choose your preferred citation format"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📄 Drop your research paper here", type=['pdf'])
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Paper statistics (if file is uploaded)
    if uploaded_file:
        st.markdown("### 📊 Paper Stats")
        st.metric("File Size", f"{round(uploaded_file.size/1024, 1)} KB")
        st.metric("File Type", uploaded_file.type)

# Directory setup
data_dir = "/content/drive/MyDrive/RAG Project/Data"

# Handle file upload and processing
if uploaded_file:
    with st.spinner("Processing paper..."):
        try:
            # Save the file and store its path
            file_path = save_uploaded_file(uploaded_file, data_dir)
            st.session_state.current_file_path = file_path
            
            # Generate arXiv URL
            arxiv_url = extract_arxiv_id(file_path)
            st.session_state.arxiv_url = arxiv_url
            
            # Load only the current PDF using the existing load_pdfs function
            st.session_state.documents = load_pdfs(file_path)
            if st.session_state.documents:
                st.session_state.processing_complete = True
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
# Research question input with example
st.markdown("### 🔍 Ask Your Research Question")
question_placeholder = "e.g., What are the main findings of this paper?"
question = st.text_input("", placeholder=question_placeholder)

# Generate button with loading state
if st.button("🚀 Generate Analysis", help="Click to analyze the paper and generate citations"):
    if question:
        if not st.session_state.documents:
            if st.session_state.current_file_path:
                st.session_state.documents = load_pdfs(st.session_state.current_file_path)
            else:
                st.error("Please upload a paper first!")
                st.stop()

        with st.spinner("Analyzing paper and generating response..."):
            # Set up QA chain
            qa_chain = setup_qa_chain(st.session_state.documents)
            
            # Generate answer
            result = qa_chain({"query": question})
            answer, sources = result["result"], result["source_documents"]

            # Extract helpful answer
            pattern = r"Helpful Answer:\s*(.*?)(?=\n\n|Unhelpful Answer|$)"
            match = re.search(pattern, answer, re.DOTALL)
            helpful_answer = match.group(1).strip() if match else answer

            # Display results in organized tabs
            tab1, tab2, tab3 = st.tabs(["📝 Answer", "📚 Sources", "🎯 Citations"])
            
            with tab1:
                st.markdown("### Analysis")
                st.markdown(f">{helpful_answer}")

            with tab2:
                st.markdown("### Summarized Source Excerpts")
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"Source {i} - Page {doc.metadata['page']}"):
                        # Generate summary of the source text
                        summary = summarize_text(doc.page_content.strip(), llm)
                        st.markdown("**Summary:**")
                        st.markdown(f">{summary}")
                        st.markdown("**Original Text:**")
                        st.markdown(f"```{doc.page_content.strip()}```")

            with tab3:
                st.markdown("### Paper Citation")
                # Generate single citation for the paper using metadata from first source
                if sources:
                    metadata_fetcher = PaperMetadata()
                    paper_url = arxiv_url
                    metadata = metadata_fetcher.get_paper_metadata(paper_url)
                    # Add arXiv URL to metadata if available
                    if hasattr(st.session_state, 'arxiv_url'):
                        metadata['url'] = st.session_state.arxiv_url
                    citation = generate_citation(metadata, format=citation_format)
                    st.markdown("**Citation:**")
                    st.markdown(f"_{citation}_")

                # Citation download option
                if citation_format in ["BibTeX", "RIS"]:
                    bibliography = generate_citation(metadata, format=citation_format)  # Only use first source for citation
                    st.download_button(
                        label=f"📥 Download {citation_format} Citation",
                        data=bibliography,
                        file_name=f"bibliography.{citation_format.lower()}",
                        mime="text/plain"
                    )
    else:
        st.warning("⚠️ Please enter a research question.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for research paper analysis")