import os
import streamlit as st
import requests
from typing import List
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFacePipeline
import torch
import re
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Helper functions for sophisticated citation handling and RAG-based system
# -----------------------------------------------------------------------------

def generate_citation(metadata, format="APA"):
    """
    Generate citations in multiple formats based on provided metadata.
    
    Args:
    - metadata (dict): Document metadata containing authors, title, etc.
    - format (str): Citation format (APA, MLA, Chicago, BibTeX, or RIS)
    
    Returns:
    - str: Formatted citation string
    """
    # Extract and clean metadata
    authors = metadata.get('authors', ['Unknown Author'])
    title = metadata.get('title', 'Untitled').strip()
    year = str(metadata.get('publication_date', {}).get('year', 'n.d.'))
    journal = metadata.get('journal', 'Unknown Journal').strip()
    doi = metadata.get('doi', '')
    
    # Format authors for different citation styles
    def format_authors_apa():
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            return f"{authors[0]} et al."
    
    def format_authors_mla():
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            return f"{authors[0]} et al."
    
    def format_authors_chicago():
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            return f"{authors[0]} et al."
    
    def format_authors_bibtex():
        return " and ".join(authors)
    
    # Generate citations based on format
    if format.upper() == "APA":
        authors_formatted = format_authors_apa()
        return f"{authors_formatted}. ({year}). {title}. {journal}. {doi}"
    
    elif format.upper() == "MLA":
        authors_formatted = format_authors_mla()
        return f"{authors_formatted}. \"{title}.\" {journal}, {year}, {doi}."
    
    elif format.upper() == "CHICAGO":
        authors_formatted = format_authors_chicago()
        return f"{authors_formatted}. {year}. \"{title}.\" {journal}. {doi}."
    
    elif format.upper() == "BIBTEX":
        authors_formatted = format_authors_bibtex()
        # Create a citation key using first author's lastname and year
        first_author_lastname = authors[0].split()[-1].lower()
        citation_key = f"{first_author_lastname}{year}"
        
        bibtex_template = "@article{" + citation_key + ",\n"
        bibtex_template += "    author = {" + authors_formatted + "},\n"
        bibtex_template += "    title = {" + title + "},\n"
        bibtex_template += "    journal = {" + journal + "},\n"
        bibtex_template += "    year = {" + year + "},\n"
        bibtex_template += "    doi = {" + doi + "}\n"
        bibtex_template += "}"
        return bibtex_template
    
    elif format.upper() == "RIS":
        ris_template = "TY  - JOUR\n"
        for author in authors:
            ris_template += f"AU  - {author}\n"
        ris_template += f"TI  - {title}\n"
        ris_template += f"JO  - {journal}\n"
        ris_template += f"PY  - {year}\n"
        ris_template += f"DO  - {doi}\n"
        ris_template += "ER  -"
        return ris_template
    
    else:
        return "Unsupported citation format"

import urllib.parse
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any

class PaperMetadata:
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/v1/paper/arXiv:"
        # New: Adding S2 API v2 endpoint which sometimes has more complete metadata
        self.s2_api_v2_base = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"

    def get_paper_metadata(self, pdf_url: str) -> Dict[str, Any]:
        """Extract metadata from arXiv and both versions of Semantic Scholar API"""
        try:
            # Extract arxiv ID
            if 'arxiv.org/pdf/' in pdf_url:
                arxiv_id = pdf_url.split('arxiv.org/pdf/')[1].replace('.pdf', '')
            elif 'arxiv.org/abs/' in pdf_url:
                arxiv_id = pdf_url.split('arxiv.org/abs/')[1]
            else:
                return {"error": "Invalid arXiv URL"}

            # Remove version number for Semantic Scholar
            base_arxiv_id = arxiv_id.split('v')[0]

            # Get metadata from all sources
            arxiv_data = self._fetch_arxiv_metadata(arxiv_id)
            semantic_data = self._fetch_semantic_scholar_metadata(base_arxiv_id)
            s2_v2_data = self._fetch_s2_v2_metadata(base_arxiv_id)
            
            # Combine metadata with priority
            return self._combine_metadata(arxiv_data, semantic_data, s2_v2_data)

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    def _fetch_arxiv_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch metadata from arXiv API"""
        api_url = f"{self.arxiv_base_url}?id_list={arxiv_id}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entry = root.find('atom:entry', namespace)
        if entry is None:
            return {}
            
        published = entry.find('atom:published', namespace).text
        pub_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
        
        journal_ref = entry.find('arxiv:journal_ref', namespace)
        doi = entry.find('arxiv:doi', namespace)
        
        return {
            "arxiv_id": arxiv_id,
            "title": entry.find('atom:title', namespace).text.strip(),
            "authors": [author.find('atom:name', namespace).text 
                       for author in entry.findall('atom:author', namespace)],
            "abstract": entry.find('atom:summary', namespace).text.strip(),
            "publication_date": {
                "full": pub_date.isoformat(),
                "year": pub_date.year,
                "month": pub_date.month,
                "formatted": pub_date.strftime('%B %Y')
            },
            "categories": [category.attrib['term'] 
                          for category in entry.findall('atom:category', namespace)],
            "primary_category": entry.find('arxiv:primary_category', namespace).attrib['term'],
            "journal_ref": journal_ref.text if journal_ref is not None else None,
            "doi": doi.text if doi is not None else None
        }

    def _fetch_semantic_scholar_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch metadata from Semantic Scholar API v1"""
        try:
            response = requests.get(f"{self.semantic_scholar_base_url}{arxiv_id}")
            if response.status_code == 200:
                data = response.json()
                return {
                    "doi": data.get("doi"),
                    "journal": data.get("venue"),
                    "volume": data.get("volume"),
                    "year": data.get("year")
                }
            return {}
        except:
            return {}

    def _fetch_s2_v2_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch metadata from Semantic Scholar API v2"""
        try:
            fields = "venue,year,volume,publicationVenue,publicationTypes,doi"
            response = requests.get(
                f"{self.s2_api_v2_base}{arxiv_id}?fields={fields}"
            )
            if response.status_code == 200:
                data = response.json()
                venue_info = data.get("publicationVenue", {})
                return {
                    "doi": data.get("doi"),
                    "journal": venue_info.get("name") or data.get("venue"),
                    "volume": venue_info.get("volume") or data.get("volume"),
                    "year": data.get("year"),
                    "publication_types": data.get("publicationTypes", [])
                }
            return {}
        except:
            return {}

    def _combine_metadata(self, arxiv: Dict, semantic: Dict, s2_v2: Dict) -> Dict:
        """Combine metadata from all sources"""
        combined = arxiv.copy()
        
        # Update with Semantic Scholar v1 data
        if semantic:
            if semantic.get("doi"):
                combined["doi"] = semantic["doi"]
            if semantic.get("journal"):
                combined["journal"] = semantic["journal"]
            if semantic.get("volume"):
                combined["volume"] = semantic["volume"]
        
        # Update with Semantic Scholar v2 data (highest priority)
        if s2_v2:
            if s2_v2.get("doi"):
                combined["doi"] = s2_v2["doi"]
            if s2_v2.get("journal"):
                combined["journal"] = s2_v2["journal"]
            if s2_v2.get("volume"):
                combined["volume"] = s2_v2["volume"]
            if s2_v2.get("publication_types"):
                combined["publication_types"] = s2_v2["publication_types"]
        
        # If we still don't have a journal name but have journal_ref, try to parse it
        if not combined.get("journal") and combined.get("journal_ref"):
            combined["journal"] = combined["journal_ref"].split(",")[0]
            
        # Special handling for known conferences/journals
        if combined.get("journal") == "Neural Information Processing Systems":
            combined["journal"] = "NeurIPS"  # More commonly used name
            if not combined.get("doi") and combined.get("publication_date"):
                # Try to construct DOI for NeurIPS papers
                year = combined["publication_date"]["year"]
                combined["doi"] = f"10.5555/nips.{year}"  # Note: This is a placeholder DOI pattern
        
        return combined

def extract_references(text: str) -> List[str]:
    """
    Extract references from text with improved pattern matching.
    Returns list of individual references.
    """
    # Common variations of reference section headers
    reference_headers = [
        "References", "REFERENCES",
        "Bibliography", "BIBLIOGRAPHY",
        "Works Cited", "WORKS CITED"
    ]

    # Find where references section starts
    start_idx = -1
    for header in reference_headers:
        if header in text:
            start_idx = text.find(header)
            break

    if start_idx == -1:
        return []

    references_text = text[start_idx:].strip()

    # Enhanced pattern to match different reference formats
    reference_patterns = [
        # [1] Author et al. Title. Journal, Year.
        r'\[\d+\][^[]+?(?=\[\d+\]|$)',

        # 1. Author et al. Title. Journal, Year.
        r'\d+\.\s+[^.]+?(?=\d+\.\s+|$)',

        # [Author1, Author2] Title. Journal, Year.
        r'\[[^\]]+\]\s+[^[]+?(?=\[|$)',

        # Author et al. (Year) Title. Journal.
        r'[A-Z][^.]+?\(\d{4}\)[^.]+?(?=\s+[A-Z]|$)'
    ]

    references = []
    for pattern in reference_patterns:
        matches = re.findall(pattern, references_text)
        if matches:
            references.extend(matches)
            break  # Use the first successful pattern

    # Clean up references
    cleaned_refs = []
    for ref in references:
        # Remove extra whitespace and line breaks
        cleaned = ' '.join(ref.split())
        # Remove standalone numbers that might appear at the start of lines
        cleaned = re.sub(r'^\d+\s+', '', cleaned)
        if cleaned:
            cleaned_refs.append(cleaned)

    return cleaned_refs

def extract_citations(text: str) -> List[int]:
    """
    Extract citation numbers from text.
    Returns a list of integers representing citation numbers, including duplicates.
    """
    citations = []

    # Pattern for different citation formats
    patterns = [
        r'\[\s*(\d+)\s*\]',                    # [1]
        r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]',     # [1,2,3]
        r'\[\s*(\d+)\s*-\s*(\d+)\s*\]'         # [1-3]
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if '-' in match.group(0):  # Handle range format
                start, end = map(int, re.findall(r'\d+', match.group(0)))
                citations.extend(range(start, end + 1))
            else:  # Handle single number or comma-separated numbers
                numbers = re.findall(r'\d+', match.group(0))
                citations.extend(map(int, numbers))

    return list(dict.fromkeys(citations))


def is_reference_continuation(text: str, prev_refs: List[str]) -> bool:
    """
    Enhanced check if the current page is a continuation of references.
    """
    # Remove common header/footer text that might appear
    text = re.sub(r'^.*?(?:page|p\.)\s+\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Check if page starts with a reference-like pattern
    ref_start_patterns = [
        r'^\s*\[\d+\]',  # [1] format
        r'^\s*\d+\.',    # 1. format
        r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:et\s+al\.)?,',  # Author name pattern
        r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\(',  # Author name with year
        r'^\s*\[',       # [Author] format
        r'.*?(?:19|20)\d{2}.*?(?:journal|conference|proceedings|arxiv)',  # Contains year and venue
    ]

    # If we have previous references, check if the last one was incomplete
    if prev_refs:
        last_ref = prev_refs[-1].strip()
        if not last_ref.endswith('.'):
            return True
        if re.search(r'\([12]\d{3}[a-z]?\)$', last_ref):  # Ends with year
            return True

    # Check if page starts with a reference pattern
    text_start = text.strip()[:200]  # Check first 200 chars
    for pattern in ref_start_patterns:
        if re.search(pattern, text_start, re.MULTILINE):
            return True

    # Check for continuation indicators
    continuation_indicators = [
        r'^\s*(?:and|&)',  # Line starts with connecting words
        r'^\s*[a-z]',      # Continues with lowercase (middle of sentence)
        r'^\s*(?:pp\.|vol\.|pages|chapter)',  # Publication details
    ]

    for pattern in continuation_indicators:
        if re.search(pattern, text_start, re.MULTILINE):
            return True

    return False

def process_references_in_content(documents):
    """
    Process references and citations across all documents with improved continuation handling.
    Once references start on a page, all subsequent pages are treated as potential reference pages.
    """
    references_found = False
    accumulated_references = []
    reference_pages = set()
    reference_start_page = -1

    # First pass: identify where references section starts
    for i, doc in enumerate(documents):
        page_content = doc.page_content

        # Extract citations for all pages
        citations = extract_citations(page_content)
        if citations:
            doc.metadata["citations"] = citations

        # Check for reference section start
        if not references_found:
            refs = extract_references(page_content)
            if refs:
                references_found = True
                reference_start_page = i
                accumulated_references.extend(refs)
                reference_pages.add(i)
                doc.metadata["references"] = refs
                doc.metadata["is_reference_page"] = True

    # If we found references, process all subsequent pages
    if reference_start_page != -1:
        for i in range(reference_start_page + 1, len(documents)):
            page_content = documents[i].page_content

            # Try to extract references from the page content
            new_refs = []

            # First try with standard reference extraction
            refs = extract_references(page_content)
            if refs:
                new_refs.extend(refs)

            # If no references found with standard extraction, try parsing the raw content
            if not new_refs:
                # Split content by possible reference delimiters
                lines = re.split(r'(?:\[\d+\]|\b\d+\.\s+)', page_content)
                lines = [line.strip() for line in lines if line.strip()]

                # Process each line as a potential reference
                for line in lines:
                    # Basic validation: check if line looks like a reference
                    if (re.search(r'\(\d{4}\)', line) or  # Has a year in parentheses
                        re.search(r'\b(?:19|20)\d{2}\b', line) or  # Has a year
                        re.search(r'(?:journal|proceedings|conference|arxiv)', line.lower())):  # Has publication venue
                        new_refs.append(line)

            if new_refs:
                accumulated_references.extend(new_refs)
                reference_pages.add(i)
                documents[i].metadata["references"] = new_refs
                documents[i].metadata["is_reference_page"] = True
            else:
                # Even if no new references found, mark as reference page if it contains
                # text that looks like a continuation
                if is_reference_continuation(page_content, accumulated_references):
                    reference_pages.add(i)
                    documents[i].metadata["is_reference_page"] = True
                    documents[i].metadata["references"] = []

    # Final pass: update all reference pages with complete reference list
    complete_references = list(dict.fromkeys(accumulated_references))  # Remove duplicates
    for i in reference_pages:
        documents[i].metadata["all_references"] = complete_references
        documents[i].metadata["total_references"] = len(complete_references)

    return documents

def load_pdfs(pdf_path: str):
    """
    Load and process a single PDF file with enhanced reference and citation extraction.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Process references and citations
        documents = process_references_in_content(documents)

        return documents

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return []

# -----------------------------------------------------------------------------
# RAG System Setup: Embedding Models, Document Loader, Text Splitter, QA System
# -----------------------------------------------------------------------------

# Initialize Hugging Face API for embeddings and LLMs
os.environ["HUGGINGFACE_API_TOKEN"] = "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"

# Load the embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define text splitter for chunking documents with section-based heuristics
class SectionBasedTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
        # Call the parent class without the 'separator' argument
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> List[str]:
        # Split text based on section headings to keep related content together
        sections = re.split(r'\n\s*(?:Introduction|Methods|Methodology|Results|Discussion|Conclusion|References)\s*\n', text, flags=re.IGNORECASE)
        chunks = []
        for section in sections:
            # Further split the section into chunks using the base class logic
            section_chunks = super().split_text(section)
            chunks.extend(section_chunks)
        return chunks

# Initialize the custom text splitter without 'separator' argument
text_splitter = SectionBasedTextSplitter(
    chunk_size=1500,  # Increased chunk size
    chunk_overlap=300  # Increased overlap
)

quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
)

# Initialize the Llama model from Hugging Face
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HUGGINGFACE_API_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config_4bit, torch_dtype=torch.float16, device_map="auto", token=os.environ["HUGGINGFACE_API_TOKEN"])
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create a LangChain wrapper for the pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create FAISS vector store and Retrieval-based QA system
def setup_qa_chain(documents):
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embed_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    return qa_chain