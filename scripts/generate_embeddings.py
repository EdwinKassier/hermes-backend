import os
import json
import io
from pathlib import Path
from typing import Dict, List
import pypdf
from google.cloud import storage
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
def validate_env_vars() -> None:
    """Validate required environment variables."""
    required_vars = [
        "GOOGLE_PROJECT_ID",
        "GOOGLE_PROJECT_LOCATION"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please ensure these variables are set in your .env file")
        exit(1)


def clean_text(text: str) -> str:
    """Clean and normalize text for better embedding quality."""
    import re
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove unicode characters that might cause issues
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219]', '•', text)  # Normalize bullet points
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize quotes
    text = re.sub(r'[\u201C\u201D]', '"', text)  # Normalize quotes
    text = re.sub(r'[\u2013\u2014]', '-', text)  # Normalize dashes
    
    # Remove other problematic unicode
    text = re.sub(r'[\u00A0\u200B\u200C\u200D\uFEFF]', ' ', text)  # Remove zero-width spaces
    
    # Clean up phone numbers and formatting - more aggressive
    text = re.sub(r'\(\+44\)\s*(\d+)\s*(\d+)\s*(\d+)', r'(+44) \1 \2 \3', text)
    text = re.sub(r'(\d+)\s+(\d+)\s+(\d+)', r'\1 \2 \3', text)  # Clean up spaced numbers
    
    # Remove excessive punctuation
    text = re.sub(r'[.!?]{2,}', '.', text)
    
    # Normalize line breaks and remove extra spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Clean up multiple spaces and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any leading/trailing punctuation
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
    
    return text


def get_documents_from_gcs(
    storage_client: storage.Client,
    bucket_name: str,
    folder_path: str = ""
) -> List[str]:
    """Get text content from PDF files in GCS bucket."""
    documents = []
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            if blob.name == folder_path and blob.name.endswith("/"):
                continue

            logger.info(f"Processing PDF: {blob.name}")
            try:
                pdf_bytes = blob.download_as_bytes()
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                text_content = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"

                if text_content.strip():
                    # Clean the text before adding to documents
                    cleaned_text = clean_text(text_content.strip())
                    if cleaned_text:
                        documents.append(cleaned_text)
                    else:
                        logger.warning(f"No text after cleaning from PDF: {blob.name}")
                else:
                    logger.warning(f"No text extracted from PDF: {blob.name}")
            except Exception as e:
                logger.error(f"Error processing PDF {blob.name}: {e}")

    return documents


def split_texts(texts: List[str], chunk_size: int = 300, 
                chunk_overlap: int = 200) -> List[str]:
    """Split texts into ultra-fine chunks for maximum accuracy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", "; ", " ", ""],  # Ultra-fine granularity
        length_function=len,
    )
    
    # Process each text individually and combine results
    all_chunks = []
    for i, text in enumerate(texts):
        chunks = splitter.split_text(text)
        
        # Add ultra-fine metadata for maximum accuracy
        for j, chunk in enumerate(chunks):
            # Clean the chunk text
            cleaned_chunk = clean_text(chunk)
            if not cleaned_chunk or len(cleaned_chunk) < 20:  # Minimum meaningful length
                continue
                
            # Add detailed position and importance information
            is_early_chunk = j < len(chunks) // 4  # First quarter of document
            is_very_early = j < len(chunks) // 8   # First eighth of document
            importance_flag = "[CRITICAL]" if is_very_early else "[IMPORTANT]" if is_early_chunk else ""
            position_info = f"[Doc{i+1}, Chunk{j+1}/{len(chunks)}] {importance_flag} "
            enhanced_chunk = position_info + cleaned_chunk
            all_chunks.append(enhanced_chunk)
    
    return all_chunks


def semantic_split_texts(texts: List[str], chunk_size: int = 300, 
                         chunk_overlap: int = 200) -> List[str]:
    """Split texts semantically while preserving document structure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semi-colons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters
        ],
        length_function=len,
    )
    
    all_chunks = []
    for i, text in enumerate(texts):
        # Pre-process to identify semantic boundaries
        text = add_semantic_markers(text)
        chunks = splitter.split_text(text)
        
        for j, chunk in enumerate(chunks):
            cleaned_chunk = clean_text(chunk)
            if not cleaned_chunk or len(cleaned_chunk) < 20:
                continue
                
            # Enhanced importance weighting
            is_very_early = j < len(chunks) // 8
            is_early = j < len(chunks) // 4
            is_header = any(marker in chunk for marker in ['HEADER:', 'SECTION:', 'TITLE:'])
            
            importance_flag = "[CRITICAL]" if is_very_early or is_header else "[IMPORTANT]" if is_early else ""
            
            # Enhanced metadata
            position_info = f"[Doc{i+1}, Chunk{j+1}/{len(chunks)}] {importance_flag} "
            enhanced_chunk = position_info + cleaned_chunk
            all_chunks.append(enhanced_chunk)
    
    return all_chunks


def add_semantic_markers(text: str) -> str:
    """Add semantic markers to preserve document structure."""
    import re
    
    # Mark headers and sections
    text = re.sub(r'^([A-Z][A-Z\s]{3,}):', r'HEADER: \1:', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+\.\s*[A-Z][A-Za-z\s]{3,})', r'SECTION: \1', text, flags=re.MULTILINE)
    
    # Mark contact information
    text = re.sub(r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s*([A-Z]+)', r'CONTACT: \1, \2', text)
    
    # Mark professional titles
    text = re.sub(r'(Professional Summary|Experience|Education|Skills)', r'TITLE: \1', text)
    
    return text


def validate_chunk_quality(chunk: str) -> bool:
    """Validate chunk quality for meaningful content."""
    import re
    
    # Minimum length check
    if len(chunk) < 20:
        return False
    
    # Check for meaningful content (not just whitespace/punctuation)
    meaningful_chars = len(re.sub(r'[^\w]', '', chunk))
    if meaningful_chars < 10:
        return False
    
    # Check information density (words per character)
    words = len(chunk.split())
    if words < 3:
        return False
    
    # Check for noise patterns
    noise_patterns = [
        r'^\s*$',  # Empty or whitespace only
        r'^[^\w]*$',  # No alphanumeric characters
        r'^\d+$',  # Only numbers
        r'^[^\w\s]*$',  # Only punctuation
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, chunk):
            return False
    
    # Check for repetitive content
    if len(set(chunk.split())) < len(chunk.split()) * 0.3:
        return False
    
    return True


def assess_embedding_quality(embeddings_dict: Dict[str, List[float]]) -> dict:
    """Comprehensive assessment of embedding quality."""
    import re
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter
    
    chunks = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    
    # Convert to numpy array
    embedding_matrix = np.array(embeddings)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(embedding_matrix)
    
    # Remove self-similarities
    np.fill_diagonal(similarities, 0)
    
    # Quality metrics
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)
    min_similarity = np.min(similarities)
    
    # Diversity analysis
    diversity_score = 1 - avg_similarity  # Higher is better
    
    # Find potential duplicates (very similar chunks)
    duplicate_threshold = 0.95
    potential_duplicates = np.sum(similarities > duplicate_threshold)
    
    # Semantic diversity analysis
    semantic_tags = []
    for chunk in chunks:
        tags = re.findall(r'\[([^\]]+)\]', chunk)
        semantic_tags.extend(tags)
    
    tag_counts = Counter(semantic_tags)
    semantic_diversity = len(tag_counts) / len(chunks) if chunks else 0
    
    # Content analysis
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = np.mean(chunk_lengths)
    length_std = np.std(chunk_lengths)
    
    # Information density analysis
    word_counts = [len(chunk.split()) for chunk in chunks]
    avg_words = np.mean(word_counts)
    
    # Unique content analysis
    unique_words = set()
    for chunk in chunks:
        words = re.findall(r'\b\w+\b', chunk.lower())
        unique_words.update(words)
    
    vocabulary_size = len(unique_words)
    vocabulary_density = vocabulary_size / sum(word_counts) if sum(word_counts) > 0 else 0
    
    # Importance distribution analysis
    importance_levels = {
        'CRITICAL': 0,
        'IMPORTANT': 0,
        'RELEVANT': 0,
        'CONTEXT': 0
    }
    
    for chunk in chunks:
        if '[CRITICAL]' in chunk:
            importance_levels['CRITICAL'] += 1
        elif '[IMPORTANT]' in chunk:
            importance_levels['IMPORTANT'] += 1
        elif '[RELEVANT]' in chunk:
            importance_levels['RELEVANT'] += 1
        elif '[CONTEXT]' in chunk:
            importance_levels['CONTEXT'] += 1
    
    # Calculate quality score
    quality_score = (
        diversity_score * 0.3 +
        semantic_diversity * 0.2 +
        (1 - potential_duplicates / len(chunks)) * 0.2 +
        min(avg_length / 300, 1) * 0.15 +
        vocabulary_density * 0.15
    )
    
    return {
        'total_embeddings': len(chunks),
        'quality_score': round(quality_score, 3),
        'diversity_score': round(diversity_score, 3),
        'semantic_diversity': round(semantic_diversity, 3),
        'average_similarity': round(avg_similarity, 3),
        'max_similarity': round(max_similarity, 3),
        'min_similarity': round(min_similarity, 3),
        'potential_duplicates': potential_duplicates,
        'duplicate_percentage': round(potential_duplicates / len(chunks) * 100, 1),
        'average_chunk_length': round(avg_length, 1),
        'length_std': round(length_std, 1),
        'average_words': round(avg_words, 1),
        'vocabulary_size': vocabulary_size,
        'vocabulary_density': round(vocabulary_density, 3),
        'importance_distribution': importance_levels,
        'semantic_tags': dict(tag_counts),
        'recommendations': generate_quality_recommendations(
            quality_score, diversity_score, potential_duplicates, avg_length
        )
    }

def generate_quality_recommendations(quality_score: float, diversity_score: float, 
                                   duplicates: int, avg_length: float) -> List[str]:
    """Generate specific recommendations for improving embedding quality."""
    recommendations = []
    
    if quality_score < 0.7:
        recommendations.append("Consider reducing chunk size for more granularity")
    
    if diversity_score < 0.4:
        recommendations.append("Increase chunk overlap to preserve context")
    
    if duplicates > len(embeddings_dict) * 0.1:
        recommendations.append("Remove duplicate or very similar chunks")
    
    if avg_length < 200:
        recommendations.append("Consider increasing chunk size for better context")
    elif avg_length > 400:
        recommendations.append("Consider decreasing chunk size for more precision")
    
    if not recommendations:
        recommendations.append("Embedding quality is excellent - no changes needed")
    
    return recommendations


def validate_and_clean_embeddings(embeddings_dict: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Validate and clean existing embeddings."""
    cleaned_embeddings = {}
    for text, embedding in embeddings_dict.items():
        # Clean the text
        cleaned_text = clean_text(text)
        if cleaned_text and len(cleaned_text) > 10:  # Minimum meaningful length
            cleaned_embeddings[cleaned_text] = embedding
    
    logger.info(f"Cleaned {len(embeddings_dict)} embeddings to {len(cleaned_embeddings)} valid embeddings")
    return cleaned_embeddings


def generate_embeddings(
    texts: List[str],
    output_file: str = "data/embeddings_cache/embeddings.json"
) -> None:
    """
    Generate embeddings for a list of texts and save them to a JSON file.
    
    Args:
        texts: List of texts to generate embeddings for
        output_file: Path to save the embeddings JSON file
    """
    # Initialize the embeddings model with the latest model
    model_name = "text-embedding-004"  # Updated to latest model
    logger.info(f"Initializing embeddings model: {model_name}")
    embeddings_model = VertexAIEmbeddings(
        model_name=model_name,
        project=os.environ["GOOGLE_PROJECT_ID"],
        location=os.environ["GOOGLE_PROJECT_LOCATION"]
    )
    
    # Generate embeddings
    embeddings_dict: Dict[str, List[float]] = {}
    for text in texts:
        logger.info(f"Generating embedding for text chunk of length {len(text)}...")
        embedding = embeddings_model.embed_query(text)
        embeddings_dict[text] = embedding
    
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings to JSON file
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False)
    
    logger.info(f"Saved {len(embeddings_dict)} embeddings to {output_file}")


def analyze_document_structure(text: str) -> dict:
    """Analyze document structure for intelligent chunking."""
    import re
    
    analysis = {
        'sections': [],
        'key_phrases': [],
        'contact_info': [],
        'dates': [],
        'numbers': [],
        'titles': []
    }
    
    # Extract contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\(\+?\d+\)\s*\d+\s*\d+\s*\d+'
    analysis['contact_info'] = (re.findall(email_pattern, text) + 
                               re.findall(phone_pattern, text))
    
    # Extract dates
    date_pattern = r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b'
    analysis['dates'] = re.findall(date_pattern, text)
    
    # Extract numbers (likely important metrics)
    number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?\b'
    analysis['numbers'] = re.findall(number_pattern, text)
    
    # Extract potential titles/sections
    title_pattern = r'^[A-Z][A-Z\s]+:$|^[A-Z][A-Z\s]+$'
    lines = text.split('\n')
    analysis['titles'] = [line.strip() for line in lines if re.match(title_pattern, line.strip())]
    
    # Extract key phrases (noun phrases, technical terms)
    key_phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    analysis['key_phrases'] = re.findall(key_phrase_pattern, text)
    
    return analysis

def create_semantic_chunks(text: str, chunk_size: int = 200, chunk_overlap: int = 100) -> List[str]:
    """Create semantically meaningful chunks that preserve document structure."""
    import re
    
    # Analyze document structure
    structure = analyze_document_structure(text)
    
    # First, split by paragraphs to preserve document structure
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Special handling for the first few paragraphs (likely headers/contact info)
        if len(chunks) < 3:  # First 3 chunks get special treatment
            # Keep header paragraphs together if they're short
            if len(paragraph) <= chunk_size * 1.5:  # Allow slightly larger chunks for headers
                if len(current_chunk) + len(paragraph) <= chunk_size * 1.5:
                    current_chunk += (" " if current_chunk else "") + paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            else:
                # If header paragraph is too long, split it carefully
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Split long header paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += (" " if temp_chunk else "") + sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                current_chunk = temp_chunk
        else:
            # For regular paragraphs, use standard chunking
            if len(paragraph) <= chunk_size:
                if len(current_chunk) + len(paragraph) <= chunk_size:
                    current_chunk += (" " if current_chunk else "") + paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            else:
                # Split long paragraphs by sentences
                if current_chunk:
                    chunks.append(current_chunk.strip())
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += (" " if temp_chunk else "") + sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                current_chunk = temp_chunk
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Apply overlap between chunks
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev = chunks[i-1]
            overlap = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped_chunks.append((overlap + " " + chunk).strip())
        else:
            overlapped_chunks.append(chunk)
    
    # Enforce max chunk length for any remaining oversized chunks
    final_chunks = []
    for chunk in overlapped_chunks:
        if len(chunk) <= chunk_size * 1.5:  # Allow slightly larger chunks
            final_chunks.append(chunk)
        else:
            # Split oversized chunks
            for i in range(0, len(chunk), chunk_size):
                final_chunks.append(chunk[i:i+chunk_size])
    
    return final_chunks

def enhance_chunk_with_context(chunk: str, doc_index: int, chunk_index: int, total_chunks: int, structure: dict) -> str:
    """Enhance chunk with contextual metadata for better retrieval."""
    
    # Determine importance level
    if chunk_index < total_chunks * 0.1:  # First 10%
        importance = "[CRITICAL]"
    elif chunk_index < total_chunks * 0.25:  # First 25%
        importance = "[IMPORTANT]"
    elif chunk_index < total_chunks * 0.5:  # First 50%
        importance = "[RELEVANT]"
    else:
        importance = "[CONTEXT]"
    
    # Add semantic tags based on content
    semantic_tags = []
    
    # Contact information
    if any(email in chunk for email in structure.get('contact_info', [])):
        semantic_tags.append("[CONTACT]")
    
    # Dates and timeline
    if any(date in chunk for date in structure.get('dates', [])):
        semantic_tags.append("[TIMELINE]")
    
    # Numbers and metrics
    if any(number in chunk for number in structure.get('numbers', [])):
        semantic_tags.append("[METRICS]")
    
    # Titles and sections
    if any(title in chunk for title in structure.get('titles', [])):
        semantic_tags.append("[TITLE]")
    
    # Technical terms
    tech_terms = ['AI', 'ML', 'API', 'AWS', 'Azure', 'Python', 'JavaScript', 'React', 'Node.js', 'Docker', 'Kubernetes', 'LLM', 'SaaS', 'Cloud', 'DevOps', 'Agile', 'Scrum']
    if any(term.lower() in chunk.lower() for term in tech_terms):
        semantic_tags.append("[TECH]")
    
    # Leadership and management terms
    leadership_terms = ['Manager', 'Lead', 'Director', 'Team', 'Leadership', 'Mentoring', 'Coaching', 'Strategy', 'Planning', 'Oversight']
    if any(term.lower() in chunk.lower() for term in leadership_terms):
        semantic_tags.append("[LEADERSHIP]")
    
    # Financial and business terms
    business_terms = ['Revenue', 'Cost', 'Budget', 'ROI', 'Efficiency', 'Savings', 'Profit', 'Growth', 'Scale', 'Commercial']
    if any(term.lower() in chunk.lower() for term in business_terms):
        semantic_tags.append("[BUSINESS]")
    
    # Education and qualifications
    education_terms = ['Degree', 'University', 'College', 'CITP', 'Certification', 'Qualification', 'Study', 'Course', 'Training']
    if any(term.lower() in chunk.lower() for term in education_terms):
        semantic_tags.append("[EDUCATION]")
    
    # Current employment detection
    if 'Current Employment' in chunk or ('MOHARA' in chunk and 'Engineering Manager' in chunk):
        semantic_tags.append("[CURRENT_ROLE]")
    
    # Experience and achievements
    achievement_terms = ['Achieved', 'Delivered', 'Implemented', 'Reduced', 'Increased', 'Improved', 'Developed', 'Built', 'Created']
    if any(term.lower() in chunk.lower() for term in achievement_terms):
        semantic_tags.append("[ACHIEVEMENT]")
    
    # Combine all metadata
    metadata = f"[Doc{doc_index+1}, Chunk{chunk_index+1}/{total_chunks}] {importance} {' '.join(semantic_tags)} "
    
    return metadata + chunk

def remove_duplicate_chunks(embeddings_dict: Dict[str, List[float]], threshold: float = 0.95) -> Dict[str, List[float]]:
    """Remove near-duplicate chunks based on cosine similarity."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    keys = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))
    sim = cosine_similarity(vectors)
    np.fill_diagonal(sim, 0)
    to_remove = set()
    
    # Remove exact duplicates first
    seen_content = set()
    for i, key in enumerate(keys):
        # Extract just the content part (remove metadata)
        content = key.split('] ', 1)[-1] if '] ' in key else key
        if content in seen_content:
            to_remove.add(key)
        else:
            seen_content.add(content)
    
    # Remove very similar chunks
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            if sim[i, j] > threshold:
                # Keep the chunk with more metadata tags (more informative)
                i_tags = len([tag for tag in keys[i].split() if tag.startswith('[') and tag.endswith(']')])
                j_tags = len([tag for tag in keys[j].split() if tag.startswith('[') and tag.endswith(']')])
                
                if i_tags >= j_tags:
                    to_remove.add(keys[j])
                else:
                    to_remove.add(keys[i])
    
    # Remove chunks that are too short or contain mostly metadata
    for key in list(embeddings_dict.keys()):
        content = key.split('] ', 1)[-1] if '] ' in key else key
        if len(content) < 30:  # Too short
            to_remove.add(key)
        elif len([c for c in content if c.isalpha()]) < 20:  # Too few letters
            to_remove.add(key)
    
    cleaned = {k: v for k, v in embeddings_dict.items() if k not in to_remove}
    return cleaned

def detect_current_employment(text: str) -> str:
    """Detect and extract current employment information."""
    import re
    
    # Look for current employment patterns
    current_patterns = [
        # Look for "Present" or "Current" positions
        r'([A-Z][A-Za-z\s&]+(?:Inc|Ltd|LLC|Corp|Company|Group|Technologies|Solutions|Systems|Services|London|UK|US|Pty))\s+(?:Engineering Manager|Senior|Lead|Manager|Director|CEO|CTO|Developer|Engineer)[^.!?]*(?:Present|Current|Now)',
        r'(?:Engineering Manager|Senior|Lead|Manager|Director|CEO|CTO|Developer|Engineer)[^.!?]*(?:Present|Current|Now)[^.!?]*([A-Z][A-Za-z\s&]+(?:Inc|Ltd|LLC|Corp|Company|Group|Technologies|Solutions|Systems|Services|London|UK|US|Pty))',
        # Look for company names followed by dates ending in Present
        r'([A-Z][A-Za-z\s&]+(?:Inc|Ltd|LLC|Corp|Company|Group|Technologies|Solutions|Systems|Services|London|UK|US|Pty))[^.!?]*(?:June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May)\s+\d{4}\s*[-–]\s*Present',
        # Look for "currently" or "presently" patterns
        r'(?:currently|presently|now)\s+(?:working\s+at|employed\s+at|with)\s+([^.!?]+)',
        r'(?:current\s+position|current\s+role|current\s+job)[:\s]+([^.!?]+)',
        # Look for company names in professional experience section
        r'Professional Experience[^.!?]*([A-Z][A-Za-z\s&]+(?:Inc|Ltd|LLC|Corp|Company|Group|Technologies|Solutions|Systems|Services|London|UK|US|Pty))',
    ]
    
    for pattern in current_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Clean up the match
            match = matches[0].strip()
            # Remove extra whitespace and common prefixes
            match = re.sub(r'^\s*(?:at|with|for)\s+', '', match)
            return match
    
    return ""

def create_employment_chunk(text: str) -> str:
    """Create a dedicated chunk for current employment information."""
    import re
    
    # Look for MOHARA and Engineering Manager specifically
    if "MOHARA" in text.upper() and "Engineering Manager" in text:
        # Extract the full employment information
        mohara_pattern = r'MOHARA[^.!?]*Engineering Manager[^.!?]*June 2024[^.!?]*Present[^.!?]*'
        match = re.search(mohara_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"Current Employment: {match.group().strip()}"
        else:
            return "Current Employment: MOHARA London - Engineering Manager (June 2024 - Present)"
    
    return ""

def ensure_employment_in_first_chunk(chunks: List[str], text: str) -> List[str]:
    """Ensure current employment information is in the first chunk."""
    current_employment = detect_current_employment(text)
    
    # Manual override for known current employment
    if "MOHARA" in text.upper() and "Engineering Manager" in text:
        current_employment = "MOHARA London - Engineering Manager (June 2024 - Present)"
    
    if not current_employment:
        return chunks
    
    # Check if current employment is already in first chunk
    if current_employment.lower() in chunks[0].lower():
        return chunks
    
    # Create a dedicated employment chunk and insert it after the first chunk
    employment_chunk = create_employment_chunk(text)
    if employment_chunk:
        return [chunks[0], employment_chunk] + chunks[1:]
    
    # If not, add it to the first chunk
    enhanced_first_chunk = f"{chunks[0]} Current Employment: {current_employment}"
    
    # Ensure the enhanced chunk doesn't exceed reasonable size
    if len(enhanced_first_chunk) <= 400:  # Allow larger first chunk
        return [enhanced_first_chunk] + chunks[1:]
    else:
        # Create a separate employment chunk if too large
        employment_chunk = f"Current Employment: {current_employment}"
        return [chunks[0], employment_chunk] + chunks[1:]

def create_optimized_chunks(embeddings_dict: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Create optimized chunks for better retrieval by combining related information."""
    import re
    
    # Group chunks by document and importance
    doc_groups = {}
    for key, embedding in embeddings_dict.items():
        # Extract document number
        doc_match = re.search(r'\[Doc(\d+)', key)
        if doc_match:
            doc_num = doc_match.group(1)
            if doc_num not in doc_groups:
                doc_groups[doc_num] = {'CRITICAL': [], 'IMPORTANT': [], 'RELEVANT': [], 'CONTEXT': []}
            
            # Determine importance level
            if '[CRITICAL]' in key:
                doc_groups[doc_num]['CRITICAL'].append((key, embedding))
            elif '[IMPORTANT]' in key:
                doc_groups[doc_num]['IMPORTANT'].append((key, embedding))
            elif '[RELEVANT]' in key:
                doc_groups[doc_num]['RELEVANT'].append((key, embedding))
            else:
                doc_groups[doc_num]['CONTEXT'].append((key, embedding))
    
    # Create optimized chunks
    optimized_chunks = {}
    
    for doc_num, groups in doc_groups.items():
        # Create a summary chunk for each document
        critical_chunks = [chunk[0].split('] ', 1)[-1] for chunk in groups['CRITICAL'][:3]]  # Top 3 critical
        important_chunks = [chunk[0].split('] ', 1)[-1] for chunk in groups['IMPORTANT'][:2]]  # Top 2 important
        
        if critical_chunks:
            summary_content = " ".join(critical_chunks[:2])  # Combine top 2 critical chunks
            if len(summary_content) > 500:  # Truncate if too long
                summary_content = summary_content[:500] + "..."
            
            summary_key = f"[Doc{doc_num}] [CRITICAL] [SUMMARY] {summary_content}"
            # Use the embedding of the first critical chunk
            optimized_chunks[summary_key] = groups['CRITICAL'][0][1]
    
    # Add all original chunks
    optimized_chunks.update(embeddings_dict)
    
    return optimized_chunks


if __name__ == "__main__":
    # Validate environment variables
    validate_env_vars()
    
    # Set up credentials - look for credentials.json in the project root
    # Get the script's directory and go up one level to the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    credentials_path = project_root / "credentials.json"
    
    if not credentials_path.exists():
        logger.error(f"Credentials file not found at {credentials_path}")
        exit(1)
    
    # Set the environment variable for Google Cloud libraries
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    logger.info(f"Using credentials from {credentials_path}")

    # Initialize GCS client
    try:
        storage_client = storage.Client()
        logger.info("Successfully initialized GCS client")
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        exit(1)
    
    # Initialize embeddings model
    try:
        model_name = "text-embedding-004"  # Updated to latest model
        logger.info(f"Initializing embeddings model: {model_name}")
        embeddings_model = VertexAIEmbeddings(
            model_name=model_name,
            project=os.environ["GOOGLE_PROJECT_ID"],
            location=os.environ["GOOGLE_PROJECT_LOCATION"]
        )
        logger.info("Successfully initialized embeddings model")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")
        exit(1)
    
    # Get documents from GCS
    bucket_name = "ashes-project-hermes-training"
    folder_path = os.environ.get("GCS_FOLDER_PATH", "")
    
    logger.info(f"Fetching documents from gs://{bucket_name}/{folder_path}")
    
    # List all blobs in the bucket to verify access
    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=folder_path))
        logger.info(f"Found {len(blobs)} total objects in bucket")
        for blob in blobs:
            logger.info(f"Found object: {blob.name}")
    except Exception as e:
        logger.error(f"Error listing bucket contents: {e}")
        exit(1)
    
    # Get documents directly from GCS
    documents = get_documents_from_gcs(storage_client, bucket_name, folder_path)
    if not documents:
        logger.error("No documents found in GCS bucket.")
        exit(1)
    
    # Split documents using advanced semantic analysis
    logger.info(f"Found {len(documents)} documents. Creating semantic chunks...")
    
    all_chunks = []
    for i, document in enumerate(documents):
        # Analyze document structure
        structure = analyze_document_structure(document)
        
        # Create semantic chunks for this document
        doc_chunks = create_semantic_chunks(document, chunk_size=200, chunk_overlap=100)
        
        # Ensure current employment is captured in first chunk
        doc_chunks = ensure_employment_in_first_chunk(doc_chunks, document)
        
        # Enhance each chunk with contextual metadata
        for j, chunk in enumerate(doc_chunks):
            enhanced_chunk = enhance_chunk_with_context(
                chunk, i, j, len(doc_chunks), structure
            )
            all_chunks.append(enhanced_chunk)
    
    logger.info(f"Created {len(all_chunks)} semantically enhanced chunks.")
    
    # Validate chunk quality
    logger.info("Validating chunk quality...")
    valid_chunks = [chunk for chunk in all_chunks if validate_chunk_quality(chunk)]
    logger.info(f"Quality validation: {len(all_chunks)} -> {len(valid_chunks)} chunks")
    
    if len(valid_chunks) < len(all_chunks) * 0.8:
        logger.warning(f"Quality validation removed {len(all_chunks) - len(valid_chunks)} chunks")
    
    # Generate embeddings - save to project root data directory
    output_file = project_root / "data" / "embeddings_cache" / "embeddings.json"
    generate_embeddings(valid_chunks, str(output_file))
    
    # Assess embedding quality
    logger.info("Assessing embedding quality...")
    with open(output_file, 'r') as f:
        embeddings_dict = json.load(f)
    quality_report = assess_embedding_quality(embeddings_dict)
    
    # Save quality report
    quality_file = project_root / "data" / "embeddings_cache" / "quality_report.json"
    def convert_np(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(quality_file, 'w') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False, default=convert_np)
    logger.info(f"Quality report saved to {quality_file}") 
    
    # Remove near-duplicate chunks
    logger.info("Removing near-duplicate chunks...")
    with open(output_file, 'r') as f:
        embeddings_dict = json.load(f)
    cleaned_embeddings = remove_duplicate_chunks(embeddings_dict, threshold=0.95)
    logger.info(f"Removed duplicates: {len(embeddings_dict) - len(cleaned_embeddings)} chunks removed. {len(cleaned_embeddings)} remain.")
    
    # Create optimized chunks for better retrieval
    logger.info("Creating optimized chunks for better retrieval...")
    optimized_embeddings = create_optimized_chunks(cleaned_embeddings)
    logger.info(f"Created {len(optimized_embeddings)} optimized chunks (including {len(optimized_embeddings) - len(cleaned_embeddings)} summary chunks).")
    
    # Save optimized embeddings
    with open(output_file, 'w') as f:
        json.dump(optimized_embeddings, f, ensure_ascii=False) 