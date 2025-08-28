import os
import re
import json
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

# Third-party imports
import numpy as np
from underthesea import word_tokenize, pos_tag, ner
from sentence_transformers import SentenceTransformer
import pandas as pd

from config import Config

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TextProcessingResult:
    """Result of text processing operations"""
    original_text: str
    cleaned_text: str
    keywords: List[str]
    entities: List[str]
    pos_tags: List[Tuple[str, str]]
    sentences: List[str]
    word_count: int
    processing_time: float

@dataclass
class FileInfo:
    """Information about a processed file"""
    file_path: str
    video_name: str
    section_name: str
    content: str
    size: int
    modified_time: float
    encoding: str

class VietnameseTextProcessor:
    """
    Advanced Vietnamese text processing utilities
    Xử lý văn bản tiếng Việt nâng cao
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Extended Vietnamese stop words
        self.stop_words = {
            # Basic stop words
            'là', 'của', 'và', 'có', 'trong', 'được', 'với', 'từ', 'theo',
            'để', 'về', 'này', 'đó', 'những', 'các', 'một', 'hai', 'ba',
            'rất', 'nhiều', 'ít', 'lớn', 'nhỏ', 'tốt', 'xấu', 'mới', 'cũ',
            'cho', 'khi', 'nếu', 'thì', 'sẽ', 'đã', 'bị', 'bởi', 'vì',
            'nên', 'mà', 'tại', 'trên', 'dưới', 'giữa', 'sau', 'trước',
            
            # Extended stop words
            'cùng', 'nhau', 'cũng', 'thêm', 'nữa', 'chỉ', 'phải', 'còn',
            'đang', 'đến', 'đi', 'lại', 'xuống', 'lên', 'ra', 'vào',
            'qua', 'khỏi', 'đây', 'kia', 'đấy', 'ấy', 'này', 'nọ',
            'gì', 'ai', 'đâu', 'nào', 'sao', 'thế', 'như', 'thế',
            'bao', 'mấy', 'bà', 'ông', 'cô', 'chú', 'anh', 'chị'
        }
        
        # Vietnamese diacritics mapping for normalization
        self.diacritics_map = {
            'àáạảãâầấậẩẫăằắặẳẵ': 'a',
            'èéẹẻẽêềếệểễ': 'e',
            'ìíịỉĩ': 'i',
            'òóọỏõôồốộổỗơờớợởỡ': 'o',
            'ùúụủũưừứựửữ': 'u',
            'ỳýỵỷỹ': 'y',
            'đ': 'd'
        }
        
        # Common Vietnamese patterns
        self.vietnamese_patterns = {
            'phone': r'(\+84|0)[1-9][0-9]{8,9}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'url': r'https?://[^\s]+',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'time': r'\d{1,2}:\d{2}(:\d{2})?',
            'money': r'\d+([.,]\d+)*\s?(VNĐ|đồng|USD|$)'
        }
        
        logger.info("VietnameseTextProcessor initialized")
    
    def clean_text(self, text: str, preserve_structure: bool = False) -> str:
        """
        Clean and normalize Vietnamese text
        Làm sạch và chuẩn hóa văn bản tiếng Việt
        
        Args:
            text: Input text
            preserve_structure: Keep line breaks and paragraph structure
        """
        if not text or not text.strip():
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+84|0)[1-9][0-9]{8,9}', ' ', text)
        
        # Normalize whitespace
        if preserve_structure:
            # Keep line breaks but normalize other whitespace
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        else:
            # Replace all whitespace with single space
            text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\sÀ-ỹ.,!?;:()\-"\']', ' ', text)
        
        # Normalize Vietnamese diacritics (optional)
        # Uncomment if you want to normalize all diacritics
        # text = self.normalize_vietnamese_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize Vietnamese diacritics
        Chuẩn hóa dấu tiếng Việt
        """
        for diacritics, base_char in self.diacritics_map.items():
            for char in diacritics:
                text = text.replace(char, base_char)
                text = text.replace(char.upper(), base_char.upper())
        return text
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Extract named entities from Vietnamese text
        Trích xuất thực thể có tên từ văn bản tiếng Việt
        """
        if not text.strip():
            return []
        
        entities = []
        
        try:
            # Use underthesea for NER
            ner_results = ner(text)
            
            current_entity = ""
            current_label = ""
            current_tokens = []
            
            for token, label in ner_results:
                if label.startswith('B-'):  # Beginning of entity
                    # Save previous entity if exists
                    if current_entity:
                        entities.append({
                            'text': current_entity.strip(),
                            'label': current_label,
                            'tokens': current_tokens,
                            'confidence': 1.0  # underthesea doesn't provide confidence
                        })
                    
                    # Start new entity
                    current_entity = token
                    current_label = label[2:]  # Remove 'B-' prefix
                    current_tokens = [token]
                    
                elif label.startswith('I-'):  # Inside entity
                    if current_label == label[2:]:  # Same entity type
                        current_entity += " " + token
                        current_tokens.append(token)
                    else:  # Different entity type, start new
                        if current_entity:
                            entities.append({
                                'text': current_entity.strip(),
                                'label': current_label,
                                'tokens': current_tokens,
                                'confidence': 1.0
                            })
                        current_entity = token
                        current_label = label[2:]
                        current_tokens = [token]
                else:  # 'O' - Outside entity
                    if current_entity:
                        entities.append({
                            'text': current_entity.strip(),
                            'label': current_label,
                            'tokens': current_tokens,
                            'confidence': 1.0
                        })
                        current_entity = ""
                        current_label = ""
                        current_tokens = []
            
            # Don't forget the last entity
            if current_entity:
                entities.append({
                    'text': current_entity.strip(),
                    'label': current_label,
                    'tokens': current_tokens,
                    'confidence': 1.0
                })
            
            # Filter by confidence and length
            filtered_entities = []
            for entity in entities:
                if (len(entity['text'].strip()) > 1 and 
                    entity['confidence'] >= confidence_threshold and
                    entity['text'].lower() not in self.stop_words):
                    
                    # Normalize entity text
                    entity['normalized_text'] = entity['text'].lower().strip()
                    filtered_entities.append(entity)
            
            # Remove duplicates
            unique_entities = []
            seen_texts = set()
            for entity in filtered_entities:
                if entity['normalized_text'] not in seen_texts:
                    unique_entities.append(entity)
                    seen_texts.add(entity['normalized_text'])
            
            return unique_entities[:self.config.MAX_ENTITIES]
        
        except Exception as e:
            logger.warning(f"Error in NER extraction: {e}")
            return []
    
    def extract_keywords(self, text: str, max_keywords: int = None) -> List[Dict[str, Any]]:
        """
        Extract keywords from Vietnamese text using POS tagging
        Trích xuất từ khóa từ văn bản tiếng Việt bằng POS tagging
        """
        if not text.strip():
            return []
        
        max_keywords = max_keywords or self.config.MAX_KEYWORDS
        
        try:
            # Tokenize and POS tag
            pos_results = pos_tag(text)
            
            keywords = []
            for word, pos in pos_results:
                # Extract nouns, proper nouns, and adjectives
                if (pos in ['N', 'Np', 'Ny', 'Nb', 'A', 'V'] and  
                    len(word) > 2 and  
                    word.lower() not in self.stop_words and
                    not word.isdigit()):
                    
                    # Calculate simple frequency-based score
                    word_count = text.lower().count(word.lower())
                    score = word_count / len(text.split()) * 1000  # Normalize
                    
                    keywords.append({
                        'word': word.lower(),
                        'pos': pos,
                        'frequency': word_count,
                        'score': score,
                        'original': word
                    })
            
            # Remove duplicates and sort by score
            unique_keywords = {}
            for kw in keywords:
                word = kw['word']
                if word not in unique_keywords or kw['score'] > unique_keywords[word]['score']:
                    unique_keywords[word] = kw
            
            # Sort by score and return top keywords
            sorted_keywords = sorted(
                unique_keywords.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            return sorted_keywords[:max_keywords]
        
        except Exception as e:
            logger.warning(f"Error in keyword extraction: {e}")
            return []
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences
        Phân đoạn văn bản thành câu
        """
        # Simple sentence segmentation for Vietnamese
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def process_text(self, text: str, extract_entities: bool = True, 
                    extract_keywords: bool = True) -> TextProcessingResult:
        """
        Comprehensive text processing
        Xử lý văn bản toàn diện
        """
        start_time = time.time()
        
        original_text = text
        cleaned_text = self.clean_text(text)
        
        # Extract entities
        entities = []
        if extract_entities:
            entity_objs = self.extract_entities(cleaned_text)
            entities = [e['normalized_text'] for e in entity_objs]
        
        # Extract keywords
        keywords = []
        if extract_keywords:
            keyword_objs = self.extract_keywords(cleaned_text)
            keywords = [k['word'] for k in keyword_objs]
        
        # POS tagging
        pos_tags = []
        try:
            pos_tags = pos_tag(cleaned_text)
        except Exception as e:
            logger.warning(f"Error in POS tagging: {e}")
        
        # Sentence segmentation
        sentences = self.segment_sentences(cleaned_text)
        
        # Word count
        word_count = len(cleaned_text.split())
        
        processing_time = time.time() - start_time
        
        return TextProcessingResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            keywords=keywords,
            entities=entities,
            pos_tags=pos_tags,
            sentences=sentences,
            word_count=word_count,
            processing_time=processing_time
        )


class FileReader:
    """
    File reading utilities for video sections
    Tiện ích đọc file cho các phần video
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.supported_encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        logger.info("FileReader initialized")
    
    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        for encoding in self.supported_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Test read
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'utf-8'  # Default fallback
    
    def read_video_files(self, base_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Read all video section files from directory structure
        Đọc tất cả các file section video từ cấu trúc thư mục
        
        Args:
            base_dir: Base directory containing video folders
            
        Returns:
            Dict with structure: {video_name: {section_name: content}}
        """
        videos_data = {}
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.error(f"Directory {base_dir} does not exist")
            return videos_data
        
        processed_files = 0
        error_files = 0
        
        # Iterate through video directories
        for video_dir in base_path.iterdir():
            if not video_dir.is_dir():
                continue
                
            video_name = video_dir.name
            videos_data[video_name] = {}
            
            # Read all .txt files in video directory
            for txt_file in video_dir.glob("*.txt"):
                try:
                    section_name = txt_file.stem  # filename without extension
                    
                    # Detect and use appropriate encoding
                    encoding = self.detect_encoding(txt_file)
                    
                    with open(txt_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # Validate content
                    if len(content.strip()) < 10:
                        logger.warning(f"File {txt_file} has very short content")
                        continue
                    
                    if len(content) > self.config.MAX_TEXT_LENGTH:
                        logger.warning(f"File {txt_file} exceeds max length, truncating")
                        content = content[:self.config.MAX_TEXT_LENGTH]
                    
                    videos_data[video_name][section_name] = content
                    processed_files += 1
                    logger.debug(f"Read {txt_file}")
                    
                except Exception as e:
                    logger.error(f"Error reading {txt_file}: {e}")
                    error_files += 1
        
        logger.info(f"Loaded {len(videos_data)} videos, "
                   f"{processed_files} files processed, "
                   f"{error_files} errors")
        
        return videos_data
    
    def get_file_info(self, file_path: Path) -> FileInfo:
        """Get detailed file information"""
        try:
            stat = file_path.stat()
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Extract video and section names from path
            section_name = file_path.stem
            video_name = file_path.parent.name
            
            return FileInfo(
                file_path=str(file_path),
                video_name=video_name,
                section_name=section_name,
                content=content,
                size=stat.st_size,
                modified_time=stat.st_mtime,
                encoding=encoding
            )
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    def get_file_stats(self, base_dir: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about video files
        Lấy thống kê toàn diện về các file video
        """
        stats = {
            'total_videos': 0,
            'total_sections': 0,
            'total_characters': 0,
            'total_words': 0,
            'average_section_length': 0,
            'videos_detail': {},
            'file_sizes': [],
            'encodings': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        videos_data = self.read_video_files(base_dir)
        
        for video_name, sections in videos_data.items():
            video_chars = sum(len(content) for content in sections.values())
            video_words = sum(len(content.split()) for content in sections.values())
            
            stats['videos_detail'][video_name] = {
                'sections_count': len(sections),
                'total_characters': video_chars,
                'total_words': video_words,
                'average_section_length': video_chars // len(sections) if sections else 0,
                'sections': {name: len(content) for name, content in sections.items()}
            }
            
            stats['total_sections'] += len(sections)
            stats['total_characters'] += video_chars
            stats['total_words'] += video_words
        
        stats['total_videos'] = len(videos_data)
        stats['average_section_length'] = (stats['total_characters'] // 
                                         stats['total_sections'] if stats['total_sections'] > 0 else 0)
        stats['processing_time'] = time.time() - start_time
        
        return stats


class EmbeddingGenerator:
    """
    Generate embeddings for Vietnamese text
    Tạo embedding cho văn bản tiếng Việt
    """
    
    def __init__(self, model_name: str = None, config: Config = None):
        self.config = config or Config()
        self.model_name = model_name or self.config.VIETNAMESE_EMBEDDING_MODEL
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
            
            # Get model dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Cache for embeddings
        self.cache = {}
        self.cache_file = Path(self.config.CACHE_DIR) / "embeddings_cache.pkl"
        
        if self.config.CACHE_EMBEDDINGS:
            self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Error loading embedding cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Error saving embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text
        Tạo embedding cho một văn bản
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        # Check cache
        text_hash = self._get_text_hash(text)
        if self.config.CACHE_EMBEDDINGS and text_hash in self.cache:
            return self.cache[text_hash]
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=normalize)
            embedding_list = embedding.tolist()
            
            # Cache the result
            if self.config.CACHE_EMBEDDINGS:
                self.cache[text_hash] = embedding_list
                
                # Periodic cache save
                if len(self.cache) % 100 == 0:
                    self._save_cache()
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def generate_batch_embeddings(self, texts: List[str], 
                                batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        Tạo embedding cho nhiều văn bản
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.BATCH_SIZE
        embeddings = []
        
        # Split into batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if not text or not text.strip():
                    cached_embeddings.append([0.0] * self.dimension)
                    continue
                
                text_hash = self._get_text_hash(text)
                if self.config.CACHE_EMBEDDINGS and text_hash in self.cache:
                    cached_embeddings.append(self.cache[text_hash])
                else:
                    cached_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    batch_embeddings = self.model.encode(
                        uncached_texts, 
                        normalize_embeddings=True,
                        batch_size=min(len(uncached_texts), batch_size)
                    )
                    
                    # Fill in the cached embeddings array
                    for idx, embedding in zip(uncached_indices, batch_embeddings):
                        embedding_list = embedding.tolist()
                        cached_embeddings[idx] = embedding_list
                        
                        # Cache the result
                        if self.config.CACHE_EMBEDDINGS:
                            text_hash = self._get_text_hash(batch_texts[idx])
                            self.cache[text_hash] = embedding_list
                
                except Exception as e:
                    logger.error(f"Error generating batch embeddings: {e}")
                    # Fill with zero embeddings
                    for idx in uncached_indices:
                        cached_embeddings[idx] = [0.0] * self.dimension
            
            embeddings.extend(cached_embeddings)
        
        # Save cache
        if self.config.CACHE_EMBEDDINGS:
            self._save_cache()
        
        return embeddings
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def __del__(self):
        """Save cache on destruction"""
        if hasattr(self, 'config') and self.config.CACHE_EMBEDDINGS:
            self._save_cache()


# Utility functions
def create_sample_data(base_dir: str = "output_files") -> str:
    """
    Create sample Vietnamese video data for testing
    Tạo dữ liệu video tiếng Việt mẫu để test
    """
    sample_dir = Path(base_dir) / "Video_01"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_texts = {
        "section01.txt": """
        Trong video này, chúng ta sẽ tìm hiểu về Machine Learning và ứng dụng của nó trong thế giới hiện đại.
        Machine Learning, hay học máy, là một phần quan trọng của trí tuệ nhân tạo (AI) cho phép máy tính 
        học hỏi và đưa ra quyết định từ dữ liệu mà không cần được lập trình cụ thể cho từng tình huống.
        
        Có ba loại chính của Machine Learning mà chúng ta cần nắm vững:
        1. Học có giám sát (Supervised Learning): sử dụng dữ liệu đã được gán nhãn
        2. Học không giám sát (Unsupervised Learning): tìm kiếm patterns trong dữ liệu không có nhãn  
        3. Học tăng cường (Reinforcement Learning): học thông qua tương tác với môi trường
        
        Các thuật toán Machine Learning phổ biến bao gồm Linear Regression, Decision Trees, 
        Random Forest, Support Vector Machines (SVM), và Neural Networks.
        """,
        
        "section02.txt": """
        Deep Learning là một nhánh con tiên tiến của Machine Learning, sử dụng mạng nơ-ron nhân tạo 
        với nhiều lớp ẩn để mô phỏng cách thức hoạt động của não bộ con người trong việc xử lý thông tin.
        
        Kiến trúc Deep Learning bao gồm:
        - Neural Networks với nhiều hidden layers
        - Convolutional Neural Networks (CNN) cho xử lý hình ảnh
        - Recurrent Neural Networks (RNN) và LSTM cho dữ liệu tuần tự
        - Transformer architecture cho xử lý ngôn ngữ tự nhiên
        
        Các ứng dụng nổi bật của Deep Learning:
        • Nhận dạng hình ảnh và Computer Vision
        • Xử lý ngôn ngữ tự nhiên (NLP)
        • Xe tự lái và autonomous systems  
        • Game AI như AlphaGo và AlphaFold
        • Sinh tạo nội dung: text, image, video
        
        TensorFlow và PyTorch là hai framework phổ biến nhất để phát triển mô hình Deep Learning.
        Google Brain team đã phát triển TensorFlow, trong khi PyTorch được tạo ra bởi Facebook AI Research.
        """,
        
        "section03.txt": """
        Computer Vision là lĩnh vực nghiên cứu interdisciplinary tập trung vào việc làm thế nào 
        để máy tính có thể hiểu, phân tích và diễn giải thông tin từ hình ảnh và video.
        
        Computer Vision sử dụng các thuật toán Machine Learning, đặc biệt là Deep Learning,
        để xử lý và phân tích dữ liệu visual một cách tự động.
        
        Các ứng dụng quan trọng của Computer Vision:
        - Phát hiện và nhận diện đối tượng (Object Detection & Recognition)
        - Nhận diện khuôn mặt và facial recognition systems
        - Optical Character Recognition (OCR) để đọc text từ hình ảnh
        - Chẩn đoán y học qua medical imaging
        - Quality control trong sản xuất industrial
        - Robot vision và autonomous navigation
        - Augmented Reality (AR) và Virtual Reality (VR)
        
        Các kỹ thuật chính: Image Classification, Object Segmentation, Feature Extraction,
        Edge Detection, và Pattern Recognition.
        
        OpenCV là thư viện mã nguồn mở phổ biến nhất cho Computer Vision,
        cung cấp tools cho image processing và computer vision tasks.
        """
    }
    
    for filename, content in sample_texts.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    logger.info(f"Created sample data in {sample_dir}")
    return str(sample_dir.parent)


# Export main classes
__all__ = [
    'VietnameseTextProcessor',
    'FileReader', 
    'EmbeddingGenerator',
    'TextProcessingResult',
    'FileInfo',
    'create_sample_data'
]
