# Tạo graph_builder.py

import json
import logging
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Neo4j and LangChain imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
# from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage

from config import Config
from utils import VietnameseTextProcessor, EmbeddingGenerator

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Triple:
    """Represent a knowledge triple (subject-predicate-object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_section: str = ""
    extraction_method: str = "llm"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"

@dataclass
class GraphStats:
    """Statistics about the knowledge graph"""
    videos_count: int = 0
    sections_count: int = 0
    entities_count: int = 0
    relationships_count: int = 0
    triples_extracted: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class VietnameseKnowledgeGraphBuilder:
    """
    Build knowledge graph from Vietnamese video content
    Xây dựng knowledge graph từ nội dung video tiếng Việt
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize Neo4j graph connection
        self._init_neo4j_connection()
        
        # Initialize LLM for triple extraction
        self._init_llm()
        
        # Initialize Vietnamese text processor
        self.text_processor = VietnameseTextProcessor(self.config)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(config=self.config)
        
        # Setup graph transformer with Vietnamese-specific schema
        self._init_graph_transformer()
        
        logger.info("VietnameseKnowledgeGraphBuilder initialized successfully")
    
    def _init_neo4j_connection(self):
        """Initialize Neo4j database connection"""
        try:
            self.graph = Neo4jGraph(
                url=self.config.NEO4J_URI,
                username=self.config.NEO4J_USERNAME,
                password=self.config.NEO4J_PASSWORD,
                database=self.config.NEO4J_DATABASE
            )
            
            # Test connection
            self.graph.query("RETURN 1 as test")
            logger.info("Neo4j connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _init_llm(self):
        """Initialize LLM for knowledge extraction"""
        try:
            self.llm = OllamaLLM(model="gpt-oss:20b", keep_alive=-1, temperature=0.1,top_p=0.95)
            logger.info(f"LLM initialized: gpt-oss:20b")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_graph_transformer(self):
        """Setup LLM Graph Transformer with Vietnamese-specific instructions"""
        
        # Vietnamese system prompt for knowledge extraction
        vietnamese_system_prompt = """
        Bạn là một chuyên gia trích xuất kiến thức từ văn bản tiếng Việt.
        Nhiệm vụ của bạn là xác định các thực thể (entities) và mối quan hệ (relationships) 
        quan trọng từ văn bản được cung cấp.

        NGUYÊN TẮC QUAN TRỌNG:
        1. Chỉ trích xuất thông tin có thực và chính xác trong văn bản
        2. Tất cả tên thực thể và quan hệ phải viết thường (lowercase)
        3. Sử dụng dấu gạch dưới (_) thay cho dấu cách
        4. Ưu tiên các khái niệm, người, tổ chức, công nghệ quan trọng
        5. Quan hệ phải có ý nghĩa và phản ánh đúng nội dung

        LOẠI THỰC THỂ (Entity Types):
        - người: tên người, tác giả, nhà khoa học
        - tổ_chức: công ty, trường học, viện nghiên cứu  
        - địa_điểm: thành phố, quốc gia, khu vực
        - khái_niệm: thuật ngữ khoa học, phương pháp, lý thuyết
        - công_nghệ: framework, thuật toán, công cụ
        - sản_phẩm: phần mềm, ứng dụng, hệ thống
        - sự_kiện: hội nghị, phát minh, sự kiện lịch sử

        LOẠI QUAN HỆ (Relationship Types):
        - là_một_loại: khái niệm A là một loại của khái niệm B
        - được_sử_dụng_trong: công nghệ/phương pháp được áp dụng
        - phát_triển_bởi: sản phẩm được tạo ra bởi tổ chức/người
        - ứng_dụng_trong: được áp dụng trong lĩnh vực nào
        - bao_gồm: tập hợp chứa các thành phần
        - liên_quan_đến: có mối quan hệ chung

        VÍ DỤ:
        Text: "Machine Learning là một phần của AI được Google phát triển"
        Entities: machine_learning (khái_niệm), ai (khái_niệm), google (tổ_chức)
        Relations: (machine_learning, là_một_phần_của, ai), (machine_learning, được_phát_triển_bởi, google)
        """
        
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.config.VIETNAMESE_ENTITY_TYPES,
            allowed_relationships=[rel[1] for rel in self.config.VIETNAMESE_RELATIONSHIP_TYPES],
            node_properties=["type", "description", "confidence"],
            relationship_properties=["confidence", "source"],
            strict_mode=self.config.STRICT_MODE
        )
        
        # Custom prompt for better Vietnamese extraction
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", vietnamese_system_prompt),
            ("human", "Hãy trích xuất entities và relationships từ văn bản sau:\\n\\n{text}")
        ])
        
        logger.info("Graph transformer initialized with Vietnamese schema")
    
    def setup_database_schema(self):
        """Setup Neo4j database constraints, indexes and schema"""
        
        logger.info("Setting up Neo4j database schema...")
        
        # Drop existing constraints and indexes if needed
        cleanup_queries = [
            "DROP CONSTRAINT video_id IF EXISTS",
            "DROP CONSTRAINT section_id IF EXISTS", 
            "DROP CONSTRAINT entity_name IF EXISTS",
            f"DROP INDEX {self.config.VECTOR_INDEX_NAME} IF EXISTS"
        ]
        
        for query in cleanup_queries:
            try:
                self.graph.query(query)
            except Exception:
                pass  # Ignore if doesn't exist
        
        # Create constraints for data integrity
        constraints = [
            "CREATE CONSTRAINT video_id IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE", 
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint creation failed: {e}")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX video_name_idx IF NOT EXISTS FOR (v:Video) ON (v.name)",
            "CREATE INDEX section_video_idx IF NOT EXISTS FOR (s:Section) ON (s.video_id)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX section_keywords_idx IF NOT EXISTS FOR (s:Section) ON (s.keywords)"
        ]
        
        for index in indexes:
            try:
                self.graph.query(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
    
    def create_vector_index(self):
        """Create vector index for semantic similarity search"""
        
        try:
            # Create vector index for section embeddings
            create_index_query = f"""
            CREATE VECTOR INDEX {self.config.VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n:Section) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.config.VECTOR_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            
            self.graph.query(create_index_query)
            logger.info(f"Created vector index: {self.config.VECTOR_INDEX_NAME}")
            
            # Wait for index to be ready
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    def extract_triples_from_text(self, text: str, section_id: str = "") -> List[Triple]:
        """
        Extract knowledge triples from Vietnamese text using LLM
        Trích xuất knowledge triples từ văn bản tiếng Việt
        """
        
        if not text or len(text.strip()) < 20:
            return []
        
        triples = []
        
        try:
            # Clean and preprocess text
            cleaned_text = self.text_processor.clean_text(text, preserve_structure=True)
            
            # Split into chunks if text is too long
            max_chunk_size = 2000  # characters
            if len(cleaned_text) > max_chunk_size:
                chunks = [cleaned_text[i:i+max_chunk_size] 
                         for i in range(0, len(cleaned_text), max_chunk_size)]
            else:
                chunks = [cleaned_text]
            
            # Process each chunk
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Create document for LLM processing
                    document = Document(
                        page_content=chunk,
                        metadata={"section_id": section_id, "chunk": chunk_idx}
                    )
                    
                    # Extract graph using LLM transformer
                    graph_docs = self.graph_transformer.convert_to_graph_documents([document])
                    
                    if graph_docs:
                        graph_doc = graph_docs[0]
                        
                        # Convert relationships to triples
                        for rel in graph_doc.relationships:
                            triple = Triple(
                                subject=self._normalize_entity_name(rel.source.id),
                                predicate=self._normalize_relation_name(rel.type),
                                object=self._normalize_entity_name(rel.target.id),
                                confidence=0.9,  # Default confidence from LLM
                                source_section=section_id,
                                extraction_method="llm_graph_transformer"
                            )
                            
                            # Validate triple
                            if self._is_valid_triple(triple):
                                triples.append(triple)
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk_idx}: {e}")
                    continue
            
            logger.info(f"Extracted {len(triples)} triples from section {section_id}")
            return triples
            
        except Exception as e:
            logger.error(f"Error extracting triples: {e}")
            return []
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name to follow Vietnamese conventions"""
        if not name:
            return ""
        
        # Convert to lowercase and replace spaces with underscores
        normalized = name.lower().strip()
        normalized = normalized.replace(' ', '_')
        normalized = normalized.replace('-', '_')
        
        # Remove special characters except Vietnamese diacritics
        import re
        normalized = re.sub(r'[^\\w_À-ỹ]', '', normalized)
        
        return normalized
    
    def _normalize_relation_name(self, relation: str) -> str:
        """Normalize relationship name"""
        if not relation:
            return "liên_quan_đến"  # Default relation
        
        normalized = relation.lower().strip()
        normalized = normalized.replace(' ', '_')
        normalized = normalized.replace('-', '_')
        
        return normalized
    
    def _is_valid_triple(self, triple: Triple) -> bool:
        """Validate if a triple is meaningful and correctly formatted"""
        # Check if subject and object are not empty
        if not triple.subject or not triple.object:
            return False
        
        # Check if subject and object are different
        if triple.subject == triple.object:
            return False
        
        # Check minimum length
        if len(triple.subject) < 2 or len(triple.object) < 2:
            return False
        
        # Check if they are not stop words
        if (triple.subject in self.text_processor.stop_words or 
            triple.object in self.text_processor.stop_words):
            return False
        
        return True
    
    def store_video_section(self, video_name: str, section_name: str, 
                          content: str, triples: List[Triple] = None) -> str:
        """
        Store video section and its knowledge in Neo4j
        Lưu trữ section video và kiến thức trong Neo4j
        """
        
        try:
            # Generate section ID
            section_id = f"{video_name}_{section_name}".lower()
            
            # Process text
            processing_result = self.text_processor.process_text(content)
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(processing_result.cleaned_text)
            
            # Extract triples if not provided
            if triples is None:
                triples = self.extract_triples_from_text(content, section_id)
            
            # Store video node
            self._store_video_node(video_name)
            
            # Store section node with all data
            self._store_section_node(
                section_id=section_id,
                video_name=video_name, 
                section_name=section_name,
                content=content,
                processing_result=processing_result,
                embedding=embedding
            )
            
            # Connect video to section
            self._connect_video_section(video_name, section_id)
            
            # Store knowledge triples
            self._store_triples(section_id, triples)
            
            logger.info(f"Successfully stored section: {section_id} with {len(triples)} triples")
            return section_id
            
        except Exception as e:
            logger.error(f"Error storing video section {video_name}/{section_name}: {e}")
            raise
    
    def _store_video_node(self, video_name: str):
        """Store or update video node"""
        video_query = """
        MERGE (v:Video {id: $video_id})
        SET v.name = $video_name,
            v.created_at = coalesce(v.created_at, datetime()),
            v.updated_at = datetime()
        """
        
        self.graph.query(video_query, {
            "video_id": video_name.lower(),
            "video_name": video_name
        })
    
    def _store_section_node(self, section_id: str, video_name: str, section_name: str,
                           content: str, processing_result, embedding: List[float]):
        """Store section node with comprehensive data"""
        
        section_query = """
        MERGE (s:Section {id: $section_id})
        SET s.name = $section_name,
            s.video_id = $video_name,
            s.content = $content,
            s.cleaned_content = $cleaned_content,
            s.keywords = $keywords,
            s.entities = $entities,
            s.word_count = $word_count,
            s.sentence_count = $sentence_count,
            s.embedding = $embedding,
            s.created_at = coalesce(s.created_at, datetime()),
            s.updated_at = datetime(),
            s.processing_time = $processing_time
        """
        
        self.graph.query(section_query, {
            "section_id": section_id,
            "section_name": section_name,
            "video_name": video_name.lower(),
            "content": content,
            "cleaned_content": processing_result.cleaned_text,
            "keywords": processing_result.keywords,
            "entities": processing_result.entities,
            "word_count": processing_result.word_count,
            "sentence_count": len(processing_result.sentences),
            "embedding": embedding,
            "processing_time": processing_result.processing_time
        })
    
    def _connect_video_section(self, video_name: str, section_id: str):
        """Connect video to section"""
        connect_query = """
        MATCH (v:Video {id: $video_name})
        MATCH (s:Section {id: $section_id})
        MERGE (v)-[:HAS_SECTION]->(s)
        """
        
        self.graph.query(connect_query, {
            "video_name": video_name.lower(),
            "section_id": section_id
        })
    
    def _store_triples(self, section_id: str, triples: List[Triple]):
        """Store extracted triples in the knowledge graph"""
        
        for triple in triples:
            try:
                # Store entities
                self._store_entity(triple.subject)
                self._store_entity(triple.object)
                
                # Store relationship
                self._store_relationship(triple, section_id)
                
                # Connect entities to source section
                self._connect_entities_to_section(section_id, [triple.subject, triple.object])
                
            except Exception as e:
                logger.warning(f"Error storing triple {triple}: {e}")
    
    def _store_entity(self, entity_name: str, entity_type: str = "unknown"):
        """Store or update entity node"""
        entity_query = """
        MERGE (e:Entity {name: $entity_name})
        SET e.type = coalesce(e.type, $entity_type),
            e.created_at = coalesce(e.created_at, datetime()),
            e.updated_at = datetime(),
            e.mention_count = coalesce(e.mention_count, 0) + 1
        """
        
        self.graph.query(entity_query, {
            "entity_name": entity_name,
            "entity_type": entity_type
        })
    
    def _store_relationship(self, triple: Triple, section_id: str):
        """Store relationship between entities"""
        
        # Dynamic relationship query
        rel_query = f"""
        MATCH (subj:Entity {{name: $subject}})
        MATCH (obj:Entity {{name: $object}})
        MERGE (subj)-[r:{triple.predicate.upper().replace('_', '_')}]->(obj)
        SET r.confidence = $confidence,
            r.source_section = $section_id,
            r.extraction_method = $extraction_method,
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        """
        
        try:
            self.graph.query(rel_query, {
                "subject": triple.subject,
                "object": triple.object,
                "confidence": triple.confidence,
                "section_id": section_id,
                "extraction_method": triple.extraction_method
            })
        except Exception as e:
            logger.warning(f"Error creating relationship {triple.predicate}: {e}")
            # Fallback to generic relation
            fallback_query = """
            MATCH (subj:Entity {name: $subject})
            MATCH (obj:Entity {name: $object})
            MERGE (subj)-[r:RELATED_TO]->(obj)
            SET r.confidence = $confidence,
                r.source_section = $section_id,
                r.original_predicate = $predicate
            """
            self.graph.query(fallback_query, {
                "subject": triple.subject,
                "object": triple.object,
                "confidence": triple.confidence,
                "section_id": section_id,
                "predicate": triple.predicate
            })
    
    def _connect_entities_to_section(self, section_id: str, entity_names: List[str]):
        """Connect entities to their source section"""
        for entity_name in entity_names:
            connect_query = """
            MATCH (s:Section {id: $section_id})
            MATCH (e:Entity {name: $entity_name})
            MERGE (s)-[:MENTIONS]->(e)
            """
            
            try:
                self.graph.query(connect_query, {
                    "section_id": section_id,
                    "entity_name": entity_name
                })
            except Exception as e:
                logger.warning(f"Error connecting entity {entity_name} to section: {e}")
    
    def build_knowledge_graph(self, videos_data: Dict[str, Dict[str, str]], 
                            parallel: bool = None) -> GraphStats:
        """
        Build complete knowledge graph from videos data
        Xây dựng knowledge graph hoàn chỉnh từ dữ liệu video
        """
        
        start_time = time.time()
        logger.info("Starting knowledge graph construction...")
        
        # Setup database schema
        self.setup_database_schema()
        
        # Initialize statistics
        stats = GraphStats()
        
        parallel = parallel if parallel is not None else self.config.PARALLEL_PROCESSING
        
        try:
            if parallel and len(videos_data) > 1:
                stats = self._build_parallel(videos_data)
            else:
                stats = self._build_sequential(videos_data)
            
            # Create vector index after all data is loaded
            self.create_vector_index()
            
            # Get final graph statistics
            final_stats = self.get_graph_statistics()
            stats.videos_count = final_stats['videos_count']
            stats.sections_count = final_stats['sections_count'] 
            stats.entities_count = final_stats['entities_count']
            stats.relationships_count = final_stats['relationships_count']
            stats.processing_time = time.time() - start_time
            
            logger.info(f"Knowledge graph construction completed in {stats.processing_time:.2f}s")
            logger.info(f"Final stats: {stats.videos_count} videos, {stats.sections_count} sections, "
                       f"{stats.entities_count} entities, {stats.relationships_count} relationships")
            
            return stats
            
        except Exception as e:
            error_msg = f"Error building knowledge graph: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
            stats.processing_time = time.time() - start_time
            return stats
    
    def _build_sequential(self, videos_data: Dict[str, Dict[str, str]]) -> GraphStats:
        """Build knowledge graph sequentially"""
        stats = GraphStats()
        
        for video_name, sections in videos_data.items():
            logger.info(f"Processing video: {video_name}")
            
            for section_name, content in sections.items():
                try:
                    logger.info(f"Processing section: {video_name}/{section_name}")
                    
                    # Extract triples from content
                    triples = self.extract_triples_from_text(content, f"{video_name}_{section_name}")
                    
                    # Store in knowledge graph
                    section_id = self.store_video_section(video_name, section_name, content, triples)
                    
                    stats.sections_count += 1
                    stats.triples_extracted += len(triples)
                    
                except Exception as e:
                    error_msg = f"Error processing {video_name}/{section_name}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
            
            stats.videos_count += 1
        
        return stats
    
    def _build_parallel(self, videos_data: Dict[str, Dict[str, str]]) -> GraphStats:
        """Build knowledge graph using parallel processing"""
        stats = GraphStats()
        
        # Prepare all section tasks
        section_tasks = []
        for video_name, sections in videos_data.items():
            for section_name, content in sections.items():
                section_tasks.append((video_name, section_name, content))
        
        # Process in parallel batches
        max_workers = min(self.config.MAX_WORKERS, len(section_tasks))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_section_task, task): task 
                for task in section_tasks
            }
            
            processed_videos = set()
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                video_name, section_name, content = task
                
                try:
                    result = future.result()
                    if result:
                        stats.sections_count += 1
                        stats.triples_extracted += result.get('triples_count', 0)
                        processed_videos.add(video_name)
                        
                        logger.info(f"Completed: {video_name}/{section_name}")
                
                except Exception as e:
                    error_msg = f"Error processing {video_name}/{section_name}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
        
        stats.videos_count = len(processed_videos)
        return stats
    
    def _process_section_task(self, task: Tuple[str, str, str]) -> Dict:
        """Process a single section task (for parallel execution)"""
        video_name, section_name, content = task
        
        try:
            # Extract triples
            triples = self.extract_triples_from_text(content, f"{video_name}_{section_name}")
            
            # Store section
            section_id = self.store_video_section(video_name, section_name, content, triples)
            
            return {
                'section_id': section_id,
                'triples_count': len(triples),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in section task {video_name}/{section_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        
        stats_queries = {
            "videos_count": "MATCH (v:Video) RETURN count(v) as count",
            "sections_count": "MATCH (s:Section) RETURN count(s) as count", 
            "entities_count": "MATCH (e:Entity) RETURN count(e) as count",
            "relationships_count": "MATCH ()-[r]->() WHERE NOT type(r) IN ['HAS_SECTION', 'MENTIONS'] RETURN count(r) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        stats = {}
        for stat_name, query in stats_queries.items():
            try:
                result = self.graph.query(query)
                stats[stat_name] = result[0]["count"] if result else 0
            except Exception as e:
                logger.error(f"Error getting {stat_name}: {e}")
                stats[stat_name] = 0
        
        # Additional detailed statistics
        try:
            # Entity type distribution
            entity_types_query = """
            MATCH (e:Entity) 
            RETURN e.type as type, count(e) as count 
            ORDER BY count DESC
            """
            entity_types_result = self.graph.query(entity_types_query)
            stats['entity_types'] = {row['type']: row['count'] for row in entity_types_result}
            
            # Top entities by mention count
            top_entities_query = """
            MATCH (e:Entity) 
            RETURN e.name as name, e.mention_count as mentions 
            ORDER BY e.mention_count DESC LIMIT 10
            """
            top_entities_result = self.graph.query(top_entities_query)
            stats['top_entities'] = [
                {'name': row['name'], 'mentions': row['mentions']} 
                for row in top_entities_result
            ]
            
        except Exception as e:
            logger.warning(f"Error getting detailed statistics: {e}")
        
        return stats
    
    def clear_database(self):
        """Clear all data from the database (use with caution!)"""
        logger.warning("Clearing all data from Neo4j database...")
        
        try:
            # Delete all nodes and relationships
            self.graph.query("MATCH (n) DETACH DELETE n")
            
            # Drop vector index
            self.graph.query(f"DROP INDEX {self.config.VECTOR_INDEX_NAME} IF EXISTS")
            
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise


# Export main classes
__all__ = [
    'VietnameseKnowledgeGraphBuilder',
    'Triple',
    'GraphStats'
]

print("graph_builder.py created successfully")
print("=" * 60)
print(graph_builder_code[:3000] + "...")