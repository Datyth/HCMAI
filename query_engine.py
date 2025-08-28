import logging
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain and Neo4j imports
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import Config
from utils import VietnameseTextProcessor, EmbeddingGenerator

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Represent a search result from the knowledge graph"""
    section_id: str
    video_name: str
    section_name: str
    content: str
    similarity_score: float
    entities: List[str]
    keywords: List[str]
    matching_triples: List[Dict[str, Any]]
    search_method: str = "hybrid"
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_content_preview(self, max_length: int = 200) -> str:
        """Get truncated content preview"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

@dataclass 
class QueryAnalysis:
    """Analysis result of user query"""
    original_query: str
    cleaned_query: str
    keywords: List[str]
    entities: List[str]
    concepts: List[str]
    relationships: List[str]
    intent: str
    confidence: float = 0.0
    
    def get_all_search_terms(self) -> List[str]:
        """Get all search terms for querying"""
        return list(set(self.keywords + self.entities + self.concepts))

class VietnameseQueryEngine:
    """
    Advanced query engine for Vietnamese video knowledge graph
    Engine truy vấn nâng cao cho knowledge graph video tiếng Việt
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize Neo4j graph connection
        self._init_neo4j_connection()
        
        # Initialize LLM for query analysis
        self._init_llm()
        
        # Initialize Vietnamese text processor
        self.text_processor = VietnameseTextProcessor(self.config)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(config=self.config)
        
        # Setup query analyzer
        self._init_query_analyzer()
        
        logger.info("VietnameseQueryEngine initialized successfully")
    
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
            logger.info("Neo4j connection established for query engine")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _init_llm(self):
        """Initialize LLM for query analysis and explanation"""
        try:
            self.llm = ChatOpenAI(
                temperature=self.config.OPENAI_TEMPERATURE,
                model=self.config.OPENAI_MODEL,
                api_key=self.config.OPENAI_API_KEY,
                max_tokens=self.config.OPENAI_MAX_TOKENS,
                request_timeout=self.config.OPENAI_REQUEST_TIMEOUT
            )
            logger.info(f"LLM initialized for query engine: {self.config.OPENAI_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_query_analyzer(self):
        """Setup query analyzer with Vietnamese-specific prompts"""
        
        # Vietnamese query analysis prompt
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Bạn là một chuyên gia phân tích câu truy vấn tiếng Việt cho hệ thống knowledge graph.
            Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và xác định:
            
            1. CÁC TỪ KHÓA (Keywords): Từ khóa chính trong câu hỏi
            2. CÁC THỰC THỂ (Entities): Tên riêng, địa danh, tổ chức, người
            3. CÁC KHÁI NIỆM (Concepts): Khái niệm, chủ đề, lĩnh vực
            4. CÁC MỐI QUAN HỆ (Relationships): Mối quan hệ mong muốn tìm kiếm
            5. Ý ĐỊNH (Intent): Loại câu hỏi (định nghĩa, so sánh, ứng dụng, v.v.)
            
            NGUYÊN TẮC:
            - Tất cả từ khóa, thực thể, khái niệm phải viết thường
            - Sử dụng dấu gạch dưới thay cho dấu cách
            - Tìm từ đồng nghĩa và các thuật ngữ liên quan
            - Xác định chính xác ý định của người dùng
            
            Trả lời theo định dạng JSON với các trường:
            - keywords: danh sách từ khóa chính
            - entities: danh sách thực thể
            - concepts: danh sách khái niệm
            - relationships: danh sách mối quan hệ có thể có
            - intent: loại câu hỏi (definition, application, comparison, explanation, list)
            - confidence: độ tin cậy (0.0-1.0)
            """),
            ("human", "Phân tích câu truy vấn sau: {query}")
        ])
        
        # Explanation generation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Bạn là một chuyên gia tạo câu trả lời từ kết quả tìm kiếm knowledge graph.
            Nhiệm vụ của bạn là tổng hợp thông tin từ các kết quả tìm kiếm và tạo ra
            một câu trả lời tự nhiên, chính xác và hữu ích bằng tiếng Việt.
            
            NGUYÊN TẮC:
            - Sử dụng thông tin từ các kết quả tìm kiếm để trả lời
            - Đề cập đến video và phần cụ thể nơi tìm thấy thông tin
            - Tránh đưa ra thông tin không có trong kết quả
            - Giữ câu trả lời ngắn gọn nhưng đầy đủ
            - Sắp xếp thông tin theo thứ tự quan trọng
            """),
            ("human", """
            Câu hỏi: {query}
            
            Kết quả tìm kiếm:
            {search_results}
            
            Hãy tạo câu trả lời tự nhiên từ các kết quả trên.
            """)
        ])
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze Vietnamese query into structured components
        Phân tích câu truy vấn tiếng Việt thành các thành phần có cấu trúc
        """
        
        start_time = time.time()
        
        try:
            # Clean query
            cleaned_query = self.text_processor.clean_text(query)
            
            # Extract basic keywords and entities using NLP
            processing_result = self.text_processor.process_text(cleaned_query, 
                                                               extract_entities=True, 
                                                               extract_keywords=True)
            
            # Use LLM for advanced analysis
            try:
                llm_response = self.llm.invoke(
                    self.query_analysis_prompt.format(query=cleaned_query)
                )
                
                # Parse LLM response as JSON
                import json
                llm_analysis = json.loads(llm_response.content)
                
                # Combine NLP and LLM results
                analysis = QueryAnalysis(
                    original_query=query,
                    cleaned_query=cleaned_query,
                    keywords=list(set(processing_result.keywords + llm_analysis.get('keywords', []))),
                    entities=list(set(processing_result.entities + llm_analysis.get('entities', []))),
                    concepts=llm_analysis.get('concepts', []),
                    relationships=llm_analysis.get('relationships', []),
                    intent=llm_analysis.get('intent', 'unknown'),
                    confidence=llm_analysis.get('confidence', 0.8)
                )
                
            except Exception as e:
                logger.warning(f"LLM analysis failed, using NLP only: {e}")
                # Fallback to NLP-only analysis
                analysis = QueryAnalysis(
                    original_query=query,
                    cleaned_query=cleaned_query,
                    keywords=processing_result.keywords,
                    entities=processing_result.entities,
                    concepts=processing_result.keywords[:5],  # Use top keywords as concepts
                    relationships=['liên_quan_đến', 'được_sử_dụng_trong', 'là_một_phần_của'],
                    intent='unknown',
                    confidence=0.7
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Query analysis completed in {processing_time:.3f}s")
            logger.debug(f"Analysis result: {analysis}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return basic analysis
            return QueryAnalysis(
                original_query=query,
                cleaned_query=query.lower(),
                keywords=query.lower().split(),
                entities=[],
                concepts=[],
                relationships=[],
                intent='unknown',
                confidence=0.5
            )
    
    def vector_similarity_search(self, query: str, k: int = 5) -> List[QueryResult]:
        """
        Perform vector similarity search on sections
        Thực hiện tìm kiếm tương tự vector trên các sections
        """
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            if not query_embedding or all(x == 0 for x in query_embedding):
                logger.warning("Failed to generate query embedding")
                return []
            
            # Vector similarity search using Neo4j vector index
            vector_search_query = f"""
            CALL db.index.vector.queryNodes(
                '{self.config.VECTOR_INDEX_NAME}', 
                $k, 
                $query_embedding
            ) YIELD node as section, score
            MATCH (v:Video)-[:HAS_SECTION]->(section)
            RETURN 
                section.id as section_id,
                v.name as video_name,
                section.name as section_name,
                section.content as content,
                section.keywords as keywords,
                section.entities as entities,
                score as similarity_score
            ORDER BY score DESC
            """
            
            results = self.graph.query(vector_search_query, {
                "k": min(k, self.config.MAX_SEARCH_RESULTS),
                "query_embedding": query_embedding
            })
            
            query_results = []
            for result in results:
                # Get matching triples for this section
                matching_triples = self._get_section_triples(result["section_id"])
                
                query_result = QueryResult(
                    section_id=result["section_id"],
                    video_name=result["video_name"],
                    section_name=result["section_name"],
                    content=result["content"] or "",
                    similarity_score=float(result["similarity_score"]),
                    entities=result["entities"] or [],
                    keywords=result["keywords"] or [],
                    matching_triples=matching_triples,
                    search_method="vector",
                    processing_time=time.time() - start_time
                )
                query_results.append(query_result)
            
            logger.info(f"Vector search found {len(query_results)} results in {time.time() - start_time:.3f}s")
            return query_results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    def knowledge_graph_search(self, query_analysis: QueryAnalysis, k: int = 5) -> List[QueryResult]:
        """
        Search knowledge graph based on entities and relationships
        Tìm kiếm knowledge graph dựa trên entities và relationships
        """
        
        start_time = time.time()
        
        try:
            search_terms = query_analysis.get_all_search_terms()
            
            if not search_terms:
                logger.warning("No search terms found in query analysis")
                return []
            
            # Build Cypher query to find relevant sections through entities
            cypher_query = """
            // Find sections that mention entities matching search terms
            MATCH (s:Section)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) IN [term IN $search_terms WHERE term IS NOT NULL | toLower(term)]
            
            WITH s, count(DISTINCT e) as entity_matches,
                 collect(DISTINCT e.name) as matched_entities
            
            MATCH (v:Video)-[:HAS_SECTION]->(s)
            
            // Calculate relevance score based on entity matches and keyword matches
            WITH s, v, entity_matches, matched_entities,
                 (entity_matches * 1.0 / size($search_terms)) as entity_score,
                 size([kw IN s.keywords WHERE toLower(kw) IN $search_terms]) as keyword_matches
            
            WITH s, v, entity_matches, matched_entities, entity_score, keyword_matches,
                 (entity_score * 0.6 + (keyword_matches * 1.0 / size(s.keywords)) * 0.4) as relevance_score
            
            RETURN 
                s.id as section_id,
                v.name as video_name,
                s.name as section_name,
                s.content as content,
                s.keywords as keywords,
                s.entities as entities,
                relevance_score,
                matched_entities,
                entity_matches,
                keyword_matches
            ORDER BY relevance_score DESC, entity_matches DESC, keyword_matches DESC
            LIMIT $k
            """
            
            # Normalize search terms
            normalized_search_terms = [term.lower() for term in search_terms if term]
            
            results = self.graph.query(cypher_query, {
                "search_terms": normalized_search_terms,
                "k": min(k, self.config.MAX_SEARCH_RESULTS)
            })
            
            query_results = []
            for result in results:
                # Skip results with very low relevance
                if result["relevance_score"] < 0.1:
                    continue
                
                # Get matching triples for this section
                matching_triples = self._get_section_triples(
                    result["section_id"], 
                    entity_filter=normalized_search_terms
                )
                
                query_result = QueryResult(
                    section_id=result["section_id"],
                    video_name=result["video_name"],
                    section_name=result["section_name"],
                    content=result["content"] or "",
                    similarity_score=float(result["relevance_score"]),
                    entities=result["entities"] or [],
                    keywords=result["keywords"] or [],
                    matching_triples=matching_triples,
                    search_method="knowledge_graph",
                    processing_time=time.time() - start_time
                )
                query_results.append(query_result)
            
            logger.info(f"Knowledge graph search found {len(query_results)} results in {time.time() - start_time:.3f}s")
            return query_results
            
        except Exception as e:
            logger.error(f"Error in knowledge graph search: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, 
                     vector_weight: float = None, kg_weight: float = None) -> List[QueryResult]:
        """
        Combine vector similarity and knowledge graph search
        Kết hợp tìm kiếm tương tự vector và knowledge graph
        """
        
        start_time = time.time()
        
        # Use configured weights if not provided
        vector_weight = vector_weight or self.config.SEARCH_WEIGHTS["vector"]
        kg_weight = kg_weight or self.config.SEARCH_WEIGHTS["knowledge_graph"]
        
        try:
            # Analyze query first
            query_analysis = self.analyze_query(query)
            
            # Perform both searches with higher k for better reranking
            extended_k = min(k * 3, self.config.MAX_SEARCH_RESULTS)
            
            # Vector search
            vector_results = self.vector_similarity_search(query, extended_k)
            
            # Knowledge graph search
            kg_results = self.knowledge_graph_search(query_analysis, extended_k)
            
            # Combine and rerank results
            combined_results = {}
            
            # Add vector results with weights
            for result in vector_results:
                section_id = result.section_id
                weighted_score = result.similarity_score * vector_weight
                
                if section_id not in combined_results:
                    result.similarity_score = weighted_score
                    result.search_method = "hybrid"
                    combined_results[section_id] = result
                else:
                    # Update existing result
                    combined_results[section_id].similarity_score += weighted_score
            
            # Add knowledge graph results with weights
            for result in kg_results:
                section_id = result.section_id
                weighted_score = result.similarity_score * kg_weight
                
                if section_id not in combined_results:
                    result.similarity_score = weighted_score
                    result.search_method = "hybrid"
                    combined_results[section_id] = result
                else:
                    # Combine scores and merge data
                    existing = combined_results[section_id]
                    existing.similarity_score += weighted_score
                    existing.search_method = "hybrid"
                    
                    # Merge matching triples (avoid duplicates)
                    existing_triple_strs = {str(t) for t in existing.matching_triples}
                    for triple in result.matching_triples:
                        if str(triple) not in existing_triple_strs:
                            existing.matching_triples.append(triple)
            
            # Sort by combined score and apply similarity threshold
            final_results = [
                result for result in combined_results.values()
                if result.similarity_score >= self.config.SIMILARITY_THRESHOLD / 10  # Lower threshold for combined scores
            ]
            
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = final_results[:k]
            
            # Update processing time
            total_time = time.time() - start_time
            for result in final_results:
                result.processing_time = total_time
            
            logger.info(f"Hybrid search found {len(final_results)} results in {total_time:.3f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _get_section_triples(self, section_id: str, entity_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get knowledge triples related to a section
        Lấy các triples kiến thức liên quan đến một section
        """
        
        try:
            base_query = """
            MATCH (s:Section {id: $section_id})-[:MENTIONS]->(e1:Entity)
            MATCH (e1)-[r]->(e2:Entity)
            WHERE r.source_section = $section_id OR s-[:MENTIONS]->(e2)
            """
            
            # Add entity filter if provided
            if entity_filter:
                base_query += """
                AND (toLower(e1.name) IN $entity_filter OR toLower(e2.name) IN $entity_filter)
                """
            
            base_query += """
            RETURN DISTINCT
                e1.name as subject,
                type(r) as predicate,
                e2.name as object,
                r.confidence as confidence,
                r.extraction_method as extraction_method
            ORDER BY r.confidence DESC
            LIMIT 20
            """
            
            params = {"section_id": section_id}
            if entity_filter:
                params["entity_filter"] = [term.lower() for term in entity_filter]
            
            results = self.graph.query(base_query, params)
            
            triples = []
            for result in results:
                triples.append({
                    "subject": result["subject"],
                    "predicate": result["predicate"],
                    "object": result["object"],
                    "confidence": result["confidence"] or 1.0,
                    "extraction_method": result["extraction_method"] or "unknown"
                })
            
            return triples
            
        except Exception as e:
            logger.warning(f"Error getting section triples for {section_id}: {e}")
            return []
    
    def query(self, query: str, k: int = None, search_method: str = None) -> List[QueryResult]:
        """
        Main query function - supports different search methods
        Hàm truy vấn chính - hỗ trợ các phương pháp tìm kiếm khác nhau
        """
        
        k = k or self.config.DEFAULT_K
        search_method = search_method or self.config.DEFAULT_SEARCH_METHOD
        
        logger.info(f"Processing query: '{query}' with method: {search_method}, k={k}")
        
        if search_method == "vector":
            return self.vector_similarity_search(query, k)
        elif search_method == "knowledge_graph":
            query_analysis = self.analyze_query(query)
            return self.knowledge_graph_search(query_analysis, k)
        elif search_method == "hybrid":
            return self.hybrid_search(query, k)
        else:
            logger.warning(f"Unknown search method: {search_method}, using hybrid")
            return self.hybrid_search(query, k)
    
    def explain_results(self, query: str, results: List[QueryResult]) -> str:
        """
        Generate explanation for search results using LLM
        Tạo giải thích cho kết quả tìm kiếm bằng LLM
        """
        
        try:
            if not results:
                return "Không tìm thấy kết quả phù hợp với câu truy vấn. Hãy thử sử dụng từ khóa khác hoặc đặt câu hỏi theo cách khác."
            
            # Prepare context from results
            context_parts = []
            for i, result in enumerate(results[:3], 1):  # Use top 3 results for explanation
                content_preview = result.get_content_preview(300)
                
                # Format matching triples
                triples_text = ""
                if result.matching_triples:
                    top_triples = result.matching_triples[:3]
                    triples_text = "\\n".join([
                        f"  - {t['subject']} {t['predicate']} {t['object']}" 
                        for t in top_triples
                    ])
                
                context_parts.append(f"""
                Kết quả {i} (Điểm tương đồng: {result.similarity_score:.3f}):
                Video: {result.video_name}
                Phần: {result.section_name}
                Nội dung: {content_preview}
                Từ khóa: {', '.join(result.keywords[:5])}
                Mối quan hệ chính:
                {triples_text}
                """)
            
            context = "\\n---\\n".join(context_parts)
            
            # Generate explanation using LLM
            try:
                response = self.llm.invoke(
                    self.explanation_prompt.format(
                        query=query,
                        search_results=context
                    )
                )
                return response.content
                
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
                # Fallback explanation
                return self._generate_fallback_explanation(query, results)
            
        except Exception as e:
            logger.error(f"Error explaining results: {e}")
            return "Có lỗi xảy ra khi tạo giải thích cho kết quả tìm kiếm."
    
    def _generate_fallback_explanation(self, query: str, results: List[QueryResult]) -> str:
        """Generate a simple fallback explanation without LLM"""
        
        if not results:
            return "Không tìm thấy kết quả nào."
        
        top_result = results[0]
        explanation = f"""
        Tìm thấy {len(results)} kết quả liên quan đến câu hỏi "{query}".
        
        Kết quả phù hợp nhất từ video "{top_result.video_name}", phần "{top_result.section_name}" 
        với độ tương đồng {top_result.similarity_score:.3f}.
        
        Nội dung chính: {top_result.get_content_preview(200)}
        """
        
        if top_result.matching_triples:
            explanation += f"\\n\\nCác mối quan hệ quan trọng được tìm thấy:"
            for triple in top_result.matching_triples[:3]:
                explanation += f"\\n- {triple['subject']} {triple['predicate']} {triple['object']}"
        
        return explanation.strip()
    
    def get_query_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """
        Get query suggestions based on existing entities and content
        Lấy gợi ý truy vấn dựa trên entities và nội dung có sẵn
        """
        
        try:
            # Analyze current query
            query_analysis = self.analyze_query(query)
            
            # Get related entities and concepts from knowledge graph
            suggestion_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS $query_term OR any(term IN $search_terms WHERE toLower(e.name) CONTAINS term)
            WITH e, e.mention_count as mentions
            ORDER BY mentions DESC
            LIMIT 10
            
            MATCH (s:Section)-[:MENTIONS]->(e)
            WITH e, collect(DISTINCT s.keywords)[..5] as related_keywords
            
            RETURN e.name as entity, related_keywords
            """
            
            search_terms = [term.lower() for term in query_analysis.get_all_search_terms()]
            query_term = query_analysis.cleaned_query.lower()
            
            results = self.graph.query(suggestion_query, {
                "query_term": query_term,
                "search_terms": search_terms
            })
            
            suggestions = []
            for result in results:
                entity = result["entity"]
                keywords = result["related_keywords"] or []
                
                # Generate suggestion templates
                if entity and keywords:
                    suggestions.extend([
                        f"{entity} là gì?",
                        f"{entity} được sử dụng như thế nào?",
                        f"{entity} có ứng dụng gì?",
                        f"Mối quan hệ giữa {entity} và {keywords[0] if keywords else 'công nghệ'}?",
                    ])
            
            # Remove duplicates and limit
            unique_suggestions = list(dict.fromkeys(suggestions))
            return unique_suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error getting query suggestions: {e}")
            return [
                "Machine learning là gì?",
                "Deep learning có ứng dụng gì?", 
                "Computer vision được sử dụng ở đâu?",
                "AI và machine learning khác nhau như thế nào?",
                "Tensorflow và PyTorch so sánh thế nào?"
            ]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search performance and content"""
        
        try:
            stats_queries = {
                "total_sections": "MATCH (s:Section) RETURN count(s) as count",
                "total_entities": "MATCH (e:Entity) RETURN count(e) as count",
                "avg_section_length": "MATCH (s:Section) RETURN avg(s.word_count) as avg_length",
                "most_mentioned_entities": """
                    MATCH (e:Entity) 
                    RETURN e.name as entity, e.mention_count as mentions 
                    ORDER BY mentions DESC LIMIT 10
                """,
                "section_keywords_distribution": """
                    MATCH (s:Section) 
                    RETURN size(s.keywords) as keyword_count, count(s) as section_count
                    ORDER BY keyword_count DESC
                """
            }
            
            stats = {}
            for stat_name, query in stats_queries.items():
                try:
                    result = self.graph.query(query)
                    if stat_name in ["total_sections", "total_entities"]:
                        stats[stat_name] = result[0]["count"] if result else 0
                    elif stat_name == "avg_section_length":
                        stats[stat_name] = round(result[0]["avg_length"] if result else 0, 2)
                    else:
                        stats[stat_name] = result
                except Exception as e:
                    logger.warning(f"Error getting {stat_name}: {e}")
                    stats[stat_name] = 0 if stat_name in ["total_sections", "total_entities", "avg_section_length"] else []
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {}


# Export main classes
__all__ = [
    'VietnameseQueryEngine',
    'QueryResult',
    'QueryAnalysis'
]

print("query_engine.py created successfully")
print("=" * 60)
print(query_engine_code[:3000] + "...")