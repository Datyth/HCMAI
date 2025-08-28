import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from utils import FileReader, VietnameseTextProcessor, create_sample_data
from graph_builder import VietnameseKnowledgeGraphBuilder
from query_engine import VietnameseQueryEngine, QueryResult

# Setup comprehensive logging
def setup_logging(config: Config):
    """Setup logging configuration"""
    
    # Create logs directory
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Path(config.LOG_DIR) / config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set third-party library log levels
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    # logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class VideoKnowledgeGraphSystem:
    """
    Main system class for Vietnamese Video Knowledge Graph
    Lớp hệ thống chính cho Vietnamese Video Knowledge Graph
    """
    
    def __init__(self, config: Config = None, environment: str = "development"):
        """Initialize the system with configuration"""
        
        self.config = config or get_config(environment)
        
        # Validate configuration
        try:
            self.config.validate_config()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components (lazy loading)
        self.file_reader = FileReader(self.config)
        self.text_processor = VietnameseTextProcessor(self.config)
        self.graph_builder = None
        self.query_engine = None
        
        # System stats
        self.system_stats = {
            'initialized_at': time.time(),
            'videos_processed': 0,
            'queries_processed': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"VideoKnowledgeGraphSystem initialized with {environment} configuration")
        self.config.print_config()
    
    def _ensure_graph_builder(self):
        """Lazy initialization of graph builder"""
        if self.graph_builder is None:
            logger.info("Initializing graph builder...")
            self.graph_builder = VietnameseKnowledgeGraphBuilder(self.config)
    
    def _ensure_query_engine(self):
        """Lazy initialization of query engine"""
        if self.query_engine is None:
            logger.info("Initializing query engine...")
            self.query_engine = VietnameseQueryEngine(self.config)
    
    def load_video_files(self, data_dir: str = None) -> Dict[str, Dict[str, str]]:
        """
        Load video files from directory
        Tải các file video từ thư mục
        """
        
        data_dir = data_dir or self.config.OUTPUT_FILES_DIR
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} does not exist")
            return {}
        
        logger.info(f"Loading video files from: {data_dir}")
        start_time = time.time()
        
        # Load videos data
        videos_data = self.file_reader.read_video_files(data_dir)
        
        # Get statistics
        stats = self.file_reader.get_file_stats(data_dir)
        
        loading_time = time.time() - start_time
        self.system_stats['total_processing_time'] += loading_time
        
        logger.info(f"Loaded {len(videos_data)} videos in {loading_time:.2f}s")
        logger.info(f"Total sections: {stats['total_sections']}, "
                   f"Total characters: {stats['total_characters']:,}")
        
        return videos_data
    
    def build_knowledge_graph(self, videos_data: Dict[str, Dict[str, str]] = None,
                            data_dir: str = None, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build knowledge graph from video data
        Xây dựng knowledge graph từ dữ liệu video
        """
        
        self._ensure_graph_builder()
        
        # Load data if not provided
        if videos_data is None:
            videos_data = self.load_video_files(data_dir)
        
        if not videos_data:
            logger.error("No video data to process")
            return {'success': False, 'error': 'No video data found'}
        
        # Clear existing data if force rebuild
        if force_rebuild:
            logger.warning("Force rebuild: clearing existing knowledge graph")
            self.graph_builder.clear_database()
        
        logger.info("Starting knowledge graph construction...")
        start_time = time.time()
        
        # Build knowledge graph
        build_stats = self.graph_builder.build_knowledge_graph(videos_data)
        
        # Get final graph statistics
        graph_stats = self.graph_builder.get_graph_statistics()
        
        construction_time = time.time() - start_time
        self.system_stats['total_processing_time'] += construction_time
        self.system_stats['videos_processed'] += build_stats.videos_count
        
        result = {
            'success': len(build_stats.errors) == 0,
            'build_stats': {
                'videos_processed': build_stats.videos_count,
                'sections_processed': build_stats.sections_count,
                'triples_extracted': build_stats.triples_extracted,
                'processing_time': build_stats.processing_time,
                'errors': build_stats.errors
            },
            'graph_stats': graph_stats,
            'construction_time': construction_time
        }
        
        if result['success']:
            logger.info(f"Knowledge graph construction completed successfully in {construction_time:.2f}s")
            logger.info(f"Graph contains: {graph_stats['videos_count']} videos, "
                       f"{graph_stats['sections_count']} sections, "
                       f"{graph_stats['entities_count']} entities, "
                       f"{graph_stats['relationships_count']} relationships")
        else:
            logger.error(f"Knowledge graph construction completed with {len(build_stats.errors)} errors")
            for error in build_stats.errors:
                logger.error(f"Build error: {error}")
        
        return result
    
    def query_system(self, query: str, k: int = None, search_method: str = None, 
                    explain: bool = True, return_analysis: bool = False) -> Dict[str, Any]:
        """
        Query the knowledge graph system
        Truy vấn hệ thống knowledge graph
        """
        
        self._ensure_query_engine()
        
        k = k or self.config.DEFAULT_K
        search_method = search_method or self.config.DEFAULT_SEARCH_METHOD
        
        logger.info(f"Processing query: '{query}' (method: {search_method}, k: {k})")
        start_time = time.time()
        
        try:
            # Perform search
            results = self.query_engine.query(query, k=k, search_method=search_method)
            
            # Generate explanation if requested
            explanation = ""
            if explain and results:
                explanation = self.query_engine.explain_results(query, results)
            
            # Get query analysis if requested
            query_analysis = None
            if return_analysis:
                query_analysis = self.query_engine.analyze_query(query)
            
            # Format results for output
            formatted_results = []
            for i, result in enumerate(results):
                formatted_result = {
                    'rank': i + 1,
                    'video': result.video_name,
                    'section': result.section_name,
                    'section_id': result.section_id,
                    'similarity_score': round(result.similarity_score, 4),
                    'content': result.content,
                    'content_preview': result.get_content_preview(300),
                    'keywords': result.keywords,
                    'entities': result.entities,
                    'knowledge_triples': result.matching_triples[:5],  # Top 5 triples
                    'search_method': result.search_method
                }
                formatted_results.append(formatted_result)
            
            query_time = time.time() - start_time
            self.system_stats['queries_processed'] += 1
            self.system_stats['total_processing_time'] += query_time
            
            query_result = {
                'success': True,
                'query': query,
                'search_method': search_method,
                'k_requested': k,
                'total_results': len(results),
                'results': formatted_results,
                'explanation': explanation,
                'processing_time': query_time
            }
            
            if return_analysis and query_analysis:
                query_result['query_analysis'] = {
                    'keywords': query_analysis.keywords,
                    'entities': query_analysis.entities,
                    'concepts': query_analysis.concepts,
                    'relationships': query_analysis.relationships,
                    'intent': query_analysis.intent,
                    'confidence': query_analysis.confidence
                }
            
            logger.info(f"Query processed successfully in {query_time:.3f}s: found {len(results)} results")
            return query_result
        
        except Exception as e:
            error_msg = f"Error processing query '{query}': {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'results': [],
                'processing_time': time.time() - start_time
            }
    
    def get_query_suggestions(self, query: str = "", max_suggestions: int = 5) -> List[str]:
        """Get query suggestions"""
        
        self._ensure_query_engine()
        
        try:
            if query:
                return self.query_engine.get_query_suggestions(query, max_suggestions)
            else:
                # Return popular/default suggestions
                return [
                    "Machine learning là gì?",
                    "Deep learning có ứng dụng gì?",
                    "Computer vision được sử dụng ở đâu?",
                    "TensorFlow và PyTorch khác nhau như thế nào?",
                    "AI và trí tuệ nhân tạo có giống nhau không?"
                ]
        except Exception as e:
            logger.error(f"Error getting query suggestions: {e}")
            return ["Machine learning là gì?"]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = self.system_stats.copy()
        
        # Add graph statistics if available
        if self.graph_builder:
            try:
                graph_stats = self.graph_builder.get_graph_statistics()
                stats['graph'] = graph_stats
            except Exception as e:
                logger.warning(f"Error getting graph statistics: {e}")
        
        # Add search statistics if available
        if self.query_engine:
            try:
                search_stats = self.query_engine.get_search_statistics()
                stats['search'] = search_stats
            except Exception as e:
                logger.warning(f"Error getting search statistics: {e}")
        
        # Calculate uptime
        stats['uptime_seconds'] = time.time() - stats['initialized_at']
        
        return stats
    
    def interactive_mode(self):
        """
        Run system in interactive mode
        Chạy hệ thống ở chế độ tương tác
        """
        
        print("\\n" + "="*70)
        print("    HỆ THỐNG TRUY VẤN VIDEO KNOWLEDGE GRAPH TIẾNG VIỆT")
        print("         Vietnamese Video Knowledge Graph Query System")
        print("="*70)
        
        # Initialize query engine
        print("\\nĐang khởi tạo hệ thống...")
        try:
            self._ensure_query_engine()
            print("✅ Hệ thống đã sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo: {e}")
            return
        
        # Show instructions
        self._show_interactive_help()
        
        # Main interactive loop
        while True:
            try:
                user_input = input("\\n❓ Câu hỏi: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'thoat', 'thoát']:
                    print("\\n👋 Tạm biệt!")
                    break
                
                elif user_input.lower() in ['help', 'huong-dan', 'hướng-dẫn']:
                    self._show_interactive_help()
                    continue
                
                elif user_input.lower() in ['stats', 'thong-ke', 'thống-kê']:
                    self._show_system_stats()
                    continue
                
                elif user_input.lower().startswith('suggest'):
                    # Get query suggestions
                    query_part = user_input[7:].strip()  # Remove 'suggest'
                    suggestions = self.get_query_suggestions(query_part)
                    print("\\n💡 Gợi ý câu hỏi:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                    continue
                
                elif user_input.lower().startswith('method'):
                    # Change search method
                    method_part = user_input[6:].strip()  # Remove 'method'
                    if method_part in ['vector', 'knowledge_graph', 'hybrid']:
                        self.config.DEFAULT_SEARCH_METHOD = method_part
                        print(f"\\n✅ Đã chuyển phương pháp tìm kiếm thành: {method_part}")
                    else:
                        print("\\n❌ Phương pháp không hợp lệ. Sử dụng: vector, knowledge_graph, hoặc hybrid")
                    continue
                
                # Process regular query
                print("\\n🔍 Đang tìm kiếm...")
                
                result = self.query_system(
                    query=user_input,
                    k=5,
                    search_method=self.config.DEFAULT_SEARCH_METHOD,
                    explain=True
                )
                
                # Display results
                self._display_query_results(result)
            
            except KeyboardInterrupt:
                print("\\n\\n👋 Tạm biệt!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\\n❌ Có lỗi xảy ra: {e}")
    
    def _show_interactive_help(self):
        """Show help information in interactive mode"""
        
        print("\\n📖 Hướng dẫn sử dụng:")
        print("   • Nhập câu hỏi bằng tiếng Việt để tìm kiếm")
        print("   • 'help' - Hiển thị hướng dẫn này")
        print("   • 'stats' - Xem thống kê hệ thống")
        print("   • 'suggest [từ khóa]' - Lấy gợi ý câu hỏi")
        print("   • 'method [vector/knowledge_graph/hybrid]' - Đổi phương pháp tìm kiếm")
        print("   • 'exit' - Thoát chương trình")
        print("\\n💡 Ví dụ câu hỏi:")
        print("   - Machine learning là gì?")
        print("   - Deep learning có ứng dụng gì trong computer vision?")
        print("   - TensorFlow và PyTorch khác nhau như thế nào?")
        print("-" * 70)
    
    def _show_system_stats(self):
        """Show system statistics in interactive mode"""
        
        print("\\n📊 Thống kê hệ thống:")
        stats = self.get_system_statistics()
        
        # Basic stats
        print(f"   📹 Videos đã xử lý: {stats.get('videos_processed', 0)}")
        print(f"   🔍 Truy vấn đã xử lý: {stats.get('queries_processed', 0)}")
        print(f"   ⏱️  Thời gian hoạt động: {stats.get('uptime_seconds', 0):.1f}s")
        print(f"   💻 Tổng thời gian xử lý: {stats.get('total_processing_time', 0):.2f}s")
        
        # Graph stats
        if 'graph' in stats:
            graph = stats['graph']
            print(f"   📊 Knowledge Graph:")
            print(f"      - Videos: {graph.get('videos_count', 0)}")
            print(f"      - Sections: {graph.get('sections_count', 0)}")
            print(f"      - Entities: {graph.get('entities_count', 0)}")
            print(f"      - Relationships: {graph.get('relationships_count', 0)}")
            
            if 'top_entities' in graph:
                print(f"   🏆 Top entities:")
                for entity in graph['top_entities'][:3]:
                    print(f"      - {entity.get('name', '')}: {entity.get('mentions', 0)} mentions")
        
        # Current configuration
        print(f"   ⚙️  Cấu hình hiện tại:")
        print(f"      - Search method: {self.config.DEFAULT_SEARCH_METHOD}")
        print(f"      - Default k: {self.config.DEFAULT_K}")
        print(f"      - Model: {self.config.OPENAI_MODEL}")
    
    def _display_query_results(self, result: Dict[str, Any]):
        """Display query results in a formatted way"""
        
        if not result.get('success'):
            print(f"\\n❌ Lỗi truy vấn: {result.get('error', 'Unknown error')}")
            return
        
        print(f"\\n📋 Kết quả cho: '{result['query']}'")
        print(f"   🔍 Phương pháp: {result['search_method']}")
        print(f"   📊 Tìm thấy: {result['total_results']} kết quả")
        print(f"   ⏱️  Thời gian: {result['processing_time']:.3f}s")
        print("-" * 70)
        
        if result['results']:
            for res in result['results']:
                print(f"\\n🎯 #{res['rank']} - {res['video']}/{res['section']}")
                print(f"   📊 Điểm tương đồng: {res['similarity_score']}")
                print(f"   📄 Nội dung: {res['content_preview']}")
                
                if res['keywords']:
                    keywords_display = ', '.join(res['keywords'][:5])
                    if len(res['keywords']) > 5:
                        keywords_display += f" (+{len(res['keywords'])-5} more)"
                    print(f"   🔑 Từ khóa: {keywords_display}")
                
                if res['entities']:
                    entities_display = ', '.join(res['entities'][:3])
                    if len(res['entities']) > 3:
                        entities_display += f" (+{len(res['entities'])-3} more)"
                    print(f"   🏷️  Thực thể: {entities_display}")
                
                if res['knowledge_triples']:
                    print(f"   🔗 Mối quan hệ chính:")
                    for triple in res['knowledge_triples'][:2]:
                        print(f"      → {triple['subject']} --{triple['predicate']}--> {triple['object']}")
                
                print("-" * 50)
        else:
            print("\\n❌ Không tìm thấy kết quả phù hợp")
            
            # Suggest alternative queries
            suggestions = self.get_query_suggestions(result['query'], 3)
            if suggestions:
                print("\\n💡 Thử các câu hỏi này:")
                for suggestion in suggestions:
                    print(f"   • {suggestion}")
        
        # Display explanation if available
        if result.get('explanation'):
            print(f"\\n💬 Giải thích:\\n{result['explanation']}")
    
    def export_results(self, query: str, results: Dict[str, Any], 
                      file_format: str = "json", output_file: str = None) -> str:
        """Export query results to file"""
        
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            output_file = f"query_results_{safe_query}_{timestamp}.{file_format}"
        
        output_path = Path(self.config.LOG_DIR) / output_file
        
        try:
            if file_format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            elif file_format.lower() == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Query Results for: {query}\\n")
                    f.write("="*50 + "\\n\\n")
                    
                    for result in results.get('results', []):
                        f.write(f"Result #{result['rank']}\\n")
                        f.write(f"Video: {result['video']}\\n")
                        f.write(f"Section: {result['section']}\\n")
                        f.write(f"Score: {result['similarity_score']}\\n")
                        f.write(f"Content: {result['content_preview']}\\n\\n")
            
            logger.info(f"Results exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Vietnamese Video Knowledge Graph Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --create-sample --mode build
  python main.py --mode query --query "Machine learning là gì?" --k 3
  python main.py --mode interactive
  python main.py --mode build --data-dir custom_data/ --force-rebuild
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['build', 'query', 'interactive'], 
                       default='interactive', help='Operation mode')
    
    # Data and configuration
    parser.add_argument('--data-dir', help='Directory containing video files')
    parser.add_argument('--config-env', choices=['development', 'production', 'testing'],
                       default='development', help='Configuration environment')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample data for testing')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of knowledge graph (deletes existing data)')
    
    # Query parameters
    parser.add_argument('--query', type=str, help='Query string for query mode')
    parser.add_argument('--k', type=int, default=5, 
                       help='Number of results to return')
    parser.add_argument('--search-method', choices=['vector', 'knowledge_graph', 'hybrid'],
                       default='hybrid', help='Search method to use')
    parser.add_argument('--no-explain', action='store_true',
                       help='Skip explanation generation')
    
    # Output options
    parser.add_argument('--export', choices=['json', 'txt'],
                       help='Export results to file')
    parser.add_argument('--output-file', help='Output file path for exported results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Get configuration
        config = get_config(args.config_env)
        
        # Override log level if verbose
        if args.verbose:
            config.LOG_LEVEL = "DEBUG"
        
        # Setup logging
        setup_logging(config)
        
        # Create system instance
        logger.info(f"Starting Vietnamese Video Knowledge Graph System in {args.mode} mode")
        system = VideoKnowledgeGraphSystem(config, args.config_env)
        
        # Create sample data if requested
        if args.create_sample:
            logger.info("Creating sample data...")
            sample_dir = create_sample_data()
            args.data_dir = sample_dir
            print(f"✅ Sample data created in: {sample_dir}")
        
        # Execute based on mode
        if args.mode == 'build':
            logger.info("Building knowledge graph...")
            result = system.build_knowledge_graph(
                data_dir=args.data_dir, 
                force_rebuild=args.force_rebuild
            )
            
            if result['success']:
                print("\\n✅ Knowledge Graph Build Completed Successfully!")
                print(f"📊 Build Statistics:")
                build_stats = result['build_stats']
                print(f"   • Videos processed: {build_stats['videos_processed']}")
                print(f"   • Sections processed: {build_stats['sections_processed']}")
                print(f"   • Triples extracted: {build_stats['triples_extracted']}")
                print(f"   • Processing time: {build_stats['processing_time']:.2f}s")
                
                print(f"\\n📈 Graph Statistics:")
                graph_stats = result['graph_stats']
                print(f"   • Videos: {graph_stats['videos_count']}")
                print(f"   • Sections: {graph_stats['sections_count']}")
                print(f"   • Entities: {graph_stats['entities_count']}")
                print(f"   • Relationships: {graph_stats['relationships_count']}")
            else:
                print("\\n❌ Knowledge Graph Build Failed!")
                for error in result['build_stats']['errors']:
                    print(f"   Error: {error}")
        
        elif args.mode == 'query':
            if not args.query:
                print("❌ Error: --query parameter is required for query mode")
                return 1
            
            logger.info(f"Processing query: {args.query}")
            result = system.query_system(
                query=args.query,
                k=args.k,
                search_method=args.search_method,
                explain=not args.no_explain
            )
            
            if result['success']:
                system._display_query_results(result)
                
                # Export results if requested
                if args.export:
                    output_file = system.export_results(
                        args.query, result, args.export, args.output_file
                    )
                    if output_file:
                        print(f"\\n📁 Results exported to: {output_file}")
            else:
                print(f"\\n❌ Query failed: {result.get('error', 'Unknown error')}")
                return 1
        
        elif args.mode == 'interactive':
            system.interactive_mode()
        
        return 0
    
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
