"""
Azure OpenAI Memory Agent - Fixed ChromaDB Compatibility
A sophisticated memory-enhanced LLM agent using Azure OpenAI with four types of memory:
1. Working Memory - Current conversation context
2. Episodic Memory - Historical experiences and learnings
3. Semantic Memory - Factual knowledge base
4. Procedural Memory - Behavioral guidelines and rules

FIXED: ChromaDB metadata compatibility issues
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
import uuid

# Core dependencies
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Azure OpenAI
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Document processing
import pypdf
from docx import Document
import openpyxl

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureMemoryAgent:
    """
    A robust memory-enhanced LLM agent using Azure OpenAI
    """
    
    def __init__(self):
        self.setup_azure_client()
        self.setup_embedding_model()
        self.setup_vector_database()
        self.setup_memory_stores()
        self.setup_reflection_chain()
        
        # Memory state
        self.current_conversation = []
        self.conversations_history = []
        self.what_worked = set()
        self.what_to_avoid = set()
        self.user_preferences = {}
        
        logger.info("Azure Memory Agent initialized successfully")
    
    def setup_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            # Azure OpenAI client for direct API calls
            self.azure_client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
            
            # LangChain Azure OpenAI client
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                model_name=os.getenv('AZURE_OPENAI_MODEL_NAME'),
                temperature=float(os.getenv('AZURE_OPENAI_TEMPERATURE', 0.1)),
                max_tokens=int(os.getenv('AZURE_OPENAI_MAX_TOKENS', 4000))
            )
            
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def setup_embedding_model(self):
        """Initialize embedding model for semantic similarity"""
        try:
            model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a smaller model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded fallback embedding model")
    
    def setup_vector_database(self):
        """Initialize ChromaDB for vector storage"""
        try:
            # Create data directory
            data_dir = Path("./memory_data")
            data_dir.mkdir(exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(data_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections for different memory types
            self.episodic_collection = self.get_or_create_collection(
                "episodic_memory",
                "Historical conversations and learnings"
            )
            
            self.semantic_collection = self.get_or_create_collection(
                "semantic_memory", 
                "Factual knowledge and documents"
            )
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def get_or_create_collection(self, name: str, description: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.chroma_client.get_collection(name)
            logger.info(f"Retrieved existing collection: {name}")
            return collection
        except:
            collection = self.chroma_client.create_collection(
                name=name,
                metadata={"description": description}
            )
            logger.info(f"Created new collection: {name}")
            return collection
    
    def setup_memory_stores(self):
        """Initialize memory storage files"""
        # Create memory directory
        self.memory_dir = Path("./memory_stores")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Procedural memory file
        self.procedural_memory_file = self.memory_dir / "procedural_memory.txt"
        if not self.procedural_memory_file.exists():
            self.initialize_procedural_memory()
        
        # User preferences file
        self.user_preferences_file = self.memory_dir / "user_preferences.json"
        if self.user_preferences_file.exists():
            with open(self.user_preferences_file, 'r') as f:
                self.user_preferences = json.load(f)
        
        logger.info("Memory stores initialized")
    
    def initialize_procedural_memory(self):
        """Initialize procedural memory with default guidelines"""
        default_guidelines = """
1. Maintain a helpful, friendly, and professional tone throughout all interactions
2. Remember and use the user's name when provided to personalize the conversation
3. Ask clarifying questions when requests are ambiguous or unclear
4. Provide structured, well-organized responses for complex topics
5. Acknowledge when you don't have sufficient information and offer alternatives
6. Adapt your communication style to match the user's preferences over time
7. Keep track of user preferences and reference them in future interactions
8. Avoid repeating the same mistakes or providing contradictory information
9. Build on previous conversations to create continuity and context
10. Continuously learn from interactions to improve response quality
""".strip()
        
        with open(self.procedural_memory_file, 'w') as f:
            f.write(default_guidelines)
        
        logger.info("Procedural memory initialized with default guidelines")
    
    def setup_reflection_chain(self):
        """Setup the reflection chain for analyzing conversations"""
        reflection_template = """
You are analyzing a conversation to create a memory that will help improve future interactions. 
Extract key insights that would be valuable for similar future conversations.

Review the conversation and create a structured reflection following these guidelines:

1. Use "N/A" for any field where information is insufficient or irrelevant
2. Be concise and actionable - focus on practical insights
3. Context tags should be specific enough to match similar situations
4. All insights should be applicable to future conversations

Output valid JSON in this exact format:
{{
    "context_tags": [
        "tag1", "tag2", "tag3"
    ],
    "conversation_summary": "Brief summary of what was accomplished",
    "what_worked": "Most effective strategies used",
    "what_to_avoid": "Important pitfalls or ineffective approaches",
    "user_preferences": "Any discovered user preferences or communication style",
    "key_insights": "Additional valuable insights for future interactions"
}}

Examples of good context tags:
- ["name_introduction", "personal_info", "greeting"]
- ["technical_question", "explanation_request", "step_by_step"]
- ["preference_setting", "customization", "user_feedback"]

Conversation to analyze:
{conversation}
"""
        
        self.reflection_prompt = ChatPromptTemplate.from_template(reflection_template)
        self.reflection_chain = self.reflection_prompt | self.llm | JsonOutputParser()
        
        logger.info("Reflection chain setup complete")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def format_conversation(self, messages: List) -> str:
        """Format conversation messages for storage and analysis"""
        formatted = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                role = msg.__class__.__name__.replace("Message", "").upper()
                formatted.append(f"{role}: {msg.content}")
            else:
                formatted.append(str(msg))
        return "\n".join(formatted)
    
    def add_to_working_memory(self, message):
        """Add message to current working memory"""
        self.current_conversation.append(message)
        
        # Limit working memory size to prevent context overflow
        max_working_memory = 20  # Adjust based on your needs
        if len(self.current_conversation) > max_working_memory:
            self.current_conversation = self.current_conversation[-max_working_memory:]
    
    def save_episodic_memory(self, conversation_id: str = None):
        """Save current conversation to episodic memory"""
        if not self.current_conversation:
            return
        
        try:
            # Format conversation for analysis
            formatted_conversation = self.format_conversation(self.current_conversation)
            
            # Generate reflection
            reflection = self.reflection_chain.invoke({"conversation": formatted_conversation})
            
            # Generate unique ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Update user preferences
            if reflection.get('user_preferences') and reflection['user_preferences'] != 'N/A':
                self.user_preferences.update({
                    'last_updated': datetime.now().isoformat(),
                    'preferences': reflection['user_preferences']
                })
                self.save_user_preferences()
            
            # Update learning sets
            if reflection.get('what_worked') and reflection['what_worked'] != 'N/A':
                self.what_worked.update(reflection['what_worked'].split('. '))
            
            if reflection.get('what_to_avoid') and reflection['what_to_avoid'] != 'N/A':
                self.what_to_avoid.update(reflection['what_to_avoid'].split('. '))
            
            # Prepare for vector storage
            search_text = f"""
            Tags: {' '.join(reflection.get('context_tags', []))}
            Summary: {reflection.get('conversation_summary', '')}
            What worked: {reflection.get('what_worked', '')}
            Insights: {reflection.get('key_insights', '')}
            """.strip()
            
            # Generate embedding
            embedding = self.embed_text(search_text)
            
            # Convert context_tags list to string for ChromaDB compatibility
            context_tags_str = ', '.join(reflection.get('context_tags', []))
            
            # Store in episodic memory with ChromaDB-compatible metadata
            self.episodic_collection.add(
                documents=[formatted_conversation],
                embeddings=[embedding],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "context_tags": context_tags_str,  # Convert list to string
                    "conversation_summary": reflection.get('conversation_summary', ''),
                    "what_worked": reflection.get('what_worked', ''),
                    "what_to_avoid": reflection.get('what_to_avoid', ''),
                    "user_preferences": reflection.get('user_preferences', ''),
                    "key_insights": reflection.get('key_insights', '')
                }],
                ids=[conversation_id]
            )
            
            # Add to conversation history
            self.conversations_history.append(formatted_conversation)
            
            logger.info(f"Conversation saved to episodic memory: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")
    
    def retrieve_episodic_memory(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant episodic memories"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search episodic memory
            results = self.episodic_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    # Convert context_tags string back to list for processing
                    if 'context_tags' in metadata and metadata['context_tags']:
                        metadata['context_tags'] = [tag.strip() for tag in metadata['context_tags'].split(',')]
                    
                    memories.append({
                        'conversation': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance  # Convert distance to similarity
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve episodic memory: {e}")
            return []
    
    def add_to_semantic_memory(self, content: str, metadata: Dict[str, Any]):
        """Add content to semantic memory"""
        try:
            # Generate embedding
            embedding = self.embed_text(content)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Ensure all metadata values are ChromaDB-compatible
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    clean_metadata[key] = value
                elif isinstance(value, list):
                    clean_metadata[key] = ', '.join(str(item) for item in value)
                else:
                    clean_metadata[key] = str(value)
            
            # Add timestamp and doc_id
            clean_metadata.update({
                'timestamp': datetime.now().isoformat(),
                'doc_id': doc_id
            })
            
            # Add to semantic memory
            self.semantic_collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Content added to semantic memory: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add to semantic memory: {e}")
            return None
    
    def retrieve_semantic_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant semantic memories"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search semantic memory
            results = self.semantic_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    memories.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve semantic memory: {e}")
            return []
    
    def load_procedural_memory(self) -> str:
        """Load procedural memory guidelines"""
        try:
            with open(self.procedural_memory_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load procedural memory: {e}")
            return ""
    
    def update_procedural_memory(self):
        """Update procedural memory based on accumulated learnings"""
        try:
            current_guidelines = self.load_procedural_memory()
            
            update_prompt = f"""
You are updating behavioral guidelines for an AI assistant based on accumulated learnings.
Current guidelines:
{current_guidelines}

New learnings:
What has worked well: {'; '.join(self.what_worked)}
What to avoid: {'; '.join(self.what_to_avoid)}

Please provide updated guidelines that:
1. Incorporate the most valuable existing guidelines
2. Add new insights from recent learnings
3. Are specific and actionable
4. Are ordered by importance
5. Do not exceed 12 guidelines

Format as a numbered list with brief explanations.
Return only the updated guidelines, no preamble.
"""
            
            response = self.llm.invoke([HumanMessage(content=update_prompt)])
            
            # Save updated guidelines
            with open(self.procedural_memory_file, 'w') as f:
                f.write(response.content)
            
            logger.info("Procedural memory updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update procedural memory: {e}")
    
    def save_user_preferences(self):
        """Save user preferences to file"""
        try:
            with open(self.user_preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
    
    def generate_enhanced_system_prompt(self, user_query: str) -> str:
        """Generate system prompt enhanced with memory context"""
        try:
            # Retrieve relevant episodic memories
            episodic_memories = self.retrieve_episodic_memory(user_query, limit=2)
            
            # Retrieve relevant semantic memories
            semantic_memories = self.retrieve_semantic_memory(user_query, limit=3)
            
            # Load procedural memory
            procedural_guidelines = self.load_procedural_memory()
            
            # Build enhanced system prompt
            prompt_parts = [
                "You are a helpful AI assistant with access to memory systems that help you provide personalized, contextual responses.",
                ""
            ]
            
            # Add procedural memory
            if procedural_guidelines:
                prompt_parts.extend([
                    "BEHAVIORAL GUIDELINES:",
                    procedural_guidelines,
                    ""
                ])
            
            # Add user preferences
            if self.user_preferences:
                prompt_parts.extend([
                    "USER PREFERENCES:",
                    json.dumps(self.user_preferences, indent=2),
                    ""
                ])
            
            # Add episodic memories
            if episodic_memories:
                prompt_parts.append("RELEVANT PAST INTERACTIONS:")
                for i, memory in enumerate(episodic_memories, 1):
                    metadata = memory['metadata']
                    prompt_parts.extend([
                        f"Memory {i}:",
                        f"- Summary: {metadata.get('conversation_summary', 'N/A')}",
                        f"- What worked: {metadata.get('what_worked', 'N/A')}",
                        f"- What to avoid: {metadata.get('what_to_avoid', 'N/A')}",
                        f"- Key insights: {metadata.get('key_insights', 'N/A')}",
                        ""
                    ])
            
            # Add semantic memories
            if semantic_memories:
                prompt_parts.append("RELEVANT KNOWLEDGE:")
                for i, memory in enumerate(semantic_memories, 1):
                    prompt_parts.extend([
                        f"Source {i}: {memory['content'][:200]}...",
                        ""
                    ])
            
            # Add final instructions
            prompt_parts.extend([
                "Use this context to provide helpful, personalized responses that:",
                "1. Build on previous interactions and learnings",
                "2. Respect user preferences and communication style",
                "3. Are grounded in relevant knowledge when appropriate",
                "4. Follow the behavioral guidelines consistently",
                ""
            ])
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced system prompt: {e}")
            return "You are a helpful AI assistant."
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        try:
            # Add user input to working memory
            user_message = HumanMessage(content=user_input)
            self.add_to_working_memory(user_message)
            
            # Generate enhanced system prompt
            system_prompt = self.generate_enhanced_system_prompt(user_input)
            system_message = SystemMessage(content=system_prompt)
            
            # Prepare messages for LLM
            messages = [system_message] + self.current_conversation[-10:]  # Last 10 messages
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Add response to working memory
            self.add_to_working_memory(response)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def start_new_conversation(self):
        """Start a new conversation, saving the previous one"""
        if self.current_conversation:
            self.save_episodic_memory()
        
        self.current_conversation = []
        logger.info("Started new conversation")
    
    def end_conversation(self):
        """End current conversation and save to memory"""
        if self.current_conversation:
            self.save_episodic_memory()
            self.update_procedural_memory()
        
        logger.info("Conversation ended and saved to memory")
    
    def load_document(self, file_path: str, doc_type: str = "general") -> bool:
        """Load document into semantic memory"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            content = ""
            
            # Extract content based on file type
            if file_path.suffix.lower() == '.pdf':
                content = self.extract_pdf_content(file_path)
            elif file_path.suffix.lower() in ['.txt']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                content = self.extract_docx_content(file_path)
            elif file_path.suffix.lower() in ['.csv']:
                content = self.extract_csv_content(file_path)
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if not content:
                logger.error("No content extracted from file")
                return False
            
            # Chunk content for better retrieval
            chunks = self.chunk_content(content)
            
            # Add chunks to semantic memory
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source_file': str(file_path),
                    'doc_type': doc_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                self.add_to_semantic_memory(chunk, metadata)
            
            logger.info(f"Document loaded successfully: {file_path} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            return False
    
    def extract_pdf_content(self, file_path: Path) -> str:
        """Extract content from PDF file"""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                return content.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return ""
    
    def extract_docx_content(self, file_path: Path) -> str:
        """Extract content from DOCX file"""
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to extract DOCX content: {e}")
            return ""
    
    def extract_csv_content(self, file_path: Path) -> str:
        """Extract content from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Failed to extract CSV content: {e}")
            return ""
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk content for better retrieval"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence ending
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunks.append(content[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            episodic_count = self.episodic_collection.count()
            semantic_count = self.semantic_collection.count()
            
            return {
                'episodic_memories': episodic_count,
                'semantic_memories': semantic_count,
                'working_memory_size': len(self.current_conversation),
                'conversation_history': len(self.conversations_history),
                'what_worked_count': len(self.what_worked),
                'what_to_avoid_count': len(self.what_to_avoid),
                'user_preferences': bool(self.user_preferences)
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

# Usage example and CLI interface
if __name__ == "__main__":
    # Initialize the memory agent
    agent = AzureMemoryAgent()
    
    print("🧠 Azure Memory Agent Initialized!")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                agent.end_conversation()
                print("👋 Goodbye! Your memories have been saved.")
                break
            
            elif user_input.lower() == 'help':
                print("""
Available commands:
- 'new' - Start a new conversation
- 'stats' - Show memory statistics
- 'load <file_path>' - Load document into semantic memory
- 'quit' - Exit and save memories
- Just type normally to chat!
                """)
                continue
            
            elif user_input.lower() == 'new':
                agent.start_new_conversation()
                print("🔄 Started new conversation")
                continue
            
            elif user_input.lower() == 'stats':
                stats = agent.get_memory_stats()
                print("📊 Memory Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            elif user_input.lower().startswith('load '):
                file_path = user_input[5:].strip()
                if agent.load_document(file_path):
                    print(f"✅ Document loaded successfully: {file_path}")
                else:
                    print(f"❌ Failed to load document: {file_path}")
                continue
            
            # Process normal conversation
            response = agent.process_user_input(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            agent.end_conversation()
            print("\n👋 Goodbye! Your memories have been saved.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print("❌ An error occurred. Please try again.")