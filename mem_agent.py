# Enhanced Memory Agent - Bug Fixes and Improvements

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import AsyncAzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# FIXED DATA MODELS
# ==============================================================================

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PREFERENCE = "preference"
    ACTIVITY = "activity"
    GOAL = "goal"
    CONSTRAINT = "constraint"

class ActivityType(Enum):
    WORK = "work"
    LEARNING = "learning"
    ENTERTAINMENT = "entertainment"
    COMMUNICATION = "communication"
    CREATION = "creation"
    PLANNING = "planning"
    RESEARCH = "research"
    SOCIAL = "social"

class PreferenceType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"

@dataclass
class UserPreference:
    """Fixed user preference model with proper datetime handling"""
    id: str
    preference_type: PreferenceType
    category: str
    content: str
    strength: float = 0.5
    confidence: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization"""
        return {
            "id": self.id,
            "preference_type": self.preference_type.value,
            "category": self.category,
            "content": self.content,
            "strength": self.strength,
            "confidence": self.confidence,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "decay_rate": self.decay_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreference':
        """Create from dictionary with proper datetime parsing"""
        return cls(
            id=data["id"],
            preference_type=PreferenceType(data["preference_type"]),
            category=data["category"],
            content=data["content"],
            strength=data["strength"],
            confidence=data["confidence"],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"],
            decay_rate=data.get("decay_rate", 0.1)
        )

@dataclass
class UserActivity:
    """Fixed user activity model"""
    id: str
    activity_type: ActivityType
    description: str
    duration: float
    context: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[str] = field(default_factory=list)
    preferences_expressed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization"""
        return {
            "id": self.id,
            "activity_type": self.activity_type.value,
            "description": self.description,
            "duration": self.duration,
            "context": self.context,
            "outcomes": self.outcomes,
            "preferences_expressed": self.preferences_expressed,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserActivity':
        """Create from dictionary with proper datetime parsing"""
        return cls(
            id=data["id"],
            activity_type=ActivityType(data["activity_type"]),
            description=data["description"],
            duration=data["duration"],
            context=data.get("context", {}),
            outcomes=data.get("outcomes", []),
            preferences_expressed=data.get("preferences_expressed", []),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            success_rate=data.get("success_rate", 1.0)
        )

@dataclass
class MemoryNode:
    """Fixed memory node model"""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    strength: float = 1.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    embeddings: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "strength": self.strength,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create from dictionary with proper parsing"""
        return cls(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            strength=data["strength"],
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if isinstance(data["last_accessed"], str) else data["last_accessed"],
            access_count=data["access_count"],
            embeddings=np.array(data["embeddings"]) if data.get("embeddings") else None
        )

# ==============================================================================
# FIXED NEO4J MEMORY STORE
# ==============================================================================

class FixedNeo4jMemoryStore:
    """Fixed Neo4j memory store with proper error handling"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = None
        self.database = database
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_connection(uri, username, password)
        self._create_constraints()
    
    def _initialize_connection(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection with proper error handling"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("✅ Neo4j connection established")
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
                logger.info("✅ Neo4j connection tested successfully")
                
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def _create_constraints(self):
        """Create constraints with proper error handling"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Preference) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON u.id",
            "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON m.user_id",
            "CREATE INDEX IF NOT EXISTS FOR (p:Preference) ON (p.user_id, p.category)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Activity) ON (a.user_id, a.timestamp)"
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for constraint in constraints:
                    session.run(constraint)
        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
    
    def _safe_datetime_convert(self, value: Any) -> str:
        """Safely convert datetime to ISO format string"""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, str):
            # Try to parse and re-format to ensure consistency
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.isoformat()
            except:
                return value
        else:
            return str(value)
    
    def _safe_datetime_parse(self, value: Any) -> datetime:
        """Safely parse datetime from various formats"""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            try:
                # Handle ISO format with Z
                if value.endswith('Z'):
                    value = value[:-1] + '+00:00'
                return datetime.fromisoformat(value)
            except:
                # Fallback to current time if parsing fails
                logger.warning(f"Could not parse datetime: {value}, using current time")
                return datetime.now()
        else:
            return datetime.now()
    
    async def initialize_user_if_not_exists(self, user_id: str):
        """Initialize user with proper error handling"""
        try:
            query = """
            MERGE (u:User {id: $user_id})
            ON CREATE SET u.created_at = datetime()
            RETURN u
            """
            
            with self.driver.session(database=self.database) as session:
                session.run(query, {"user_id": user_id})
                logger.info(f"✅ User {user_id} initialized")
                
        except Exception as e:
            logger.error(f"Error initializing user {user_id}: {e}")
            raise
    
    async def store_user_preference(self, user_id: str, preference: UserPreference):
        """Store user preference with fixed datetime handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode([preference.content])[0]
            
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (p:Preference {id: $pref_id})
            SET p.user_id = $user_id,
                p.preference_type = $pref_type,
                p.category = $category,
                p.content = $content,
                p.strength = $strength,
                p.confidence = $confidence,
                p.context = $context,
                p.created_at = $created_at,
                p.updated_at = $updated_at,
                p.decay_rate = $decay_rate,
                p.embeddings = $embeddings
            MERGE (u)-[:HAS_PREFERENCE]->(p)
            RETURN p
            """
            
            params = {
                "user_id": user_id,
                "pref_id": preference.id,
                "pref_type": preference.preference_type.value,
                "category": preference.category,
                "content": preference.content,
                "strength": preference.strength,
                "confidence": preference.confidence,
                "context": json.dumps(preference.context),
                "created_at": self._safe_datetime_convert(preference.created_at),
                "updated_at": self._safe_datetime_convert(preference.updated_at),
                "decay_rate": preference.decay_rate,
                "embeddings": embeddings.tolist()
            }
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                result.single()
                logger.info(f"✅ Stored preference for user {user_id}: {preference.content}")
                
        except Exception as e:
            logger.error(f"Error storing preference for user {user_id}: {e}")
            raise
    
    async def store_user_activity(self, user_id: str, activity: UserActivity):
        """Store user activity with fixed datetime handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode([activity.description])[0]
            
            query = """
            MERGE (u:User {id: $user_id})
            CREATE (a:Activity {
                id: $activity_id,
                user_id: $user_id,
                activity_type: $activity_type,
                description: $description,
                duration: $duration,
                context: $context,
                outcomes: $outcomes,
                preferences_expressed: $preferences_expressed,
                timestamp: $timestamp,
                success_rate: $success_rate,
                embeddings: $embeddings
            })
            MERGE (u)-[:PERFORMED_ACTIVITY]->(a)
            RETURN a
            """
            
            params = {
                "user_id": user_id,
                "activity_id": activity.id,
                "activity_type": activity.activity_type.value,
                "description": activity.description,
                "duration": activity.duration,
                "context": json.dumps(activity.context),
                "outcomes": activity.outcomes,
                "preferences_expressed": activity.preferences_expressed,
                "timestamp": self._safe_datetime_convert(activity.timestamp),
                "success_rate": activity.success_rate,
                "embeddings": embeddings.tolist()
            }
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                result.single()
                logger.info(f"✅ Stored activity for user {user_id}: {activity.description}")
                
        except Exception as e:
            logger.error(f"Error storing activity for user {user_id}: {e}")
            raise
    
    async def store_memory_node(self, user_id: str, memory: MemoryNode):
        """Store memory node with fixed datetime handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            # Generate embeddings if not present
            if memory.embeddings is None:
                content_text = ""
                if memory.content.get('text'):
                    content_text = memory.content['text']
                elif memory.content.get('user_message'):
                    content_text = memory.content['user_message']
                elif memory.content.get('description'):
                    content_text = memory.content['description']
                
                if content_text:
                    memory.embeddings = self.embeddings_model.encode([content_text])[0]
                else:
                    memory.embeddings = np.zeros(384)  # Default embedding size
            
            query = """
            MERGE (u:User {id: $user_id})
            CREATE (m:Memory {
                id: $memory_id,
                user_id: $user_id,
                memory_type: $memory_type,
                content: $content,
                strength: $strength,
                last_accessed: $last_accessed,
                access_count: $access_count,
                embeddings: $embeddings
            })
            MERGE (u)-[:HAS_MEMORY]->(m)
            RETURN m
            """
            
            params = {
                "user_id": user_id,
                "memory_id": memory.id,
                "memory_type": memory.memory_type.value,
                "content": json.dumps(memory.content),
                "strength": memory.strength,
                "last_accessed": self._safe_datetime_convert(memory.last_accessed),
                "access_count": memory.access_count,
                "embeddings": memory.embeddings.tolist() if memory.embeddings is not None else []
            }
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                result.single()
                logger.info(f"✅ Stored memory for user {user_id}: {memory.memory_type.value}")
                
        except Exception as e:
            logger.error(f"Error storing memory for user {user_id}: {e}")
            raise
    
    async def get_user_preferences(self, user_id: str, category: str = None) -> List[UserPreference]:
        """Get user preferences with proper error handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_PREFERENCE]->(p:Preference)
            WHERE ($category IS NULL OR p.category = $category)
            RETURN p
            ORDER BY p.strength DESC, p.updated_at DESC
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {"user_id": user_id, "category": category})
                preferences = []
                
                for record in result:
                    try:
                        p = record["p"]
                        preferences.append(UserPreference(
                            id=p["id"],
                            preference_type=PreferenceType(p["preference_type"]),
                            category=p["category"],
                            content=p["content"],
                            strength=p["strength"],
                            confidence=p["confidence"],
                            context=json.loads(p["context"]) if p["context"] else {},
                            created_at=self._safe_datetime_parse(p["created_at"]),
                            updated_at=self._safe_datetime_parse(p["updated_at"]),
                            decay_rate=p["decay_rate"]
                        ))
                    except Exception as e:
                        logger.warning(f"Error parsing preference record: {e}")
                        continue
                
                logger.info(f"✅ Retrieved {len(preferences)} preferences for user {user_id}")
                return preferences
                
        except Exception as e:
            logger.error(f"Error getting preferences for user {user_id}: {e}")
            return []
    
    async def get_recent_activities(self, user_id: str, hours: int = 24) -> List[UserActivity]:
        """Get recent activities with proper error handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = """
            MATCH (u:User {id: $user_id})-[:PERFORMED_ACTIVITY]->(a:Activity)
            WHERE a.timestamp >= $cutoff_time
            RETURN a
            ORDER BY a.timestamp DESC
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {
                    "user_id": user_id,
                    "cutoff_time": self._safe_datetime_convert(cutoff_time)
                })
                activities = []
                
                for record in result:
                    try:
                        a = record["a"]
                        activities.append(UserActivity(
                            id=a["id"],
                            activity_type=ActivityType(a["activity_type"]),
                            description=a["description"],
                            duration=a["duration"],
                            context=json.loads(a["context"]) if a["context"] else {},
                            outcomes=a["outcomes"],
                            preferences_expressed=a["preferences_expressed"],
                            timestamp=self._safe_datetime_parse(a["timestamp"]),
                            success_rate=a["success_rate"]
                        ))
                    except Exception as e:
                        logger.warning(f"Error parsing activity record: {e}")
                        continue
                
                logger.info(f"✅ Retrieved {len(activities)} activities for user {user_id}")
                return activities
                
        except Exception as e:
            logger.error(f"Error getting activities for user {user_id}: {e}")
            return []
    
    async def get_user_memories(self, user_id: str, memory_type: MemoryType = None, limit: int = 10) -> List[MemoryNode]:
        """Get user memories with proper error handling"""
        try:
            await self.initialize_user_if_not_exists(user_id)
            
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE ($memory_type IS NULL OR m.memory_type = $memory_type)
            RETURN m
            ORDER BY m.strength DESC, m.last_accessed DESC
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {
                    "user_id": user_id,
                    "memory_type": memory_type.value if memory_type else None,
                    "limit": limit
                })
                memories = []
                
                for record in result:
                    try:
                        m = record["m"]
                        memories.append(MemoryNode(
                            id=m["id"],
                            memory_type=MemoryType(m["memory_type"]),
                            content=json.loads(m["content"]) if m["content"] else {},
                            strength=m["strength"],
                            last_accessed=self._safe_datetime_parse(m["last_accessed"]),
                            access_count=m["access_count"],
                            embeddings=np.array(m["embeddings"]) if m.get("embeddings") else None
                        ))
                    except Exception as e:
                        logger.warning(f"Error parsing memory record: {e}")
                        continue
                
                logger.info(f"✅ Retrieved {len(memories)} memories for user {user_id}")
                return memories
                
        except Exception as e:
            logger.error(f"Error getting memories for user {user_id}: {e}")
            return []
    
    async def update_preference_strength(self, user_id: str, preference_id: str, new_strength: float):
        """Update preference strength with proper error handling"""
        try:
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_PREFERENCE]->(p:Preference {id: $preference_id})
            SET p.strength = $new_strength, p.updated_at = $updated_at
            RETURN p
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {
                    "user_id": user_id,
                    "preference_id": preference_id,
                    "new_strength": new_strength,
                    "updated_at": self._safe_datetime_convert(datetime.now())
                })
                
                if result.single():
                    logger.info(f"✅ Updated preference strength for user {user_id}")
                else:
                    logger.warning(f"⚠️ No preference found to update for user {user_id}")
                    
        except Exception as e:
            logger.error(f"Error updating preference strength for user {user_id}: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("🔚 Neo4j connection closed")

# ==============================================================================
# SIMPLE MEMORY AGENT FOR TESTING
# ==============================================================================

class SimpleMemoryAgent:
    """Simplified memory agent for testing and demonstration"""
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        self.memory_store = FixedNeo4jMemoryStore(neo4j_uri, neo4j_username, neo4j_password)
        self.current_user_id = "default_user"
    
    async def process_message(self, message: str, user_id: str = None) -> str:
        """Process a message and return response"""
        user_id = user_id or self.current_user_id
        
        try:
            # Store the message as a memory
            memory = MemoryNode(
                id=str(uuid.uuid4()),
                memory_type=MemoryType.EPISODIC,
                content={"user_message": message, "timestamp": datetime.now().isoformat()},
                strength=1.0
            )
            await self.memory_store.store_memory_node(user_id, memory)
            
            # Check for preferences in the message
            await self._extract_and_store_preferences(user_id, message)
            
            # Check for activities in the message
            await self._extract_and_store_activities(user_id, message)
            
            # Generate response based on context
            response = await self._generate_response(user_id, message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error processing your message: {str(e)}"
    
    async def _extract_and_store_preferences(self, user_id: str, message: str):
        """Extract and store preferences from message"""
        message_lower = message.lower()
        
        # Simple preference detection
        if "prefer" in message_lower or "like" in message_lower:
            preference = UserPreference(
                id=str(uuid.uuid4()),
                preference_type=PreferenceType.EXPLICIT,
                category="stated_preference",
                content=message,
                strength=0.8,
                confidence=0.9
            )
            await self.memory_store.store_user_preference(user_id, preference)
            logger.info(f"✅ Extracted preference: {message}")
        
        # Detect detailed explanations preference
        if "detailed" in message_lower and "explanation" in message_lower:
            preference = UserPreference(
                id=str(uuid.uuid4()),
                preference_type=PreferenceType.EXPLICIT,
                category="response_style",
                content="Prefers detailed explanations with examples",
                strength=0.9,
                confidence=0.95
            )
            await self.memory_store.store_user_preference(user_id, preference)
            logger.info("✅ Extracted detailed explanation preference")
        
        # Detect hands-on learning preference
        if "hands-on" in message_lower and "learn" in message_lower:
            preference = UserPreference(
                id=str(uuid.uuid4()),
                preference_type=PreferenceType.EXPLICIT,
                category="learning_style",
                content="Prefers hands-on practice for learning",
                strength=0.85,
                confidence=0.9
            )
            await self.memory_store.store_user_preference(user_id, preference)
            logger.info("✅ Extracted hands-on learning preference")
    
    async def _extract_and_store_activities(self, user_id: str, message: str):
        """Extract and store activities from message"""
        message_lower = message.lower()
        
        # Detect work activities
        if "working on" in message_lower or "project" in message_lower:
            activity = UserActivity(
                id=str(uuid.uuid4()),
                activity_type=ActivityType.WORK,
                description=message,
                duration=0.0,
                success_rate=1.0,
                context={"extracted_from": "message"}
            )
            await self.memory_store.store_user_activity(user_id, activity)
            logger.info(f"✅ Extracted work activity: {message}")
        
        # Detect learning activities
        if "learn" in message_lower or "study" in message_lower:
            activity = UserActivity(
                id=str(uuid.uuid4()),
                activity_type=ActivityType.LEARNING,
                description=message,
                duration=0.0,
                success_rate=1.0,
                context={"extracted_from": "message"}
            )
            await self.memory_store.store_user_activity(user_id, activity)
            logger.info(f"✅ Extracted learning activity: {message}")
    
    async def _generate_response(self, user_id: str, message: str) -> str:
        """Generate response based on user context"""
        try:
            # Get user preferences and activities
            preferences = await self.memory_store.get_user_preferences(user_id)
            activities = await self.memory_store.get_recent_activities(user_id)
            memories = await self.memory_store.get_user_memories(user_id)
            
            # Check if asking about name
            if "name" in message.lower():
                # Look for name in memories
                for memory in memories:
                    if memory.content.get("user_message") and "name" in memory.content["user_message"].lower():
                        if "my name is" in memory.content["user_message"].lower():
                            name = memory.content["user_message"].split("my name is")[-1].strip()
                            return f"Yes, your name is {name}!"
                
                return "I don't seem to have your name on record yet. If you'd like, you can share it with me, and I'll make sure to personalize my responses for you moving forward!"
            
            # Check if asking about preferences
            if "preference" in message.lower():
                if preferences:
                    pref_list = []
                    for pref in preferences[:5]:  # Top 5 preferences
                        pref_list.append(f"- {pref.content} (strength: {pref.strength:.2f})")
                    return f"Here are your preferences I've learned:\n" + "\n".join(pref_list)
                else:
                    return "I haven't learned any specific preferences yet. Share what you like and I'll remember it!"
            
            # Check if asking about activities
            if "activity" in message.lower() or "activities" in message.lower():
                if activities:
                    activity_list = []
                    for activity in activities[:5]:  # Recent 5 activities
                        activity_list.append(f"- {activity.description} ({activity.activity_type.value})")
                    return f"Here are your recent activities:\n" + "\n".join(activity_list)
                else:
                    return "I haven't recorded any activities yet. Tell me what you're working on!"
            
            # Response based on preferences
            response_style = "standard"
            for pref in preferences:
                if "detailed" in pref.content.lower():
                    response_style = "detailed"
                elif "brief" in pref.content.lower():
                    response_style = "brief"
            
            # Generate contextual response
            if "machine learning" in message.lower():
                if response_style == "detailed":
                    return """That's exciting! Based on your interest in machine learning, I'd love to help you make progress on your project. Could you share a bit more about what you're working on? Are you building a model, analyzing data, or perhaps fine-tuning an existing algorithm? 

For example, if you're working on:
- **Classification tasks**: I can help with feature selection, model evaluation metrics, and hyperparameter tuning
- **Data preprocessing**: We can discuss techniques like normalization, handling missing values, and feature engineering
- **Model optimization**: I can suggest approaches for improving accuracy, reducing overfitting, or speeding up training

If you're facing any specific challenges—like dealing with imbalanced datasets, debugging code, or choosing the right algorithm—I can provide tailored advice or resources. What aspect would you like to focus on first?"""
                else:
                    return "That's exciting! Based on your interest in machine learning, I'd love to help you make progress on your project. Could you share a bit more about what you're working on? Are you building a model, analyzing data, or perhaps fine-tuning an existing algorithm? If you're facing any challenges—like feature selection, hyperparameter tuning, or debugging code—I can provide tailored advice or resources."
            
            # Default responses based on message patterns
            if "thank" in message.lower() or "perfect" in message.lower() or "exactly" in message.lower():
                # Reinforce preferences when user gives positive feedback
                for pref in preferences:
                    if pref.strength < 1.0:
                        new_strength = min(1.0, pref.strength * 1.1)
                        await self.memory_store.update_preference_strength(user_id, pref.id, new_strength)
                
                return "I'm glad I could help! I've noted your positive feedback and will continue to tailor my responses to your preferences."
            
            if "context" in message.lower():
                context_info = []
                context_info.append(f"📊 **Your Profile Summary:**")
                context_info.append(f"- Preferences learned: {len(preferences)}")
                context_info.append(f"- Recent activities: {len(activities)}")
                context_info.append(f"- Memory entries: {len(memories)}")
                
                if preferences:
                    context_info.append(f"\n**Top Preferences:**")
                    for pref in preferences[:3]:
                        context_info.append(f"- {pref.content} (strength: {pref.strength:.2f})")
                
                if activities:
                    context_info.append(f"\n**Recent Activities:**")
                    for activity in activities[:3]:
                        context_info.append(f"- {activity.description}")
                
                return "\n".join(context_info)
            
            # Default response
            return "I'm here to help and learn about your preferences! Feel free to share what you're working on, what you like, or ask me about what I know about you so far."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm here to help! What would you like to talk about?"
    
    async def get_user_summary(self, user_id: str = None) -> Dict[str, Any]:
        """Get a summary of what we know about the user"""
        user_id = user_id or self.current_user_id
        
        try:
            preferences = await self.memory_store.get_user_preferences(user_id)
            activities = await self.memory_store.get_recent_activities(user_id, hours=168)  # Past week
            memories = await self.memory_store.get_user_memories(user_id)
            
            return {
                "user_id": user_id,
                "preferences": {
                    "count": len(preferences),
                    "categories": list(set([p.category for p in preferences])),
                    "top_preferences": [
                        {"content": p.content, "strength": p.strength, "category": p.category}
                        for p in preferences[:5]
                    ]
                },
                "activities": {
                    "count": len(activities),
                    "types": list(set([a.activity_type.value for a in activities])),
                    "recent": [
                        {"description": a.description, "type": a.activity_type.value, "timestamp": a.timestamp.isoformat()}
                        for a in activities[:5]
                    ]
                },
                "memories": {
                    "count": len(memories),
                    "types": list(set([m.memory_type.value for m in memories])),
                    "recent": [
                        {"content": m.content, "type": m.memory_type.value, "strength": m.strength}
                        for m in memories[:5]
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error getting user summary: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the memory store"""
        self.memory_store.close()

# ==============================================================================
# TESTING FUNCTIONS
# ==============================================================================

async def test_fixed_memory_agent():
    """Test the fixed memory agent"""
    print("🚀 Testing Fixed Memory Agent...")
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "dataforge123"
    
    agent = None
    try:
        # Initialize agent
        agent = SimpleMemoryAgent(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        test_user = "test_user_fixed"
        
        print("✅ Agent initialized successfully!")
        
        # Test conversations
        test_messages = [
            "Hi, my name is veera",
            "I prefer detailed explanations with examples",
            "I'm currently working on a prompt engineering",
            "I like to learn through hands-on practice",
            "do you know my name?",
            "what are my preferences?",
            "tell me about my context",
            "That explanation was perfect, exactly what I needed!",
            "what activities have I mentioned?",
            "Can you help me with top prompting techniques?"
            "what are my preferences"
        ]
        
        print("\n💬 Testing conversations...")
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"User: {message}")
            
            try:
                response = await agent.process_message(message, test_user)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            # Small delay
            await asyncio.sleep(0.5)
        
        # Get user summary
        print("\n📊 User Summary:")
        try:
            summary = await agent.get_user_summary(test_user)
            print(json.dumps(summary, indent=2, default=str))
        except Exception as e:
            print(f"❌ Error getting summary: {e}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if agent:
            agent.close()
            print("\n🔚 Agent closed successfully")

# ==============================================================================
# DIAGNOSTIC FUNCTIONS
# ==============================================================================

async def diagnose_memory_issues():
    """Diagnose memory-related issues"""
    print("🔍 Diagnosing Memory Issues...")
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "dataforge123"
    
    memory_store = None
    try:
        # Test Neo4j connection
        print("1. Testing Neo4j connection...")
        memory_store = FixedNeo4jMemoryStore(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        print("✅ Neo4j connection successful")
        
        # Test user initialization
        print("2. Testing user initialization...")
        test_user = "diagnostic_user"
        await memory_store.initialize_user_if_not_exists(test_user)
        print("✅ User initialization successful")
        
        # Test preference storage
        print("3. Testing preference storage...")
        test_pref = UserPreference(
            id=str(uuid.uuid4()),
            preference_type=PreferenceType.EXPLICIT,
            category="test",
            content="Test preference content",
            strength=0.8
        )
        await memory_store.store_user_preference(test_user, test_pref)
        print("✅ Preference storage successful")
        
        # Test preference retrieval
        print("4. Testing preference retrieval...")
        prefs = await memory_store.get_user_preferences(test_user)
        print(f"✅ Retrieved {len(prefs)} preferences")
        
        # Test activity storage
        print("5. Testing activity storage...")
        test_activity = UserActivity(
            id=str(uuid.uuid4()),
            activity_type=ActivityType.WORK,
            description="Test activity",
            duration=1.0
        )
        await memory_store.store_user_activity(test_user, test_activity)
        print("✅ Activity storage successful")
        
        # Test activity retrieval
        print("6. Testing activity retrieval...")
        activities = await memory_store.get_recent_activities(test_user)
        print(f"✅ Retrieved {len(activities)} activities")
        
        # Test memory storage
        print("7. Testing memory storage...")
        test_memory = MemoryNode(
            id=str(uuid.uuid4()),
            memory_type=MemoryType.EPISODIC,
            content={"test": "memory content"},
            strength=0.9
        )
        await memory_store.store_memory_node(test_user, test_memory)
        print("✅ Memory storage successful")
        
        # Test memory retrieval
        print("8. Testing memory retrieval...")
        memories = await memory_store.get_user_memories(test_user)
        print(f"✅ Retrieved {len(memories)} memories")
        
        # Test embeddings
        print("9. Testing embeddings...")
        test_text = "This is a test sentence for embeddings"
        embedding = memory_store.embeddings_model.encode([test_text])[0]
        print(f"✅ Generated embedding with shape: {embedding.shape}")
        
        print("\n✅ All diagnostic tests passed!")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if memory_store:
            memory_store.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

async def main():
    """Main execution function"""
    print("=" * 60)
    print("🧠 ENHANCED MEMORY AGENT - FIXES & IMPROVEMENTS")
    print("=" * 60)
    
    # Run diagnostics first
    await diagnose_memory_issues()
    
    print("\n" + "=" * 60)
    
    # Run the fixed memory agent test
    await test_fixed_memory_agent()

if __name__ == "__main__":
    asyncio.run(main())