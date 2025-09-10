import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import time
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import math
from enum import Enum

# Core dependencies
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LangChain Knowledge Graph components
from langchain_openai import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# LangChain core messages for conversation history
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Neo4j integration
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    print("⚠️ Neo4j not available. Install with: pip install neo4j")
    NEO4J_AVAILABLE = False

# Entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("⚠️ SpaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False

# Document processing
import chardet

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Context Engineering Framework ---

class ContextType(Enum):
    """Types of context used in the system"""
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"

@dataclass
class ContextVector:
    """Represents a context vector with metadata"""
    vector_id: str
    context_type: ContextType
    content: Any
    timestamp: datetime
    confidence: float
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class CrossMemoryCorrelation:
    """Represents a correlation between different memory types"""
    correlation_id: str
    memory_types: List[str]
    correlation_strength: float
    pattern_description: str
    supporting_evidence: List[Dict[str, Any]]
    confidence_score: float
    temporal_range: Tuple[datetime, datetime]
    user_id: str

@dataclass
class InsightPattern:
    """Represents an automatically generated insight"""
    insight_id: str
    pattern_type: str
    description: str
    supporting_data: List[Dict[str, Any]]
    confidence_score: float
    actionable_recommendations: List[str]
    temporal_context: Dict[str, Any]
    user_relevance_score: float

@dataclass
class PredictiveModel:
    """Represents a predictive behavior model"""
    model_id: str
    user_id: str
    prediction_type: str
    model_features: List[str]
    accuracy_metrics: Dict[str, float]
    last_updated: datetime
    predictions: List[Dict[str, Any]]

class AdvancedContextEngine:
    """Advanced Context Engineering Engine for Memory Synthesis"""
    
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        self.context_vectors: Dict[str, ContextVector] = {}
        self.correlations: Dict[str, CrossMemoryCorrelation] = {}
        self.insights: Dict[str, InsightPattern] = {}
        self.predictive_models: Dict[str, PredictiveModel] = {}
        
        # Context fusion parameters
        self.temporal_decay_factor = 0.95
        self.correlation_threshold = 0.7
        self.insight_confidence_threshold = 0.8
        
        logger.info("Advanced Context Engineering Engine initialized")
    
    def extract_temporal_context(self, activities: List[Dict[str, Any]]) -> ContextVector:
        """Extract temporal patterns from user activities"""
        if not activities:
            return None
            
        # Analyze temporal patterns
        timestamps = [datetime.fromisoformat(a.get('timestamp', datetime.now().isoformat())) 
                     for a in activities if a.get('timestamp')]
        
        if not timestamps:
            return None
            
        # Calculate temporal features
        activity_hours = [t.hour for t in timestamps]
        activity_days = [t.weekday() for t in timestamps]
        
        temporal_features = {
            'peak_hours': self._find_peak_hours(activity_hours),
            'active_days': self._find_active_days(activity_days),
            'activity_frequency': len(activities),
            'time_span_days': (max(timestamps) - min(timestamps)).days,
            'regularity_score': self._calculate_regularity(timestamps)
        }
        
        # Create embedding for temporal patterns
        temporal_text = f"User active during hours {temporal_features['peak_hours']}, " \
                       f"most active on days {temporal_features['active_days']}, " \
                       f"regularity score {temporal_features['regularity_score']:.2f}"
        
        embedding = self.embedding_model.encode(temporal_text).tolist()
        
        context_vector = ContextVector(
            vector_id=f"temporal_{uuid.uuid4().hex[:8]}",
            context_type=ContextType.TEMPORAL,
            content=temporal_features,
            timestamp=datetime.now(),
            confidence=0.8,
            source="activity_analysis",
            metadata={'activity_count': len(activities)},
            embedding=embedding
        )
        
        self.context_vectors[context_vector.vector_id] = context_vector
        return context_vector
    
    def extract_behavioral_context(self, activities: List[Dict[str, Any]]) -> ContextVector:
        """Extract behavioral patterns from user activities"""
        if not activities:
            return None
            
        # Analyze behavioral patterns
        activity_types = [a.get('type', 'unknown') for a in activities]
        satisfaction_scores = [a.get('satisfaction_score', 0.5) for a in activities 
                              if a.get('satisfaction_score') is not None]
        durations = [a.get('duration', 0) for a in activities if a.get('duration')]
        
        behavioral_features = {
            'preferred_activities': self._get_top_activities(activity_types),
            'average_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0.5,
            'engagement_patterns': self._analyze_engagement(satisfaction_scores, durations),
            'diversity_score': len(set(activity_types)) / len(activity_types) if activity_types else 0,
            'persistence_score': self._calculate_persistence(activities)
        }
        
        # Create embedding for behavioral patterns
        behavioral_text = f"User prefers {behavioral_features['preferred_activities']}, " \
                         f"average satisfaction {behavioral_features['average_satisfaction']:.2f}, " \
                         f"engagement level {behavioral_features['engagement_patterns']}"
        
        embedding = self.embedding_model.encode(behavioral_text).tolist()
        
        context_vector = ContextVector(
            vector_id=f"behavioral_{uuid.uuid4().hex[:8]}",
            context_type=ContextType.BEHAVIORAL,
            content=behavioral_features,
            timestamp=datetime.now(),
            confidence=0.85,
            source="behavioral_analysis",
            metadata={'activities_analyzed': len(activities)},
            embedding=embedding
        )
        
        self.context_vectors[context_vector.vector_id] = context_vector
        return context_vector
    
    def extract_semantic_context(self, documents: List[Dict[str, Any]], 
                                entities: List[Dict[str, Any]]) -> ContextVector:
        """Extract semantic context from documents and entities"""
        if not documents and not entities:
            return None
            
        # Analyze semantic patterns
        entity_types = [e.get('label', 'UNKNOWN') for e in entities]
        entity_texts = [e.get('text', '') for e in entities]
        
        # Extract topics and themes
        all_text = ' '.join([doc.get('content', '') for doc in documents])
        
        semantic_features = {
            'dominant_entity_types': self._get_top_items(entity_types),
            'key_entities': self._get_top_items(entity_texts),
            'topic_diversity': len(set(entity_types)) / len(entity_types) if entity_types else 0,
            'semantic_density': len(entities) / max(len(all_text.split()), 1),
            'content_themes': self._extract_themes(all_text)
        }
        
        # Create embedding for semantic patterns
        semantic_text = f"Content focuses on {semantic_features['dominant_entity_types']}, " \
                       f"key entities include {semantic_features['key_entities']}, " \
                       f"main themes: {semantic_features['content_themes']}"
        
        embedding = self.embedding_model.encode(semantic_text).tolist()
        
        context_vector = ContextVector(
            vector_id=f"semantic_{uuid.uuid4().hex[:8]}",
            context_type=ContextType.SEMANTIC,
            content=semantic_features,
            timestamp=datetime.now(),
            confidence=0.9,
            source="semantic_analysis",
            metadata={'documents_count': len(documents), 'entities_count': len(entities)},
            embedding=embedding
        )
        
        self.context_vectors[context_vector.vector_id] = context_vector
        return context_vector
    
    def find_cross_memory_correlations(self, user_id: str) -> List[CrossMemoryCorrelation]:
        """Find correlations across different memory types"""
        correlations = []
        
        # Get context vectors for analysis
        context_vectors = [cv for cv in self.context_vectors.values()]
        
        if len(context_vectors) < 2:
            return correlations
            
        # Compare embeddings between different context types
        for i, cv1 in enumerate(context_vectors):
            for cv2 in context_vectors[i+1:]:
                if cv1.context_type != cv2.context_type and cv1.embedding and cv2.embedding:
                    similarity = cosine_similarity([cv1.embedding], [cv2.embedding])[0][0]
                    
                    if similarity > self.correlation_threshold:
                        correlation = self._create_correlation(cv1, cv2, similarity, user_id)
                        correlations.append(correlation)
                        self.correlations[correlation.correlation_id] = correlation
        
        return correlations
    
    def generate_automatic_insights(self, user_id: str, 
                                  context_vectors: List[ContextVector]) -> List[InsightPattern]:
        """Generate automatic insights from context analysis"""
        insights = []
        
        for cv in context_vectors:
            if cv.confidence > self.insight_confidence_threshold:
                insight = self._generate_insight_from_context(cv, user_id)
                if insight:
                    insights.append(insight)
                    self.insights[insight.insight_id] = insight
        
        # Generate compound insights from correlations
        for correlation in self.correlations.values():
            if correlation.user_id == user_id and correlation.confidence_score > 0.8:
                compound_insight = self._generate_compound_insight(correlation)
                if compound_insight:
                    insights.append(compound_insight)
                    self.insights[compound_insight.insight_id] = compound_insight
        
        return insights
    
    def build_predictive_model(self, user_id: str, activities: List[Dict[str, Any]]) -> PredictiveModel:
        """Build predictive model for user behavior"""
        if len(activities) < 10:  # Need minimum data
            return None
            
        # Extract features for prediction
        features = self._extract_predictive_features(activities)
        
        # Simple prediction model for next activity type
        activity_transitions = self._analyze_activity_transitions(activities)
        next_activity_probabilities = self._calculate_next_activity_probabilities(activity_transitions)
        
        # Predict optimal engagement times
        engagement_predictions = self._predict_engagement_windows(activities)
        
        predictions = [
            {
                'type': 'next_activity',
                'probabilities': next_activity_probabilities,
                'confidence': 0.75
            },
            {
                'type': 'engagement_windows',
                'predictions': engagement_predictions,
                'confidence': 0.8
            }
        ]
        
        model = PredictiveModel(
            model_id=f"model_{user_id}_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            prediction_type="behavioral_prediction",
            model_features=list(features.keys()),
            accuracy_metrics={'mse': 0.1, 'r2_score': 0.85},
            last_updated=datetime.now(),
            predictions=predictions
        )
        
        self.predictive_models[model.model_id] = model
        return model
    
    def generate_context_aware_recommendations(self, user_id: str, 
                                             current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate context-aware recommendations"""
        recommendations = []
        
        # Get user's context vectors
        user_contexts = [cv for cv in self.context_vectors.values()]
        user_insights = [insight for insight in self.insights.values()]
        user_model = next((model for model in self.predictive_models.values() 
                          if model.user_id == user_id), None)
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        temporal_recommendations = self._get_temporal_recommendations(user_contexts, current_hour)
        recommendations.extend(temporal_recommendations)
        
        # Behavioral recommendations
        behavioral_recommendations = self._get_behavioral_recommendations(user_insights)
        recommendations.extend(behavioral_recommendations)
        
        # Predictive recommendations
        if user_model:
            predictive_recommendations = self._get_predictive_recommendations(user_model)
            recommendations.extend(predictive_recommendations)
        
        # Rank recommendations by relevance
        ranked_recommendations = self._rank_recommendations(recommendations, current_context)
        
        return ranked_recommendations[:10]  # Return top 10
    
    # Helper methods
    def _find_peak_hours(self, hours: List[int]) -> List[int]:
        """Find peak activity hours"""
        if not hours:
            return []
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]
    
    def _find_active_days(self, days: List[int]) -> List[str]:
        """Find most active days of week"""
        if not days:
            return []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = defaultdict(int)
        for day in days:
            day_counts[day] += 1
        sorted_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        return [day_names[day] for day, count in sorted_days[:3]]
    
    def _calculate_regularity(self, timestamps: List[datetime]) -> float:
        """Calculate regularity score based on timestamp intervals"""
        if len(timestamps) < 2:
            return 0.0
        
        intervals = []
        sorted_timestamps = sorted(timestamps)
        for i in range(1, len(sorted_timestamps)):
            interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
            
        # Lower coefficient of variation indicates higher regularity
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        
        # Convert to 0-1 scale (higher is more regular)
        regularity = max(0, 1 - min(cv, 2) / 2)
        return regularity
    
    def _get_top_activities(self, activity_types: List[str], top_n: int = 3) -> List[str]:
        """Get top N most frequent activities"""
        counts = defaultdict(int)
        for activity in activity_types:
            counts[activity] += 1
        sorted_activities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [activity for activity, count in sorted_activities[:top_n]]
    
    def _get_top_items(self, items: List[str], top_n: int = 5) -> List[str]:
        """Get top N most frequent items"""
        counts = defaultdict(int)
        for item in items:
            if item:  # Skip empty items
                counts[item] += 1
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [item for item, count in sorted_items[:top_n]]
    
    def _analyze_engagement(self, satisfaction_scores: List[float], 
                          durations: List[float]) -> str:
        """Analyze engagement patterns"""
        if not satisfaction_scores:
            return "unknown"
            
        avg_satisfaction = np.mean(satisfaction_scores)
        avg_duration = np.mean(durations) if durations else 0
        
        if avg_satisfaction > 0.8 and avg_duration > 300:  # 5 minutes
            return "high_engagement"
        elif avg_satisfaction > 0.6:
            return "moderate_engagement"
        else:
            return "low_engagement"
    
    def _calculate_persistence(self, activities: List[Dict[str, Any]]) -> float:
        """Calculate user persistence score"""
        if len(activities) < 3:
            return 0.5
            
        # Look for patterns of continued engagement
        consecutive_sessions = 0
        max_consecutive = 0
        
        sorted_activities = sorted(activities, key=lambda x: x.get('timestamp', ''))
        
        for i in range(1, len(sorted_activities)):
            prev_time = datetime.fromisoformat(sorted_activities[i-1].get('timestamp', datetime.now().isoformat()))
            curr_time = datetime.fromisoformat(sorted_activities[i].get('timestamp', datetime.now().isoformat()))
            
            # If activities are within 24 hours, consider consecutive
            if (curr_time - prev_time).total_seconds() < 86400:
                consecutive_sessions += 1
            else:
                max_consecutive = max(max_consecutive, consecutive_sessions)
                consecutive_sessions = 0
        
        max_consecutive = max(max_consecutive, consecutive_sessions)
        return min(max_consecutive / len(activities), 1.0)
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract main themes from text"""
        if not text:
            return []
            
        # Simple theme extraction based on word frequency
        words = text.lower().split()
        word_counts = defaultdict(int)
        
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_counts[word] += 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]
    
    def _create_correlation(self, cv1: ContextVector, cv2: ContextVector, 
                          similarity: float, user_id: str) -> CrossMemoryCorrelation:
        """Create a cross-memory correlation"""
        correlation_id = f"corr_{uuid.uuid4().hex[:8]}"
        
        pattern_description = f"Strong correlation between {cv1.context_type.value} and {cv2.context_type.value} patterns"
        
        supporting_evidence = [
            {
                'type': 'similarity_score',
                'value': similarity,
                'context_1': cv1.context_type.value,
                'context_2': cv2.context_type.value
            }
        ]
        
        return CrossMemoryCorrelation(
            correlation_id=correlation_id,
            memory_types=[cv1.context_type.value, cv2.context_type.value],
            correlation_strength=similarity,
            pattern_description=pattern_description,
            supporting_evidence=supporting_evidence,
            confidence_score=similarity * 0.9,  # Slightly lower than similarity
            temporal_range=(min(cv1.timestamp, cv2.timestamp), max(cv1.timestamp, cv2.timestamp)),
            user_id=user_id
        )
    
    def _generate_insight_from_context(self, cv: ContextVector, user_id: str) -> Optional[InsightPattern]:
        """Generate insight from a context vector"""
        if cv.context_type == ContextType.TEMPORAL:
            return self._generate_temporal_insight(cv, user_id)
        elif cv.context_type == ContextType.BEHAVIORAL:
            return self._generate_behavioral_insight(cv, user_id)
        elif cv.context_type == ContextType.SEMANTIC:
            return self._generate_semantic_insight(cv, user_id)
        return None
    
    def _generate_temporal_insight(self, cv: ContextVector, user_id: str) -> InsightPattern:
        """Generate temporal insight"""
        content = cv.content
        peak_hours = content.get('peak_hours', [])
        regularity = content.get('regularity_score', 0)
        
        if regularity > 0.7:
            description = f"User shows high regularity in activity patterns, most active during hours {peak_hours}"
            recommendations = [
                "Schedule important tasks during peak activity hours",
                "Maintain consistent daily routines for optimal productivity"
            ]
        else:
            description = f"User shows irregular activity patterns with peak hours {peak_hours}"
            recommendations = [
                "Consider establishing more consistent routines",
                "Focus on flexible scheduling to accommodate varying patterns"
            ]
        
        return InsightPattern(
            insight_id=f"temporal_insight_{uuid.uuid4().hex[:8]}",
            pattern_type="temporal_pattern",
            description=description,
            supporting_data=[asdict(cv)],
            confidence_score=cv.confidence,
            actionable_recommendations=recommendations,
            temporal_context={'analysis_period': 'recent_activities'},
            user_relevance_score=0.8
        )
    
    def _generate_behavioral_insight(self, cv: ContextVector, user_id: str) -> InsightPattern:
        """Generate behavioral insight"""
        content = cv.content
        preferred_activities = content.get('preferred_activities', [])
        avg_satisfaction = content.get('average_satisfaction', 0.5)
        engagement = content.get('engagement_patterns', 'unknown')
        
        description = f"User shows {engagement} with preferred activities: {preferred_activities}. Average satisfaction: {avg_satisfaction:.2f}"
        
        if avg_satisfaction > 0.8:
            recommendations = [
                f"Continue focusing on activities like {', '.join(preferred_activities[:2])}",
                "Explore similar high-satisfaction activities",
                "Share successful patterns with others"
            ]
        else:
            recommendations = [
                "Explore new activity types to improve satisfaction",
                "Analyze what makes certain activities more engaging",
                "Consider adjusting approach to current activities"
            ]
        
        return InsightPattern(
            insight_id=f"behavioral_insight_{uuid.uuid4().hex[:8]}",
            pattern_type="behavioral_pattern",
            description=description,
            supporting_data=[asdict(cv)],
            confidence_score=cv.confidence,
            actionable_recommendations=recommendations,
            temporal_context={'satisfaction_trend': 'recent_period'},
            user_relevance_score=0.9
        )
    
    def _generate_semantic_insight(self, cv: ContextVector, user_id: str) -> InsightPattern:
        """Generate semantic insight"""
        content = cv.content
        dominant_entities = content.get('dominant_entity_types', [])
        themes = content.get('content_themes', [])
        diversity = content.get('topic_diversity', 0)
        
        if diversity > 0.7:
            description = f"User engages with diverse topics including {dominant_entities} and themes like {themes}"
            recommendations = [
                "Leverage diverse knowledge for creative problem-solving",
                "Consider interdisciplinary approaches to projects",
                "Explore connections between different domains"
            ]
        else:
            description = f"User shows focused engagement with topics: {dominant_entities}, themes: {themes}"
            recommendations = [
                "Deepen expertise in current focus areas",
                "Consider gradual expansion to related topics",
                "Become a specialist in current domains"
            ]
        
        return InsightPattern(
            insight_id=f"semantic_insight_{uuid.uuid4().hex[:8]}",
            pattern_type="semantic_pattern",
            description=description,
            supporting_data=[asdict(cv)],
            confidence_score=cv.confidence,
            actionable_recommendations=recommendations,
            temporal_context={'content_analysis': 'recent_documents'},
            user_relevance_score=0.85
        )
    
    def _generate_compound_insight(self, correlation: CrossMemoryCorrelation) -> InsightPattern:
        """Generate insight from cross-memory correlation"""
        memory_types = correlation.memory_types
        strength = correlation.correlation_strength
        
        description = f"Strong correlation ({strength:.2f}) found between {' and '.join(memory_types)} patterns"
        
        recommendations = [
            f"Leverage synergies between {memory_types[0]} and {memory_types[1]} patterns",
            "Use correlated patterns for more accurate predictions",
            "Consider integrated approaches that address both domains"
        ]
        
        return InsightPattern(
            insight_id=f"compound_insight_{uuid.uuid4().hex[:8]}",
            pattern_type="cross_memory_correlation",
            description=description,
            supporting_data=[asdict(correlation)],
            confidence_score=correlation.confidence_score,
            actionable_recommendations=recommendations,
            temporal_context={'correlation_period': correlation.temporal_range},
            user_relevance_score=0.9
        )
    
    def _extract_predictive_features(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for predictive modeling"""
        if not activities:
            return {}
            
        # Temporal features
        timestamps = [datetime.fromisoformat(a.get('timestamp', datetime.now().isoformat())) 
                     for a in activities if a.get('timestamp')]
        hours = [t.hour for t in timestamps]
        days = [t.weekday() for t in timestamps]
        
        # Activity features
        activity_types = [a.get('type', 'unknown') for a in activities]
        durations = [a.get('duration', 0) for a in activities if a.get('duration')]
        satisfactions = [a.get('satisfaction_score', 0.5) for a in activities 
                        if a.get('satisfaction_score') is not None]
        
        return {
            'avg_hour': np.mean(hours) if hours else 12,
            'most_common_day': max(set(days), key=days.count) if days else 0,
            'activity_diversity': len(set(activity_types)) / len(activity_types) if activity_types else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'avg_satisfaction': np.mean(satisfactions) if satisfactions else 0.5,
            'total_activities': len(activities)
        }
    
    def _analyze_activity_transitions(self, activities: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Analyze transitions between activity types"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        sorted_activities = sorted(activities, key=lambda x: x.get('timestamp', ''))
        
        for i in range(len(sorted_activities) - 1):
            current_type = sorted_activities[i].get('type', 'unknown')
            next_type = sorted_activities[i + 1].get('type', 'unknown')
            transitions[current_type][next_type] += 1
        
        return dict(transitions)
    
    def _calculate_next_activity_probabilities(self, transitions: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """Calculate probabilities for next activities"""
        probabilities = {}
        
        for current_activity, next_activities in transitions.items():
            total = sum(next_activities.values())
            if total > 0:
                probabilities[current_activity] = {
                    next_act: count / total 
                    for next_act, count in next_activities.items()
                }
        
        return probabilities
    
    def _predict_engagement_windows(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict optimal engagement time windows"""
        if not activities:
            return []
            
        # Analyze historical engagement by hour and satisfaction
        hour_satisfaction = defaultdict(list)
        
        for activity in activities:
            if activity.get('timestamp') and activity.get('satisfaction_score') is not None:
                timestamp = datetime.fromisoformat(activity['timestamp'])
                hour_satisfaction[timestamp.hour].append(activity['satisfaction_score'])
        
        # Calculate average satisfaction by hour
        optimal_windows = []
        for hour, satisfactions in hour_satisfaction.items():
            avg_satisfaction = np.mean(satisfactions)
            if avg_satisfaction > 0.7:  # High satisfaction threshold
                optimal_windows.append({
                    'hour': hour,
                    'predicted_satisfaction': avg_satisfaction,
                    'confidence': min(len(satisfactions) / 5, 1.0),  # More data = higher confidence
                    'sample_size': len(satisfactions)
                })
        
        return sorted(optimal_windows, key=lambda x: x['predicted_satisfaction'], reverse=True)
    
    def _get_temporal_recommendations(self, contexts: List[ContextVector], current_hour: int) -> List[Dict[str, Any]]:
        """Get time-based recommendations"""
        recommendations = []
        
        for cv in contexts:
            if cv.context_type == ContextType.TEMPORAL:
                peak_hours = cv.content.get('peak_hours', [])
                if current_hour in peak_hours:
                    recommendations.append({
                        'type': 'temporal_opportunity',
                        'message': f"Current time ({current_hour}:00) is one of your peak activity hours",
                        'action': 'Consider tackling important tasks now',
                        'confidence': cv.confidence,
                        'priority': 'high'
                    })
                elif any(abs(current_hour - ph) <= 1 for ph in peak_hours):
                    recommendations.append({
                        'type': 'temporal_preparation',
                        'message': f"Approaching peak activity time",
                        'action': 'Prepare for high-productivity period',
                        'confidence': cv.confidence * 0.8,
                        'priority': 'medium'
                    })
        
        return recommendations
    
    def _get_behavioral_recommendations(self, insights: List[InsightPattern]) -> List[Dict[str, Any]]:
        """Get behavior-based recommendations"""
        recommendations = []
        
        for insight in insights:
            if insight.pattern_type == "behavioral_pattern":
                for rec_text in insight.actionable_recommendations:
                    recommendations.append({
                        'type': 'behavioral_guidance',
                        'message': rec_text,
                        'action': 'Consider implementing this behavioral change',
                        'confidence': insight.confidence_score,
                        'priority': 'medium',
                        'insight_id': insight.insight_id
                    })
        
        return recommendations
    
    def _get_predictive_recommendations(self, model: PredictiveModel) -> List[Dict[str, Any]]:
        """Get predictive model-based recommendations"""
        recommendations = []
        
        for prediction in model.predictions:
            if prediction['type'] == 'next_activity':
                probabilities = prediction['probabilities']
                if probabilities:
                    top_activity = max(probabilities.items(), key=lambda x: max(x[1].values()) if x[1] else 0)
                    recommendations.append({
                        'type': 'activity_prediction',
                        'message': f"Based on patterns, you might want to engage in {top_activity[0]} activities",
                        'action': 'Consider this activity type for your next session',
                        'confidence': prediction['confidence'],
                        'priority': 'low'
                    })
            
            elif prediction['type'] == 'engagement_windows':
                windows = prediction['predictions']
                if windows:
                    best_window = windows[0]
                    recommendations.append({
                        'type': 'engagement_timing',
                        'message': f"Hour {best_window['hour']}:00 predicted as optimal engagement time",
                        'action': 'Schedule important activities during this window',
                        'confidence': best_window['confidence'],
                        'priority': 'high'
                    })
        
        return recommendations
    
    def _rank_recommendations(self, recommendations: List[Dict[str, Any]], 
                            current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank recommendations by relevance and priority"""
        priority_weights = {'high': 3, 'medium': 2, 'low': 1}
        
        for rec in recommendations:
            base_score = priority_weights.get(rec.get('priority', 'low'), 1)
            confidence_score = rec.get('confidence', 0.5)
            
            # Add context relevance scoring
            context_bonus = 0
            if rec['type'] == 'temporal_opportunity' and current_context.get('immediate_task', False):
                context_bonus = 0.5
            elif rec['type'] == 'behavioral_guidance' and current_context.get('seeking_improvement', False):
                context_bonus = 0.3
                
            rec['relevance_score'] = base_score * confidence_score + context_bonus
        
        return sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)


# --- Enhanced Knowledge Graph Manager with Context Integration ---
class LangChainKnowledgeGraphManager:
    """LangChain-based Knowledge Graph Manager with Context Engineering integration"""
    
    def __init__(self, llm, neo4j_url: str, neo4j_username: str, neo4j_password: str, database: str = "neo4j"):
        self.llm = llm
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.database = database
        
        try:
            # Initialize LangChain Neo4j Graph
            self.graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database=database
            )
            
            # Initialize LLMGraphTransformer with enhanced configuration (removed 'id' as it's reserved)
            self.llm_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["User", "Activity", "Document", "Entity", "Session", "Preference", 
                             "Skill", "Context", "File", "Project", "Conversation", "Topic", "Interest",
                             "FavoriteView", "QueryPattern", "InsightType", "ContextVector", 
                             "Correlation", "PredictiveModel", "Recommendation"],
                allowed_relationships=["PERFORMED", "ACCESSED", "INVOLVES", "HAS_PREFERENCE", "HAS_SKILL", 
                                     "PARTICIPATED_IN", "CREATED", "INTERACTED_WITH", "BELONGS_TO", "RELATED_TO",
                                     "WORKED_ON", "DISCUSSED", "INTERESTED_IN", "SIMILAR_TO", "COLLABORATED_WITH",
                                     "HAS_FAVORITE_VIEW", "FOLLOWS_PATTERN", "PREFERS_INSIGHT", "GENERATED_BY",
                                     "HAS_CONTEXT", "CORRELATES_WITH", "PREDICTS", "RECOMMENDS"],
                node_properties=["name", "type", "timestamp", "satisfaction", "frequency", "importance", "category",
                                 "description", "cypher_query", "user_id", "created_at", "updated_at",
                                 "strength", "activity_count", "engagement_level", "total_activities", 
                                 "average_satisfaction", "most_common_activity", "total_duration_seconds",
                                 "pattern_definition", "pattern_example", "insight_category", "insight_description",
                                 "context_type", "confidence", "correlation_strength", "prediction_accuracy"]
            )
            
            # Initialize direct Neo4j driver for advanced queries and schema management
            self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
            
            # Initialize database constraints and indexes
            self._initialize_database_schema()
            
            logger.info("Enhanced LangChain Knowledge Graph Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Graph Manager: {e}", exc_info=True)
            self.graph = None
            self.llm_transformer = None
            self.driver = None
    
    def _initialize_database_schema(self):
        """Initialize database schema with constraints and indexes for enhanced node types."""
        if not self.driver:
            return
            
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints for essential nodes
                constraints_to_create = [
                    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                    "CREATE CONSTRAINT activity_id_unique IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE",
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
                    "CREATE CONSTRAINT context_vector_id_unique IF NOT EXISTS FOR (cv:ContextVector) REQUIRE cv.id IS UNIQUE",
                    "CREATE CONSTRAINT correlation_id_unique IF NOT EXISTS FOR (c:Correlation) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT predictive_model_id_unique IF NOT EXISTS FOR (pm:PredictiveModel) REQUIRE pm.id IS UNIQUE"
                ]
                
                for constraint in constraints_to_create:
                    try:
                        session.run(constraint)
                        logger.debug(f"Executed constraint: {constraint}")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Failed to execute constraint '{constraint}': {e}")
                        
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
    
    def store_context_vectors(self, user_id: str, context_vectors: List[ContextVector]):
        """Store context vectors in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j driver not available, cannot store context vectors.")
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                for cv in context_vectors:
                    session.run("""
                        MERGE (u:User {id: $user_id})
                        MERGE (cv:ContextVector {id: $cv_id})
                        ON CREATE SET cv.id = $cv_id, cv.context_type = $context_type, 
                                     cv.confidence = $confidence, cv.source = $source,
                                     cv.created_at = $timestamp
                        SET cv.content = $content, cv.metadata = $metadata, cv.updated_at = $timestamp
                        MERGE (u)-[:HAS_CONTEXT]->(cv)
                    """, 
                    user_id=user_id, cv_id=cv.vector_id, context_type=cv.context_type.value,
                    confidence=cv.confidence, source=cv.source, content=json.dumps(cv.content),
                    metadata=json.dumps(cv.metadata), timestamp=cv.timestamp.isoformat())
                    
                logger.info(f"Stored {len(context_vectors)} context vectors for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error storing context vectors for {user_id}: {e}", exc_info=True)
    
    def store_correlations(self, correlations: List[CrossMemoryCorrelation]):
        """Store cross-memory correlations in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j driver not available, cannot store correlations.")
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                for corr in correlations:
                    session.run("""
                        MERGE (u:User {id: $user_id})
                        MERGE (c:Correlation {id: $corr_id})
                        ON CREATE SET c.id = $corr_id, c.memory_types = $memory_types,
                                     c.correlation_strength = $strength, c.confidence_score = $confidence,
                                     c.created_at = $timestamp
                        SET c.pattern_description = $description, c.supporting_evidence = $evidence,
                            c.temporal_range = $temporal_range, c.updated_at = $timestamp
                        MERGE (u)-[:HAS_CORRELATION]->(c)
                    """,
                    user_id=corr.user_id, corr_id=corr.correlation_id, memory_types=corr.memory_types,
                    strength=corr.correlation_strength, confidence=corr.confidence_score,
                    description=corr.pattern_description, evidence=json.dumps(corr.supporting_evidence),
                    temporal_range=f"{corr.temporal_range[0].isoformat()},{corr.temporal_range[1].isoformat()}",
                    timestamp=datetime.now().isoformat())
                    
                logger.info(f"Stored {len(correlations)} correlations")
                
        except Exception as e:
            logger.error(f"Error storing correlations: {e}", exc_info=True)
    
    def store_insights(self, insights: List[InsightPattern]):
        """Store generated insights in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j driver not available, cannot store insights.")
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                for insight in insights:
                    session.run("""
                        MERGE (i:InsightPattern {id: $insight_id})
                        ON CREATE SET i.id = $insight_id, i.pattern_type = $pattern_type,
                                     i.confidence_score = $confidence, i.user_relevance_score = $relevance,
                                     i.created_at = $timestamp
                        SET i.description = $description, i.supporting_data = $supporting_data,
                            i.actionable_recommendations = $recommendations, 
                            i.temporal_context = $temporal_context, i.updated_at = $timestamp
                    """,
                    insight_id=insight.insight_id, pattern_type=insight.pattern_type,
                    confidence=insight.confidence_score, relevance=insight.user_relevance_score,
                    description=insight.description, supporting_data=json.dumps(insight.supporting_data),
                    recommendations=json.dumps(insight.actionable_recommendations),
                    temporal_context=json.dumps(insight.temporal_context),
                    timestamp=datetime.now().isoformat())
                    
                logger.info(f"Stored {len(insights)} insights")
                
        except Exception as e:
            logger.error(f"Error storing insights: {e}", exc_info=True)
    
    def store_predictive_model(self, model: PredictiveModel):
        """Store predictive model in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j driver not available, cannot store predictive model.")
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (pm:PredictiveModel {id: $model_id})
                    ON CREATE SET pm.id = $model_id, pm.user_id = $user_id, 
                                 pm.prediction_type = $prediction_type, pm.created_at = $timestamp
                    SET pm.model_features = $features, pm.accuracy_metrics = $accuracy,
                        pm.predictions = $predictions, pm.last_updated = $last_updated,
                        pm.updated_at = $timestamp
                    MERGE (u)-[:HAS_MODEL]->(pm)
                """,
                user_id=model.user_id, model_id=model.model_id, 
                prediction_type=model.prediction_type, features=json.dumps(model.model_features),
                accuracy=json.dumps(model.accuracy_metrics), predictions=json.dumps(model.predictions),
                last_updated=model.last_updated.isoformat(), timestamp=datetime.now().isoformat())
                
                logger.info(f"Stored predictive model {model.model_id}")
                
        except Exception as e:
            logger.error(f"Error storing predictive model: {e}", exc_info=True)

    # ... (rest of the existing methods remain the same)
    def create_user_activity_graph(self, user_id: str, activities_data: List[Dict[str, Any]]) -> bool:
        """Create comprehensive knowledge graph from user activities, integrating new node types."""
        if not self.llm_transformer or not self.graph:
            logger.warning("LLM Transformer or Neo4j graph not initialized, skipping graph creation.")
            return False
        
        try:
            # Create comprehensive text representation of user activities
            user_activity_text = self._create_comprehensive_user_activity_text(user_id, activities_data)
            
            # Create document for LLMGraphTransformer
            documents = [Document(page_content=user_activity_text, metadata={"user_id": user_id})]
            
            # Transform text to graph documents using LLM
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            
            # Add graph documents to Neo4j
            self.graph.add_graph_documents(graph_documents)
            
            # Add enhanced user-specific data and relationships
            self._add_comprehensive_user_data(user_id, activities_data)
            
            # Update user preferences and skills derived from activities
            self._update_user_preferences_and_skills(user_id, activities_data)
            
            logger.info(f"Comprehensive user activity graph updated for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating or updating user activity graph for {user_id}: {e}", exc_info=True)
            return False
    
    def _create_comprehensive_user_activity_text(self, user_id: str, activities_data: List[Dict[str, Any]]) -> str:
        """Create comprehensive text representation for better knowledge graph extraction."""
        text_parts = []
        text_parts.append(f"User {user_id} is an active system user involved in various activities.")
        
        for activity in activities_data:
            activity_type = activity.get('type', 'unknown')
            subtype = activity.get('subtype', '')
            timestamp = activity.get('timestamp', '')
            duration = activity.get('duration', 0)
            satisfaction = activity.get('satisfaction_score', 0.5)
            entities = activity.get('entities_involved', [])
            files = activity.get('files_accessed', [])
            
            activity_desc = f"Activity: {activity_type}"
            if subtype: activity_desc += f" (focus: {subtype})"
            if timestamp:
                date_part = timestamp[:10] if isinstance(timestamp, str) and len(timestamp) > 10 else str(timestamp)
                activity_desc += f" on {date_part}"
            if duration > 0:
                if duration > 3600: activity_desc += f" for {duration/3600:.1f} hours"
                elif duration > 60: activity_desc += f" for {duration/60:.1f} minutes"
                else: activity_desc += f" for {duration} seconds"
            
            # Add satisfaction context
            if satisfaction > 0.8: activity_desc += ", with high satisfaction."
            elif satisfaction > 0.6: activity_desc += ", with good satisfaction."
            elif satisfaction < 0.4: activity_desc += ", with low satisfaction."
            
            # Include entities and files if significant
            if entities:
                entity_names = [e.get('text') for e in entities if isinstance(e, dict) and e.get('text')]
                if entity_names: activity_desc += f" Involved entities: {', '.join(entity_names[:3])}."
            if files: activity_desc += f" Accessed files: {', '.join(files[:2])}."
                
            text_parts.append(activity_desc)
        
        return " ".join(text_parts)
    
    def _add_comprehensive_user_data(self, user_id: str, activities_data: List[Dict[str, Any]]):
        """Add/update user node and link activities using direct Neo4j driver for precision."""
        if not self.driver:
            logger.warning("Neo4j driver not available, cannot add comprehensive user data.")
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                user_properties = self._extract_comprehensive_user_properties(user_id, activities_data)
                
                session.run("""
                    MERGE (u:User {id: $user_id})
                    ON CREATE SET u.id = $user_id, u.created_at = $timestamp
                    SET u += $properties, u.updated_at = $timestamp
                """, user_id=user_id, properties=user_properties, timestamp=datetime.now().isoformat())
                
                for i, activity in enumerate(activities_data):
                    activity_id = activity.get('activity_id', f"{user_id}_act_{int(time.time())}_{i}_{uuid.uuid4().hex[:4]}")
                    activity_props = {
                        'id': activity_id, 'type': activity.get('type', 'unknown'), 'subtype': activity.get('subtype', ''),
                        'timestamp': activity.get('timestamp', datetime.now().isoformat()), 'duration': activity.get('duration', 0),
                        'satisfaction_score': activity.get('satisfaction_score', 0.5), 'user_id': user_id
                    }
                    
                    session.run("""
                        MERGE (a:Activity {id: $activity_id})
                        ON CREATE SET a = $properties, a.created_at = $timestamp
                        SET a += $properties, a.updated_at = $timestamp
                        WITH a
                        MATCH (u:User {id: $user_id})
                        MERGE (u)-[r:PERFORMED]->(a)
                        ON CREATE SET r.timestamp = a.timestamp, r.created_at = $timestamp
                        SET r.activity_data_json = $activity_props_json, r.updated_at = $timestamp
                    """, 
                    activity_id=activity_id, properties=activity_props, timestamp=datetime.now().isoformat(),
                    user_id=user_id, activity_props_json=json.dumps({k: v for k, v in activity_props.items() if k not in ['id', 'user_id']}))

                    entities = activity.get('entities_involved', [])
                    if entities:
                        for entity in entities:
                            if isinstance(entity, dict) and entity.get('text'):
                                entity_text = entity['text']
                                entity_label = entity.get('label', 'UnknownEntity')
                                entity_node_id = f"{entity_label}_{entity_text.replace(' ', '_')[:50]}" 

                                session.run("""
                                    MERGE (e:Entity {id: $entity_node_id})
                                    ON CREATE SET e.id = $entity_node_id, e.name = $entity_text, e.label = $entity_label, e.created_at = $timestamp
                                    SET e.name = $entity_text, e.label = $entity_label, e.updated_at = $timestamp
                                    WITH e
                                    MATCH (a:Activity {id: $activity_id})
                                    MERGE (a)-[:INVOLVES]->(e)
                                """, entity_node_id=entity_node_id, entity_text=entity_text, entity_label=entity_label,
                                     activity_id=activity_id, timestamp=datetime.now().isoformat())
        
        except Exception as e:
            logger.error(f"Error adding comprehensive user data for {user_id}: {e}", exc_info=True)
    
    def _extract_comprehensive_user_properties(self, user_id: str, activities_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive user properties from activities for the User node"""
        total_activities = len(activities_data)
        satisfaction_scores = [a.get('satisfaction_score') for a in activities_data if a.get('satisfaction_score') is not None]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.5
        
        activity_types = defaultdict(int)
        total_duration = 0
        for activity in activities_data:
            activity_types[activity.get('type', 'unknown')] += 1
            total_duration += activity.get('duration', 0)
        
        most_common_activity = max(activity_types.items(), key=lambda x: x[1])[0] if activity_types else 'none'
        
        if avg_satisfaction > 0.8 and total_activities > 5: engagement_level = 'high'
        elif avg_satisfaction > 0.6 and total_activities > 2: engagement_level = 'medium'
        else: engagement_level = 'low'
        
        return {
            'name': user_id,
            'total_activities': total_activities, 'average_satisfaction': round(avg_satisfaction, 3),
            'most_common_activity': most_common_activity, 'total_duration_seconds': total_duration,
            'engagement_level': engagement_level, 'last_active': datetime.now().isoformat(),
        }
    
    def _update_user_preferences_and_skills(self, user_id: str, activities_data: List[Dict[str, Any]]):
        """Infer and store user preferences and skills based on activities."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot update preferences/skills."); return
        
        try:
            with self.driver.session(database=self.database) as session:
                # Analyze preferences from high-satisfaction activities
                high_satisfaction_activities = [a for a in activities_data if a.get('satisfaction_score', 0) > 0.75]
                preference_types = defaultdict(int)
                for activity in high_satisfaction_activities: preference_types[activity.get('type', 'unknown')] += 1
                
                for pref_type, count in preference_types.items():
                    if count >= 1:
                        preference_strength = min(count / len(activities_data), 1.0)
                        session.run("""
                            MERGE (u:User {id: $user_id})
                            MERGE (u)-[:HAS_PREFERENCE]->(p:Preference {type: $pref_type})
                            ON CREATE SET p.type = $pref_type, p.user_id = $user_id, p.created_at = $timestamp
                            SET p.strength = $strength, p.activity_count = $count, p.updated_at = $timestamp
                        """, pref_type=pref_type, user_id=user_id, strength=preference_strength, count=count, timestamp=datetime.now().isoformat())
                
                # Infer skills (simplified)
                skill_counts = defaultdict(int)
                for activity in activities_data:
                    if activity.get('type') == 'coding': skill_counts['coding'] += 1
                    if activity.get('type') == 'writing': skill_counts['writing'] += 1
                        
                for skill_name, count in skill_counts.items():
                    if count > 0:
                        skill_level = min(count / len(activities_data), 1.0)
                        session.run("""
                            MERGE (u:User {id: $user_id})
                            MERGE (u)-[:HAS_SKILL]->(s:Skill {name: $skill_name})
                            ON CREATE SET s.name = $skill_name, s.user_id = $user_id, s.created_at = $timestamp
                            SET s.level = $level, s.activity_count = $count, s.updated_at = $timestamp
                        """, skill_name=skill_name, user_id=user_id, level=skill_level, count=count, timestamp=datetime.now().isoformat())

        except Exception as e:
            logger.error(f"Error updating user preferences and skills for {user_id}: {e}", exc_info=True)
    
    # Methods for explicit preferences (same as before)
    def save_explicit_favorite_view(self, user_id: str, view_name: str, cypher_query: str, description: str) -> bool:
        """Saves an explicit favorite view as a distinct node type."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot save favorite view."); return False
        try:
            favorite_view_id = f"{user_id}_{view_name.replace(' ', '_').lower()}"
            with self.driver.session(database=self.database) as session:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (u)-[:HAS_EXPLICIT_FAVORITE]->(fv:FavoriteView {id: $fv_id})
                    ON CREATE SET fv.id = $fv_id, fv.user_id = $user_id, fv.name = $view_name, fv.created_at = $timestamp
                    SET fv.cypher_query = $cypher_query, fv.description = $description, fv.updated_at = $timestamp
                """, user_id=user_id, fv_id=favorite_view_id, view_name=view_name, cypher_query=cypher_query, description=description, timestamp=datetime.now().isoformat())
            logger.info(f"Saved explicit favorite view '{view_name}' for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving explicit favorite view '{view_name}' for user {user_id}: {e}", exc_info=True)
            return False

    def get_explicit_favorite_views(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieves explicit favorite views for a user."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot get favorite views."); return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:HAS_EXPLICIT_FAVORITE]->(fv:FavoriteView)
                    RETURN fv.name AS name, fv.cypher_query AS cypher_query, fv.description AS description
                    ORDER BY fv.updated_at DESC
                """, user_id=user_id)
                return [record for record in result]
        except Exception as e:
            logger.error(f"Error getting explicit favorite views for user {user_id}: {e}", exc_info=True)
            return []
    
    # --- Methods for Query Patterns and Insight Types ---
    def save_query_pattern(self, user_id: str, pattern_name: str, pattern_definition: str, pattern_example: str = "") -> bool:
        """Saves a user-defined query pattern."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot save query pattern."); return False
        try:
            pattern_id = f"{user_id}_{pattern_name.replace(' ', '_').lower()}"
            with self.driver.session(database=self.database) as session:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (u)-[:FOLLOWS_PATTERN]->(qp:QueryPattern {id: $pattern_id})
                    ON CREATE SET qp.id = $pattern_id, qp.user_id = $user_id, qp.name = $pattern_name, qp.created_at = $timestamp
                    SET qp.pattern_definition = $pattern_definition, qp.pattern_example = $pattern_example, qp.updated_at = $timestamp
                """, user_id=user_id, pattern_id=pattern_id, pattern_name=pattern_name, pattern_definition=pattern_definition,
                    pattern_example=pattern_example, timestamp=datetime.now().isoformat())
            logger.info(f"Saved query pattern '{pattern_name}' for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving query pattern '{pattern_name}' for user {user_id}: {e}", exc_info=True)
            return False

    def get_query_patterns(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieves user-defined query patterns."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot get query patterns."); return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:FOLLOWS_PATTERN]->(qp:QueryPattern)
                    RETURN qp.name AS name, qp.pattern_definition AS definition, qp.pattern_example AS example
                    ORDER BY qp.updated_at DESC
                """, user_id=user_id)
                return [record for record in result]
        except Exception as e:
            logger.error(f"Error getting query patterns for user {user_id}: {e}", exc_info=True)
            return []

    def save_insight_type(self, user_id: str, category: str, description: str = "") -> bool:
        """Saves a preferred insight type for the user."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot save insight type."); return False
        try:
            insight_id = f"{user_id}_{category.replace(' ', '_').lower()}"
            with self.driver.session(database=self.database) as session:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (u)-[:PREFERS_INSIGHT]->(it:InsightType {id: $insight_id})
                    ON CREATE SET it.id = $insight_id, it.user_id = $user_id, it.insight_category = $category, it.created_at = $timestamp
                    SET it.description = $description, it.updated_at = $timestamp
                """, user_id=user_id, insight_id=insight_id, category=category, description=description, timestamp=datetime.now().isoformat())
            logger.info(f"Saved insight type '{category}' for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving insight type '{category}' for user {user_id}: {e}", exc_info=True)
            return False

    def get_insight_types(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieves preferred insight types for the user."""
        if not self.driver: logger.warning("Neo4j driver not available, cannot get insight types."); return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:PREFERS_INSIGHT]->(it:InsightType)
                    RETURN it.insight_category AS category, it.description AS description
                    ORDER BY it.updated_at DESC
                """, user_id=user_id)
                return [record for record in result]
        except Exception as e:
            logger.error(f"Error getting insight types for user {user_id}: {e}", exc_info=True)
            return []
    
    # --- Methods for Visualization Data Retrieval ---
    def get_user_graph_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user graph data for visualization, including favorites, patterns, and insights."""
        if not self.driver: 
            logger.warning("Neo4j driver not available, returning empty graph data.")
            return {'nodes': [], 'edges': [], 'user_id': user_id, 'total_nodes': 0, 'total_edges': 0}
        
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                    MATCH (u:User {id: $user_id})
                    OPTIONAL MATCH (u)-[r1]-(connected) WHERE NOT type(r1) IN ['HAS_EXPLICIT_FAVORITE', 'FOLLOWS_PATTERN', 'PREFERS_INSIGHT']
                    OPTIONAL MATCH (u)-[:HAS_EXPLICIT_FAVORITE]->(fv:FavoriteView)
                    OPTIONAL MATCH (u)-[:FOLLOWS_PATTERN]->(qp:QueryPattern)
                    OPTIONAL MATCH (u)-[:PREFERS_INSIGHT]->(it:InsightType)
                    
                    RETURN u,
                           collect(DISTINCT {node: connected, relationship: r1, rel_type: type(r1)}) as directConnections,
                           collect(DISTINCT {node: fv, relationship_type: 'HAS_EXPLICIT_FAVORITE'}) as favoriteViews,
                           collect(DISTINCT {node: qp, relationship_type: 'FOLLOWS_PATTERN'}) as queryPatterns,
                           collect(DISTINCT {node: it, relationship_type: 'PREFERS_INSIGHT'}) as insightTypes
                """
                result = session.run(query, user_id=user_id)
                record = result.single()
                if not record: 
                    logger.info(f"No user node found for ID: {user_id}, creating empty graph data")
                    return {'nodes': [], 'edges': [], 'user_id': user_id, 'total_nodes': 0, 'total_edges': 0}
                
                nodes = []
                edges = []
                node_ids_added = set()
                
                # Add user node
                user_node_data_dict = dict(record['u'])
                user_node_style = self._get_node_style('User', user_node_data_dict)
                user_node_data = {'id': user_id, 'label': self._create_node_label('User', user_node_data_dict), 'type': 'User', 'properties': user_node_data_dict, 'size': user_node_style[0], 'color': user_node_style[1]}
                nodes.append(user_node_data)
                node_ids_added.add(user_id)
                
                # Process direct connections (activities, preferences, etc.)
                for conn_info in record['directConnections']:
                    node, rel_type = conn_info['node'], conn_info['rel_type']
                    if node:
                        node_dict = dict(node) if hasattr(node, 'items') else {}
                        node_labels = list(node.labels) if hasattr(node, 'labels') else []
                        node_type = node_labels[0] if node_labels else 'Unknown'
                        node_id = self._get_node_id(node_type, node_dict)
                        
                        if node_id not in node_ids_added:
                            size, color = self._get_node_style(node_type, node_dict)
                            nodes.append({'id': node_id, 'label': self._create_node_label(node_type, node_dict), 'type': node_type, 'properties': node_dict, 'size': size, 'color': color})
                            node_ids_added.add(node_id)
                        
                        edges.append({'source': user_id, 'target': node_id, 'relationship': rel_type, 'label': rel_type.replace('_', ' ').title(), 'properties': {}})
                
                # Process explicit favorite views
                for fav_info in record['favoriteViews']:
                    fav_node = fav_info['node']
                    if fav_node:
                        fav_dict = dict(fav_node) if hasattr(fav_node, 'items') else {}
                        fav_node_id = self._get_node_id('FavoriteView', fav_dict)
                        if fav_node_id not in node_ids_added:
                            size, color = self._get_node_style('FavoriteView', fav_dict)
                            nodes.append({'id': fav_node_id, 'label': self._create_node_label('FavoriteView', fav_dict), 'type': 'FavoriteView', 'properties': fav_dict, 'size': size, 'color': color})
                            node_ids_added.add(fav_node_id)
                        edges.append({'source': user_id, 'target': fav_node_id, 'relationship': 'HAS_EXPLICIT_FAVORITE', 'label': 'Has Favorite View', 'properties': {}})
                
                # Process query patterns
                for qp_info in record['queryPatterns']:
                    qp_node = qp_info['node']
                    if qp_node:
                        qp_dict = dict(qp_node) if hasattr(qp_node, 'items') else {}
                        qp_node_id = self._get_node_id('QueryPattern', qp_dict)
                        if qp_node_id not in node_ids_added:
                            size, color = self._get_node_style('QueryPattern', qp_dict)
                            nodes.append({'id': qp_node_id, 'label': self._create_node_label('QueryPattern', qp_dict), 'type': 'QueryPattern', 'properties': qp_dict, 'size': size, 'color': color})
                            node_ids_added.add(qp_node_id)
                        edges.append({'source': user_id, 'target': qp_node_id, 'relationship': 'FOLLOWS_PATTERN', 'label': 'Follows Pattern', 'properties': {}})

                # Process insight types
                for it_info in record['insightTypes']:
                    it_node = it_info['node']
                    if it_node:
                        it_dict = dict(it_node) if hasattr(it_node, 'items') else {}
                        it_node_id = self._get_node_id('InsightType', it_dict)
                        if it_node_id not in node_ids_added:
                            size, color = self._get_node_style('InsightType', it_dict)
                            nodes.append({'id': it_node_id, 'label': self._create_node_label('InsightType', it_dict), 'type': 'InsightType', 'properties': it_dict, 'size': size, 'color': color})
                            node_ids_added.add(it_node_id)
                        edges.append({'source': user_id, 'target': it_node_id, 'relationship': 'PREFERS_INSIGHT', 'label': 'Prefers Insight', 'properties': {}})
                
                return {'nodes': nodes, 'edges': edges, 'user_id': user_id, 'total_nodes': len(nodes), 'total_edges': len(edges)}
                
        except Exception as e:
            logger.error(f"Error getting user graph data for {user_id}: {e}", exc_info=True)
            return {'nodes': [], 'edges': [], 'user_id': user_id, 'total_nodes': 0, 'total_edges': 0}
    
    def _get_node_id(self, node_type: str, node_dict: Dict) -> str:
        """Generate a consistent node ID for visualization."""
        if node_type == 'User': return node_dict.get('id', f"User_{node_dict.get('name', '')}")
        elif node_type == 'Activity': return node_dict.get('id', f"Activity_{node_dict.get('timestamp', '')}")
        elif node_type == 'FavoriteView': return node_dict.get('id', f"FavoriteView_{node_dict.get('user_id', '')}_{node_dict.get('name', '')}")
        elif node_type == 'QueryPattern': return node_dict.get('id', f"QueryPattern_{node_dict.get('user_id', '')}_{node_dict.get('name', '')}")
        elif node_type == 'InsightType': return node_dict.get('id', f"InsightType_{node_dict.get('user_id', '')}_{node_dict.get('insight_category', '')}")
        elif node_type == 'Entity': return node_dict.get('id', f"Entity_{node_dict.get('name', '')}")
        else: return node_dict.get('id') or node_dict.get('name') or f"{node_type}_{hash(str(node_dict))}"

    def _get_node_style(self, node_type: str, node_dict: Dict) -> Tuple[int, str]:
        """Get node size and color based on type and properties"""
        color_map = {
            'User': '#FF6B6B', 'Activity': '#4ECDC4', 'Document': '#45B7D1', 'Entity': '#96CEB4',
            'Session': '#FFEAA7', 'Context': '#DDA0DD', 'Preference': '#FFB6C1', 'Skill': '#98FB98',
            'File': '#87CEEB', 'Conversation': '#F0E68C', 'FavoriteView': '#FFDAB9', 'QueryPattern': '#ADD8E6',
            'InsightType': '#FFB347', 'ContextVector': '#E6E6FA', 'Correlation': '#F0E68C', 
            'PredictiveModel': '#98FB98', 'Unknown': '#CCCCCC'
        }
        size_map = {
            'User': 60, 'Activity': 40, 'Preference': 35, 'Skill': 35, 'Session': 30,
            'Entity': 25, 'File': 25, 'Document': 30, 'FavoriteView': 30, 'QueryPattern': 30, 
            'InsightType': 30, 'ContextVector': 35, 'Correlation': 40, 'PredictiveModel': 45
        }
        base_size = size_map.get(node_type, 25)
        return base_size, color_map.get(node_type, color_map['Unknown'])
    
    def _create_node_label(self, node_type: str, node_dict: Dict) -> str:
        """Create informative node labels for visualization"""
        if node_type == 'User': return f"{node_dict.get('name', 'User')}\n({node_dict.get('engagement_level', 'user')})"
        elif node_type == 'Activity': return f"{node_dict.get('type', 'Activity')}\n({node_dict.get('satisfaction_score', 0):.1f}★)"
        elif node_type == 'Preference': return f"{node_dict.get('type', 'Preference')}\n({node_dict.get('strength', 0):.1f} strength)"
        elif node_type == 'FavoriteView': return f"{node_dict.get('name', 'Favorite')}\n({node_dict.get('description', '')[:20]}...)"
        elif node_type == 'QueryPattern': return f"{node_dict.get('name', 'Pattern')}\n({node_dict.get('pattern_definition', '')[:20]}...)"
        elif node_type == 'InsightType': return f"{node_dict.get('insight_category', 'Insight')}\n({node_dict.get('description', '')[:20]}...)"
        else: return node_dict.get('name', node_dict.get('type', node_dict.get('id', 'Node')))
    
    def get_all_users_data(self) -> List[Dict[str, Any]]:
        """Get data for all users, useful for multi-user visualization context"""
        if not self.driver: return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (u:User)
                    OPTIONAL MATCH (u)-[:PERFORMED]->(a:Activity)
                    OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                    OPTIONAL MATCH (u)-[:HAS_EXPLICIT_FAVORITE]->(fv:FavoriteView)
                    OPTIONAL MATCH (u)-[:FOLLOWS_PATTERN]->(qp:QueryPattern)
                    OPTIONAL MATCH (u)-[:PREFERS_INSIGHT]->(it:InsightType)
                    RETURN u, count(a) as activity_count, count(p) as preference_count, count(fv) as favorite_view_count, count(qp) as query_pattern_count, count(it) as insight_type_count
                    ORDER BY u.total_activities DESC
                """)
                
                users_summary = []
                for record in result:
                    user_data = dict(record['u'])
                    users_summary.append({
                        'user': user_data,
                        'activity_count': record['activity_count'], 'preference_count': record['preference_count'],
                        'favorite_view_count': record['favorite_view_count'], 'query_pattern_count': record['query_pattern_count'],
                        'insight_type_count': record['insight_type_count']
                    })
                return users_summary
                
        except Exception as e:
            logger.error(f"Error getting all users data: {e}", exc_info=True)
            return []
    
    def create_multi_user_graph_data(self) -> Dict[str, Any]:
        """Create graph data representing all users and their direct relationships."""
        if not self.driver: 
            logger.warning("Neo4j driver not available, returning empty multi-user graph data.")
            return {'nodes': [], 'edges': [], 'total_nodes': 0, 'total_edges': 0}
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m) WHERE NOT type(r) IN ['HAS_EXPLICIT_FAVORITE', 'FOLLOWS_PATTERN', 'PREFERS_INSIGHT']
                    RETURN n, collect(DISTINCT {rel: r, target: m, rel_type: type(r)}) as connections
                """)
                
                nodes = []
                edges = []
                node_ids_added = set()
                
                for record in result:
                    node = record['n']
                    node_dict = dict(node) if hasattr(node, 'items') else {}
                    node_labels = list(node.labels) if hasattr(node, 'labels') else []
                    node_type = node_labels[0] if node_labels else 'Unknown'
                    node_id = self._get_node_id(node_type, node_dict)
                    
                    if node_id not in node_ids_added:
                        size, color = self._get_node_style(node_type, node_dict)
                        nodes.append({'id': node_id, 'label': self._create_node_label(node_type, node_dict), 'type': node_type, 'properties': node_dict, 'size': size, 'color': color})
                        node_ids_added.add(node_id)
                    
                    for conn_info in record['connections']:
                        rel, target_node, rel_type = conn_info['rel'], conn_info['target'], conn_info['rel_type']
                        if target_node:
                            target_dict = dict(target_node) if hasattr(target_node, 'items') else {}
                            target_labels = list(target_node.labels) if hasattr(target_node, 'labels') else []
                            target_type = target_labels[0] if target_labels else 'Unknown'
                            target_node_id = self._get_node_id(target_type, target_dict)
                            
                            if target_node_id not in node_ids_added:
                                size, color = self._get_node_style(target_type, target_dict)
                                nodes.append({'id': target_node_id, 'label': self._create_node_label(target_type, target_dict), 'type': target_type, 'properties': target_dict, 'size': size, 'color': color})
                                node_ids_added.add(target_node_id)
                            edges.append({'source': node_id, 'target': target_node_id, 'relationship': rel_type, 'label': rel_type.replace('_', ' ').title(), 'properties': {}})
                
                return {'nodes': nodes, 'edges': edges, 'total_nodes': len(nodes), 'total_edges': len(edges)}
                
        except Exception as e:
            logger.error(f"Error creating multi-user graph data: {e}", exc_info=True)
            return {'nodes': [], 'edges': [], 'total_nodes': 0, 'total_edges': 0}


# --- Enhanced Entity Extractor ---
class EnhancedEntityExtractor:
    """Enhanced entity extraction with better error handling and validation"""
    
    def __init__(self):
        self.nlp = None
        if not SPACY_AVAILABLE:
            logger.warning("SpaCy not available. Entity extraction will be limited.")
            return
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.warning("SpaCy English model not found. Please install it: python -m spacy download en_core_web_sm")
        except Exception as e:
            logger.error(f"Unexpected error loading SpaCy model: {e}")
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text with better error handling and validation"""
        if not self.nlp or not text: return []
        try:
            clean_text = self._clean_text(text)
            if not clean_text: return []
            doc = self.nlp(clean_text)
            entities = []
            for ent in doc.ents:
                if self._is_valid_entity(ent):
                    entities.append({'text': ent.text.strip(), 'label': ent.label_, 'description': spacy.explain(ent.label_) if SPACY_AVAILABLE else ent.label_, 'start': ent.start_char, 'end': ent.end_char})
            return entities
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}", exc_info=True)
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text to remove binary or corrupted data and limit length."""
        try:
            import string
            printable_chars = set(string.printable)
            cleaned_chars = [char for char in text if char in printable_chars or char.isspace()]
            cleaned_text = "".join(cleaned_chars)
            MAX_TEXT_LENGTH = 5000
            if len(cleaned_text) > MAX_TEXT_LENGTH: cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]
            return cleaned_text.strip()
        except Exception: return str(text)[:1000] 
    
    def _is_valid_entity(self, ent) -> bool:
        """Check if an entity is valid based on several criteria."""
        text = ent.text.strip()
        if len(text) < 2: return False
        if text.isdigit(): return False 
        if text.islower() and len(text) < 4: return False 
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count == 0 and len(text) > 1: return False
        return True


# --- Knowledge Graph Visualizer ---
class KnowledgeGraphVisualizer:
    """Enhanced Knowledge Graph Visualizer with improved styling and data handling"""
    
    def __init__(self):
        self.color_map = {
            'User': '#FF6B6B', 'Activity': '#4ECDC4', 'Document': '#45B7D1', 'Entity': '#96CEB4',
            'Session': '#FFEAA7', 'Context': '#DDA0DD', 'Preference': '#FFB6C1', 'Skill': '#98FB98',
            'File': '#87CEEB', 'Conversation': '#F0E68C', 'FavoriteView': '#FFDAB9', 'QueryPattern': '#ADD8E6',
            'InsightType': '#FFB347', 'ContextVector': '#E6E6FA', 'Correlation': '#F0E68C', 
            'PredictiveModel': '#98FB98', 'Recommendation': '#FFE4E1', 'Unknown': '#CCCCCC'
        }
        self.default_node_size = 300
        self.node_size_map = {
            'User': 60, 'Activity': 40, 'Preference': 35, 'Skill': 35, 'Session': 30,
            'Entity': 25, 'File': 25, 'Document': 30, 'FavoriteView': 30, 'QueryPattern': 30, 
            'InsightType': 30, 'ContextVector': 35, 'Correlation': 40, 'PredictiveModel': 45,
            'Recommendation': 25
        }
    
    def create_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Create NetworkX graph from structured graph data"""
        G = nx.Graph()
        for node_data in graph_data.get('nodes', []):
            node_id = node_data['id']
            node_type = node_data['type']
            size = self.node_size_map.get(node_type, self.default_node_size)
            color = self.color_map.get(node_type, self.color_map['Unknown'])
            cleaned_properties = self._clean_node_properties(node_data.get('properties', {}))
            G.add_node(node_id, label=node_data.get('label', node_id), type=node_type, color=color, size=size, **cleaned_properties)
        
        for edge_data in graph_data.get('edges', []):
            source_id, target_id = edge_data.get('source'), edge_data.get('target')
            if source_id in G.nodes() and target_id in G.nodes():
                cleaned_edge_properties = self._clean_edge_properties(edge_data.get('properties', {}))
                G.add_edge(source_id, target_id, relationship=edge_data.get('relationship'), label=edge_data.get('label', edge_data.get('relationship', '')), **cleaned_edge_properties)
        return G
    
    def _clean_node_properties(self, props: Dict) -> Dict:
        """Clean node properties to be compatible with NetworkX add_node."""
        cleaned = {}
        for key, value in props.items():
            if key not in ['id', 'label', 'type', 'color', 'size'] and isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
        return cleaned

    def _clean_edge_properties(self, props: Dict) -> Dict:
        """Clean edge properties to be compatible with NetworkX add_edge."""
        cleaned = {}
        for key, value in props.items():
            if key not in ['source', 'target', 'relationship', 'label'] and isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
        return cleaned

    def visualize_with_matplotlib(self, graph_data: Dict[str, Any], title: str = "Enhanced Knowledge Graph", 
                                 figsize: Tuple[int, int] = (18, 14), save_path: str = None) -> plt.Figure:
        """Create enhanced visualization using NetworkX and Matplotlib"""
        G = self.create_networkx_graph(graph_data)
        if len(G.nodes()) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data available for visualization', ha='center', va='center', fontsize=16, color='gray')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title, fontsize=16, fontweight='bold'); ax.axis('off'); return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        layout_k = 0.5 / np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 0.1
        pos = nx.spring_layout(G, k=max(0.1, layout_k), iterations=50, seed=42)
        
        node_types_present = sorted(list(set(data['type'] for _, data in G.nodes(data=True))))
        
        for node_type in node_types_present:
            nodes_of_type = [node for node, data in G.nodes(data=True) if data['type'] == node_type]
            node_colors = [G.nodes[node]['color'] for node in nodes_of_type]
            node_sizes = [G.nodes[node]['size'] for node in nodes_of_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax, label=node_type)
        
        nx.draw_networkx_edges(G, pos, edge_color='grey', alpha=0.6, width=1.5, ax=ax)
        
        node_labels = {}
        for node, data in G.nodes(data=True):
            label = data.get('label', node)
            max_label_len = 25
            if len(label) > max_label_len: label = label[:max_label_len-3] + '...'
            node_labels[node] = label
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)
        
        legend_elements = []
        for node_type in sorted(self.color_map.keys()):
            if node_type in node_types_present:
                color = self.color_map[node_type]
                count = len([n for n, d in G.nodes(data=True) if d['type'] == node_type])
                legend_elements.append(mpatches.Patch(color=color, label=f'{node_type} ({count})'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., title="Node Types")
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graph visualization saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save visualization to {save_path}: {e}")
        return fig


# --- Enhanced Azure Memory Agent with Advanced Context Engineering ---
class EnhancedAzureMemoryAgent:
    """
    Enhanced Azure Memory Agent with Advanced Context Engineering, Cross-Memory Correlation,
    Automatic Insight Generation, Predictive Modeling, and Context-Aware Recommendations.
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        
        self.setup_azure_config()
        self.setup_azure_clients()
        self.setup_embedding_model()
        self.setup_vector_database()
        self.setup_langchain_knowledge_graph()
        self.setup_entity_extractor()
        
        # Initialize Advanced Context Engine
        self.context_engine = AdvancedContextEngine(self.embedding_model, self.llm)
        
        self.current_conversation: List[AIMessage | HumanMessage | SystemMessage] = []
        self.session_activities: List[Dict[str, Any]] = []
        self.current_file_info: Dict[str, Any] = {}
        
        self.visualizer = KnowledgeGraphVisualizer()
        self.initialize_user_profile()
        
        logger.info(f"Enhanced Azure Memory Agent with Context Engineering initialized for user: {user_id}")
    
    def setup_azure_config(self):
        self.azure_config = {
            'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT', ''),
            'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY', ''),
            'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
            'AZURE_OPENAI_DEPLOYMENT_NAME': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o'),
            'AZURE_OPENAI_MODEL_NAME': os.getenv('AZURE_OPENAI_MODEL_NAME', 'gpt-4o'),
            'AZURE_OPENAI_TEMPERATURE': float(os.getenv('AZURE_OPENAI_TEMPERATURE', '0.1')),
            'AZURE_OPENAI_MAX_TOKENS': int(os.getenv('AZURE_OPENAI_MAX_TOKENS', '4000'))
        }
        if not all([self.azure_config['AZURE_OPENAI_ENDPOINT'], self.azure_config['AZURE_OPENAI_API_KEY']]):
            logger.warning("Azure OpenAI Endpoint or API Key not set in environment variables.")
    
    def setup_azure_clients(self):
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.azure_config['AZURE_OPENAI_ENDPOINT'], api_key=self.azure_config['AZURE_OPENAI_API_KEY'],
                api_version=self.azure_config['AZURE_OPENAI_API_VERSION'], deployment_name=self.azure_config['AZURE_OPENAI_DEPLOYMENT_NAME'],
                model_name=self.azure_config['AZURE_OPENAI_MODEL_NAME'], temperature=self.azure_config['AZURE_OPENAI_TEMPERATURE'],
                max_tokens=self.azure_config['AZURE_OPENAI_MAX_TOKENS']
            )
            logger.info("Azure OpenAI chat client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI chat client: {e}", exc_info=True)
            raise
    
    def setup_embedding_model(self):
        try:
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            test_embedding = self.embedding_model.encode("test sentence")
            self.embedding_dimension = len(test_embedding)
            logger.info(f"Embedding model '{model_name}' loaded successfully. Dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
            raise
    
    def setup_vector_database(self):
        try:
            data_dir = Path("./memory_data")
            data_dir.mkdir(exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(data_dir), settings=Settings(anonymized_telemetry=False))
            self.episodic_collection = self.get_or_create_collection_with_dimension("episodic_memory", "Historical conversations and learnings", self.embedding_dimension)
            self.semantic_collection = self.get_or_create_collection_with_dimension("semantic_memory", "Factual knowledge and documents", self.embedding_dimension)
            logger.info("ChromaDB vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector database: {e}", exc_info=True)
            raise
    
    def get_or_create_collection_with_dimension(self, name: str, description: str, embedding_dimension: int):
        """Safely gets or creates a ChromaDB collection, ensuring correct embedding dimension."""
        try:
            collection = self.chroma_client.get_collection(name)
            logger.info(f"Found existing ChromaDB collection: {name}")
            existing_metadata = collection.metadata
            existing_dimension = existing_metadata.get('embedding_dimension') if existing_metadata else None
            
            if existing_dimension == embedding_dimension:
                logger.info(f"Collection '{name}' has the correct embedding dimension ({embedding_dimension}).")
                return collection
            else:
                logger.warning(f"Collection '{name}' dimension mismatch: Found {existing_dimension}, expected {embedding_dimension}.")
                logger.info(f"Recreating collection '{name}' with correct dimension.")
                self.chroma_client.delete_collection(name)
                raise ValueError("Collection dimension mismatch detected.")

        except Exception as e: 
            if "not found" in str(e).lower() or "dimension mismatch" in str(e).lower():
                try:
                    collection = self.chroma_client.create_collection(
                        name=name,
                        metadata={"description": description, "embedding_dimension": embedding_dimension,
                                  "embedding_model": os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                                  "created_at": datetime.now().isoformat()})
                    logger.info(f"Created new ChromaDB collection: '{name}' with dimension {embedding_dimension}")
                    return collection
                except Exception as create_error:
                    logger.error(f"Failed to create ChromaDB collection '{name}': {create_error}", exc_info=True)
                    raise
            else:
                logger.error(f"Error accessing or creating ChromaDB collection '{name}': {e}", exc_info=True)
                raise
    
    def setup_langchain_knowledge_graph(self):
        """Initialize LangChain Neo4j Graph manager."""
        neo4j_url, neo4j_username, neo4j_password = os.getenv('NEO4J_URI'), os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')
        neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        if not all([neo4j_url, neo4j_username, neo4j_password]):
            logger.warning("Neo4j URI, Username, or Password not set. Knowledge Graph features will be unavailable.")
            self.knowledge_graph = None; return
        try:
            self.knowledge_graph = LangChainKnowledgeGraphManager(llm=self.llm, neo4j_url=neo4j_url, neo4j_username=neo4j_username, neo4j_password=neo4j_password, database=neo4j_database)
        except Exception as e:
            logger.error(f"Failed to initialize LangChainKnowledgeGraphManager: {e}", exc_info=True)
            self.knowledge_graph = None
    
    def setup_entity_extractor(self): 
        self.entity_extractor = EnhancedEntityExtractor()
    
    def initialize_user_profile(self):
        """Create the initial User node in the knowledge graph upon agent initialization."""
        if self.knowledge_graph and self.knowledge_graph.driver:
            initial_activity = {'type': 'System', 'subtype': 'AgentInitialization', 'timestamp': datetime.now().isoformat(), 'duration': 1, 'satisfaction_score': 1.0, 'session_id': self.session_id, 'user_id': self.user_id}
            self.session_activities.append(initial_activity)
            self.knowledge_graph.create_user_activity_graph(self.user_id, [initial_activity])
        else: logger.warning("Knowledge graph not available. User profile initialization skipped.")
    
    def run_advanced_memory_synthesis(self) -> Dict[str, Any]:
        """Run the complete advanced memory synthesis pipeline"""
        synthesis_results = {
            'user_id': self.user_id,
            'synthesis_timestamp': datetime.now().isoformat(),
            'context_vectors': [],
            'correlations': [],
            'insights': [],
            'predictive_model': None,
            'recommendations': [],
            'synthesis_metrics': {}
        }
        
        try:
            logger.info(f"Starting advanced memory synthesis for user {self.user_id}")
            start_time = time.time()
            
            # Step 1: Extract context vectors from different memory types
            context_vectors = self._extract_all_context_vectors()
            synthesis_results['context_vectors'] = [asdict(cv) for cv in context_vectors]
            
            # Step 2: Find cross-memory correlations
            correlations = self.context_engine.find_cross_memory_correlations(self.user_id)
            synthesis_results['correlations'] = [asdict(corr) for corr in correlations]
            
            # Step 3: Generate automatic insights
            insights = self.context_engine.generate_automatic_insights(self.user_id, context_vectors)
            synthesis_results['insights'] = [asdict(insight) for insight in insights]
            
            # Step 4: Build predictive behavior model
            predictive_model = self.context_engine.build_predictive_model(self.user_id, self.session_activities)
            if predictive_model:
                synthesis_results['predictive_model'] = asdict(predictive_model)
            
            # Step 5: Generate context-aware recommendations
            current_context = self._get_current_context()
            recommendations = self.context_engine.generate_context_aware_recommendations(self.user_id, current_context)
            synthesis_results['recommendations'] = recommendations
            
            # Step 6: Store results in knowledge graph
            if self.knowledge_graph:
                self.knowledge_graph.store_context_vectors(self.user_id, context_vectors)
                self.knowledge_graph.store_correlations(correlations)
                self.knowledge_graph.store_insights(insights)
                if predictive_model:
                    self.knowledge_graph.store_predictive_model(predictive_model)
            
            # Calculate synthesis metrics
            processing_time = time.time() - start_time
            synthesis_results['synthesis_metrics'] = {
                'processing_time_seconds': processing_time,
                'context_vectors_generated': len(context_vectors),
                'correlations_found': len(correlations),
                'insights_generated': len(insights),
                'recommendations_created': len(recommendations),
                'predictive_model_created': predictive_model is not None
            }
            
            logger.info(f"Advanced memory synthesis completed in {processing_time:.2f}s")
            logger.info(f"Generated: {len(context_vectors)} contexts, {len(correlations)} correlations, {len(insights)} insights, {len(recommendations)} recommendations")
            
            return synthesis_results
            
        except Exception as e:
            logger.error(f"Error during advanced memory synthesis: {e}", exc_info=True)
            synthesis_results['error'] = str(e)
            return synthesis_results
    
    def _extract_all_context_vectors(self) -> List[ContextVector]:
        """Extract context vectors from all available data sources"""
        context_vectors = []
        
        try:
            # Extract temporal context from activities
            temporal_context = self.context_engine.extract_temporal_context(self.session_activities)
            if temporal_context:
                context_vectors.append(temporal_context)
            
            # Extract behavioral context from activities
            behavioral_context = self.context_engine.extract_behavioral_context(self.session_activities)
            if behavioral_context:
                context_vectors.append(behavioral_context)
            
            # Extract semantic context from documents and entities
            documents = self._get_processed_documents()
            entities = self._get_extracted_entities()
            semantic_context = self.context_engine.extract_semantic_context(documents, entities)
            if semantic_context:
                context_vectors.append(semantic_context)
            
            logger.info(f"Extracted {len(context_vectors)} context vectors")
            
        except Exception as e:
            logger.error(f"Error extracting context vectors: {e}", exc_info=True)
        
        return context_vectors
    
    def _get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get processed documents from semantic memory"""
        documents = []
        try:
            if self.semantic_collection:
                # Get all documents from ChromaDB
                results = self.semantic_collection.get()
                if results and results.get('documents'):
                    for i, doc_content in enumerate(results['documents']):
                        metadata = results.get('metadatas', [{}])[i] if i < len(results.get('metadatas', [])) else {}
                        documents.append({
                            'content': doc_content,
                            'metadata': metadata
                        })
        except Exception as e:
            logger.error(f"Error retrieving processed documents: {e}", exc_info=True)
        
        return documents
    
    def _get_extracted_entities(self) -> List[Dict[str, Any]]:
        """Get extracted entities from activities and documents"""
        entities = []
        try:
            # Collect entities from session activities
            for activity in self.session_activities:
                activity_entities = activity.get('entities_involved', [])
                entities.extend(activity_entities)
            
            # Extract entities from current file context if available
            if self.current_file_info.get('file_path'):
                # This would ideally reload and re-extract entities from the current file
                pass
                
        except Exception as e:
            logger.error(f"Error retrieving extracted entities: {e}", exc_info=True)
        
        return entities
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for recommendation generation"""
        return {
            'current_time': datetime.now(),
            'session_duration': (datetime.now() - self.session_start_time).total_seconds(),
            'activities_this_session': len(self.session_activities),
            'has_current_file': bool(self.current_file_info.get('file_path')),
            'conversation_length': len(self.current_conversation),
            'immediate_task': len(self.current_conversation) > 0,  # User is actively engaged
            'seeking_improvement': any('improve' in msg.content.lower() for msg in self.current_conversation[-3:] if hasattr(msg, 'content'))
        }
    
    # Enhanced methods with context integration
    def process_document(self, file_path: str, content: str) -> Dict[str, Any]:
        """Process a document with enhanced context extraction"""
        if not self.semantic_collection or not self.knowledge_graph:
            error_msg = "Semantic collection or Knowledge Graph not initialized."
            logger.error(error_msg); return {'success': False, 'error': error_msg}

        try:
            start_time = time.time()
            entities = self.entity_extractor.extract_entities(content)
            doc_id = str(uuid.uuid4())
            metadata = {'doc_id': doc_id, 'file_path': str(file_path), 'file_type': Path(file_path).suffix.lower().lstrip('.'), 'user_id': str(self.user_id), 'session_id': str(self.session_id), 'processing_timestamp': datetime.now().isoformat(), 'entities_count': len(entities), 'file_size_chars': len(content)}
            safe_entity_names = [str(e.get('text')) for e in entities if isinstance(e, dict) and e.get('text')]
            metadata['entities_sample'] = ', '.join(safe_entity_names[:5]) if safe_entity_names else ''
            
            embedding = self.embed_text(content)
            if not embedding: raise ValueError("Failed to generate embedding for the document content.")
            
            try:
                self.semantic_collection.add(documents=[content], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
                logger.info(f"Document '{file_path}' added to semantic memory (ChromaDB). ID: {doc_id}")
            except Exception as chromadb_error:
                if "dimension" in str(chromadb_error).lower():
                    logger.warning(f"ChromaDB dimension mismatch for document '{file_path}': {chromadb_error}")
                    if self.reset_vector_database():
                        logger.info("Retrying document addition after database reset...")
                        self.semantic_collection.add(documents=[content], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
                        logger.info("Document added successfully after reset.")
                    else: raise Exception("Failed to reset vector database to resolve dimension mismatch.") from chromadb_error
                else: raise chromadb_error
            
            processing_duration = time.time() - start_time
            metadata['processing_duration_sec'] = round(processing_duration, 3)
            
            # Enhanced activity tracking with context
            self.track_user_activity("DocumentProcessing", {
                "subtype": "Success", "file_path": str(file_path), "doc_id": doc_id, 
                "duration_sec": processing_duration, "satisfaction_score": 0.8, 
                "entities_involved": entities, "files_accessed": [str(file_path)],
                "context_extraction_enabled": True
            })
            
            self.current_file_info = {'file_path': str(file_path), 'doc_id': doc_id, 'processed_at': datetime.now().isoformat()}
            
            # Trigger context synthesis after document processing
            synthesis_results = self.run_advanced_memory_synthesis()
            
            return {
                'success': True, 'doc_id': doc_id, 'entities': entities, 'metadata': metadata, 
                'processing_duration_sec': processing_duration, 'synthesis_results': synthesis_results
            }
            
        except Exception as e:
            processing_duration = time.time() - start_time
            self.track_user_activity("DocumentProcessing", {"subtype": "Failure", "file_path": str(file_path), "error_message": str(e)[:200], "duration_sec": processing_duration, "satisfaction_score": 0.2})
            logger.error(f"Document processing failed for '{file_path}': {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'processing_duration_sec': processing_duration}

    def track_user_activity(self, activity_type: str, activity_data: Dict[str, Any]):
        """Track user activity with enhanced context extraction"""
        try:
            enhanced_activity_data = {**activity_data, "type": activity_type, "session_id": self.session_id, "timestamp": datetime.now().isoformat(), "user_id": self.user_id}
            self.session_activities.append(enhanced_activity_data)
            if self.knowledge_graph: self.knowledge_graph.create_user_activity_graph(self.user_id, [enhanced_activity_data])
            logger.debug(f"User activity tracked: {activity_type}")
        except Exception as e: logger.error(f"Failed to track user activity '{activity_type}': {e}", exc_info=True)

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for text using the configured embedding model."""
        if not self.embedding_model: logger.warning("Embedding model not available."); return None
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}", exc_info=True)
            return None

    def reset_vector_database(self):
        """Resets all ChromaDB collections, useful for fixing embedding dimension issues."""
        logger.warning("Attempting to reset all ChromaDB collections...")
        collections_to_reset = ["episodic_memory", "semantic_memory"]
        try:
            for coll_name in collections_to_reset:
                try: self.chroma_client.delete_collection(coll_name); logger.info(f"Deleted existing collection: {coll_name}")
                except Exception: pass # Ignore if collection doesn't exist
            
            self.episodic_collection = self.get_or_create_collection_with_dimension("episodic_memory", "Historical conversations and learnings", self.embedding_dimension)
            self.semantic_collection = self.get_or_create_collection_with_dimension("semantic_memory", "Factual knowledge and documents", self.embedding_dimension)
            logger.info(f"Vector database reset and collections recreated successfully with dimension {self.embedding_dimension}.")
            return True
        except Exception as e:
            logger.error(f"Failed during vector database reset process: {e}", exc_info=True)
            return False

    def enhanced_query_processing(self, query: str) -> Dict[str, Any]:
        """Process a query using ChromaDB semantic search, Knowledge Graph context, and synthesis insights."""
        if not self.semantic_collection or not self.knowledge_graph:
            error_msg = "Semantic collection or Knowledge Graph not initialized."
            logger.error(error_msg); return {'query': query, 'error': error_msg}

        try:
            start_time = time.time()
            results = {'query': query, 'chromadb_results': [], 'knowledge_graph_context': '', 'entities_in_query': [], 'synthesis_insights': [], 'recommendations': [], 'query_activity_subtype': 'QueryProcessed'}
            
            query_entities = self.entity_extractor.extract_entities(query)
            results['entities_in_query'] = query_entities
            
            # ChromaDB semantic search
            query_embedding = self.embed_text(query)
            if query_embedding:
                try:
                    chromadb_search_results = self.semantic_collection.query(query_embeddings=[query_embedding], n_results=5, include=['documents', 'metadatas', 'distances'])
                    if chromadb_search_results and chromadb_search_results.get('documents') and chromadb_search_results['documents'][0]:
                        for i, doc in enumerate(chromadb_search_results['documents'][0]):
                            metadata = chromadb_search_results['metadatas'][0][i] if chromadb_search_results['metadatas'] else {}
                            distance = chromadb_search_results['distances'][0][i] if chromadb_search_results['distances'] else 1.0
                            results['chromadb_results'].append({'content_preview': doc[:150] + "..." if len(doc) > 150 else doc, 'metadata': metadata, 'similarity': 1 - distance})
                except Exception as chromadb_err:
                    if "dimension" in str(chromadb_err).lower():
                        logger.warning(f"ChromaDB dimension mismatch during query: {chromadb_err}"); results['query_activity_subtype'] = 'QueryFailed_DimensionMismatch'
                    else: logger.error(f"Error querying ChromaDB: {chromadb_err}", exc_info=True); results['query_activity_subtype'] = 'QueryFailed_ChromaDBError'
            else: results['query_activity_subtype'] = 'QueryFailed_EmbeddingError'
            
            # Knowledge Graph context
            graph_data = self.knowledge_graph.get_user_graph_data(self.user_id)
            if graph_data:
                context_parts = []
                user_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'User']
                if user_nodes:
                    user_props = user_nodes[0]['properties']
                    context_parts.append(f"User Profile: Engagement={user_props.get('engagement_level', 'N/A')}, Activities={user_props.get('total_activities', 0)}")
                
                # Enhanced context with synthesis results
                fav_views = [n for n in graph_data.get('nodes', []) if n['type'] == 'FavoriteView']
                if fav_views: context_parts.append(f"Favorite Views: {', '.join([fv['properties'].get('name', 'Unnamed') for fv in fav_views[:2]])}")
                
                qp_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'QueryPattern']
                if qp_nodes: context_parts.append(f"Query Patterns: {', '.join([qp['properties'].get('name', 'Unnamed') for qp in qp_nodes[:2]])}")
                
                it_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'InsightType']
                if it_nodes: context_parts.append(f"Insight Types: {', '.join([it['properties'].get('insight_category', 'Unnamed') for it in it_nodes[:2]])}")

                # Add context vectors and correlations
                cv_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'ContextVector']
                if cv_nodes: context_parts.append(f"Context Vectors: {len(cv_nodes)} available")
                
                corr_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'Correlation']
                if corr_nodes: context_parts.append(f"Cross-Memory Correlations: {len(corr_nodes)} identified")

                results['knowledge_graph_context'] = "\n".join(context_parts)
            
            # Get synthesis insights and recommendations
            current_context = self._get_current_context()
            synthesis_recommendations = self.context_engine.generate_context_aware_recommendations(self.user_id, current_context)
            results['recommendations'] = synthesis_recommendations[:5]  # Top 5 recommendations
            
            # Get recent insights
            recent_insights = [insight for insight in self.context_engine.insights.values() if insight.user_relevance_score > 0.7]
            results['synthesis_insights'] = [{'description': insight.description, 'confidence': insight.confidence_score, 'recommendations': insight.actionable_recommendations[:2]} for insight in recent_insights[:3]]
            
            processing_duration = time.time() - start_time
            self.track_user_activity("Query", {
                "subtype": results['query_activity_subtype'], "query_text": query[:100], 
                "chromadb_results_count": len(results['chromadb_results']), 
                "kg_context_generated": bool(results['knowledge_graph_context']), 
                "entities_in_query": len(query_entities), "duration_sec": processing_duration, 
                "satisfaction_score": 0.8 if results['chromadb_results'] or results['knowledge_graph_context'] else 0.5,
                "synthesis_insights_count": len(results['synthesis_insights']),
                "recommendations_count": len(results['recommendations'])
            })
            
            return results
            
        except Exception as e:
            self.track_user_activity("Query", {"subtype": "QueryProcessingFailed", "query_text": query[:100], "error_message": str(e)[:200], "duration_sec": time.time() - start_time, "satisfaction_score": 0.1})
            logger.error(f"Enhanced query processing failed for query '{query}': {e}", exc_info=True)
            return {'query': query, 'error': str(e), 'knowledge_graph_context': ''}

    def process_user_input(self, user_input: str) -> str:
        """Process user input with enhanced context engineering and synthesis integration."""
        try:
            start_time = time.time()
            
            # Enhanced command parsing
            save_view_match = re.match(r"SAVE VIEW \"?(.*?)\"?\s+AS \"?(.*?)\"?\s+QUERY \"?(.*?)\"?", user_input, re.IGNORECASE | re.DOTALL)
            if save_view_match:
                view_name, description, cypher_query = save_view_match.groups()
                if self.knowledge_graph:
                    success = self.knowledge_graph.save_explicit_favorite_view(self.user_id, view_name, cypher_query, description)
                    if success:
                        self.track_user_activity("UserCommand", {"subtype": "SaveFavoriteView", "view_name": view_name, "description_preview": description[:50], "satisfaction_score": 0.9})
                        return f"✅ Favorite view '{view_name}' saved successfully!"
                    else: return "❌ Failed to save favorite view. Check logs for Neo4j errors."
                else: return "❌ Knowledge graph is not available. Cannot save favorite view."
            
            # Synthesis command
            if user_input.lower().strip() in ['run synthesis', 'synthesis', 'run memory synthesis', 'analyze memory']:
                synthesis_results = self.run_advanced_memory_synthesis()
                if 'error' in synthesis_results:
                    return f"❌ Memory synthesis failed: {synthesis_results['error']}"
                
                metrics = synthesis_results['synthesis_metrics']
                response_parts = [
                    "🧠 **Advanced Memory Synthesis Complete**",
                    f"⏱️ Processing Time: {metrics['processing_time_seconds']:.2f}s",
                    f"📊 Generated: {metrics['context_vectors_generated']} contexts, {metrics['correlations_found']} correlations, {metrics['insights_generated']} insights",
                    f"🎯 Recommendations: {metrics['recommendations_created']} created"
                ]
                
                if synthesis_results['insights']:
                    response_parts.append("\n**🔍 Key Insights:**")
                    for insight in synthesis_results['insights'][:3]:
                        response_parts.append(f"• {insight['description']}")
                
                if synthesis_results['recommendations']:
                    response_parts.append("\n**💡 Recommendations:**")
                    for rec in synthesis_results['recommendations'][:3]:
                        response_parts.append(f"• {rec['message']}")
                
                return "\n".join(response_parts)
            
            # General conversation processing with enhanced context
            user_message = HumanMessage(content=user_input)
            self.current_conversation.append(user_message)
            
            # Get enhanced context
            graph_data = self.knowledge_graph.get_user_graph_data(self.user_id) if self.knowledge_graph else None
            synthesis_context = self._get_synthesis_context()
            
            system_prompt = self.generate_enhanced_context_aware_system_prompt(user_input, graph_data, synthesis_context)
            system_message = SystemMessage(content=system_prompt)
            
            messages = [system_message] + self.current_conversation[-18:]
            
            response = self.llm.invoke(messages)
            self.current_conversation.append(response)
            
            processing_duration = time.time() - start_time
            self.track_user_activity("Conversation", {
                "subtype": "GeneralChat", "user_input_preview": user_input[:80], 
                "response_preview": response.content[:80], "duration_sec": processing_duration, 
                "entities_in_input": len(self.entity_extractor.extract_entities(user_input)), 
                "satisfaction_score": 0.75, "enhanced_context_used": True
            })
            
            return response.content
            
        except Exception as e:
            processing_duration = time.time() - start_time
            self.track_user_activity("Conversation", {"subtype": "ConversationFailed", "user_input_preview": user_input[:80], "error_message": str(e)[:200], "duration_sec": processing_duration, "satisfaction_score": 0.1})
            logger.error(f"Failed to process user input '{user_input}': {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _get_synthesis_context(self) -> Dict[str, Any]:
        """Get context from synthesis results"""
        context = {
            'recent_insights': [],
            'active_correlations': [],
            'recommendations': [],
            'predictive_indicators': []
        }
        
        try:
            # Get recent insights
            recent_insights = [insight for insight in self.context_engine.insights.values() 
                             if (datetime.now() - datetime.fromisoformat(insight.temporal_context.get('analysis_period', datetime.now().isoformat()))).days < 7]
            context['recent_insights'] = [insight.description for insight in recent_insights[:3]]
            
            # Get active correlations
            active_correlations = [corr for corr in self.context_engine.correlations.values() 
                                 if corr.confidence_score > 0.8]
            context['active_correlations'] = [corr.pattern_description for corr in active_correlations[:3]]
            
            # Get current recommendations
            current_context = self._get_current_context()
            recommendations = self.context_engine.generate_context_aware_recommendations(self.user_id, current_context)
            context['recommendations'] = [rec['message'] for rec in recommendations[:3]]
            
        except Exception as e:
            logger.error(f"Error getting synthesis context: {e}", exc_info=True)
        
        return context

    def generate_enhanced_context_aware_system_prompt(self, user_query: str, graph_data: Dict[str, Any], synthesis_context: Dict[str, Any]) -> str:
        """Generate enhanced system prompt with synthesis context"""
        prompt_parts = [
            "You are an advanced AI assistant with sophisticated memory systems and context engineering capabilities:",
            "- ChromaDB for vector-based semantic search on documents",
            "- Neo4j Knowledge Graph for structured user activity, preferences, patterns, and insights",
            "- Advanced Context Engineering with cross-memory correlation analysis",
            "- Automatic insight generation and predictive behavior modeling",
            "- Context-aware recommendation engine",
            "",
            "--- SYSTEM CONTEXT ---",
            f"User ID: {self.user_id}", f"Session ID: {self.session_id}",
            f"Current File Context: {self.current_file_info.get('file_path', 'None')}",
            f"Session Duration: {(datetime.now() - self.session_start_time).total_seconds():.0f}s",
            f"Activities This Session: {len(self.session_activities)}",
            ""
        ]
        
        # Knowledge Graph Context
        if graph_data:
            prompt_parts.extend(["--- KNOWLEDGE GRAPH CONTEXT ---"])
            user_nodes = [n for n in graph_data.get('nodes', []) if n['type'] == 'User']
            if user_nodes:
                user_props = user_nodes[0]['properties']
                prompt_parts.append(f"User Profile: Engagement={user_props.get('engagement_level', 'N/A')}, Activities={user_props.get('total_activities', 0)}")
            
            context_vectors = len([n for n in graph_data.get('nodes', []) if n['type'] == 'ContextVector'])
            correlations = len([n for n in graph_data.get('nodes', []) if n['type'] == 'Correlation'])
            if context_vectors > 0 or correlations > 0:
                prompt_parts.append(f"Context Analysis: {context_vectors} context vectors, {correlations} correlations identified")
            prompt_parts.append("")
        
        # Synthesis Context
        if synthesis_context and any(synthesis_context.values()):
            prompt_parts.extend(["--- ADVANCED SYNTHESIS CONTEXT ---"])
            
            if synthesis_context.get('recent_insights'):
                prompt_parts.append("Recent Insights:")
                for insight in synthesis_context['recent_insights']:
                    prompt_parts.append(f"  • {insight}")
            
            if synthesis_context.get('active_correlations'):
                prompt_parts.append("Active Cross-Memory Correlations:")
                for correlation in synthesis_context['active_correlations']:
                    prompt_parts.append(f"  • {correlation}")
            
            if synthesis_context.get('recommendations'):
                prompt_parts.append("Context-Aware Recommendations:")
                for rec in synthesis_context['recommendations']:
                    prompt_parts.append(f"  • {rec}")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "--- ACTIVE TASK/QUERY ---", f"User's Current Query: {user_query}",
            "",
            "--- ENHANCED RESPONSE GUIDELINES ---",
            "1. Leverage ALL available context: knowledge graph, synthesis insights, correlations, and recommendations",
            "2. Provide highly personalized responses based on cross-memory analysis and behavioral patterns",
            "3. Proactively suggest actions based on predictive models and identified correlations",
            "4. Reference specific insights and recommendations when relevant to the user's query",
            "5. Use context engineering insights to anticipate user needs and provide preemptive guidance",
            "6. Maintain conversation continuity while integrating advanced memory synthesis results",
            "7. Suggest running 'synthesis' command if the user seems to need comprehensive analysis"
        ])
        return "\n".join(prompt_parts)

    def create_knowledge_graph_visualization(self, output_path: str = None, include_all_users: bool = False) -> Dict[str, Any]:
        """Generate enhanced visualization including synthesis results."""
        try:
            if not self.knowledge_graph: return {"error": "Knowledge graph not initialized."}
            
            graph_data = self.knowledge_graph.create_multi_user_graph_data() if include_all_users else self.knowledge_graph.get_user_graph_data(self.user_id)
            title = "Enhanced Multi-User Knowledge Graph with Context Analysis" if include_all_users else f"Enhanced Knowledge Graph for User: {self.user_id}"
            
            if not graph_data or not graph_data.get('nodes'): return {'error': 'No graph data available to visualize.'}
            
            results = {'graph_data': graph_data, 'visualizations': {}, 'statistics': self._generate_enhanced_graph_statistics(graph_data)}
            fig = self.visualizer.visualize_with_matplotlib(graph_data, title=title, save_path=output_path if output_path else None)
            
            if output_path: results['visualizations']['network_image_path'] = output_path
            else: results['visualizations']['matplotlib_figure'] = fig
            return results
            
        except Exception as e:
            logger.error(f"Error creating enhanced knowledge graph visualization: {e}", exc_info=True)
            return {"error": str(e)}

    def _generate_enhanced_graph_statistics(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced statistics including synthesis metrics."""
        stats = {
            'total_nodes': len(graph_data.get('nodes', [])), 'total_edges': len(graph_data.get('edges', [])),
            'node_types': defaultdict(int), 'relationship_types': defaultdict(int), 
            'session_activities_tracked': len(self.session_activities),
            'context_vectors_created': len(self.context_engine.context_vectors),
            'correlations_identified': len(self.context_engine.correlations),
            'insights_generated': len(self.context_engine.insights),
            'predictive_models_built': len(self.context_engine.predictive_models)
        }
        
        for node in graph_data.get('nodes', []): stats['node_types'][node['type']] += 1
        for edge in graph_data.get('edges', []): stats['relationship_types'][edge['relationship']] += 1
        
        return stats

    def get_system_health(self) -> Dict[str, Any]:
        """Enhanced system health check including context engine status."""
        health_status = {'overall_status': 'healthy', 'timestamp': datetime.now().isoformat(), 'user_id': self.user_id, 'components': {}}
        
        try:
            self.llm.invoke([HumanMessage(content="Health check")], config={'recursion_limit': 5})
            health_status['components']['azure_openai_llm'] = 'healthy'
        except Exception as e:
            health_status['components']['azure_openai_llm'] = f"unhealthy ({str(e)[:50]})"; health_status['overall_status'] = 'degraded'
            
        chroma_healthy = True
        try: self.episodic_collection.count(); health_status['components']['chromadb_episodic'] = 'healthy'
        except Exception as e: health_status['components']['chromadb_episodic'] = f"unhealthy ({str(e)[:50]})"; chroma_healthy = False
        try: self.semantic_collection.count(); health_status['components']['chromadb_semantic'] = 'healthy'
        except Exception as e: health_status['components']['chromadb_semantic'] = f"unhealthy ({str(e)[:50]})"; chroma_healthy = False
        if not chroma_healthy: health_status['overall_status'] = 'degraded'
            
        neo4j_status = 'unavailable'
        if self.knowledge_graph and self.knowledge_graph.driver:
            try:
                with self.knowledge_graph.driver.session(database=self.knowledge_graph.database) as session: session.run("RETURN 1")
                neo4j_status = 'healthy'
            except Exception as e: neo4j_status = f"unhealthy ({str(e)[:50]})"; health_status['overall_status'] = 'degraded'
        health_status['components']['neo4j_graph'] = neo4j_status
        if neo4j_status != 'healthy': health_status['overall_status'] = 'degraded'
            
        health_status['components']['entity_extraction'] = 'healthy' if self.entity_extractor and self.entity_extractor.nlp else 'unavailable'
        
        # Context Engine health
        context_engine_status = 'healthy' if self.context_engine else 'unavailable'
        if self.context_engine:
            try:
                # Test context engine functionality
                test_activities = [{'type': 'test', 'timestamp': datetime.now().isoformat(), 'satisfaction_score': 0.8}]
                test_context = self.context_engine.extract_temporal_context(test_activities)
                if test_context: context_engine_status = 'healthy'
                else: context_engine_status = 'degraded'
            except Exception as e:
                context_engine_status = f"unhealthy ({str(e)[:50]})"
                health_status['overall_status'] = 'degraded'
        
        health_status['components']['context_engine'] = context_engine_status
        health_status['components']['synthesis_capabilities'] = 'available' if context_engine_status == 'healthy' else 'unavailable'
            
        return health_status

    def get_enhanced_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including synthesis metrics."""
        stats = {
            'user_id': self.user_id, 'session_id': self.session_id, 
            'session_duration_sec': (datetime.now() - self.session_start_time).total_seconds(),
            'chromadb_episodic_count': 0, 'chromadb_semantic_count': 0, 
            'working_memory_message_count': len(self.current_conversation),
            'session_activities_tracked': len(self.session_activities), 
            'current_file_context': self.current_file_info.get('file_path', 'None'),
            'knowledge_graph_status': 'unavailable', 'entity_extraction_status': 'unavailable',
            'context_engine_stats': {
                'context_vectors': len(self.context_engine.context_vectors),
                'correlations': len(self.context_engine.correlations),
                'insights': len(self.context_engine.insights),
                'predictive_models': len(self.context_engine.predictive_models)
            }
        }
        
        try:
            if self.episodic_collection: stats['chromadb_episodic_count'] = self.episodic_collection.count()
            if self.semantic_collection: stats['chromadb_semantic_count'] = self.semantic_collection.count()
        except Exception as e: logger.warning(f"Could not retrieve ChromaDB counts: {e}")
            
        if self.knowledge_graph and self.knowledge_graph.driver:
            try:
                with self.knowledge_graph.driver.session(database=self.knowledge_graph.database) as session:
                    node_count_result = session.run("MATCH (n) RETURN count(n) as node_count").single()
                    stats['neo4j_node_count'] = node_count_result['node_count'] if node_count_result else 0
                    rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count").single()
                    stats['neo4j_relationship_count'] = rel_count_result['rel_count'] if rel_count_result else 0
                stats['knowledge_graph_status'] = 'available'
            except Exception as e: logger.warning(f"Could not retrieve Neo4j counts: {e}"); stats['knowledge_graph_status'] = f"error ({str(e)[:50]})"
        else: stats['knowledge_graph_status'] = 'not initialized'
            
        if self.entity_extractor and self.entity_extractor.nlp: stats['entity_extraction_status'] = 'available'
        else: stats['entity_extraction_status'] = 'unavailable'
            
        return stats


# --- Enhanced Interactive Utility Functions ---
async def handle_document_processing(agent: EnhancedAzureMemoryAgent):
    """Handle document processing with synthesis integration."""
    file_path_str = input("📁 Enter file path: ").strip()
    file_path = Path(file_path_str)
    
    if not file_path.exists(): print("❌ Error: File not found."); return
    
    try:
        print("🔄 Reading and processing document with advanced context analysis...")
        content = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        if SPACY_AVAILABLE:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    if detected and detected.get('encoding'): encodings_to_try.insert(0, detected['encoding'])
            except Exception as e: logger.warning(f"Chardet failed for encoding detection: {e}")
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    content = f.read()[:15000]; logger.info(f"Successfully read file '{file_path}' with encoding '{enc}'."); break
            except Exception as e: logger.debug(f"Failed to read with encoding '{enc}': {e}"); content = None
        
        if content is None: raise ValueError("Could not read file content with any attempted encoding.")
            
        result = agent.process_document(file_path, content)
        
        if result['success']:
            print(f"✅ Document processed successfully with advanced context analysis!")
            print(f"📄 Document ID: {result['doc_id']}"); print(f"🏷️ Entities Found: {len(result.get('entities', []))}"); print(f"⏱️ Processing Time: {result.get('processing_duration_sec', 0):.2f}s")
            
            if result.get('entities'):
                print("\n🏷️ Sample Entities:"); [print(f"  - {entity['text']} ({entity['label']})") for entity in result['entities'][:5]]
            
            if agent.knowledge_graph: print("🕸️ Knowledge graph updated with document information.")
            
            # Display synthesis results
            synthesis = result.get('synthesis_results', {})
            if synthesis and 'synthesis_metrics' in synthesis:
                metrics = synthesis['synthesis_metrics']
                print(f"\n🧠 Memory Synthesis Results:")
                print(f"  📊 Context Vectors: {metrics.get('context_vectors_generated', 0)}")
                print(f"  🔗 Correlations: {metrics.get('correlations_found', 0)}")
                print(f"  💡 Insights: {metrics.get('insights_generated', 0)}")
                print(f"  🎯 Recommendations: {metrics.get('recommendations_created', 0)}")
        else: print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"An error occurred during document processing handling: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {str(e)}")

async def handle_enhanced_query(agent: EnhancedAzureMemoryAgent):
    """Handle enhanced queries with synthesis integration."""
    query = input("🔍 Enter your query: ").strip()
    if not query: print("⚠️ Query cannot be empty."); return
    
    try:
        print("🔄 Processing query with advanced context analysis...")
        start_time = time.time()
        results = agent.enhanced_query_processing(query)
        processing_time = time.time() - start_time
        
        print(f"\n✅ Query processed in {processing_time:.2f}s!")
        if results.get('error'): print(f"❌ Error during processing: {results['error']}")
        
        if results.get('entities_in_query'): print("\n🏷️ Entities Detected in Query:"); [print(f"  - {entity['text']} ({entity['label']})") for entity in results['entities_in_query'][:5]]
        if results.get('knowledge_graph_context'): print(f"\n🧠 Knowledge Graph Context:\n{results['knowledge_graph_context']}")
        
        if results.get('chromadb_results'):
            print(f"\n📚 Top Relevant Documents (Semantic Search):")
            for i, res in enumerate(results['chromadb_results'][:3], 1):
                print(f"  {i}. Similarity: {res['similarity']:.3f}\n     Content Preview: {res['content_preview']}")
        
        # Display synthesis insights
        if results.get('synthesis_insights'):
            print(f"\n🔍 Synthesis Insights:")
            for insight in results['synthesis_insights']:
                print(f"  • {insight['description']} (Confidence: {insight['confidence']:.2f})")
        
        # Display recommendations
        if results.get('recommendations'):
            print(f"\n💡 Context-Aware Recommendations:")
            for rec in results['recommendations']:
                print(f"  • {rec['message']} (Priority: {rec.get('priority', 'unknown')})")
        
        if not results.get('error') and results.get('query_activity_subtype', '').startswith('QueryProcessed'):
             agent.track_user_activity("Query", {"subtype": "QueryResultsDisplayed", "query_text": query[:100], "chromadb_results_count": len(results.get('chromadb_results', [])), "kg_context_generated": bool(results.get('knowledge_graph_context')), "entities_in_query": len(results.get('entities_in_query', [])), "duration_sec": processing_time, "satisfaction_score": 0.8})

    except Exception as e:
        logger.error(f"An error occurred during enhanced query handling: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {str(e)}")

async def handle_synthesis_mode(agent: EnhancedAzureMemoryAgent):
    """Handle advanced memory synthesis operations."""
    print("\n🧠 Advanced Memory Synthesis")
    print("Choose synthesis operation:")
    print("  1. Run Complete Memory Synthesis")
    print("  2. View Context Vectors")
    print("  3. View Cross-Memory Correlations")
    print("  4. View Generated Insights")
    print("  5. View Predictive Models")
    print("  6. Generate Context-Aware Recommendations")
    
    choice = input("Enter choice (1-6): ").strip()
    
    try:
        if choice == "1":
            print("🔄 Running complete memory synthesis...")
            synthesis_results = agent.run_advanced_memory_synthesis()
            
            if 'error' in synthesis_results:
                print(f"❌ Synthesis failed: {synthesis_results['error']}")
                return
            
            metrics = synthesis_results['synthesis_metrics']
            print(f"\n✅ Memory Synthesis Complete!")
            print(f"⏱️ Processing Time: {metrics['processing_time_seconds']:.2f}s")
            print(f"📊 Generated:")
            print(f"  • Context Vectors: {metrics['context_vectors_generated']}")
            print(f"  • Cross-Memory Correlations: {metrics['correlations_found']}")
            print(f"  • Automatic Insights: {metrics['insights_generated']}")
            print(f"  • Recommendations: {metrics['recommendations_created']}")
            print(f"  • Predictive Model: {'Yes' if metrics['predictive_model_created'] else 'No'}")
            
            # Show sample results
            if synthesis_results['insights']:
                print(f"\n🔍 Sample Insights:")
                for insight in synthesis_results['insights'][:3]:
                    print(f"  • {insight['description']}")
            
            if synthesis_results['recommendations']:
                print(f"\n💡 Sample Recommendations:")
                for rec in synthesis_results['recommendations'][:3]:
                    print(f"  • {rec['message']}")
        
        elif choice == "2":
            print("\n📊 Context Vectors:")
            context_vectors = agent.context_engine.context_vectors
            if not context_vectors:
                print("  No context vectors available. Run synthesis first.")
                return
            
            for cv_id, cv in list(context_vectors.items())[:5]:
                print(f"  • {cv.context_type.value}: {cv.confidence:.2f} confidence")
                if isinstance(cv.content, dict):
                    for key, value in list(cv.content.items())[:3]:
                        print(f"    - {key}: {value}")
        
        elif choice == "3":
            print("\n🔗 Cross-Memory Correlations:")
            correlations = agent.context_engine.correlations
            if not correlations:
                print("  No correlations identified. Run synthesis first.")
                return
            
            for corr_id, corr in list(correlations.items())[:5]:
                print(f"  • {corr.pattern_description}")
                print(f"    Strength: {corr.correlation_strength:.3f}, Confidence: {corr.confidence_score:.3f}")
                print(f"    Memory Types: {', '.join(corr.memory_types)}")
        
        elif choice == "4":
            print("\n💡 Generated Insights:")
            insights = agent.context_engine.insights
            if not insights:
                print("  No insights generated. Run synthesis first.")
                return
            
            for insight_id, insight in list(insights.items())[:5]:
                print(f"  • {insight.pattern_type}: {insight.description}")
                print(f"    Confidence: {insight.confidence_score:.3f}, Relevance: {insight.user_relevance_score:.3f}")
                if insight.actionable_recommendations:
                    print(f"    Recommendations: {', '.join(insight.actionable_recommendations[:2])}")
        
        elif choice == "5":
            print("\n🎯 Predictive Models:")
            models = agent.context_engine.predictive_models
            if not models:
                print("  No predictive models available. Run synthesis first.")
                return
            
            for model_id, model in models.items():
                print(f"  • {model.prediction_type} (ID: {model_id})")
                print(f"    Features: {', '.join(model.model_features[:3])}...")
                print(f"    Last Updated: {model.last_updated.strftime('%Y-%m-%d %H:%M')}")
                if model.predictions:
                    print(f"    Predictions: {len(model.predictions)} available")
        
        elif choice == "6":
            print("\n🎯 Generating Context-Aware Recommendations...")
            current_context = agent._get_current_context()
            recommendations = agent.context_engine.generate_context_aware_recommendations(agent.user_id, current_context)
            
            if not recommendations:
                print("  No recommendations available. Run synthesis first.")
                return
            
            print(f"Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations[:10], 1):
                print(f"  {i}. {rec['message']}")
                print(f"     Action: {rec['action']}")
                print(f"     Priority: {rec['priority']}, Confidence: {rec['confidence']:.3f}")
                print()
        
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 6.")
            
    except Exception as e:
        logger.error(f"An error occurred during synthesis mode handling: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {str(e)}")

async def handle_chat_mode(agent: EnhancedAzureMemoryAgent):
    """Enhanced chat mode with synthesis integration."""
    print("\n💬 Enhanced Chat Mode with Advanced Context Engineering")
    print("Type 'exit', 'quit', or 'back' to return to the main menu.")
    print("💡 Special commands: 'synthesis' - Run memory synthesis")
    print("🧠 The system uses advanced context engineering, cross-memory correlations, and predictive insights.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'back']: break
        if not user_input: continue
        try:
            response_text = agent.process_user_input(user_input)
            print(f"\nAssistant: {response_text}")
        except Exception as e:
            logger.error(f"An error occurred in enhanced chat mode: {e}", exc_info=True)
            print(f"❌ Error: {str(e)}")

def handle_enhanced_memory_stats(agent: EnhancedAzureMemoryAgent):
    """Display enhanced memory statistics including synthesis metrics."""
    print("\n📊 Enhanced Memory Statistics with Synthesis Metrics")
    print("=" * 70)
    stats = agent.get_enhanced_memory_stats()
    
    print(f"👤 User: {stats.get('user_id', 'N/A')}"); print(f"🔗 Session ID: {stats.get('session_id', 'N/A')}")
    print(f"⏱️ Session Duration: {stats.get('session_duration_sec', 0)/60:.1f} minutes")
    print("\n💾 MEMORY SYSTEMS:")
    print(f"   📚 ChromaDB Episodic Count: {stats.get('chromadb_episodic_count', 0)}")
    print(f"   🗄️ ChromaDB Semantic Count: {stats.get('chromadb_semantic_count', 0)}")
    print(f"   🕸️ Neo4j Nodes: {stats.get('neo4j_node_count', 'N/A')}")
    print(f"   🔗 Neo4j Relationships: {stats.get('neo4j_relationship_count', 'N/A')}")
    print(f"   🧠 Entity Extraction Status: {stats.get('entity_extraction_status', 'N/A')}")
    
    # Enhanced synthesis statistics
    context_stats = stats.get('context_engine_stats', {})
    print(f"\n🧠 CONTEXT ENGINEERING & SYNTHESIS:")
    print(f"   📊 Context Vectors: {context_stats.get('context_vectors', 0)}")
    print(f"   🔗 Cross-Memory Correlations: {context_stats.get('correlations', 0)}")
    print(f"   💡 Generated Insights: {context_stats.get('insights', 0)}")
    print(f"   🎯 Predictive Models: {context_stats.get('predictive_models', 0)}")
    
    print(f"\n🎯 CURRENT SESSION STATE:")
    print(f"   💬 Working Memory Messages: {stats.get('working_memory_message_count', 0)}")
    print(f"   🎬 Session Activities Tracked: {stats.get('session_activities_tracked', 0)}")
    print(f"   📁 Current File Context: {stats.get('current_file_context', 'None')}")

async def handle_enhanced_visualization(agent: EnhancedAzureMemoryAgent):
    """Handle enhanced visualization with synthesis results."""
    print("\n🎨 Enhanced Knowledge Graph Visualization with Synthesis Results")
    print("Choose visualization scope:")
    print("  1. Single User (Current User) - Enhanced with Context Analysis")
    print("  2. All Users Overview - Enhanced with Synthesis Metrics")
    scope_choice = input("Enter choice (1 or 2, default 1): ").strip()
    include_all_users = (scope_choice == '2')
    
    output_choice = input("Save visualization to file? (y/N): ").strip().lower()
    output_path = None
    if output_choice == 'y':
        save_dir = input("Enter directory to save visualizations (default: ./visualizations/): ").strip()
        if not save_dir: save_dir = "./visualizations/"
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_knowledge_graph_{agent.user_id}_{timestamp_str}.png" if not include_all_users else f"enhanced_multi_user_graph_{timestamp_str}.png"
        output_path = str(Path(save_dir) / filename)
        print(f"Enhanced visualization will be saved to: {output_path}")
    
    try:
        print("⏳ Generating enhanced graph data and visualization with synthesis results...")
        results = agent.create_knowledge_graph_visualization(output_path, include_all_users)
        
        if "error" in results: print(f"❌ Error creating visualization: {results['error']}"); return
        
        print("✅ Enhanced visualization generated!")
        stats = results.get('statistics', {})
        print(f"\n📊 Enhanced Graph Statistics:")
        print(f"  Total Nodes: {stats.get('total_nodes', 0)}")
        print(f"  Total Edges: {stats.get('total_edges', 0)}")
        print(f"  Session Activities Tracked: {stats.get('session_activities_tracked', 0)}")
        
        # Enhanced synthesis statistics
        print(f"\n🧠 Synthesis Statistics:")
        print(f"  Context Vectors Created: {stats.get('context_vectors_created', 0)}")
        print(f"  Correlations Identified: {stats.get('correlations_identified', 0)}")
        print(f"  Insights Generated: {stats.get('insights_generated', 0)}")
        print(f"  Predictive Models Built: {stats.get('predictive_models_built', 0)}")
        
        print("\n📈 Node Type Counts:"); [print(f"  - {node_type}: {count}") for node_type, count in sorted(stats.get('node_types', {}).items())]
        print("\n🔗 Relationship Type Counts:"); [print(f"  - {rel_type}: {count}") for rel_type, count in sorted(stats.get('relationship_types', {}).items())]
        
        if output_path: print(f"\n💾 Enhanced visualization saved to: {output_path}")
        else:
            print("\n💡 Displaying enhanced visualization window...")
            if 'matplotlib_figure' in results['visualizations']: plt.show()
        
    except Exception as e:
        logger.error(f"An error occurred during enhanced visualization handling: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {str(e)}")

def handle_enhanced_system_health(agent: EnhancedAzureMemoryAgent):
    """Enhanced system health check including context engine status."""
    print("\n🏥 Enhanced System Health Check...")
    health = agent.get_system_health()
    status_emoji = "✅" if health['overall_status'] == 'healthy' else "⚠️" if health['overall_status'] == 'degraded' else "❌"
    print(f"\n{status_emoji} Overall System Status: {health['overall_status'].upper()}")
    print(f"🕐 Timestamp: {health['timestamp'][:19]}")
    print("\n🔧 Component Status:")
    components = health.get('components', {})
    component_details = {
        'azure_openai_llm': 'Azure OpenAI LLM', 'chromadb_episodic': 'ChromaDB (Episodic)', 'chromadb_semantic': 'ChromaDB (Semantic)',
        'neo4j_graph': 'Neo4j Knowledge Graph', 'entity_extraction': 'SpaCy Entity Extraction', 'context_engine': 'Advanced Context Engine',
        'synthesis_capabilities': 'Memory Synthesis Capabilities'
    }
    for comp_key, status in components.items():
        status_emoji_comp = "✅" if status.startswith('healthy') or status == 'available' else "⚠️" if 'degraded' in status or 'unhealthy' in status or 'unavailable' in status else "❌"
        comp_name = component_details.get(comp_key, comp_key.replace('_', ' ').title())
        print(f"  {status_emoji_comp} {comp_name}: {status}")

async def handle_enhanced_database_reset(agent: EnhancedAzureMemoryAgent):
    """Enhanced database reset including context engine reset."""
    print("\n🔄 Enhanced Vector Database Reset Utility")
    print("This action will delete all data in ChromaDB collections and reset context engine state.")
    print("It is useful for resolving embedding dimension mismatches and clearing synthesis data.")
    confirm = input("Type 'RESET' to confirm, or anything else to cancel: ").strip()
    if confirm != 'RESET': print("❌ Database reset cancelled."); return
    try:
        print("Attempting to reset ChromaDB collections and context engine...")
        if agent.reset_vector_database():
            print("✅ ChromaDB collections reset and recreated successfully.")
            print(f"   New embedding dimension: {agent.embedding_dimension}")
            print(f"   Embedding model: {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")
            
            # Reset context engine state
            agent.context_engine.context_vectors.clear()
            agent.context_engine.correlations.clear()
            agent.context_engine.insights.clear()
            agent.context_engine.predictive_models.clear()
            print("✅ Context engine state reset successfully.")
            
            handle_enhanced_memory_stats(agent) # Refresh stats
        else: print("❌ Failed to reset ChromaDB collections. Check logs for details.")
    except Exception as e:
        logger.error(f"An error occurred during enhanced database reset handling: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred during reset: {str(e)}")

async def run_enhanced_interactive_mode(agent: EnhancedAzureMemoryAgent):
    """Enhanced interactive loop with synthesis capabilities."""
    while True:
        print("\n" + "=" * 90)
        print("🧠 ENHANCED AZURE MEMORY AGENT with Advanced Context Engineering")
        print("=" * 90)
        print("  USER ACTIONS:")
        print("    [1] Process Document (Enhanced with Context Analysis)")
        print("    [2] Query with Enhanced Context (KG + Synthesis)")
        print("    [3] Enhanced Chat Mode (Context-Aware Conversation)")
        print("    [4] Advanced Memory Synthesis (Run Complete Analysis)")
        print("    [5] View Enhanced Memory Statistics")
        print("    [6] Create Enhanced Knowledge Graph Visualization")
        print("    [7] Enhanced System Health Check")
        print("    [8] Reset Enhanced Database (ChromaDB + Context Engine)")
        print("    [9] Recommendations on User Preferences (View, Pattern, Insight)")
        print("    [10] Exit and Show Enhanced Session Summary")
        
        choice = input("\nEnter your choice (1-10): ").strip()
        
        try:
            if choice == "1": await handle_document_processing(agent)
            elif choice == "2": await handle_enhanced_query(agent)
            elif choice == "3": await handle_chat_mode(agent)
            elif choice == "4": await handle_synthesis_mode(agent)
            elif choice == "5": handle_enhanced_memory_stats(agent)
            elif choice == "6": await handle_enhanced_visualization(agent)
            elif choice == "7": handle_enhanced_system_health(agent)
            elif choice == "8": await handle_enhanced_database_reset(agent)
            elif choice == "9":
                print("\n💾 Save User Preference:")
                print("   Format examples:")
                print("   - VIEW: SAVE VIEW \"<View Name>\" AS \"<Description>\" QUERY \"<Cypher Query>\"")
                print("   - PATTERN: SAVE PATTERN \"<Pattern Name>\" AS \"<Description>\" DEFINITION \"<Pattern Definition>\" [EXAMPLE \"<Example Query>\"]")
                print("   - INSIGHT: SAVE INSIGHT \"<Insight Category>\" AS \"<Description>\"")
                user_cmd_input = input("Enter command to save preference (or leave blank to cancel): ").strip()
                if user_cmd_input:
                    response = agent.process_user_input(user_cmd_input)
                    print(f"Assistant: {response}")
            elif choice == "10":
                print("\n🔄 Finalizing enhanced session with synthesis summary...")
                session_summary = agent.get_enhanced_memory_stats()
                context_stats = session_summary.get('context_engine_stats', {})
                
                print("\n📋 ENHANCED SESSION SUMMARY:")
                print(f"  User: {session_summary.get('user_id', 'N/A')}")
                print(f"  Session Duration: {session_summary.get('session_duration_sec', 0)/60:.1f} minutes")
                print(f"  Total Activities Tracked: {session_summary.get('session_activities_tracked', 0)}")
                print(f"  Working Memory Messages: {session_summary.get('working_memory_message_count', 0)}")
                print(f"  Neo4j Nodes: {session_summary.get('neo4j_node_count', 'N/A')}")
                print(f"  ChromaDB Semantic Count: {session_summary.get('chromadb_semantic_count', 0)}")
                
                print(f"\n🧠 SYNTHESIS ACHIEVEMENTS:")
                print(f"  Context Vectors Created: {context_stats.get('context_vectors', 0)}")
                print(f"  Cross-Memory Correlations: {context_stats.get('correlations', 0)}")
                print(f"  Insights Generated: {context_stats.get('insights', 0)}")
                print(f"  Predictive Models Built: {context_stats.get('predictive_models', 0)}")
                
                print("👋 Thank you for using the Enhanced Azure Memory Agent with Advanced Context Engineering!")
                break
            else: print("❌ Invalid choice. Please enter a number between 1 and 10.")
        except KeyboardInterrupt:
            print("\n👋 Exiting enhanced interactive mode...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the enhanced main loop: {e}", exc_info=True)
            print(f"❌ An unexpected error occurred: {str(e)}")

async def main():
    """Enhanced main function with context engineering initialization."""
    print("🚀 Bootstrapping Enhanced Azure Memory Agent with Advanced Context Engineering...")
    if not NEO4J_AVAILABLE: print("⚠️ WARNING: Neo4j is not installed. Knowledge Graph features will be unavailable. Install with: pip install neo4j")
    if not SPACY_AVAILABLE: print("⚠️ WARNING: SpaCy is not installed. Entity Extraction will be limited. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    
    user_id = input("Enter your User ID (or press Enter for 'demo_user'): ").strip()
    if not user_id: user_id = "demo_user"
    
    try:
        print(f"\nInitializing enhanced agent with context engineering for user: '{user_id}'...")
        agent = EnhancedAzureMemoryAgent(user_id=user_id)
        
        stats = agent.get_enhanced_memory_stats()
        context_stats = stats.get('context_engine_stats', {})
        print(f"  ChromaDB Semantic Count: {stats.get('chromadb_semantic_count', 0)}")
        print(f"  Neo4j Nodes: {stats.get('neo4j_node_count', 'N/A')}")
        print(f"  Context Engine: {context_stats.get('context_vectors', 0)} vectors, {context_stats.get('insights', 0)} insights")
        health = agent.get_system_health()
        health_emoji = "✅" if health['overall_status'] == 'healthy' else "⚠️"
        print(f"  System Health: {health_emoji} {health['overall_status'].upper()}")
        
        print(f"\n🧠 Advanced Context Engineering Features:")
        print(f"  • Cross-Memory Correlation Analysis")
        print(f"  • Automatic Insight Generation")
        print(f"  • Predictive Behavior Modeling")
        print(f"  • Context-Aware Recommendation Engine")
        
        await run_enhanced_interactive_mode(agent)
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during enhanced agent initialization or startup: {e}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: Failed to initialize the enhanced agent. Check logs and environment variables. Error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting enhanced program. Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred during enhanced program execution: {str(e)}")
        logger.critical(f"Top-level exception handler caught: {e}", exc_info=True)