import pandas as pd
import io
import os
import time
import sqlite3
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from neo4j import GraphDatabase
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
import json
import logging
from datetime import datetime
import networkx as nx
import statistics
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set LangSmith environment variables to avoid warnings
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

class InsightGenerator:
    """Generate insights and recommendations from analysis results"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate_insights(self, query: str, results: str, data_context: Dict, analysis_type: str) -> Dict[str, Any]:
        """Generate actionable insights and recommendations from analysis results"""
        
        # Create a comprehensive prompt for insight generation
        insight_prompt = f"""
        You are an expert data analyst. Based on the following analysis, provide comprehensive insights and actionable recommendations.
        
        **User Query:** {query}
        **Analysis Type:** {analysis_type}
        **Data Context:** 
        - Dataset: {data_context.get('table_name', 'Unknown')}
        - Shape: {data_context.get('data_shape', 'Unknown')}
        - Columns: {', '.join(data_context.get('columns', []))}
        
        **Analysis Results:**
        {results}
        
        Please provide:
        1. **Key Findings**: 2-3 most important discoveries from the analysis
        2. **Business Insights**: What do these findings mean in a business context?
        3. **Actionable Recommendations**: 3-5 specific actions that could be taken based on these findings
        4. **Data Quality Assessment**: Any observations about data completeness or quality
        5. **Further Analysis Suggestions**: What additional analysis might be valuable
        6. **Executive Summary**: A concise 2-sentence summary for leadership
        
        Format your response as a structured analysis that's both technical and business-friendly.
        """
        
        try:
            response = self.llm.invoke(insight_prompt)
            insights = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the insights into structured format
            return self._parse_insights(insights, query, analysis_type)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "key_findings": ["Analysis completed but insights generation failed"],
                "recommendations": ["Review the raw analysis results"],
                "executive_summary": "Analysis completed with technical results available.",
                "confidence_score": 0.5,
                "insights_error": str(e)
            }
    
    def _parse_insights(self, insights_text: str, query: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM-generated insights into structured format"""
        
        # Extract sections using simple text parsing
        sections = {
            "key_findings": self._extract_section(insights_text, "Key Findings", "Business Insights"),
            "business_insights": self._extract_section(insights_text, "Business Insights", "Actionable Recommendations"),
            "recommendations": self._extract_section(insights_text, "Actionable Recommendations", "Data Quality Assessment"),
            "data_quality": self._extract_section(insights_text, "Data Quality Assessment", "Further Analysis"),
            "further_analysis": self._extract_section(insights_text, "Further Analysis", "Executive Summary"),
            "executive_summary": self._extract_section(insights_text, "Executive Summary", None)
        }
        
        # Calculate confidence score based on analysis type and result quality
        confidence_score = self._calculate_confidence(insights_text, analysis_type)
        
        return {
            "key_findings": self._parse_list_items(sections["key_findings"]),
            "business_insights": sections["business_insights"].strip() if sections["business_insights"] else "Business context analysis pending.",
            "recommendations": self._parse_list_items(sections["recommendations"]),
            "data_quality_notes": sections["data_quality"].strip() if sections["data_quality"] else "Data quality assessment needed.",
            "further_analysis_suggestions": self._parse_list_items(sections["further_analysis"]),
            "executive_summary": sections["executive_summary"].strip() if sections["executive_summary"] else f"Analysis of '{query}' completed using {analysis_type} approach.",
            "confidence_score": confidence_score,
            "analysis_metadata": {
                "query": query,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _extract_section(self, text: str, start_marker: str, end_marker: Optional[str]) -> str:
        """Extract text between markers"""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            
            start_idx = text.find(":", start_idx) + 1
            
            if end_marker:
                end_idx = text.find(end_marker, start_idx)
                if end_idx == -1:
                    return text[start_idx:].strip()
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
        except Exception:
            return ""
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Parse text into list items"""
        if not text:
            return []
        
        # Split by common list indicators
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line.startswith(('-', '•', '*')) or line[0].isdigit()):
                # Remove list markers
                clean_line = line.lstrip('-•*0123456789. ').strip()
                if clean_line:
                    items.append(clean_line)
        
        return items if items else [text.strip()]
    
    def _calculate_confidence(self, insights_text: str, analysis_type: str) -> float:
        """Calculate confidence score for the insights"""
        base_confidence = 0.7
        
        # Adjust based on analysis type
        type_modifiers = {
            "sql": 0.1,
            "graph": 0.05,
            "hybrid": 0.15
        }
        
        confidence = base_confidence + type_modifiers.get(analysis_type, 0)
        
        # Adjust based on content quality indicators
        quality_indicators = [
            "specific", "data shows", "analysis reveals", "recommend", 
            "significant", "pattern", "trend", "correlation"
        ]
        
        found_indicators = sum(1 for indicator in quality_indicators if indicator in insights_text.lower())
        confidence += (found_indicators / len(quality_indicators)) * 0.2
        
        return min(confidence, 1.0)

class RecommendationEngine:
    """Advanced recommendation engine for data analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        self.recommendation_history = []
        
    def generate_recommendations(self, query: str, analysis_results: str, data_context: Dict, 
                               previous_queries: List[str] = None) -> Dict[str, Any]:
        """Generate contextual recommendations based on analysis and user behavior"""
        
        recommendations = {
            "immediate_actions": [],
            "follow_up_analyses": [],
            "data_improvement_suggestions": [],
            "business_opportunities": [],
            "risk_factors": [],
            "visualization_suggestions": []
        }
        
        # Analyze query patterns and suggest follow-ups
        if previous_queries:
            recommendations["follow_up_analyses"].extend(
                self._suggest_follow_up_queries(query, previous_queries, data_context)
            )
        
        # Suggest visualizations based on data types
        recommendations["visualization_suggestions"].extend(
            self._suggest_visualizations(query, data_context)
        )
        
        # Generate business-focused recommendations
        business_recs = self._generate_business_recommendations(query, analysis_results, data_context)
        recommendations.update(business_recs)
        
        # Generate LLM-powered strategic recommendations
        strategic_recs = self._generate_strategic_recommendations(query, analysis_results, data_context)
        
        return {
            **recommendations,
            "strategic_recommendations": strategic_recs,
            "priority_score": self._calculate_priority_score(query, analysis_results),
            "recommendation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "based_on_query": query,
                "confidence": self._calculate_recommendation_confidence(analysis_results)
            }
        }
    
    def _suggest_follow_up_queries(self, current_query: str, previous_queries: List[str], 
                                 data_context: Dict) -> List[str]:
        """Suggest logical follow-up queries"""
        suggestions = []
        current_lower = current_query.lower()
        
        # Pattern-based suggestions
        if "count" in current_lower or "how many" in current_lower:
            suggestions.extend([
                "What are the trends over time?",
                "How does this compare to industry benchmarks?",
                "What factors influence these numbers?"
            ])
        
        if "average" in current_lower or "mean" in current_lower:
            suggestions.extend([
                "What's the distribution of values?",
                "Are there any outliers?",
                "How does this vary by category?"
            ])
        
        # Context-specific suggestions based on available columns
        columns = data_context.get('columns', [])
        if any('date' in col.lower() or 'time' in col.lower() for col in columns):
            suggestions.append("Show me the time-based trends")
        
        if any('amount' in col.lower() or 'cost' in col.lower() or 'price' in col.lower() for col in columns):
            suggestions.append("Analyze the financial patterns")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _suggest_visualizations(self, query: str, data_context: Dict) -> List[str]:
        """Suggest appropriate visualizations"""
        suggestions = []
        query_lower = query.lower()
        
        if "trend" in query_lower or "over time" in query_lower:
            suggestions.append("Line chart to show trends over time")
        
        if "compare" in query_lower or "versus" in query_lower:
            suggestions.append("Bar chart for category comparisons")
        
        if "distribution" in query_lower or "spread" in query_lower:
            suggestions.append("Histogram to show data distribution")
        
        if "relationship" in query_lower or "correlation" in query_lower:
            suggestions.append("Scatter plot to visualize relationships")
        
        # Default suggestions based on data context
        if not suggestions:
            suggestions.extend([
                "Bar chart for categorical analysis",
                "Pie chart for proportional breakdown",
                "Table view for detailed data examination"
            ])
        
        return suggestions[:3]
    
    def _generate_business_recommendations(self, query: str, analysis_results: str, 
                                         data_context: Dict) -> Dict[str, List[str]]:
        """Generate business-focused recommendations"""
        
        business_recs = {
            "immediate_actions": [],
            "business_opportunities": [],
            "risk_factors": []
        }
        
        query_lower = query.lower()
        results_lower = analysis_results.lower()
        
        # Healthcare/medical context recommendations
        if any(term in query_lower for term in ['patient', 'doctor', 'medical', 'treatment', 'diagnosis']):
            business_recs["immediate_actions"].extend([
                "Review patient care protocols for identified patterns",
                "Analyze resource allocation based on patient distribution"
            ])
            business_recs["business_opportunities"].extend([
                "Identify opportunities for specialized care programs",
                "Optimize appointment scheduling based on demand patterns"
            ])
        
        # Financial context recommendations
        if any(term in query_lower for term in ['revenue', 'cost', 'profit', 'sales', 'financial']):
            business_recs["immediate_actions"].extend([
                "Investigate cost drivers identified in analysis",
                "Review pricing strategies for underperforming segments"
            ])
            business_recs["risk_factors"].extend([
                "Monitor financial metrics showing declining trends",
                "Assess sustainability of current cost structure"
            ])
        
        # Operational context recommendations
        if any(term in query_lower for term in ['meeting', 'franchise', 'performance', 'efficiency']):
            business_recs["immediate_actions"].extend([
                "Standardize best practices from high-performing units",
                "Implement regular performance monitoring"
            ])
            business_recs["business_opportunities"].extend([
                "Scale successful franchise models",
                "Develop training programs based on performance gaps"
            ])
        
        return business_recs
    
    def _generate_strategic_recommendations(self, query: str, analysis_results: str, 
                                          data_context: Dict) -> List[str]:
        """Generate high-level strategic recommendations using LLM"""
        
        strategic_prompt = f"""
        As a senior business strategist, analyze the following data analysis results and provide 3-5 high-level strategic recommendations.
        
        Query: {query}
        Analysis Results: {analysis_results}
        Data Context: {data_context.get('table_name', 'Dataset')} with {data_context.get('data_shape', ['Unknown', 'Unknown'])[0]} records
        
        Focus on:
        1. Strategic implications of the findings
        2. Long-term business value creation opportunities  
        3. Competitive advantages that could be developed
        4. Resource allocation recommendations
        5. Risk mitigation strategies
        
        Provide concise, actionable strategic recommendations (1-2 sentences each).
        """
        
        try:
            response = self.llm.invoke(strategic_prompt)
            strategic_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse into list of recommendations
            recommendations = []
            for line in strategic_text.split('\n'):
                line = line.strip()
                if line and (line.startswith(('-', '•', '*')) or line[0].isdigit()):
                    clean_line = line.lstrip('-•*0123456789. ').strip()
                    if clean_line and len(clean_line) > 20:  # Filter out very short items
                        recommendations.append(clean_line)
            
            return recommendations[:5]  # Return top 5 strategic recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            return [
                "Conduct deeper analysis of the identified patterns",
                "Develop action plans based on key findings",
                "Monitor relevant metrics regularly for continued optimization"
            ]
    
    def _calculate_priority_score(self, query: str, analysis_results: str) -> float:
        """Calculate priority score for recommendations"""
        priority_score = 0.5  # Base score
        
        # Increase priority for certain keywords
        high_priority_terms = ['urgent', 'critical', 'significant', 'major', 'important', 'revenue', 'cost']
        for term in high_priority_terms:
            if term in query.lower() or term in analysis_results.lower():
                priority_score += 0.1
        
        # Increase priority for numerical results (indicates concrete findings)
        if any(char.isdigit() for char in analysis_results):
            priority_score += 0.2
        
        return min(priority_score, 1.0)
    
    def _calculate_recommendation_confidence(self, analysis_results: str) -> float:
        """Calculate confidence in recommendations"""
        confidence = 0.6  # Base confidence
        
        # Increase confidence based on result quality indicators
        quality_indicators = ['shows', 'indicates', 'demonstrates', 'reveals', 'significant', 'pattern']
        found_indicators = sum(1 for indicator in quality_indicators if indicator in analysis_results.lower())
        
        confidence += (found_indicators / len(quality_indicators)) * 0.3
        
        return min(confidence, 1.0)

class EntityRelationshipExtractor:
    """Extract entities and relationships from tabular data"""

    def __init__(self, llm):
        self.llm = llm

    def extract_schema(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Extract potential entities and relationships from DataFrame schema"""
        schema_info = {
            "table_name": table_name,
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "sample_data": df.head(3).to_dict('records'),
            "shape": df.shape
        }

        # Analyze column relationships
        relationships = self._infer_relationships(df)

        return {
            "schema": schema_info,
            "relationships": relationships,
            "entities": self._identify_entities(df)
        }

    def _identify_entities(self, df: pd.DataFrame) -> List[Dict]:
        """Identify potential entities in the data"""
        entities = []

        for column in df.columns:
            # Check if column might represent an entity
            unique_ratio = df[column].nunique() / len(df) if len(df) > 0 else 0

            entity_indicators = [
                'id', 'name', 'code', 'number', 'key', 'identifier',
                'customer', 'product', 'category', 'type', 'class',
                'provider', 'doctor', 'patient', 'policy', 'coverage'
            ]

            if any(indicator in column.lower() for indicator in entity_indicators) or unique_ratio > 0.7:
                sample_values = df[column].dropna().unique()[:5]
                sample_values = [str(val) for val in sample_values]

                entities.append({
                    "column": column,
                    "type": "entity",
                    "unique_values": int(df[column].nunique()),
                    "unique_ratio": float(unique_ratio),
                    "sample_values": sample_values
                })

        return entities

    def _infer_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Infer potential relationships between columns"""
        relationships = []
        columns = df.columns.tolist()

        # Look for foreign key relationships
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if self._might_be_related(col1, col2):
                    confidence = self._calculate_relationship_confidence(df, col1, col2)
                    if confidence > 0.1:  # Only include relationships with some confidence
                        relationships.append({
                            "from": col1,
                            "to": col2,
                            "type": "related_to",
                            "confidence": float(confidence)
                        })

        return relationships

    def _might_be_related(self, col1: str, col2: str) -> bool:
        """Enhanced heuristic to determine if columns might be related"""
        patterns = [
            ('customer', 'order'), ('product', 'category'), ('user', 'account'),
            ('invoice', 'payment'), ('id', 'name'), ('patient', 'doctor'),
            ('policy', 'provider'), ('coverage', 'provider'), ('doctor', 'patient')
        ]

        col1_lower = col1.lower()
        col2_lower = col2.lower()

        # Check for ID relationships
        if 'id' in col1_lower and col2_lower.replace('id', '') in col1_lower.replace('id', ''):
            return True
        if 'id' in col2_lower and col1_lower.replace('id', '') in col2_lower.replace('id', ''):
            return True

        # Check for pattern matches
        for pattern in patterns:
            if (pattern[0] in col1_lower and pattern[1] in col2_lower) or \
               (pattern[1] in col1_lower and pattern[0] in col2_lower):
                return True

        return False

    def _calculate_relationship_confidence(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Calculate confidence score for relationship between two columns"""
        try:
            # For numeric columns, use correlation
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                correlation = df[col1].corr(df[col2])
                return abs(correlation) if pd.notna(correlation) else 0.0

            # For categorical/text columns, check value overlap
            unique_col1 = set(df[col1].dropna().astype(str))
            unique_col2 = set(df[col2].dropna().astype(str))
            
            if len(unique_col1) == 0 or len(unique_col2) == 0:
                return 0.0
                
            overlap = len(unique_col1.intersection(unique_col2))
            total_unique = len(unique_col1.union(unique_col2))
            
            return overlap / total_unique if total_unique > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating relationship confidence: {e}")
            return 0.0

class Neo4jKnowledgeGraph:
    """Enhanced Neo4j integration with knowledge graph capabilities"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """Clear all data from the database"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all data from Neo4j database")

    def create_schema_from_dataframe(self, df: pd.DataFrame, table_name: str, extractor: EntityRelationshipExtractor):
        """Create Neo4j schema based on DataFrame structure"""
        schema_info = extractor.extract_schema(df, table_name)

        with self.driver.session(database=self.database) as session:
            # Create constraints and indexes for entity columns
            for entity in schema_info["entities"]:
                column = entity["column"].replace(' ', '_').replace('-', '_')
                try:
                    node_label = table_name.title().replace('_', '').replace('-', '')
                    index_query = f"""
                    CREATE INDEX {table_name}_{column}_index IF NOT EXISTS
                    FOR (n:{node_label}) ON (n.{column})
                    """
                    session.run(index_query)
                    logger.info(f"Created index for {table_name}.{column}")
                except Exception as e:
                    logger.warning(f"Could not create index for {table_name}.{column}: {e}")

    def load_dataframe_to_graph(self, df: pd.DataFrame, table_name: str, extractor: EntityRelationshipExtractor):
        """Load DataFrame data into Neo4j as a knowledge graph"""
        schema_info = extractor.extract_schema(df, table_name)
        node_label = table_name.title().replace('_', '').replace('-', '')

        with self.driver.session(database=self.database) as session:
            # Clear existing data for this table
            logger.info(f"Clearing existing nodes with label: {node_label}")
            session.run(f"MATCH (n:{node_label}) DETACH DELETE n")

            # Load data in batches
            batch_size = 100
            for start_idx in range(0, len(df), batch_size):
                batch_df = df.iloc[start_idx:start_idx + batch_size]
                self._load_batch_to_neo4j(session, batch_df, node_label)

            # Create relationships based on inferred schema
            self._create_relationships(session, df, node_label, schema_info)

            logger.info(f"Loaded {len(df)} records into Neo4j as {node_label} nodes")

    def _load_batch_to_neo4j(self, session, batch_df: pd.DataFrame, node_label: str):
        """Load a batch of data into Neo4j"""
        for index, row in batch_df.iterrows():
            properties = {'_row_id': int(index)}
            
            for col, value in row.items():
                if pd.notna(value):
                    col_clean = col.replace(' ', '_').replace('-', '_')
                    if isinstance(value, (int, float)):
                        if not pd.isna(value):
                            properties[col_clean] = float(value) if isinstance(value, float) else int(value)
                    else:
                        properties[col_clean] = str(value)

            if len(properties) > 1:  # More than just _row_id
                # Use parameterized queries for safety
                query = f"CREATE (n:{node_label} $props)"
                try:
                    session.run(query, props=properties)
                except Exception as e:
                    logger.warning(f"Could not create node for row {index}: {e}")

    def _create_relationships(self, session, df: pd.DataFrame, node_label: str, schema_info: Dict):
        """Create relationships between nodes based on inferred relationships"""
        for relationship in schema_info["relationships"]:
            if relationship["confidence"] > 0.2:  # Threshold for creating relationships
                from_col = relationship["from"].replace(' ', '_').replace('-', '_')
                to_col = relationship["to"].replace(' ', '_').replace('-', '_')

                # Create relationships between nodes that share common values
                # This is a simplified approach - in practice, you'd want more sophisticated logic
                rel_query = f"""
                MATCH (a:{node_label}), (b:{node_label})
                WHERE a.{from_col} IS NOT NULL AND b.{to_col} IS NOT NULL
                AND a.{from_col} = b.{to_col}
                AND a._row_id <> b._row_id
                WITH a, b, '{relationship["type"]}' as rel_type, {relationship["confidence"]} as confidence
                LIMIT 100
                CREATE (a)-[:RELATED_TO {{
                    type: rel_type,
                    confidence: confidence,
                    from_field: '{from_col}',
                    to_field: '{to_col}'
                }}]->(b)
                """

                try:
                    result = session.run(rel_query)
                    logger.info(f"Created relationships between '{from_col}' and '{to_col}' for {node_label}")
                except Exception as e:
                    logger.warning(f"Could not create relationship between '{from_col}' and '{to_col}': {e}")

    def execute_cypher_query(self, query: str) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Cypher query error: {e}")
                return [{"error": str(e)}]

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        stats_queries = {
            "node_count": "MATCH (n) RETURN count(n) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_labels": "CALL db.labels()",
            "relationship_types": "CALL db.relationshipTypes()"
        }

        stats = {}
        with self.driver.session(database=self.database) as session:
            for stat_name, query in stats_queries.items():
                try:
                    result = session.run(query)
                    if stat_name in ["node_count", "relationship_count"]:
                        record = result.single()
                        stats[stat_name] = record["count"] if record else 0
                    elif stat_name in ["node_labels", "relationship_types"]:
                        stats[stat_name] = [record.values()[0] for record in result]
                except Exception as e:
                    stats[stat_name] = f"Error: {e}"
        
        return stats

class EnhancedFileProcessor:
    """Enhanced file processor with Neo4j knowledge graph integration and recommendations"""

    def __init__(self, azure_config: Dict[str, str], neo4j_config: Dict[str, str]):
        """Initialize with Azure OpenAI and Neo4j configurations"""

        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            deployment_name=azure_config["deployment_name"],
            model_name=azure_config["model_name"],
            temperature=float(azure_config.get("temperature", 0.1)),
            max_tokens=int(azure_config.get("max_tokens", 4000))
        )

        # Initialize database components
        self.db_path = "file_database.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.db = SQLDatabase(engine=self.engine)

        # Initialize Neo4j knowledge graph
        self.knowledge_graph = Neo4jKnowledgeGraph(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"]
        )

        # Initialize entity relationship extractor
        self.entity_extractor = EntityRelationshipExtractor(self.llm)

        # Initialize new components for insights and recommendations
        self.insight_generator = InsightGenerator(self.llm)
        self.recommendation_engine = RecommendationEngine(self.llm)

        # Create SQL agent
        self.sql_agent = self._create_sql_agent()

        self.current_table_name = None
        self.current_df = None
        self.column_descriptions = {}
        self.query_history = []  # Track user queries for better recommendations

    def _create_sql_agent(self):
        """Create the SQL agent for querying the database"""
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        return create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="tool-calling",
            verbose=True
        )

    def _execute_sql_query(self, query: str) -> str:
        """Execute SQL query with better error handling"""
        try:
            if self.current_table_name is None:
                return "No data has been loaded. Please load a file first."

            # Provide context about available columns
            column_info = f"Available columns in table '{self.current_table_name}': {', '.join(self.current_df.columns)}"
            enhanced_query = f"{column_info}\n\nUser query: {query}"
            
            result = self.sql_agent.invoke({"input": enhanced_query})
            return str(result.get("output", "No results from SQL agent"))
            
        except Exception as e:
            logger.error(f"SQL Query Execution Error: {e}")
            return f"SQL Query Error: {str(e)}"

    def _execute_graph_query(self, query: str) -> str:
        """Execute Cypher query with improved natural language processing"""
        try:
            cypher_query = self._convert_to_cypher(query)
            
            if not cypher_query or "error" in cypher_query.lower():
                return f"Could not convert query to Cypher: {cypher_query}"

            results = self.knowledge_graph.execute_cypher_query(cypher_query)

            if results and not any("error" in str(result) for result in results):
                return f"Graph Query Results:\n{json.dumps(results, indent=2)}"
            else:
                return f"Graph Query Error or No Results: {results}"
                
        except Exception as e:
            logger.error(f"Graph Query Execution Error: {e}")
            return f"Graph Query Error: {str(e)}"

    def _convert_to_cypher(self, natural_query: str) -> str:
        """Improved natural language to Cypher conversion"""
        query_lower = natural_query.lower()
        
        if self.current_table_name:
            node_label = self.current_table_name.title().replace('_', '').replace('-', '')
            
            if "count" in query_lower or "how many" in query_lower:
                return f"MATCH (n:{node_label}) RETURN count(n) as total_count"
            elif "all" in query_lower or "show me" in query_lower or "list" in query_lower:
                return f"MATCH (n:{node_label}) RETURN n LIMIT 10"
            elif "relationship" in query_lower or "connected" in query_lower:
                return f"MATCH (a:{node_label})-[r]->(b:{node_label}) RETURN a, type(r), b LIMIT 10"
            elif "doctor" in query_lower and "patient" in query_lower:
                return f"""
                MATCH (p:{node_label}) 
                WHERE p.DoctorID IS NOT NULL 
                RETURN p.DoctorID as doctor, count(p) as patient_count 
                ORDER BY patient_count DESC 
                LIMIT 10
                """
            else:
                return f"MATCH (n:{node_label}) RETURN n LIMIT 5"
        
        return "MATCH (n) RETURN count(n) as total_nodes"

    def _execute_hybrid_analysis(self, analysis_request: str) -> str:
        """Execute hybrid analysis combining SQL and Graph insights"""
        try:
            sql_insights = self._execute_sql_query(f"Analyze: {analysis_request}")
            graph_insights = self._execute_graph_query(f"Find patterns for: {analysis_request}")
            
            return f"""
**Statistical Analysis (SQL):**
{sql_insights}

**Relationship Analysis (Graph):**
{graph_insights}

**Combined Insights:**
Both SQL and Graph analysis provide complementary views of the data structure and relationships.
            """
        except Exception as e:
            logger.error(f"Hybrid Analysis Error: {e}")
            return f"Hybrid Analysis Error: {str(e)}"

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixed preprocessing function"""
        processed_df = df.copy()
        
        for column in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[column]):
                # Use proper assignment instead of inplace
                processed_df.loc[:, column] = processed_df[column].fillna(0)
            else:
                # Use proper assignment instead of inplace
                processed_df.loc[:, column] = processed_df[column].fillna('unknown').astype(str)
        
        return processed_df

    def preprocess_and_load_file(self, file_path: str, table_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess and load file with better error handling"""
        try:
            # Load DataFrame with better encoding handling
            if file_path.endswith(".csv"):
                try:
                    original_df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
                except UnicodeDecodeError:
                    try:
                        original_df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
                    except UnicodeDecodeError:
                        original_df = pd.read_csv(file_path, encoding="cp1252", low_memory=False)
            elif file_path.endswith((".xls", ".xlsx")):
                original_df = pd.read_excel(file_path)
            else:
                return pd.DataFrame(), {"error": "Unsupported file format. Please provide CSV or Excel."}
            
            logger.info(f"Loaded DataFrame shape: {original_df.shape}")
            logger.info(f"Columns: {list(original_df.columns)}")
            
            # Preprocess data
            processed_df = self._preprocess_dataframe(original_df)
            
            # Generate table name
            if table_name is None:
                base_name = os.path.basename(file_path)
                table_name = os.path.splitext(base_name)[0].lower()
                table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
                if not table_name[0].isalpha():
                    table_name = 'data_' + table_name

            # Store current data
            self.current_table_name = table_name
            self.current_df = processed_df
            
            # Store column descriptions for better query understanding
            self.column_descriptions = {col: self._describe_column(processed_df, col) for col in processed_df.columns}
            
            # Load into SQL database
            processed_df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Loaded data into SQL table: {table_name}")
            
            # Refresh database connection
            self.db = SQLDatabase(engine=self.engine)
            self.sql_agent = self._create_sql_agent()
            
            # Extract schema information
            schema_info = self.entity_extractor.extract_schema(processed_df, table_name)
            
            # Load into Neo4j
            try:
                self.knowledge_graph.create_schema_from_dataframe(processed_df, table_name, self.entity_extractor)
                self.knowledge_graph.load_dataframe_to_graph(processed_df, table_name, self.entity_extractor)
                logger.info("Successfully loaded data into Neo4j knowledge graph")
            except Exception as e:
                logger.warning(f"Neo4j loading failed: {e}")
                schema_info["neo4j_error"] = str(e)
            
            return processed_df, schema_info
            
        except Exception as e:
            error_msg = f"Error processing file '{file_path}': {str(e)}"
            logger.error(error_msg)
            self.current_table_name = None
            self.current_df = None
            raise

    def _describe_column(self, df: pd.DataFrame, column: str) -> str:
        """Generate a description of a column's content"""
        col_data = df[column]
        unique_count = col_data.nunique()
        total_count = len(col_data)
        
        if pd.api.types.is_numeric_dtype(col_data):
            return f"Numeric column with {unique_count} unique values, range: {col_data.min()} to {col_data.max()}"
        else:
            sample_values = col_data.dropna().unique()[:3]
            return f"Text column with {unique_count} unique values, examples: {', '.join(map(str, sample_values))}"

    def chat(self, query: str) -> Dict[str, Any]:
        """Enhanced chat interface with insights and recommendations"""
        try:
            if self.current_table_name is None:
                return {
                    "response": "No data has been loaded. Please load a CSV or Excel file first.",
                    "type": "error"
                }
            
            # Add query to history for better recommendations
            self.query_history.append(query)
            if len(self.query_history) > 10:  # Keep only recent queries
                self.query_history = self.query_history[-10:]
            
            # Provide data context to help the user understand what's available
            data_context = {
                "table_name": self.current_table_name,
                "columns": list(self.current_df.columns),
                "data_shape": list(self.current_df.shape)
            }
            
            query_lower = query.lower()
            
            # Enhanced keyword detection
            graph_keywords = ['relationship', 'connected', 'network', 'pattern', 'link', 'graph']
            sql_keywords = ['sum', 'count', 'average', 'total', 'group', 'filter', 'maximum', 'minimum']
            
            # Check if query mentions columns that don't exist
            mentioned_columns = [col for col in ['provider', 'insurance', 'coverage', 'policy'] 
                               if col in query_lower]
            
            if mentioned_columns and not any(col.lower() in [c.lower() for c in self.current_df.columns] for col in mentioned_columns):
                missing_info = f"The current dataset doesn't contain columns for: {', '.join(mentioned_columns)}. "
                available_info = f"Available columns are: {', '.join(self.current_df.columns)}. "
                suggestion = "Please rephrase your query using the available columns or load a different dataset."
                
                return {
                    "response": missing_info + available_info + suggestion,
                    "type": "data_mismatch",
                    "table_name": self.current_table_name,
                    "data_context": data_context,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Route to appropriate analysis type
            has_graph_keywords = any(keyword in query_lower for keyword in graph_keywords)
            has_sql_keywords = any(keyword in query_lower for keyword in sql_keywords)
            
            if has_graph_keywords and has_sql_keywords:
                analysis_type = "hybrid"
                result = self._execute_hybrid_analysis(query)
            elif has_graph_keywords:
                analysis_type = "graph"
                result = self._execute_graph_query(query)
            else:
                analysis_type = "sql"
                result = self._execute_sql_query(query)
            
            # Generate insights and recommendations
            try:
                insights = self.insight_generator.generate_insights(
                    query, result, data_context, analysis_type
                )
                
                recommendations = self.recommendation_engine.generate_recommendations(
                    query, result, data_context, self.query_history[:-1]  # Exclude current query
                )
            except Exception as e:
                logger.error(f"Error generating insights/recommendations: {e}")
                insights = {
                    "executive_summary": "Analysis completed successfully.",
                    "key_findings": ["Analysis results available in response"],
                    "confidence_score": 0.7
                }
                recommendations = {
                    "follow_up_analyses": ["Consider exploring related data patterns"],
                    "priority_score": 0.5
                }
            
            # Get graph statistics
            try:
                graph_stats = self.knowledge_graph.get_graph_statistics()
            except Exception as e:
                graph_stats = {"error": str(e)}
            
            return {
                "response": result,
                "type": analysis_type,
                "table_name": self.current_table_name,
                "data_shape": list(self.current_df.shape),
                "data_context": data_context,
                "graph_stats": graph_stats,
                "insights": insights,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "response": error_msg,
                "type": "error",
                "timestamp": datetime.now().isoformat()
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Fixed system status with proper SQL connection testing"""
        status = {
            "sql_database": {
                "connected": False,
                "tables": [],
                "current_table": self.current_table_name
            },
            "knowledge_graph": {
                "connected": False,
                "statistics": {}
            },
            "current_data": {
                "loaded": self.current_df is not None,
                "shape": list(self.current_df.shape) if self.current_df is not None else None,
                "columns": list(self.current_df.columns) if self.current_df is not None else []
            },
            "query_history_count": len(self.query_history)
        }
        
        # Fixed SQL database connection test
        try:
            with self.engine.connect() as conn:
                # Use SQLAlchemy's text() function for raw SQL
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = result.fetchall()
                status["sql_database"]["tables"] = [table[0] for table in tables]
                status["sql_database"]["connected"] = True
        except Exception as e:
            status["sql_database"]["connected"] = False
            status["sql_database"]["error"] = str(e)
        
        # Test Neo4j connection
        try:
            graph_stats = self.knowledge_graph.get_graph_statistics()
            status["knowledge_graph"]["connected"] = True
            status["knowledge_graph"]["statistics"] = graph_stats
        except Exception as e:
            status["knowledge_graph"]["connected"] = False
            status["knowledge_graph"]["error"] = str(e)
        
        return status

    def get_analysis_summary(self, query: str = None) -> Dict[str, Any]:
        """Generate comprehensive analysis summary with insights"""
        if self.current_df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "dataset_overview": {
                "name": self.current_table_name,
                "shape": list(self.current_df.shape),
                "columns": list(self.current_df.columns),
                "data_types": {col: str(dtype) for col, dtype in self.current_df.dtypes.items()}
            },
            "data_quality": self._assess_data_quality(),
            "key_statistics": self._generate_key_statistics(),
            "potential_analyses": self._suggest_potential_analyses()
        }
        
        if query:
            # Generate query-specific insights
            try:
                insights = self.insight_generator.generate_insights(
                    query, "Dataset summary", {"table_name": self.current_table_name, "columns": list(self.current_df.columns)}, "summary"
                )
                summary["query_insights"] = insights
            except Exception as e:
                logger.error(f"Error generating query-specific insights: {e}")
        
        return summary
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality metrics"""
        quality_metrics = {}
        
        for col in self.current_df.columns:
            col_data = self.current_df[col]
            quality_metrics[col] = {
                "missing_count": int(col_data.isnull().sum()),
                "missing_percentage": float(col_data.isnull().sum() / len(col_data) * 100),
                "unique_count": int(col_data.nunique()),
                "data_type": str(col_data.dtype)
            }
        
        return quality_metrics
    
    def _generate_key_statistics(self) -> Dict[str, Any]:
        """Generate key statistics for the dataset"""
        stats = {
            "total_rows": int(len(self.current_df)),
            "total_columns": int(len(self.current_df.columns)),
            "numeric_columns": int(len(self.current_df.select_dtypes(include=[int, float]).columns)),
            "categorical_columns": int(len(self.current_df.select_dtypes(include=['object']).columns)),
            "memory_usage_mb": float(self.current_df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Add summary statistics for numeric columns
        numeric_df = self.current_df.select_dtypes(include=[int, float])
        if not numeric_df.empty:
            stats["numeric_summary"] = {
                col: {
                    "mean": float(numeric_df[col].mean()),
                    "median": float(numeric_df[col].median()),
                    "std": float(numeric_df[col].std()),
                    "min": float(numeric_df[col].min()),
                    "max": float(numeric_df[col].max())
                } for col in numeric_df.columns
            }
        
        return stats
    
    def _suggest_potential_analyses(self) -> List[str]:
        """Suggest potential analyses based on data structure"""
        suggestions = []
        
        columns = list(self.current_df.columns)
        numeric_cols = list(self.current_df.select_dtypes(include=[int, float]).columns)
        categorical_cols = list(self.current_df.select_dtypes(include=['object']).columns)
        
        # Time-based analysis suggestions
        date_cols = [col for col in columns if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month'])]
        if date_cols:
            suggestions.append(f"Time series analysis using {date_cols[0]}")
        
        # Categorical analysis suggestions
        if categorical_cols:
            suggestions.append(f"Category distribution analysis for {categorical_cols[0]}")
        
        # Numeric analysis suggestions
        if len(numeric_cols) >= 2:
            suggestions.append(f"Correlation analysis between {numeric_cols[0]} and {numeric_cols[1]}")
        
        # Business-specific suggestions based on column names
        if any('revenue' in col.lower() or 'sales' in col.lower() for col in columns):
            suggestions.append("Revenue/Sales performance analysis")
        
        if any('customer' in col.lower() or 'client' in col.lower() for col in columns):
            suggestions.append("Customer segmentation analysis")
        
        return suggestions

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
            self.knowledge_graph.close()

def create_enhanced_file_processor(azure_config: Dict[str, str], neo4j_config: Dict[str, str]) -> EnhancedFileProcessor:
    """Create an enhanced file processor with Neo4j integration"""
    return EnhancedFileProcessor(azure_config, neo4j_config)

# Example usage with proper error handling
if __name__ == "__main__":
    azure_config = {
       
    }
    
    neo4j_config = {
    
    }
    
    try:
        # Create processor
        processor = create_enhanced_file_processor(azure_config, neo4j_config)
        print("Enhanced File Processor with Recommendations initialized successfully!")
        
        # Clean Neo4j database to start fresh
        print("Cleaning Neo4j database...")
        processor.knowledge_graph.clear_database()
        
        # Check system status
        print("\nSystem Status:")
        status = processor.get_system_status()
        print(json.dumps(status, indent=2))
        
        # Process file
        file_to_process = r"C:\Users\PhaniKumarMedapati\workspace\memory_agent\2023_HCP_Meeting_Report_by_Franchise_20231016_dataset_forNLP_Demo.xlsx"
        
        if os.path.exists(file_to_process):
            print(f"\nProcessing file: {file_to_process}")
            try:
                df, schema_info = processor.preprocess_and_load_file(file_to_process, "patient_data")
                print(f"Successfully loaded {df.shape[0]} rows with {df.shape[1]} columns")
                print(f"Columns: {list(df.columns)}")
                
                # Show sample data
                print("\nSample data:")
                print(df.head(3).to_string())
                
                # Generate dataset summary with insights
                print("\n" + "="*50)
                print("DATASET ANALYSIS SUMMARY")
                print("="*50)
                summary = processor.get_analysis_summary("What are the key patterns in this healthcare data?")
                print(json.dumps(summary, indent=2, default=str))
                
            except Exception as e:
                print(f"Error loading file: {e}")
                exit(1)
        else:
            print(f"File not found: {file_to_process}")
            print("Please update the file path or create a sample CSV file.")
            
            # Create sample data for testing
            sample_data = {
                'franchise': ['HCV', 'HCV', 'Oncology', 'Oncology', 'Neurology', 'Neurology'],
                'meetings_completed': [45, 32, 28, 41, 35, 22],
                'region': ['North', 'South', 'North', 'South', 'East', 'West'],
                'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2'],
                'revenue': [125000, 98000, 156000, 134000, 87000, 65000]
            }
            
            sample_df = pd.DataFrame(sample_data)
            sample_file = "sample_franchise_data.csv"
            sample_df.to_csv(sample_file, index=False)
            print(f"Created sample file: {sample_file}")
            
            # Load the sample file
            try:
                df, schema_info = processor.preprocess_and_load_file(sample_file, "franchise_data")
                print(f"Successfully loaded sample data: {df.shape[0]} rows with {df.shape[1]} columns")
                file_to_process = sample_file
                
                # Generate dataset summary
                summary = processor.get_analysis_summary("What insights can we derive from this franchise data?")
                print("\nDataset Summary with Insights:")
                print(json.dumps(summary, indent=2, default=str))
                
            except Exception as e:
                print(f"Error loading sample file: {e}")
                exit(1)

        print("\n" + "="*50)
        print("TESTING ENHANCED QUERIES WITH RECOMMENDATIONS")
        print("="*50)

        # Test queries that demonstrate the new features
        test_queries = [
            "What is the total estimated number of attendees for all completed meetings?",
 
             "Which project had the highest honorarium amount paid to a speaker?",
             "number of meetings completed each franchise"                        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 60)
            
            try:
                result = processor.chat(query)
                print(f"Analysis Type: {result['type']}")
                print(f"Response: {result['response']}")
                
                # Display insights
                if 'insights' in result:
                    insights = result['insights']
                    print(f"\n📊 INSIGHTS:")
                    print(f"Executive Summary: {insights.get('executive_summary', 'N/A')}")
                    print(f"Key Findings: {', '.join(insights.get('key_findings', []))}")
                    print(f"Confidence Score: {insights.get('confidence_score', 'N/A')}")
                
                # Display recommendations
                if 'recommendations' in result:
                    recs = result['recommendations']
                    print(f"\n💡 RECOMMENDATIONS:")
                    if recs.get('follow_up_analyses'):
                        print(f"Follow-up Analyses: {', '.join(recs['follow_up_analyses'][:2])}")
                    if recs.get('strategic_recommendations'):
                        print(f"Strategic Recommendations: {recs['strategic_recommendations'][0] if recs['strategic_recommendations'] else 'None'}")
                    if recs.get('visualization_suggestions'):
                        print(f"Visualization Suggestions: {', '.join(recs['visualization_suggestions'][:2])}")
                    print(f"Priority Score: {recs.get('priority_score', 'N/A')}")
                
                if result['type'] == 'data_mismatch':
                    print(f"Data Context: {result.get('data_context', 'N/A')}")
                    
            except Exception as e:
                print(f"Error processing query: {e}")

        # Show final system status
        print("\n" + "="*50)
        print("FINAL SYSTEM STATUS WITH ENHANCEMENTS")
        print("="*50)
        final_status = processor.get_system_status()
        print(json.dumps(final_status, indent=2))

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        if 'processor' in locals():
            processor.cleanup()
            print("\nResources cleaned up successfully.")