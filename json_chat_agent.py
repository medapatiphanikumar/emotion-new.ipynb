# app/agents/json_chat_agent.py
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Add the parent directory to the path to fix import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.config import settings
except ImportError:
    from core.config import settings

class JsonChatAgent:
    """Enhanced JSON chat agent using LangChain JSON toolkit"""
    
    def __init__(self):
        self.logger = logging.getLogger("document_processor")
        self.current_schema_path = None
        self.json_agent = None
        self.json_spec = None
        self.schema_data = None
        
        # Validate Azure OpenAI configuration
        if not settings.validate_azure_config():
            raise ValueError("Azure OpenAI configuration is incomplete. Check your .env file.")
        
        # Initialize Azure OpenAI Chat LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
    
    def _load_schema(self, schema_path: str) -> bool:
        """Load schema and create JSON agent"""
        try:
            # Check if schema exists
            if not Path(schema_path).exists():
                self.logger.error(f"Schema file not found: {schema_path}")
                return False
            
            # Load schema data
            with open(schema_path, 'r') as f:
                self.schema_data = json.load(f)
            
            # Create JsonSpec from schema data
            self.json_spec = JsonSpec(dict_=self.schema_data, max_value_length=4000)
            
            # Create JSON toolkit
            json_toolkit = JsonToolkit(spec=self.json_spec)
            
            # Create JSON agent
            self.json_agent = create_json_agent(
                llm=self.llm,
                toolkit=json_toolkit,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            self.current_schema_path = schema_path
            self.logger.info(f"Successfully loaded schema: {schema_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading schema: {str(e)}")
            return False
    
    async def chat_with_data(self, schema_path: str, message: str, file_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Chat with JSON data using schema information"""
        try:
            self.logger.info(f"Processing chat message for {file_id}")
            
            # Load schema if not already loaded or if different schema
            if self.current_schema_path != schema_path:
                if not self._load_schema(schema_path):
                    raise Exception("Failed to load schema")
            
            # Analyze user query intent
            query_analysis = self._analyze_user_query(message)
            
            # Create enhanced query with context
            enhanced_query = self._create_enhanced_query(message, query_analysis, context)
            
            # Execute query through JSON agent
            self.logger.info(f"Enhanced query: {enhanced_query}")
            result = self.json_agent.invoke({"input": enhanced_query})
            
            # Structure the agent result
            structured_result = self._structure_agent_result(result, message, query_analysis)
            
            return structured_result
            
        except Exception as e:
            self.logger.error(f"Error in chat interaction: {str(e)}")
            # Fallback to direct schema analysis
            return await self._fallback_schema_analysis(schema_path, message, file_id, context)
    
    def _create_enhanced_query(self, message: str, query_analysis: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Create enhanced query with context and intent"""
        
        # Base context about the data
        data_context = f"""
        You are analyzing a dataset schema with the following information:
        - Document Type: {self.schema_data.get('source_type', 'unknown')}
        - Table Name: {self.schema_data.get('table_name', 'unknown')}
        - Total Rows: {self.schema_data.get('row_count', 0)}
        - Total Columns: {len(self.schema_data.get('columns', []))}
        
        Query Intent: {query_analysis['intent']}
        """
        
        # Add specific instructions based on query intent
        if query_analysis["intent"] == "data_quality":
            data_context += """
            
            Focus on data quality aspects:
            - Analyze nullable columns and potential missing data issues
            - Identify data type consistency
            - Suggest data validation strategies
            """
        elif query_analysis["intent"] == "schema_inquiry":
            data_context += """
            
            Focus on schema structure:
            - Describe column names, types, and characteristics
            - Explain data organization and relationships
            - Provide detailed column information
            """
        elif query_analysis["intent"] == "data_analysis":
            data_context += """
            
            Focus on analytical possibilities:
            - Identify numeric columns for statistical analysis
            - Suggest categorical analysis opportunities
            - Recommend analysis techniques based on data types
            """
        
        # Combine context with user message
        enhanced_query = f"{data_context}\n\nUser Question: {message}\n\nPlease provide a detailed, helpful response based on the schema information."
        
        return enhanced_query
    
    def _structure_agent_result(self, result: Dict[str, Any], original_message: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the agent result for consistent response format"""
        try:
            # Extract information from agent result
            agent_response = result.get("output", "")
            
            # Generate additional insights
            insights = self._generate_insights_from_schema(original_message, query_analysis)
            
            # Create schema summary
            schema_summary = self._create_schema_summary()
            
            return {
                "message": original_message,
                "response": agent_response,
                "query_analysis": query_analysis,
                "insights": insights,
                "schema_summary": schema_summary,
                "schema_used": True,
                "agent_type": "json_toolkit",
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error structuring result: {str(e)}")
            return {
                "message": original_message,
                "response": str(result),
                "query_analysis": query_analysis,
                "success": False,
                "error": str(e)
            }
    
    async def _fallback_schema_analysis(self, schema_path: str, message: str, file_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback analysis when JSON agent fails"""
        try:
            self.logger.info("Using fallback schema analysis")
            
            # Load schema directly
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            # Analyze user query
            query_analysis = self._analyze_user_query(message)
            
            # Generate response using direct LLM call
            prompt = ChatPromptTemplate.from_template("""
            You are a data analysis assistant. Analyze the following dataset schema and answer the user's question.
            
            Schema Information:
            {schema_info}
            
            User Question: {user_message}
            
            Query Intent: {query_intent}
            
            Please provide a detailed, helpful response based on the schema information.
            Include specific insights about the data structure, quality, and analysis possibilities.
            """)
            
            # Format schema information
            schema_info = self._format_schema_for_prompt(schema)
            
            # Get response from LLM
            messages = prompt.format_messages(
                schema_info=schema_info,
                user_message=message,
                query_intent=query_analysis["intent"]
            )
            
            response = await self.llm.ainvoke(messages)
            
            # Generate insights
            insights = self._generate_insights_from_schema(message, query_analysis)
            schema_summary = self._create_schema_summary_from_data(schema)
            
            return {
                "message": message,
                "response": response.content,
                "query_analysis": query_analysis,
                "insights": insights,
                "schema_summary": schema_summary,
                "schema_used": True,
                "agent_type": "fallback_direct",
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback analysis: {str(e)}")
            raise
    
    def _analyze_user_query(self, message: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and information needs"""
        message_lower = message.lower()
        
        analysis = {
            "intent": "general_inquiry",
            "needs_schema_details": False,
            "needs_data_quality": False,
            "needs_column_info": False,
            "specific_columns": []
        }
        
        # Determine query intent
        if any(word in message_lower for word in ["quality", "clean", "missing", "null", "empty"]):
            analysis["intent"] = "data_quality"
            analysis["needs_data_quality"] = True
        elif any(word in message_lower for word in ["column", "field", "structure", "schema"]):
            analysis["intent"] = "schema_inquiry" 
            analysis["needs_column_info"] = True
        elif any(word in message_lower for word in ["insight", "analysis", "pattern", "trend"]):
            analysis["intent"] = "data_analysis"
            analysis["needs_schema_details"] = True
        elif any(word in message_lower for word in ["type", "format", "data type"]):
            analysis["intent"] = "type_inquiry"
            analysis["needs_column_info"] = True
        else:
            analysis["intent"] = "general_inquiry"
            analysis["needs_schema_details"] = True
        
        return analysis
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema information for LLM prompt"""
        try:
            formatted = f"""
Dataset Schema:
- Source Type: {schema.get('source_type', 'unknown')}
- Table Name: {schema.get('table_name', 'unknown')}
- Total Rows: {schema.get('row_count', 0):,}
- Total Columns: {len(schema.get('columns', []))}

Columns:
"""
            for i, col in enumerate(schema.get('columns', []), 1):
                formatted += f"""
{i}. {col.get('name', 'unknown')}
   - Type: {col.get('data_type', 'unknown')}
   - Nullable: {col.get('nullable', True)}
   - Unique Values: {col.get('unique_values', 0)}
   - Sample Values: {col.get('sample_values', [])}
"""
            
            return formatted
        except Exception as e:
            self.logger.error(f"Error formatting schema: {str(e)}")
            return "Schema information unavailable"
    
    def _generate_insights_from_schema(self, message: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights based on schema analysis"""
        insights = {
            "data_overview": {},
            "recommendations": [],
            "analysis_suggestions": []
        }
        
        if not self.schema_data:
            return insights
        
        try:
            columns = self.schema_data.get('columns', [])
            
            # Basic data overview
            insights["data_overview"] = {
                "total_columns": len(columns),
                "total_rows": self.schema_data.get('row_count', 0),
                "document_type": self.schema_data.get('source_type', 'unknown'),
                "table_name": self.schema_data.get('table_name', 'unknown')
            }
            
            # Generate recommendations based on query intent
            if query_analysis["intent"] == "data_quality":
                nullable_count = sum(1 for col in columns if col.get('nullable', True))
                if nullable_count > 0:
                    insights["recommendations"].extend([
                        f"Found {nullable_count} nullable columns - consider data validation",
                        "Check for missing values in critical columns",
                        "Validate data types for consistency"
                    ])
                
            elif query_analysis["intent"] == "data_analysis":
                numeric_cols = [col["name"] for col in columns if col.get("data_type") in ["INTEGER", "REAL", "NUMERIC"]]
                text_cols = [col["name"] for col in columns if col.get("data_type") == "TEXT"]
                
                if numeric_cols:
                    insights["analysis_suggestions"].extend([
                        f"Statistical analysis on {len(numeric_cols)} numeric columns",
                        "Correlation analysis between numeric variables",
                        "Distribution analysis for outlier detection"
                    ])
                
                if text_cols:
                    insights["analysis_suggestions"].extend([
                        f"Categorical analysis on {len(text_cols)} text columns",
                        "Frequency analysis for categorical variables"
                    ])
            
            elif query_analysis["intent"] == "schema_inquiry":
                insights["analysis_suggestions"].extend([
                    "Review column naming conventions",
                    "Analyze data type distribution",
                    "Identify potential primary keys"
                ])
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _create_schema_summary(self) -> Dict[str, Any]:
        """Create a concise schema summary from loaded data"""
        if not self.schema_data:
            return {}
        
        return self._create_schema_summary_from_data(self.schema_data)
    
    def _create_schema_summary_from_data(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise schema summary from schema data"""
        columns = schema.get('columns', [])
        
        summary = {
            "table_name": schema.get('table_name', 'unknown'),
            "total_columns": len(columns),
            "total_rows": schema.get('row_count', 0),
            "column_types": {},
            "nullable_columns": 0
        }
        
        # Count column types
        for col in columns:
            data_type = col.get('data_type', 'unknown')
            summary["column_types"][data_type] = summary["column_types"].get(data_type, 0) + 1
            
            if col.get('nullable', True):
                summary["nullable_columns"] += 1
        
        return summary
    
    # Legacy methods for compatibility
    def generate_data_insights(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from schema (legacy method for compatibility)"""
        try:
            insights = {
                "data_quality": self._assess_data_quality(schema),
                "column_analysis": self._analyze_columns(schema),
                "recommendations": self._generate_recommendations(schema)
            }
            return insights
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return {}
    
    def _assess_data_quality(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality from schema"""
        columns = schema.get('columns', [])
        total_rows = schema.get('row_count', 0)
        
        quality_metrics = {
            "completeness": 0,
            "column_count": len(columns),
            "nullable_columns": 0,
            "high_cardinality_columns": 0
        }
        
        if columns:
            nullable_count = sum(1 for col in columns if col.get('nullable', True))
            high_cardinality_count = sum(1 for col in columns if col.get('unique_values', 0) > total_rows * 0.8)
            
            quality_metrics.update({
                "nullable_columns": nullable_count,
                "high_cardinality_columns": high_cardinality_count,
                "completeness": (len(columns) - nullable_count) / len(columns) * 100 if columns else 0
            })
        
        return quality_metrics
    
    def _analyze_columns(self, schema: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Analyze individual columns"""
        columns = schema.get('columns', [])
        analysis = []
        
        for col in columns:
            col_analysis = {
                "name": col.get('name', 'unknown'),
                "type": col.get('data_type', 'unknown'),
                "uniqueness": col.get('unique_values', 0),
                "nullable": col.get('nullable', True),
                "characteristics": []
            }
            
            # Add characteristics based on analysis
            if col.get('unique_values', 0) == schema.get('row_count', 0):
                col_analysis["characteristics"].append("Unique identifier")
            
            if col.get('data_type') == 'TEXT' and col.get('unique_values', 0) < 10:
                col_analysis["characteristics"].append("Categorical")
            
            analysis.append(col_analysis)
        
        return analysis
    
    def _generate_recommendations(self, schema: Dict[str, Any]) -> list[str]:
        """Generate recommendations based on schema"""
        recommendations = []
        columns = schema.get('columns', [])
        
        # Check for potential issues and recommendations
        nullable_columns = [col for col in columns if col.get('nullable', True)]
        if nullable_columns:
            recommendations.append(f"Consider data validation for {len(nullable_columns)} nullable columns")
        
        text_columns = [col for col in columns if col.get('data_type') == 'TEXT']
        if len(text_columns) > len(columns) * 0.7:
            recommendations.append("Consider data type optimization - many TEXT columns detected")
        
        if schema.get('row_count', 0) > 10000:
            recommendations.append("Consider indexing for large dataset performance")
        
        return recommendations

# Factory function for creating JSON chat agent
def create_json_file_agent(llm=None):
    """Create a JSON chat agent for schema-based queries"""
    return JsonChatAgent()