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
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic

class FileProcessor:
    """Class to handle file preprocessing and database operations"""
    
    def __init__(self, llm, project="insights-452718", location="us-east5"):
        """Initialize the processor with LLM and project settings"""
        self.llm = llm
        self.project = project
        self.location = location
        self.db_path = "file_database.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.db = SQLDatabase(engine=self.engine)
        self.preprocessing_agent = self._create_preprocessing_agent()
        self.sql_agent = self._create_sql_agent()
        self.current_table_name = None
        
    def _create_preprocessing_agent(self):
        """Create the agent for preprocessing files"""
        # Define the preprocessing prompt
        preprocessing_prompt = hub.pull("hwchase17/react").partial(instructions="""
        You are an agent that preprocesses and cleans Excel/CSV files using Langchain, automatically detecting data types for each column and applying appropriate transformations.

        ## Column Classification and Processing

        ### 1. Numeric Conversion
        - **Currency Detection**:
          - Identify columns with currency symbols ($, €, ₹, etc.)
          - Extract and document the predominant currency symbol
          - Remove currency symbols and commas from values
          - Convert parenthetical values to positive numbers: ($45,95,040) → 4595040
          - Convert all cleaned values to float data type
          - Append the identified currency symbol to column headers for reference
          
        - **Percentage Handling**:
          - Convert percentage values (e.g., 50%) to numeric values (50)

        ### 2. Non-Numeric Columns
        - Preserve text format for identifier columns:
          - ID, Address, Zip, Phone, Invoice Number
        - Maintain text format for mixed alphanumeric fields (e.g., "123 Main St")
                        
        ### 3. Missing Value Processing
        - **For numeric columns**:
          - Fill missing values with column zero or '0'.
          
        - **For text columns**:
          - Fill missing values with 'unknown'.

        ### 4. Date and Time Standardization
        - Convert all date/datetime columns to standard format: 'YYYY-MM-DD HH:MM:SS'
        - Replace invalid date values with 'unknown'
                                                     
        Generate the cleaned dataset and return it as a pandas DataFrame for loading into a SQL database.
        """)
        
        # Define preprocessing tools
        preprocessing_tools = [PythonREPLTool(name="python_repl")]
        
        # Create preprocessing agent
        preprocessing_agent = create_react_agent(
            llm=self.llm,
            tools=preprocessing_tools,
            prompt=preprocessing_prompt
        )
        
        # Create agent executor
        return AgentExecutor(
            agent=preprocessing_agent, 
            tools=preprocessing_tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    
    def _create_sql_agent(self):
        """Create the SQL agent for querying the database"""
        # Create the SQL toolkit
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create the SQL agent
        return create_sql_agent(
            llm=self.llm, 
            db=self.db, 
            agent_type="tool-calling", 
            verbose=True
        )
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(anthropic.APIStatusError)
    )
    def _invoke_agent_with_retry(self, agent_executor, input_text):
        """Invoke an agent with retry mechanism"""
        return agent_executor.invoke({"input": input_text})
    
    def preprocess_file(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess a file and load it into the SQL database
        
        Args:
            file_path (str): Path to the CSV or Excel file
            table_name (str, optional): Name for the SQL table. If None, uses filename
            
        Returns:
            pd.DataFrame: The preprocessed DataFrame
        """
        try:
            # Load the DataFrame
            if file_path.endswith(".csv"):
                original_df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
            else:
                original_df = pd.read_excel(file_path)
            
            print(f"Input DataFrame shape: {original_df.shape}")
            print("Input DataFrame columns:", list(original_df.columns))
            
            # Save to temporary CSV
            temp_file = "temp_dataset.csv"
            original_df.to_csv(temp_file, index=False)
            
            # Process with agent
            print("Invoking preprocessing agent...")
            preprocessing_response = self._invoke_agent_with_retry(
                self.preprocessing_agent,
                f"Dataset is saved as '{temp_file}'. Clean and preprocess this file for SQL loading."
            )
            
            # Check if cleaneddata.csv was created
            output_file = "cleaneddata.csv"
            if os.path.exists(output_file):
                print(f"Found output file: {output_file}")
                processed_df = pd.read_csv(output_file)
            else:
                # Try to find a DataFrame in the agent's output
                try:
                    # Execute any DataFrame code in the agent's response
                    exec_locals = {'pd': pd}
                    exec("processed_df = pd.read_csv(temp_file)", globals(), exec_locals)
                    processed_df = exec_locals.get('processed_df', original_df)
                except Exception:
                    processed_df = original_df
            
            print("Processed DataFrame shape:", processed_df.shape)
            print("Processed DataFrame columns:", list(processed_df.columns))
            
            # Generate table name if not provided
            if table_name is None:
                base_name = os.path.basename(file_path)
                table_name = os.path.splitext(base_name)[0].lower().replace(" ", "_")
            
            # Store current table name
            self.current_table_name = table_name
            
            # Load into SQL database
            processed_df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            print(f"Loaded processed data into SQL table: {table_name}")
            
            # Refresh the database connection to see the new table
            self.db = SQLDatabase(engine=self.engine)
            self.sql_agent = self._create_sql_agent()
            
            return processed_df
        
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(error_msg)
            raise
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a natural language query against the database
        
        Args:
            query (str): Natural language query
            
        Returns:
            Dict: Results of the query
        """
        try:
            # If no table is loaded yet, raise an error
            if self.current_table_name is None:
                return {"error": "No data has been loaded. Please load a file first."}
            
            # Add table context to the query
            enhanced_query = f"Using the {self.current_table_name} table, {query}"
            
            # Run the query through the SQL agent
            print(f"Executing query: {enhanced_query}")
            result = self.sql_agent.invoke({"input": enhanced_query})
            
            return result
        
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

# Main implementation
def create_sql_file_agent(llm):
    """Create a SQL agent for preprocessed files"""
    return FileProcessor(llm=llm)

# Example usage
if __name__ == "__main__":
    # Initialize Vertex AI LLM
    project = "insights-452718"
    location = "us-east5"
    llm = ChatAnthropicVertex(
        model_name="claude-3-5-sonnet-v2@20241022",
        project=project,
        location=location,
    )
    
    # Create the file processor
    processor = create_sql_file_agent(llm)
    
    # Example: Process a file
    file_path = r"C:\Users\PhaniKumarMedapati\workspace\autopreprocessing of csv or excel files using vertexai\2023_HCP_Meeting_Report_by_Franchise_20231016_dataset_forNLP_Demo.xlsx"  # Change to your file path
    processor.preprocess_file(file_path)
    
    # Example: Run a query
    result = processor.execute_query("plot a bar chart for number of meetings completed by each franchise")
    print("Query result:", result)