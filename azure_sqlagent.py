import os
import openai
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import io
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from openai import RateLimitError

AZURE_OPENAI_ENDPOINT = "https://newkeyss.openai.azure.com/"
AZURE_OPENAI_API_KEY = "4076f9d9575544e384c754dd47ecf804"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
OPENAI_API_VERSION = "2024-08-01-preview" # add the openai api version to the global variables

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT_NAME
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION #add this to the environment variables

# Instantiate AzureOpenAI client, no need for api_version here
client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

model= AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)




# Set up Azure OpenAI environment variables

model=model



tools = [PythonREPLTool()]

# Instructions for the agent
instructions = """You are an agent that preprocesses and cleans Excel/CSV files using Langchain.
Automatically detect the data type of each column
You will analyze the dataset and classify columns into four categories:
Numeric Conversion:
Convert numeric columns with currency symbols ($, €, ₹) into pure numbers.
Identify the most common currency symbol in the column and Remove currency symbols ($, €, ₹) and commas from numerical values.(e.g.,Amount= $5000.0 becomes Amount ($) = 5000.0 )
Move currency symbols to the column header.Add the special character (e.g., $ annual salary, $ amount, % increase) to the left side of the column header (annual salary $, amount $)
Convert percentages (50% → 50).
Non-Numeric Columns:
Ensure "ID", "Address", "Zip", "Phone", "Invoice Number", and similar fields remain text.
If a column has mixed text and numbers (e.g., "123 Main St"), treat it as text.
Missing Value Handling:
Fill missing values in numeric columns with the median. if there is no median use 'unknown'.
Fill missing values in text columns with the most frequent value (mode). if there is no most frequent value use 'unknown'.
Date and Time Columns:
For all columns in the dataset, check if they contain date or datetime values. Convert these columns to a standard format: 'YYYY-MM-DD(e.g., 25-02-2023) HH:MM:SS (e.g., 09:30:40)'. If a value cannot be converted to a valid date, replace it with 'unknown'. Ensure consistency across all date columns to avoid format discrepancies.
Generate and Save Outputs:
Save the cleaned dataset in the cleaneddata.csv."""
# Load base prompt and modify it
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)




agent = create_openai_functions_agent(model, tools=tools, prompt=prompt)

# Output parsing
class Output(BaseModel):
    output: str = Field(description="The result of the Python code.")
parser = PydanticOutputParser(pydantic_object=Output)
output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

def handle_error(error: OutputParserException):
    return output_fixing_parser.parse(error.llm_output)

# Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=handle_error)


def preprocess_file(file_path, prompt, output_format="dataframe"):
    """Loads and preprocesses an Excel or CSV file based on user prompts."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path,encoding="ISO-8859-1",low_memory=False)
    else:
        df = pd.ExcelFile(file_path).parse(sheet_name=0)


    #df_json = df.to_json(orient="records")  # Converts DataFrame to a JSON string

    df.to_csv("temp_dataset.csv", index=False)

    # Change here: instead of passing entire prompt as input, pass a simpler instruction
    response = agent_executor.invoke({"input": f"Dataset is saved as 'temp_dataset.csv'. **{prompt}**"})


    #response = agent_executor.invoke({"input": f"Dataset:\n{df_json}\nUser Request: {prompt}"})
    # Invoke the agent with user prompt
    #response = agent_executor.invoke({"input": f"Dataset preview: {df.head().to_string()}\nUser Request: {prompt}"})

    return response

    processed_data = pd.read_csv(io.StringIO(response["output"]))

    # Return based on the desired format
    if output_format == "csv":
        processed_data.to_csv("processed_output.csv", index=False)
        return "Processed file saved as processed_output.csv"
    elif output_format == "excel":
        processed_data.to_excel("processed_output.xlsx", index=False)
        return "Processed file saved as processed_output.xlsx"
    else:
        return processed_data


file_path = r"C:\Users\PhaniKumarMedapati\workspace\autopreprocessing of csv or excel with Azure\2023_HCP_Meeting_Report_by_Franchise_20231016_dataset_forNLP_Demo.xlsx"

processed_data = preprocess_file(file_path, prompt,output_format="dataframe")


print(processed_data)



preprocess_file(file_path, prompt, output_format="csv")

# To save as Excel
# preprocess_file(file_path, user_prompt, output_format="excel")

print(processed_data)


##MCP integrated needed to be tested###

import pandas as pd
import io
import os
import time
import sqlite3
import json
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
from langchain_openai import AzureChatOpenAI

# MCP Protocol Structures
@dataclass
class MCPResource:
    """Represents an MCP resource (readable data)"""
    uri: str
    name: str
    description: str
    mimeType: str = "application/json"

@dataclass
class MCPTool:
    """Represents an MCP tool (executable action)"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPResponse:
    """Base class for MCP responses"""
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error
        }

class SQLAgentMCPServer:
    """MCP Server implementation for SQL Agent"""
    
    def __init__(self):
        # Azure OpenAI Configuration
        self.AZURE_OPENAI_ENDPOINT = "https://newkeyss.openai.azure.com/"
        self.AZURE_OPENAI_API_KEY = "4076f9d9575544e384c754dd47ecf804"
        self.AZURE_DEPLOYMENT_NAME = "gpt-4o"
        self.OPENAI_API_VERSION = "2024-08-01-preview"
        
        # Set environment variables
        os.environ["AZURE_OPENAI_API_KEY"] = self.AZURE_OPENAI_API_KEY
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.AZURE_OPENAI_ENDPOINT
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = self.AZURE_DEPLOYMENT_NAME
        os.environ["OPENAI_API_VERSION"] = self.OPENAI_API_VERSION
        
        # Initialize Azure OpenAI client
        self.client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        # Initialize LLM model
        self.model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )
        
        # Initialize file processor
        self.file_processor = FileProcessor(llm=self.model)
        
        # Define MCP resources and tools
        self.resources = self._define_resources()
        self.tools = self._define_tools()
    
    def _define_resources(self) -> List[MCPResource]:
        """Define available MCP resources"""
        return [
            MCPResource(
                uri="sql://database/tables",
                name="Database Tables",
                description="List of all available database tables and their schemas"
            ),
            MCPResource(
                uri="sql://database/current_table",
                name="Current Table Info",
                description="Information about the currently loaded table"
            ),
            MCPResource(
                uri="sql://files/processed",
                name="Processed Files",
                description="List of files that have been processed and loaded into the database"
            )
        ]
    
    def _define_tools(self) -> List[MCPTool]:
        """Define available MCP tools"""
        return [
            MCPTool(
                name="preprocess_file",
                description="Preprocess and load a CSV or Excel file into the SQL database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV or Excel file to process"
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Optional name for the SQL table. If not provided, uses filename"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            MCPTool(
                name="execute_sql_query",
                description="Execute a natural language query against the loaded database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to execute against the database"
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="get_table_schema",
                description="Get the schema information for a specific table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to get schema for"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            MCPTool(
                name="list_tables",
                description="List all available tables in the database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> MCPResponse:
        """Handle incoming MCP requests"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "resources/list":
                return MCPResponse(success=True, data=self.resources)
            
            elif method == "resources/read":
                uri = params.get("uri")
                return await self._read_resource(uri)
            
            elif method == "tools/list":
                return MCPResponse(success=True, data=self.tools)
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                return await self._call_tool(tool_name, arguments)
            
            else:
                return MCPResponse(success=False, error=f"Unknown method: {method}")
        
        except Exception as e:
            return MCPResponse(success=False, error=str(e))
    
    async def _read_resource(self, uri: str) -> MCPResponse:
        """Read MCP resource data"""
        try:
            if uri == "sql://database/tables":
                # Get list of tables and their schemas
                tables_info = self._get_all_tables_info()
                return MCPResponse(success=True, data=tables_info)
            
            elif uri == "sql://database/current_table":
                # Get current table information
                current_table_info = self._get_current_table_info()
                return MCPResponse(success=True, data=current_table_info)
            
            elif uri == "sql://files/processed":
                # Get list of processed files
                processed_files = self._get_processed_files_info()
                return MCPResponse(success=True, data=processed_files)
            
            else:
                return MCPResponse(success=False, error=f"Unknown resource URI: {uri}")
        
        except Exception as e:
            return MCPResponse(success=False, error=str(e))
    
    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Execute MCP tool"""
        try:
            if tool_name == "preprocess_file":
                file_path = arguments.get("file_path")
                table_name = arguments.get("table_name")
                
                result_df = self.file_processor.preprocess_file(file_path, table_name)
                
                return MCPResponse(
                    success=True, 
                    data={
                        "message": f"File processed successfully. Loaded {len(result_df)} rows into table '{self.file_processor.current_table_name}'",
                        "table_name": self.file_processor.current_table_name,
                        "rows_loaded": len(result_df),
                        "columns": list(result_df.columns)
                    }
                )
            
            elif tool_name == "execute_sql_query":
                query = arguments.get("query")
                result = self.file_processor.execute_query(query)
                
                return MCPResponse(success=True, data=result)
            
            elif tool_name == "get_table_schema":
                table_name = arguments.get("table_name")
                schema_info = self._get_table_schema(table_name)
                
                return MCPResponse(success=True, data=schema_info)
            
            elif tool_name == "list_tables":
                tables = self._list_database_tables()
                
                return MCPResponse(success=True, data=tables)
            
            else:
                return MCPResponse(success=False, error=f"Unknown tool: {tool_name}")
        
        except Exception as e:
            return MCPResponse(success=False, error=str(e))
    
    def _get_all_tables_info(self) -> Dict[str, Any]:
        """Get information about all tables in the database"""
        try:
            if hasattr(self.file_processor, 'db') and self.file_processor.db:
                tables = self.file_processor.db.get_usable_table_names()
                tables_info = {}
                
                for table in tables:
                    try:
                        schema = self.file_processor.db.get_table_info_no_throw([table])
                        tables_info[table] = {
                            "schema": schema,
                            "exists": True
                        }
                    except Exception as e:
                        tables_info[table] = {
                            "schema": None,
                            "exists": False,
                            "error": str(e)
                        }
                
                return tables_info
            else:
                return {"message": "No database connection available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_current_table_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded table"""
        if self.file_processor.current_table_name:
            return {
                "current_table": self.file_processor.current_table_name,
                "schema": self._get_table_schema(self.file_processor.current_table_name)
            }
        else:
            return {"message": "No table currently loaded"}
    
    def _get_processed_files_info(self) -> Dict[str, Any]:
        """Get information about processed files"""
        # This would typically track processed files in a more sophisticated way
        return {
            "current_database": self.file_processor.db_path,
            "current_table": self.file_processor.current_table_name,
            "message": "File processing history not tracked in current implementation"
        }
    
    def _get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema for a specific table"""
        try:
            if hasattr(self.file_processor, 'db') and self.file_processor.db:
                schema_info = self.file_processor.db.get_table_info_no_throw([table_name])
                return {
                    "table_name": table_name,
                    "schema": schema_info
                }
            else:
                return {"error": "No database connection available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _list_database_tables(self) -> Dict[str, Any]:
        """List all tables in the database"""
        try:
            if hasattr(self.file_processor, 'db') and self.file_processor.db:
                tables = self.file_processor.db.get_usable_table_names()
                return {
                    "tables": tables,
                    "count": len(tables)
                }
            else:
                return {"tables": [], "count": 0, "message": "No database connection available"}
        except Exception as e:
            return {"error": str(e)}

class FileProcessor:
    """Enhanced File Processor with Azure OpenAI integration"""
    
    def __init__(self, llm):
        """Initialize the processor with LLM"""
        self.llm = llm
        self.db_path = "file_database.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.db = SQLDatabase(engine=self.engine)
        self.preprocessing_agent = self._create_preprocessing_agent()
        self.sql_agent = self._create_sql_agent()
        self.current_table_name = None
        
    def _create_preprocessing_agent(self):
        """Create the agent for preprocessing files"""
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
        
        preprocessing_tools = [PythonREPLTool(name="python_repl")]
        
        preprocessing_agent = create_react_agent(
            llm=self.llm,
            tools=preprocessing_tools,
            prompt=preprocessing_prompt
        )
        
        return AgentExecutor(
            agent=preprocessing_agent, 
            tools=preprocessing_tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    
    def _create_sql_agent(self):
        """Create the SQL agent for querying the database"""
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        return create_sql_agent(
            llm=self.llm, 
            db=self.db, 
            agent_type="tool-calling", 
            verbose=True
        )
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception)
    )
    def _invoke_agent_with_retry(self, agent_executor, input_text):
        """Invoke an agent with retry mechanism"""
        return agent_executor.invoke({"input": input_text})
    
    def preprocess_file(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """Preprocess a file and load it into the SQL database"""
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
                try:
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
        """Execute a natural language query against the database"""
        try:
            if self.current_table_name is None:
                return {"error": "No data has been loaded. Please load a file first."}
            
            enhanced_query = f"Using the {self.current_table_name} table, {query}"
            
            print(f"Executing query: {enhanced_query}")
            result = self.sql_agent.invoke({"input": enhanced_query})
            
            return result
        
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

# MCP Client Example
class MCPClient:
    """Example MCP client to interact with the SQL Agent server"""
    
    def __init__(self, server: SQLAgentMCPServer):
        self.server = server
    
    async def list_resources(self):
        """List available resources"""
        request = {"method": "resources/list", "params": {}}
        response = await self.server.handle_mcp_request(request)
        return response
    
    async def read_resource(self, uri: str):
        """Read a specific resource"""
        request = {"method": "resources/read", "params": {"uri": uri}}
        response = await self.server.handle_mcp_request(request)
        return response
    
    async def list_tools(self):
        """List available tools"""
        request = {"method": "tools/list", "params": {}}
        response = await self.server.handle_mcp_request(request)
        return response
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool"""
        request = {
            "method": "tools/call", 
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        response = await self.server.handle_mcp_request(request)
        return response

# Example usage
async def main():
    # Create MCP server
    mcp_server = SQLAgentMCPServer()
    
    # Create MCP client
    client = MCPClient(mcp_server)
    
    # Example 1: List available resources
    print("=== Available Resources ===")
    resources_response = await client.list_resources()
    if resources_response.success:
        for resource in resources_response.data:
            print(f"- {resource.name}: {resource.description}")
    
    # Example 2: List available tools
    print("\n=== Available Tools ===")
    tools_response = await client.list_tools()
    if tools_response.success:
        for tool in tools_response.data:
            print(f"- {tool.name}: {tool.description}")
    
    # Example 3: Process a file (replace with your file path)
    file_path = "your_file.xlsx"  # Replace with actual file path
    if os.path.exists(file_path):
        print(f"\n=== Processing File: {file_path} ===")
        process_response = await client.call_tool(
            "preprocess_file", 
            {"file_path": file_path}
        )
        print("Process result:", process_response.to_dict())
        
        # Example 4: Execute a query
        print("\n=== Executing Query ===")
        query_response = await client.call_tool(
            "execute_sql_query",
            {"query": "Show me the first 5 rows of data"}
        )
        print("Query result:", query_response.to_dict())
    
    # Example 5: Read database tables resource
    print("\n=== Reading Database Tables Resource ===")
    tables_response = await client.read_resource("sql://database/tables")
    print("Tables info:", tables_response.to_dict())

if __name__ == "__main__":
    asyncio.run(main())