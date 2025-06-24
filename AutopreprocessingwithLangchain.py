import pandas as pd
import io
import os
import psycopg2
import time
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic

# Initialize Vertex AI
project = "insights-452718"
location = "us-east5"
llm = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet-v2@20241022",
    project=project,
    location=location,
)

# Retry decorator for API calls
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(anthropic.APIStatusError)
)
def invoke_agent_with_retry(agent_executor, input_text):
    return agent_executor.invoke({"input": input_text})

# Define the ReAct agent prompt
prompt = hub.pull("hwchase17/react").partial(instructions="""
You are an agent that preprocesses and cleans Excel/CSV files using Langchain, automatically detecting data types for each column and applying appropriate transformations.

## Column Classification and Processing

### 1. Numeric Conversion
- **Currency Detection**:
  - Identify columns with currency symbols ($, €, ₹, etc.)
  - Extract and document the predominant currency symbol
  - Remove currency symbols and commas from values
  - Convert parenthetical values to positive numbers: ($45,95,040) → 4595040
  - Preserve negative signs when present: -$4,595,040 → -4595040
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
                                             
Generate and Save Outputs:
Save the cleaned dataset in the cleaneddata.csv.
""")

# Define the tool
tools = [PythonREPLTool(name="python_repl")]

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Output parser setup
class Output(BaseModel):
    output: str = Field(description="The result of the Python code.")

parser = PydanticOutputParser(pydantic_object=Output)
output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

def handle_error(error: OutputParserException):
    return output_fixing_parser.parse(error.llm_output)

def preprocess_file(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """
    Preprocesses a DataFrame based on the given prompt.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be preprocessed
        prompt (str): Preprocessing instructions
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Print initial DataFrame info
        print(f"Input DataFrame shape: {df.shape}")
        print("Input DataFrame columns:", list(df.columns))
        
        # Save to temporary CSV
        temp_file = "temp_dataset.csv"
        df.to_csv(temp_file, index=False)
        print(f"Saved temporary file to {temp_file}")
        
        # Process with agent
        print("Invoking agent with retry mechanism...")
        response = invoke_agent_with_retry(agent_executor, 
                                          f"Dataset is saved as '{temp_file}'. **{prompt}**")
        
        # Check if cleaneddata.csv was created
        output_file = "cleaneddata.csv"
        if os.path.exists(output_file):
            print(f"Found output file: {output_file}")
            processed_data = pd.read_csv(output_file)
            print("Processed DataFrame shape:", processed_data.shape)
            print("Processed DataFrame columns:", list(processed_data.columns))
        else:
            print("Output file not found, returning original DataFrame")
            processed_data = df
        
        return processed_data
            
    except Exception as e:
        error_msg = f"Error processing DataFrame: {str(e)}"
        print(error_msg)
        return df

# Main execution (example usage)
if __name__ == "__main__":
    # File path
    file_path = r"C:\Users\PhaniKumarMedapati\workspace\autopreprocessing of csv or excel files using vertexai\2023_HCP_Meeting_Report_by_Franchise_20231016_dataset_forNLP_Demo.xlsx"
    
    try:
        # Load the DataFrame first
        if file_path.endswith(".csv"):
            original_df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
        else:
            original_df = pd.read_excel(file_path)
        
        print("Processing DataFrame in a single attempt without batching.")
        result_df = preprocess_file(original_df, prompt)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")