# Query Pipeline Demonstration Suite

This directory contains comprehensive demonstrations of every component in the run_query_pipeline and its connected parts. Each demo is designed to showcase specific functionality and can be run independently or as part of the complete suite.

## 📁 Demo Files Overview

### Core Component Demos

1. **`demo_01_embedder_initialization.py`** - Embedder Setup
   - Initializes the EnhancedCSVEmbedder
   - Loads IRI mappings from CSV files
   - Shows embedding statistics and sample data
   - Demonstrates the foundation of semantic matching

2. **`demo_02_embedding_matching.py`** - Query-to-IRI Matching
   - Tests embedding similarity search
   - Extracts relevant IRIs from user queries
   - Shows matching confidence scores
   - Demonstrates semantic understanding capabilities

3. **`demo_03_clause_extraction.py`** - Structured Query Processing
   - Runs extract_chunk.py on sample queries
   - Shows clause extraction and parsing
   - Demonstrates compression techniques
   - Analyzes extraction warnings and confidence

4. **`demo_04_prompt_building.py`** - Template Processing
   - Builds LLM prompts from templates
   - Integrates user queries, extracted clauses, and turtle data
   - Shows placeholder replacement
   - Analyzes prompt structure and content

5. **`demo_05_llm_interaction.py`** - SPARQL Generation (Mocked)
   - Simulates LLM API calls and responses
   - Shows different SPARQL generation scenarios
   - Analyzes query structure and validity
   - Demonstrates response processing

6. **`demo_06_graphdb_query.py`** - Knowledge Graph Querying
   - Tests SPARQL execution against GraphDB
   - Shows different query types and results
   - Demonstrates result processing patterns
   - Tests connectivity and error scenarios

7. **`demo_07_error_handling.py`** - Retry and Recovery
   - Simulates various error scenarios
   - Shows error classification and correction
   - Demonstrates retry mechanisms
   - Tests recovery strategies

8. **`demo_08_complete_pipeline.py`** - End-to-End Integration
   - Runs the complete pipeline workflow
   - Shows component integration points
   - Demonstrates data flow between stages
   - Provides comprehensive output analysis

### Utility Files

- **`run_all_demos.py`** - Master Demo Runner
  - Executes all demos in sequence
  - Provides comprehensive demonstration
  - Shows success/failure statistics
  - Pauses between demos for review

## 🚀 How to Run

### Run Individual Demos
```cmd
cd demo
python demo_01_embedder_initialization.py
python demo_02_embedding_matching.py
# ... etc for each demo
```

### Run Complete Demo Suite
```cmd
cd demo
python run_all_demos.py
```

## 📊 What Each Demo Demonstrates

### Pipeline Components Covered:

1. **Embedding System**
   - CSV data loading and preprocessing
   - Semantic similarity matching
   - IRI resolution and mapping
   - Multi-category embedding search

2. **Query Processing**
   - Natural language clause extraction
   - Structured data representation
   - Compression and optimization
   - Query analysis and validation

3. **Prompt Engineering**
   - Template-based prompt construction
   - Dynamic content insertion
   - Reference data integration
   - Prompt optimization techniques

4. **LLM Integration**
   - API interaction patterns
   - Response parsing and validation
   - SPARQL query generation
   - Quality assessment metrics

5. **Knowledge Graph Interaction**
   - SPARQL query execution
   - Result processing and formatting
   - Connection management
   - Performance monitoring

6. **Error Management**
   - Error detection and classification
   - Automatic retry mechanisms
   - Query correction strategies
   - Failure recovery patterns

7. **Data Persistence**
   - Output file organization
   - Debug information capture
   - Run history maintenance
   - Performance analytics

## 📋 Expected Outputs

Each demo generates various output files:

- **Sample queries and results** (`.txt`, `.json`, `.sparql`)
- **Intermediate processing data** (extracted clauses, prompts)
- **Error examples and corrections** (error messages, fixed queries)
- **Performance metrics** (timing, success rates, compression ratios)
- **Debug information** (component status, configuration data)

## 🔧 Prerequisites

Before running the demos, ensure you have:

1. **Required Python packages:**
   - openai
   - python-dotenv
   - numpy (for embeddings)
   - Any other dependencies from requirements.txt

2. **Data files in parent directory:**
   - `label_iri_mappings.csv`
   - `industries_enhanced.csv`
   - `material_caps_enhanced.csv`
   - `certificates_enhanced_fixed.csv`
   - `process_embeddings_enhanced_with_def.csv`
   - `sparql_prompt_template.txt`
   - `reference_turtle.txt`

3. **External scripts:**
   - `extract_chunk.py`
   - `query_Graph.py`
   - `compress_chunk.py`
   - `enhanced_csv_embedder.py`

## 🎯 Key Insights Demonstrated

### Component Integration
- How embeddings enhance clause extraction with semantic IRIs
- How templates provide structure while allowing dynamic content
- How error handling creates robust query processing pipelines
- How output persistence enables debugging and analysis

### Performance Characteristics
- Embedding search efficiency across multiple categories
- Clause extraction accuracy and compression benefits
- Prompt optimization techniques and size management
- Query execution patterns and retry effectiveness

### Quality Assurance
- Semantic matching confidence and validation
- SPARQL syntax correctness and optimization
- Error recovery success rates and patterns
- End-to-end pipeline reliability metrics

## 🚨 Troubleshooting

### Common Issues:

1. **Missing data files**: Ensure all CSV and template files exist in parent directory
2. **Import errors**: Check that all required Python packages are installed
3. **File permissions**: Ensure write access for output file generation
4. **Path issues**: Run demos from within the demo/ directory

### Mock vs Real Execution:

Some demos use mocked responses to avoid:
- External API costs (OpenAI)
- Network connectivity requirements
- Long execution times
- External service dependencies

For real execution, configure appropriate API keys and service endpoints.

## 📈 Using Demo Results

The demonstration outputs can be used for:

- **Pipeline optimization**: Analyze performance bottlenecks
- **Quality improvement**: Review extraction accuracy and SPARQL correctness
- **Error analysis**: Study failure patterns and recovery strategies
- **Integration testing**: Verify component compatibility and data flow
- **Documentation**: Generate examples and usage patterns

## 🔄 Extending the Demos

To add new demonstrations:

1. Create `demo_XX_new_component.py` following the existing pattern
2. Add comprehensive logging and output generation
3. Include error handling and edge case testing
4. Update `run_all_demos.py` to include the new demo
5. Document the new demo in this README

Each demo should be self-contained but follow the established patterns for consistency and maintainability.
