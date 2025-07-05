# AI Agent with Internet Access

An intelligent AI agent built with LangChain and Azure OpenAI that can access the internet to provide comprehensive, iterative answers to user queries.

## Features

- 🤖 **Iterative Processing**: The agent analyzes queries and iteratively improves answers
- 🌐 **Internet Access**: Searches the web using DuckDuckGo and extracts content from web pages
- 📚 **Wikipedia Integration**: Searches Wikipedia for factual information
- 🔄 **Self-Improving**: Evaluates its own answers and decides if more research is needed
- 💬 **Interactive Interface**: Easy-to-use command-line interface with conversation history
- 📝 **Detailed Logging**: Shows step-by-step process of how the agent works

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and fill in your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
```

### 3. Run the Agent

```bash
python agent.py
```

## How It Works

The agent follows an iterative process for each user query:

### 1. Query Analysis
- Analyzes the user's question to determine what type of information is needed
- Decides whether to search the web, Wikipedia, or both
- Identifies optimal search terms

### 2. Information Gathering
- **Web Search**: Uses DuckDuckGo to find recent and relevant information
- **Content Extraction**: Extracts full content from promising web pages
- **Wikipedia Search**: Searches Wikipedia for factual, encyclopedic information

### 3. Answer Synthesis
- Combines information from all sources
- Creates a comprehensive answer with proper citations
- Provides source URLs for further reading

### 4. Iterative Improvement
- Evaluates the quality and completeness of the current answer
- Decides if additional research would improve the response
- Performs up to 3 iterations for complex queries

## Usage Examples

### Basic Usage
```
💬 Your question: What are the latest developments in artificial intelligence?

🔄 Iteration 1/3
📝 Step 1: Analyzing your query...
🔍 Step 2: Gathering information from relevant sources...
🌐 Searching the web...
🧠 Step 3: Synthesizing comprehensive answer...
🤔 Step 4: Evaluating if more research is needed...
✅ Answer is comprehensive. No further iteration needed.

📋 FINAL ANSWER:
Based on recent web sources and current information, here are the latest developments in AI...
```

### Commands
- `quit` or `exit` - Stop the agent
- `history` - View conversation history
- `clear` - Clear conversation history

## Architecture

```
agent.py
├── IterativeAIAgent (Main class)
│   ├── Query Analysis
│   ├── Information Gathering
│   ├── Answer Synthesis
│   └── Iteration Control
│
tools.py
├── WebSearchTool (DuckDuckGo search)
├── WebContentExtractor (Web page content)
└── WikipediaTool (Wikipedia search)
```

## Key Components

### IterativeAIAgent
The main agent class that orchestrates the entire process:
- Manages conversation flow
- Coordinates different tools
- Handles iterative improvement logic

### WebSearchTool
Performs web searches using DuckDuckGo:
- Returns search results with titles, snippets, and URLs
- Handles search errors gracefully

### WebContentExtractor
Extracts content from web pages:
- Removes HTML markup and formatting
- Extracts clean text content
- Handles various web page formats

### WikipediaTool
Searches Wikipedia for factual information:
- Searches Wikipedia articles
- Handles disambiguation pages
- Returns summaries and full article URLs

## Configuration

### Environment Variables
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-02-15-preview)
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your model deployment name
- `LOG_LEVEL`: Logging level (default: INFO)

### Customization
You can customize various aspects of the agent:
- Maximum iterations per query (default: 3)
- Number of search results (default: 5 for web, 3 for Wikipedia)
- Content extraction length (default: 2000 characters)
- LLM parameters (temperature, max tokens, etc.)

## New Configuration System

The AI Agent now features a flexible JSON-based configuration system that allows you to customize tool behavior without modifying source code.

### Key Features:
- **🔧 Tool Management**: Enable/disable tools via configuration
- **⚙️ Customizable Settings**: Adjust search limits, timeouts, and other parameters
- **🔄 Runtime Reload**: Update configuration without restarting the agent
- **📊 Tool Status**: View current tool status and configurations

### Configuration File

Edit `tools_config.json` to customize:
- Which tools are enabled
- Tool-specific settings (timeouts, result limits, etc.)
- Agent behavior (max iterations, search strategy)

See [CONFIGURATION.md](CONFIGURATION.md) for detailed documentation.

### Interactive Commands

- `tools` - Display tool status and configuration
- `reload` - Reload configuration from file
- `history` - View conversation history
- `clear` - Clear conversation history

## Error Handling

The agent includes comprehensive error handling:
- Network connectivity issues
- API rate limits
- Malformed web pages
- Missing environment variables
- Azure OpenAI service errors

## Logging

Detailed logging shows the agent's thought process:
- 🤔 Query analysis
- 🔍 Information gathering
- 📚 Wikipedia searches
- 🌐 Web searches
- 📄 Content extraction
- 🧠 Answer synthesis
- ✅ Completion status

## Dependencies

- **langchain**: Core LangChain framework
- **langchain-openai**: Azure OpenAI integration
- **requests**: HTTP requests for web content
- **beautifulsoup4**: HTML parsing and content extraction
- **duckduckgo-search**: Web search capabilities
- **wikipedia**: Wikipedia API access
- **python-dotenv**: Environment variable management

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **Azure OpenAI Connection**: Verify your credentials in the `.env` file

3. **Search Failures**: Check internet connectivity and try again

4. **Rate Limits**: The agent includes built-in retry logic for API limits

### Debug Mode
Set `LOG_LEVEL=DEBUG` in your `.env` file for verbose logging.

## License

This project is open source and available under the MIT License.
