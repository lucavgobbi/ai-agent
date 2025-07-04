# AI Agent Setup and Usage Guide

## Quick Start

### 1. Test the Installation
```bash
python test_setup.py
```

### 2. Try the Demo (No Azure OpenAI Required)
```bash
python demo.py
```

### 3. Interactive Research Example
```bash
python example.py
```

### 4. Full AI Agent (Requires Azure OpenAI)
```bash
# First, configure your credentials in .env file
python agent.py
```

## Azure OpenAI Setup

### Prerequisites
- Azure subscription
- Azure OpenAI service deployed
- A chat model deployed (e.g., GPT-4, GPT-3.5-turbo)

### Configuration Steps

1. **Get your Azure OpenAI credentials:**
   - API Key
   - Endpoint URL
   - Deployment name
   - API version (usually `2024-02-15-preview`)

2. **Edit the `.env` file:**
   ```bash
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

3. **Test the connection:**
   ```bash
   python test_setup.py
   ```

## How the AI Agent Works

### Core Components

1. **Query Analysis**: The agent analyzes user queries to determine what information is needed
2. **Information Gathering**: Searches Wikipedia and the web for relevant information
3. **Content Extraction**: Extracts detailed content from promising web sources
4. **Answer Synthesis**: Uses Azure OpenAI to combine all information into a comprehensive answer
5. **Iterative Improvement**: Evaluates if the answer is sufficient or needs more research

### Search Capabilities

- **Wikipedia Search**: Factual, encyclopedic information
- **Web Search**: Current information using DuckDuckGo
- **Content Extraction**: Full text extraction from web pages
- **Multi-source Integration**: Combines information from multiple sources

### Iterative Process

The agent can perform up to 3 iterations per query:
1. **First iteration**: Basic research and initial answer
2. **Second iteration**: Additional research if needed
3. **Third iteration**: Final refinement

## Usage Examples

### Basic Usage
```
💬 Your question: What are the latest developments in AI?

🔄 Iteration 1/3
📝 Step 1: Analyzing your query...
🔍 Step 2: Gathering information from relevant sources...
🧠 Step 3: Synthesizing comprehensive answer...
✅ Answer is comprehensive. No further iteration needed.

📋 FINAL ANSWER:
[Comprehensive answer with sources]
```

### Available Commands
- `quit` or `exit`: Stop the agent
- `history`: View conversation history
- `clear`: Clear conversation history

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Azure OpenAI Connection Issues**
   - Verify credentials in `.env` file
   - Check if your Azure OpenAI service is running
   - Ensure the deployment name is correct

3. **Search Failures**
   - Check internet connection
   - Some searches may fail due to rate limits

4. **Content Extraction Issues**
   - Some websites may block content extraction
   - This is normal and the agent will continue with other sources

### Debug Mode
Set `LOG_LEVEL=DEBUG` in your `.env` file for verbose logging.

## Advanced Usage

### Customization Options

You can modify the agent behavior by editing `agent.py`:

- **Maximum iterations**: Change `max_iterations` parameter
- **Search result limits**: Modify `max_results` parameters
- **Content extraction length**: Adjust `max_length` parameter
- **LLM parameters**: Modify temperature, max_tokens, etc.

### Using Individual Tools

You can use the search tools independently:

```python
from tools import WebSearchTool, WikipediaTool, WebContentExtractor

# Web search
web_tool = WebSearchTool()
results = web_tool.search("your query", max_results=5)

# Wikipedia search
wiki_tool = WikipediaTool()
results = wiki_tool.search("your query", max_results=3)

# Content extraction
extractor = WebContentExtractor()
content = extractor.extract_content("https://example.com")
```

## Performance Tips

1. **Use specific queries**: More specific queries yield better results
2. **Limit iterations**: Set appropriate max_iterations for your use case
3. **Monitor API usage**: Azure OpenAI has usage limits and costs
4. **Cache results**: Consider implementing caching for repeated queries

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Content Filtering**: Be aware that web content may contain inappropriate material
3. **Rate Limits**: Respect API rate limits to avoid being blocked
4. **Data Privacy**: Be mindful of what information you're querying

## Contributing

To extend the agent:

1. **Add new search tools**: Implement new classes in `tools.py`
2. **Enhance analysis**: Improve query analysis logic
3. **Add new LLM providers**: Extend beyond Azure OpenAI
4. **Improve content extraction**: Handle more website types

## Files Overview

- `agent.py`: Main AI agent with iterative processing
- `tools.py`: Internet search and content extraction tools
- `demo.py`: Demonstration of search capabilities
- `example.py`: Standalone research assistant example
- `test_setup.py`: Setup verification script
- `requirements.txt`: Python dependencies
- `.env`: Configuration file (create from .env.example)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run `python test_setup.py` to verify your setup
3. Try the demo first: `python demo.py`
