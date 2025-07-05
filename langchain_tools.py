"""
LangChain-compatible tools for the AI agent.
Fixed version that works with Pydantic validation.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Type
from duckduckgo_search import DDGS
import wikipedia
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="The search query to look for on the web")
    max_results: Optional[int] = Field(default=None, description="Maximum number of results to return")


class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo."""
    
    name: str = "web_search"
    description: str = "Search the web for current information using DuckDuckGo. Use this when you need recent or current information from the internet."
    
    def __init__(self, max_results_default: int = 5, timeout: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = WebSearchInput
        # Store config in a dict to avoid Pydantic field conflicts
        self._config = {
            'max_results_default': max_results_default,
            'timeout': timeout
        }
        self._ddgs = DDGS()
    
    def _run(self, query: str, max_results: Optional[int] = None) -> str:
        """Search the web and return formatted results."""
        if max_results is None:
            max_results = self._config['max_results_default']
            
        try:
            logger.info(f"🔍 Searching the web for: '{query}'")
            results = []
            
            # Use DuckDuckGo search
            search_results = self._ddgs.text(query, max_results=max_results)
            
            for i, result in enumerate(search_results, 1):
                results.append(f"""
Result {i}:
Title: {result.get('title', '')}
Description: {result.get('body', '')}
URL: {result.get('href', '')}
""")
            
            if not results:
                return "No web search results found for the query."
            
            formatted_results = "\n".join(results)
            logger.info(f"✅ Found {len(results)} search results")
            return f"Web search results for '{query}':\n{formatted_results}"
            
        except Exception as e:
            logger.error(f"❌ Error during web search: {str(e)}")
            return f"Error performing web search: {str(e)}"


class ContentExtractionInput(BaseModel):
    """Input schema for content extraction tool."""
    url: str = Field(description="The URL to extract content from")
    max_length: Optional[int] = Field(default=None, description="Maximum length of extracted content")


class WebContentExtractor(BaseTool):
    """Tool for extracting content from web pages."""
    
    name: str = "extract_content"
    description: str = "Extract the main text content from a web page URL. Use this to get detailed information from specific web pages."
    
    def __init__(self, max_length_default: int = 2000, timeout: int = 10, 
                 user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                 **kwargs):
        super().__init__(**kwargs)
        self.args_schema = ContentExtractionInput
        # Store config in a dict to avoid Pydantic field conflicts
        self._config = {
            'max_length_default': max_length_default,
            'timeout': timeout,
            'user_agent': user_agent
        }
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': user_agent
        })
    
    def _run(self, url: str, max_length: Optional[int] = None) -> str:
        """Extract text content from a web page."""
        if max_length is None:
            max_length = self._config['max_length_default']
            
        try:
            logger.info(f"📄 Extracting content from: {url}")
            
            response = self._session.get(url, timeout=self._config['timeout'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if max_length and len(text) > max_length:
                text = text[:max_length] + "..."
            
            logger.info(f"✅ Extracted {len(text)} characters from {url}")
            return f"Content extracted from {url}:\n{text}"
            
        except Exception as e:
            logger.error(f"❌ Error extracting content from {url}: {str(e)}")
            return f"Failed to extract content from {url}: {str(e)}"


class WikipediaSearchInput(BaseModel):
    """Input schema for Wikipedia search tool."""
    query: str = Field(description="The search query to look for on Wikipedia")
    max_results: Optional[int] = Field(default=None, description="Maximum number of results to return")


class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia."""
    
    name: str = "wikipedia_search"
    description: str = "Search Wikipedia for factual and encyclopedic information. Use this for general knowledge, historical facts, and well-established information."
    
    def __init__(self, max_results_default: int = 3, summary_sentences: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = WikipediaSearchInput
        # Store config in a dict to avoid Pydantic field conflicts
        self._config = {
            'max_results_default': max_results_default,
            'summary_sentences': summary_sentences
        }
    
    def _run(self, query: str, max_results: Optional[int] = None) -> str:
        """Search Wikipedia and return formatted results."""
        if max_results is None:
            max_results = self._config['max_results_default']
            
        try:
            logger.info(f"📚 Searching Wikipedia for: '{query}'")
            
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            results = []
            
            for i, title in enumerate(search_results, 1):
                try:
                    page = wikipedia.page(title)
                    summary = wikipedia.summary(title, sentences=self._config['summary_sentences'])
                    
                    results.append(f"""
Result {i}:
Title: {page.title}
Summary: {summary}
URL: {page.url}
""")
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option from disambiguation
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=self._config['summary_sentences'])
                        results.append(f"""
Result {i}:
Title: {page.title}
Summary: {summary}
URL: {page.url}
""")
                    except:
                        continue
                except:
                    continue
            
            if not results:
                return "No Wikipedia results found for the query."
            
            formatted_results = "\n".join(results)
            logger.info(f"✅ Found {len(results)} Wikipedia results")
            return f"Wikipedia search results for '{query}':\n{formatted_results}"
            
        except Exception as e:
            logger.error(f"❌ Error during Wikipedia search: {str(e)}")
            return f"Error performing Wikipedia search: {str(e)}"
