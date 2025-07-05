"""
LangChain-compatible tools for the AI agent.
Modern version using @tool decorator for cleaner implementation.
"""
import requests
from bs4 import BeautifulSoup
from typing import Optional
import wikipedia
import logging
import os
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Global session for HTTP requests
_session = requests.Session()
_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# Configuration constants
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_CONTENT_LENGTH = 2000
DEFAULT_TIMEOUT = 10
DEFAULT_SUMMARY_SENTENCES = 3


@tool
def web_search(query: str, max_results: Optional[int] = None) -> str:
    """Search the web for current information using Brave Search. 
    
    Use this when you need recent or current information from the internet.
    
    Args:
        query: The search query to look for on the web
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        String containing formatted search results with titles, descriptions, and URLs
    """
    if max_results is None:
        max_results = DEFAULT_MAX_RESULTS
        
    try:
        logger.info(f"🔍 Searching the web for: '{query}'")
        
        # Get Brave Search API key from environment
        api_key = os.getenv('BRAVE_SEARCH_API_KEY')
        if not api_key:
            return "Error: BRAVE_SEARCH_API_KEY environment variable not set. Please configure your Brave Search API key."
        
        # Brave Search API endpoint
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': api_key
        }
        params = {
            'q': query,
            'count': max_results,
            'safesearch': 'moderate',
            'search_lang': 'en',
            'country': 'US'
        }
        
        response = _session.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        web_results = data.get('web', {}).get('results', [])
        
        if not web_results:
            return "No web search results found for the query."
        
        results = []
        for i, result in enumerate(web_results, 1):
            results.append(f"""
Result {i}:
Title: {result.get('title', '')}
Description: {result.get('description', '')}
URL: {result.get('url', '')}
""")
        
        formatted_results = "\n".join(results)
        logger.info(f"✅ Found {len(results)} search results")
        return f"Web search results for '{query}':\n{formatted_results}"
        
    except Exception as e:
        logger.error(f"❌ Error during web search: {str(e)}")
        return f"Error performing web search: {str(e)}"


@tool
def extract_content(url: str, max_length: Optional[int] = None) -> str:
    """Extract the main text content from a web page URL. 
    
    Use this to get detailed information from specific web pages.
    
    Args:
        url: The URL to extract content from
        max_length: Maximum length of extracted content (default: 2000)
    
    Returns:
        String containing the extracted text content from the web page
    """
    if max_length is None:
        max_length = DEFAULT_MAX_CONTENT_LENGTH
        
    try:
        logger.info(f"📄 Extracting content from: {url}")
        
        response = _session.get(url, timeout=DEFAULT_TIMEOUT)
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


@tool
def wikipedia_search(query: str, max_results: Optional[int] = None) -> str:
    """Search Wikipedia for factual and encyclopedic information. 
    
    Use this for general knowledge, historical facts, and well-established information.
    
    Args:
        query: The search query to look for on Wikipedia
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        String containing formatted Wikipedia search results with titles, summaries, and URLs
    """
    if max_results is None:
        max_results = 3
        
    try:
        logger.info(f"📚 Searching Wikipedia for: '{query}'")
        
        # Search for pages
        search_results = wikipedia.search(query, results=max_results)
        results = []
        
        for i, title in enumerate(search_results, 1):
            try:
                page = wikipedia.page(title)
                summary = wikipedia.summary(title, sentences=DEFAULT_SUMMARY_SENTENCES)
                
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
                    summary = wikipedia.summary(e.options[0], sentences=DEFAULT_SUMMARY_SENTENCES)
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


# Export tools for easy access
__all__ = ['web_search', 'extract_content', 'wikipedia_search']
