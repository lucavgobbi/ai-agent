"""
Internet search tools for the AI agent.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
import wikipedia
import logging

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self, max_results_default: int = 5, timeout: int = 10):
        self.ddgs = DDGS()
        self.max_results_default = max_results_default
        self.timeout = timeout
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the web and return results."""
        if max_results is None:
            max_results = self.max_results_default
        try:
            logger.info(f"🔍 Searching the web for: '{query}'")
            results = []
            
            # Use DuckDuckGo search
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'source': 'DuckDuckGo'
                })
            
            logger.info(f"✅ Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during web search: {str(e)}")
            return []

class WebContentExtractor:
    """Tool for extracting content from web pages."""
    
    def __init__(self, max_length_default: int = 2000, timeout: int = 10, 
                 user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'):
        self.max_length_default = max_length_default
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
    
    def extract_content(self, url: str, max_length: Optional[int] = None) -> str:
        """Extract text content from a web page."""
        if max_length is None:
            max_length = self.max_length_default
        try:
            logger.info(f"📄 Extracting content from: {url}")
            
            response = self.session.get(url, timeout=self.timeout)
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
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            logger.info(f"✅ Extracted {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"❌ Error extracting content from {url}: {str(e)}")
            return f"Failed to extract content from {url}: {str(e)}"

class WikipediaTool:
    """Tool for searching Wikipedia."""
    
    def __init__(self, max_results_default: int = 3, summary_sentences: int = 3):
        self.max_results_default = max_results_default
        self.summary_sentences = summary_sentences
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search Wikipedia and return results."""
        if max_results is None:
            max_results = self.max_results_default
        try:
            logger.info(f"📚 Searching Wikipedia for: '{query}'")
            
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            results = []
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    summary = wikipedia.summary(title, sentences=self.summary_sentences)
                    
                    results.append({
                        'title': page.title,
                        'snippet': summary,
                        'url': page.url,
                        'source': 'Wikipedia'
                    })
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option from disambiguation
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=self.summary_sentences)
                        results.append({
                            'title': page.title,
                            'snippet': summary,
                            'url': page.url,
                            'source': 'Wikipedia'
                        })
                    except:
                        continue
                except:
                    continue
            
            logger.info(f"✅ Found {len(results)} Wikipedia results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during Wikipedia search: {str(e)}")
            return []
