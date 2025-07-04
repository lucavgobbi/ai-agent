#!/usr/bin/env python3
"""
Test script to verify the AI agent setup and dependencies.
"""
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test importing required modules."""
    print("🧪 Testing imports...")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import python-dotenv: {e}")
        return False
    
    try:
        from langchain_openai import AzureChatOpenAI
        print("✅ langchain-openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain-openai: {e}")
        return False
    
    try:
        from tools import WebSearchTool, WebContentExtractor, WikipediaTool
        print("✅ Tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import requests: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import beautifulsoup4: {e}")
        return False
    
    try:
        from duckduckgo_search import DDGS
        print("✅ duckduckgo-search imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import duckduckgo-search: {e}")
        return False
    
    try:
        import wikipedia
        print("✅ wikipedia imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import wikipedia: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\n🔧 Testing environment variables...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var} is not set")
        else:
            print(f"✅ {var} is set")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {missing_vars}")
        print("Please copy .env.example to .env and fill in your Azure OpenAI credentials.")
        return False
    
    return True

def test_tools():
    """Test basic tool functionality."""
    print("\n🛠️  Testing tools...")
    
    try:
        from tools import WebSearchTool, WikipediaTool
        
        # Test Wikipedia search
        wiki_tool = WikipediaTool()
        results = wiki_tool.search("Python programming", max_results=1)
        if results:
            print(f"✅ Wikipedia search works: Found {len(results)} results")
        else:
            print("⚠️  Wikipedia search returned no results")
        
        # Test web search
        web_tool = WebSearchTool()
        results = web_tool.search("test query", max_results=1)
        if results:
            print(f"✅ Web search works: Found {len(results)} results")
        else:
            print("⚠️  Web search returned no results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tools: {e}")
        return False

def test_llm_connection():
    """Test LLM connection if environment variables are set."""
    print("\n🤖 Testing LLM connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if all required env vars are present
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_DEPLOYMENT_NAME'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                print(f"⚠️  Skipping LLM test: {var} not set")
                return True
        
        from langchain_openai import AzureChatOpenAI
        
        llm = AzureChatOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
            deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            temperature=0.1,
            max_tokens=100,
        )
        
        response = llm.invoke("Hello! This is a test.")
        print(f"✅ LLM connection works: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM connection: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Agent Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Environment Tests", test_environment),
        ("Tool Tests", test_tools),
        ("LLM Connection Tests", test_llm_connection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI agent is ready to use.")
        print("Run: python agent.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
