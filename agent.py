"""
AI Agent using LangChain and Azure OpenAI with internet access capabilities.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from tools import WebSearchTool, WebContentExtractor, WikipediaTool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IterativeAIAgent:
    """An AI agent that can iteratively process user queries with internet access."""
    
    def __init__(self):
        """Initialize the AI agent with Azure OpenAI and tools."""
        self.llm = self._setup_llm()
        self.web_search = WebSearchTool()
        self.content_extractor = WebContentExtractor()
        self.wikipedia = WikipediaTool()
        self.conversation_history = []
        
    def _setup_llm(self) -> AzureChatOpenAI:
        """Setup Azure OpenAI LLM."""
        try:
            # Check for required environment variables
            required_vars = [
                'AZURE_OPENAI_API_KEY',
                'AZURE_OPENAI_ENDPOINT',
                'AZURE_OPENAI_DEPLOYMENT_NAME'
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            llm = AzureChatOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                temperature=0.1,
                max_tokens=1000,
            )
            logger.info("✅ Azure OpenAI LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {str(e)}")
            raise
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query to determine what information is needed."""
        analysis_prompt = f"""
        Analyze this user query and determine:
        1. Does it require current/recent information from the internet?
        2. Does it require factual information that might be found on Wikipedia?
        3. What specific search terms would be most effective?
        4. What type of answer format would be most helpful?
        
        Query: "{query}"
        
        Respond in this format:
        NEEDS_WEB_SEARCH: yes/no
        NEEDS_WIKIPEDIA: yes/no
        SEARCH_TERMS: [comma-separated list of search terms]
        ANSWER_TYPE: [brief description of expected answer format]
        """
        
        try:
            logger.info("🤔 Analyzing user query...")
            invoke_response = self.llm.invoke(analysis_prompt)
            response = invoke_response.content
            # Parse the response
            response_text = response if isinstance(response, str) else str(response)
            lines = response_text.strip().split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis[key.strip()] = value.strip()
            
            logger.info(f"📋 Query analysis complete: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing query: {str(e)}")
            return {
                'NEEDS_WEB_SEARCH': 'yes',
                'NEEDS_WIKIPEDIA': 'no',
                'SEARCH_TERMS': [query],
                'ANSWER_TYPE': 'comprehensive answer'
            }
    
    def _gather_information(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather information from various sources based on the analysis."""
        all_results = []
        
        search_terms_str = analysis.get('SEARCH_TERMS', '')
        search_terms = [term.strip() for term in search_terms_str.split(',') if term.strip()]
        
        if not search_terms:
            return all_results
        
        # Use the first search term as primary
        primary_term = search_terms[0]
        
        # Search Wikipedia if needed
        if analysis.get('NEEDS_WIKIPEDIA', '').lower() == 'yes':
            logger.info("📚 Searching Wikipedia...")
            wiki_results = self.wikipedia.search(primary_term, max_results=2)
            all_results.extend(wiki_results)
        
        # Search the web if needed
        if analysis.get('NEEDS_WEB_SEARCH', '').lower() == 'yes':
            logger.info("🌐 Searching the web...")
            web_results = self.web_search.search(primary_term, max_results=3)
            all_results.extend(web_results)
            
            # Extract content from top results
            for result in web_results[:2]:  # Only extract from top 2 results
                if result.get('url'):
                    content = self.content_extractor.extract_content(result['url'])
                    result['full_content'] = content
        
        logger.info(f"📊 Gathered {len(all_results)} information sources")
        return all_results
    
    def _synthesize_answer(self, query: str, information: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Synthesize a comprehensive answer using the gathered information."""
        
        # Prepare context from gathered information
        context_parts = []
        for i, info in enumerate(information, 1):
            source_info = f"Source {i} ({info.get('source', 'Unknown')}):\n"
            source_info += f"Title: {info.get('title', 'N/A')}\n"
            source_info += f"Content: {info.get('snippet', '')}\n"
            
            if info.get('full_content'):
                source_info += f"Full Content: {info.get('full_content', '')[:1000]}...\n"
            
            source_info += f"URL: {info.get('url', 'N/A')}\n"
            context_parts.append(source_info)
        
        context = "\n---\n".join(context_parts)
        
        synthesis_prompt = f"""
        You are a helpful AI assistant. Using the provided information sources, answer the user's query comprehensively and accurately.
        
        User Query: "{query}"
        
        Available Information:
        {context}
        
        Instructions:
        1. Provide a comprehensive and accurate answer based on the information gathered
        2. Cite your sources by mentioning them (e.g., "According to Wikipedia..." or "Based on recent web sources...")
        3. If information is conflicting or uncertain, acknowledge this
        4. If the gathered information is insufficient, say so honestly
        5. Provide URLs when relevant for further reading
        6. Structure your answer clearly with proper formatting
        
        Answer:
        """
        
        try:
            logger.info("🧠 Synthesizing comprehensive answer...")
            invoke_response = self.llm.invoke(synthesis_prompt)
            response = invoke_response.content
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing answer: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def _should_iterate(self, query: str, current_answer: str) -> bool:
        """Determine if another iteration would improve the answer."""
        
        iteration_prompt = f"""
        Evaluate whether the current answer adequately addresses the user's query, or if additional research would be beneficial.
        
        Original Query: "{query}"
        
        Current Answer: "{current_answer[:500]}..."
        
        Does this answer:
        1. Fully address all aspects of the user's query?
        2. Provide sufficient detail and accuracy?
        3. Include relevant current information if needed?
        
        Respond with only "SUFFICIENT" or "NEEDS_MORE_RESEARCH" followed by a brief reason.
        """
        
        try:
            invoke_response = self.llm.invoke(iteration_prompt)
            response = invoke_response.content
            result = response if isinstance(response, str) else str(response)
            
            if "NEEDS_MORE_RESEARCH" in result:
                logger.info(f"🔄 Iteration needed: {result}")
                return True
            else:
                logger.info(f"✅ Answer is sufficient: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error evaluating iteration need: {str(e)}")
            return False
    
    def process_query(self, query: str, max_iterations: int = 3) -> str:
        """Process a user query with iterative improvement."""
        
        print(f"\n🤖 Processing your query: '{query}'")
        print("=" * 60)
        
        iteration = 1
        current_answer = ""
        
        while iteration <= max_iterations:
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            print("-" * 40)
            
            # Step 1: Analyze the query
            print("📝 Step 1: Analyzing your query...")
            analysis = self._analyze_query(query)
            
            # Step 2: Gather information
            print("🔍 Step 2: Gathering information from relevant sources...")
            information = self._gather_information(analysis)
            
            if not information:
                print("⚠️  No additional information found. Using existing knowledge...")
            
            # Step 3: Synthesize answer
            print("🧠 Step 3: Synthesizing comprehensive answer...")
            current_answer = self._synthesize_answer(query, information, analysis)
            
            # Step 4: Check if iteration is needed (except for last iteration)
            if iteration < max_iterations:
                print("🤔 Step 4: Evaluating if more research is needed...")
                if not self._should_iterate(query, current_answer):
                    print("✅ Answer is comprehensive. No further iteration needed.")
                    break
                else:
                    print("🔄 Additional research may improve the answer...")
                    # For next iteration, modify the query to be more specific
                    query = f"{query} (provide more detailed and current information)"
            
            iteration += 1
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': current_answer,
            'iterations': iteration - 1
        })
        
        return current_answer
    
    def run_interactive_loop(self):
        """Run the interactive loop for user queries."""
        
        print("🤖 AI Agent with Internet Access")
        print("=" * 50)
        print("I'm an AI agent that can search the internet and iteratively")
        print("improve my answers. Ask me anything!")
        print("\nCommands:")
        print("- Type 'quit' or 'exit' to stop")
        print("- Type 'history' to see conversation history")
        print("- Type 'clear' to clear conversation history")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using the AI Agent!")
                    break
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️  Conversation history cleared!")
                    continue
                
                elif not user_input:
                    print("⚠️  Please enter a question or command.")
                    continue
                
                # Process the query
                answer = self.process_query(user_input)
                
                print("\n" + "=" * 60)
                print("📋 FINAL ANSWER:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the AI Agent!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                logger.error(f"Error in interactive loop: {str(e)}")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 40)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. Query: {entry['query'][:100]}...")
            print(f"   Iterations: {entry['iterations']}")
            print(f"   Answer: {entry['answer'][:200]}...")
            print("-" * 40)

def main():
    """Main function to run the AI agent."""
    try:
        # Check for required environment variables
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_DEPLOYMENT_NAME'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nPlease copy .env.example to .env and fill in your Azure OpenAI credentials.")
            return
        
        # Initialize and run the agent
        agent = IterativeAIAgent()
        agent.run_interactive_loop()
        
    except Exception as e:
        logger.error(f"Failed to start AI agent: {str(e)}")
        print(f"❌ Failed to start AI agent: {str(e)}")

if __name__ == "__main__":
    main()
