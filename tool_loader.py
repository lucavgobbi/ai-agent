"""
Dynamic tool loader for the AI agent.
Loads tools based on configuration from tools_config.json.
Supports both LangChain tools and legacy tools.
"""
import json
import logging
import importlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class ToolLoader:
    """Dynamically loads and manages tools based on configuration."""
    
    def __init__(self, config_path: str = "tools_config.json"):
        """Initialize the tool loader with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.loaded_tools = {}
        self._load_tools()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file {self.config_path} not found")
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"✅ Loaded configuration from {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {str(e)}")
            raise
    
    def _load_tools(self):
        """Load all enabled tools from configuration."""
        tools_config = self.config.get('tools', {})
        
        for tool_name, tool_config in tools_config.items():
            if tool_config.get('enabled', False):
                try:
                    self._load_single_tool(tool_name, tool_config)
                except Exception as e:
                    logger.error(f"❌ Failed to load tool '{tool_name}': {str(e)}")
            else:
                logger.info(f"⏭️  Tool '{tool_name}' is disabled in configuration")
    
    def _load_single_tool(self, tool_name: str, tool_config: Dict[str, Any]):
        """Load a single tool based on its configuration."""
        module_name = tool_config.get('module')
        class_name = tool_config.get('class_name')
        config = tool_config.get('config', {})
        
        if not module_name or not class_name:
            raise ValueError(f"Tool '{tool_name}' missing 'module' or 'class_name' in configuration")
        
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the class
        tool_class = getattr(module, class_name)
        
        # Initialize the tool with configuration
        if config:
            tool_instance = tool_class(**config)
        else:
            tool_instance = tool_class()
        
        self.loaded_tools[tool_name] = {
            'instance': tool_instance,
            'config': tool_config,
            'description': tool_config.get('description', ''),
            'is_langchain_tool': isinstance(tool_instance, BaseTool)
        }
        
        logger.info(f"✅ Loaded tool '{tool_name}' ({class_name})")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a loaded tool instance by name."""
        tool_data = self.loaded_tools.get(tool_name)
        return tool_data['instance'] if tool_data else None
    
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names."""
        return list(self.loaded_tools.keys())
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description of a tool."""
        tool_data = self.loaded_tools.get(tool_name)
        return tool_data['description'] if tool_data else ''
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration of a tool."""
        tool_data = self.loaded_tools.get(tool_name)
        return tool_data['config'] if tool_data else {}
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config.get('agent_config', {})
    
    def get_search_strategy(self) -> Dict[str, Any]:
        """Get search strategy configuration."""
        return self.config.get('search_strategy', {})
    
    def reload_config(self):
        """Reload configuration and tools."""
        logger.info("🔄 Reloading configuration...")
        self.loaded_tools.clear()
        self.config = self._load_config()
        self._load_tools()
        logger.info("✅ Configuration reloaded successfully")
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return tool_name in self.loaded_tools
    
    def get_available_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available tools."""
        return {
            name: {
                'description': data['description'],
                'enabled': True,
                'config': data['config'].get('config', {})
            }
            for name, data in self.loaded_tools.items()
        }
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Get all enabled LangChain tools."""
        langchain_tools = []
        for tool_name, tool_data in self.loaded_tools.items():
            if tool_data.get('is_langchain_tool', False):
                langchain_tools.append(tool_data['instance'])
        return langchain_tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all enabled tools."""
        return [tool_data['instance'].name if tool_data.get('is_langchain_tool') 
                else tool_name for tool_name, tool_data in self.loaded_tools.items()]
