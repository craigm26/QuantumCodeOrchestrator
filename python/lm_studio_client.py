"""
LM Studio Client for Quantum Code Orchestrator
Handles communication with local LM Studio API server
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class LMStudioConfig:
    """Configuration for LM Studio connection"""
    api_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    model_name: str = "gpt-oss-20b"
    max_tokens: int = 2048
    timeout: int = 30

class LMStudioClient:
    """Client for communicating with LM Studio API"""
    
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.session = None
        self.base_url = config.api_url.rstrip('/')
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_text(self, prompt: str, temperature: float = 0.7, 
                          max_tokens: Optional[int] = None) -> str:
        """Generate text using LM Studio API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Use OpenAI-compatible API format
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logging.error(f"LM Studio API error: {response.status} - {error_text}")
                    return f"Error: {response.status} - {error_text}"
        except Exception as e:
            logging.error(f"LM Studio API request failed: {e}")
            return f"Error: {str(e)}"
    
    async def test_connection(self) -> bool:
        """Test if LM Studio API is available"""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                return response.status == 200
        except Exception as e:
            logging.error(f"LM Studio connection test failed: {e}")
            return False

class LMStudioAgent:
    """Agent that uses LM Studio for text generation"""
    
    def __init__(self, name: str, config: LMStudioConfig, temperature: float = 0.7):
        self.name = name
        self.config = config
        self.temperature = temperature
        self.client = None
    
    async def initialize(self):
        """Initialize the LM Studio client"""
        self.client = LMStudioClient(self.config)
        await self.client.__aenter__()
        
        # Test connection
        if not await self.client.test_connection():
            logging.warning(f"LM Studio connection failed for {self.name}")
            return False
        return True
    
    async def generate_response(self, prompt: str, context: List[str] = None) -> str:
        """Generate response using LM Studio"""
        if not self.client:
            return "LM Studio client not initialized"
        
        # Combine context and prompt
        full_prompt = prompt
        if context:
            context_text = "\n".join(context[-5:])  # Last 5 context items
            full_prompt = f"Context:\n{context_text}\n\nTask:\n{prompt}"
        
        return await self.client.generate_text(
            full_prompt, 
            temperature=self.temperature
        )
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.__aexit__(None, None, None) 