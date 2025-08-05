#!/usr/bin/env python3
"""
Quantum Code Orchestrator with LM Studio Integration
Modified version that uses LM Studio's local API instead of loading models directly
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import subprocess
import sys
import argparse

# Web server imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# File system monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Graph and visualization
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Import our LM Studio client
from python.lm_studio_client import LMStudioConfig, LMStudioAgent

# Import core components (without model loading)
try:
    from python.quantum_ai_backend import (
        CompressedVectorDB, QuantumOptimizer, CodeContext
    )
except ImportError:
    # Fallback if quantum_ai_backend is not available
    print("Warning: quantum_ai_backend not available, using demo mode")
    CompressedVectorDB = None
    QuantumOptimizer = None
    CodeContext = None

@dataclass
class AgentFocus:
    """Real-time agent focus tracking"""
    agent_id: str
    current_task: str
    focus_area: str
    progress: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class WorkflowNode:
    """Node in the workflow graph"""
    id: str
    type: str  # 'task', 'agent', 'decision', 'data'
    label: str
    status: str  # 'pending', 'active', 'completed', 'failed'
    agent_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class WorkflowEdge:
    """Edge in the workflow graph"""
    source: str
    target: str
    type: str  # 'data', 'control', 'dependency'
    weight: float = 1.0
    metadata: Dict[str, Any] = None

class RealTimeTracker:
    """Tracks agent activities and system state in real-time"""
    
    def __init__(self):
        self.agent_focuses: Dict[str, AgentFocus] = {}
        self.workflow_nodes: Dict[str, WorkflowNode] = {}
        self.workflow_edges: List[WorkflowEdge] = []
        self.active_tasks: Dict[str, Dict] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.event_log: List[Dict] = []
        self.connected_clients: Set[WebSocket] = set()

    def update_agent_focus(self, agent_id: str, task: str, focus_area: str, progress: float, metadata: Dict = None):
        """Update agent's current focus"""
        self.agent_focuses[agent_id] = AgentFocus(
            agent_id=agent_id,
            current_task=task,
            focus_area=focus_area,
            progress=progress,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Log event
        self.log_event("agent_focus_update", {
            "agent_id": agent_id,
            "task": task,
            "focus_area": focus_area,
            "progress": progress
        })

    def add_workflow_node(self, node: WorkflowNode):
        """Add node to workflow graph"""
        self.workflow_nodes[node.id] = node
        self.log_event("workflow_node_added", {"node_id": node.id, "type": node.type})

    def add_workflow_edge(self, edge: WorkflowEdge):
        """Add edge to workflow graph"""
        self.workflow_edges.append(edge)
        self.log_event("workflow_edge_added", {"source": edge.source, "target": edge.target})

    def update_task_status(self, task_id: str, status: str, metadata: Dict = None):
        """Update task status"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = status
            self.active_tasks[task_id]["updated_at"] = time.time()
            if metadata:
                self.active_tasks[task_id]["metadata"].update(metadata)
        
        # Update workflow node if exists
        if task_id in self.workflow_nodes:
            self.workflow_nodes[task_id].status = status
            if status == 'completed':
                self.workflow_nodes[task_id].end_time = time.time()
        
        self.log_event("task_status_update", {"task_id": task_id, "status": status})

    def log_event(self, event_type: str, data: Dict):
        """Log system event"""
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        self.event_log.append(event)
        
        # Keep only last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]

    def get_realtime_state(self) -> Dict:
        """Get complete real-time state"""
        return {
            "agent_focuses": {k: asdict(v) for k, v in self.agent_focuses.items()},
            "workflow_nodes": {k: asdict(v) for k, v in self.workflow_nodes.items()},
            "workflow_edges": [asdict(e) for e in self.workflow_edges],
            "active_tasks": self.active_tasks,
            "system_metrics": self.system_metrics,
            "recent_events": self.event_log[-50:],  # Last 50 events
            "timestamp": time.time()
        }

class LMStudioMultiAgentOrchestrator:
    """Multi-agent orchestrator using LM Studio"""
    
    def __init__(self, config: Dict, tracker: RealTimeTracker):
        self.config = config
        self.tracker = tracker
        self.agents = {}
        self.lm_config = LMStudioConfig(
            api_url=config.get("lm_studio", {}).get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("lm_studio", {}).get("api_key", "not-needed"),
            model_name=config.get("lm_studio", {}).get("model_name", "gpt-oss-20b"),
            max_tokens=config.get("lm_studio", {}).get("max_tokens", 2048),
            timeout=config.get("lm_studio", {}).get("timeout", 30)
        )
        
        # Initialize agents with different temperatures
        agent_temps = config.get("agent_temperatures", {})
        self.agents = {
            'researcher': LMStudioAgent('researcher', self.lm_config, agent_temps.get('researcher', 0.7)),
            'engineer': LMStudioAgent('engineer', self.lm_config, agent_temps.get('engineer', 0.3)),
            'analyst': LMStudioAgent('analyst', self.lm_config, agent_temps.get('analyst', 0.5))
        }

    async def initialize(self):
        """Initialize all agents"""
        logging.info("Initializing LM Studio agents...")
        for name, agent in self.agents.items():
            success = await agent.initialize()
            if success:
                logging.info(f"Agent {name} initialized successfully")
            else:
                logging.warning(f"Agent {name} initialization failed")
        
        # Initialize vector database if available
        if CompressedVectorDB:
            self.vector_db = CompressedVectorDB(self.config.get("db_path", "./vector_db"))
        else:
            self.vector_db = None
            logging.warning("Vector database not available")

    async def process_request(self, request: Dict) -> Dict:
        """Process a multi-agent request"""
        request_id = request.get('id', str(uuid.uuid4()))
        
        # Track request workflow
        self.tracker.add_workflow_node(WorkflowNode(
            id=request_id,
            type='request',
            label=f"Request: {request.get('type', 'unknown')}",
            status='active',
            start_time=time.time()
        ))
        
        # Decompose request into tasks
        tasks = self._decompose_request(request)
        
        # Execute tasks with different agents
        results = {}
        for i, task in enumerate(tasks):
            task_id = f"task_{i}_{request_id}"
            agent_name = self._select_agent_for_task(task)
            
            if agent_name in self.agents:
                # Update agent focus
                self.tracker.update_agent_focus(
                    agent_name, 
                    task.get('type', 'unknown'),
                    'processing',
                    0.0,
                    {'task_id': task_id}
                )
                
                # Process task
                result = await self._process_task_with_agent(agent_name, task, task_id)
                results[task_id] = result
                
                # Update completion
                self.tracker.update_agent_focus(
                    agent_name,
                    task.get('type', 'unknown'),
                    'completed',
                    100.0,
                    {'task_id': task_id}
                )
        
        # Synthesize results
        final_result = self._synthesize_results(results, request)
        
        # Update completion
        self.tracker.update_task_status(request_id, 'completed')
        
        return {
            'request_id': request_id,
            'results': final_result,
            'agent_contributions': {name: len([r for r in results.values() if r.get('agent') == name]) 
                                  for name in self.agents.keys()}
        }

    def _decompose_request(self, request: Dict) -> List[Dict]:
        """Decompose complex request into tasks"""
        request_type = request.get('type', 'general')
        
        if request_type == 'code_review':
            return [
                {'type': 'research', 'code': request.get('code', ''), 'query': 'analyze code patterns'},
                {'type': 'analysis', 'code': request.get('code', ''), 'metrics': ['complexity', 'security', 'maintainability']},
                {'type': 'engineering', 'requirement': 'suggest improvements', 'existing_code': request.get('code', '')}
            ]
        elif request_type == 'feature_implementation':
            return [
                {'type': 'research', 'query': f"research {request.get('feature', '')} implementation patterns"},
                {'type': 'engineering', 'requirement': request.get('feature', ''), 'existing_code': request.get('context', '')},
                {'type': 'analysis', 'code': '<!-- generated code -->', 'metrics': ['complexity', 'maintainability']}
            ]
        else:
            return [{'type': 'research', 'query': request.get('query', 'general assistance')}]

    def _select_agent_for_task(self, task: Dict) -> str:
        """Select appropriate agent for task"""
        task_type = task.get('type', 'general')
        
        if task_type == 'research':
            return 'researcher'
        elif task_type == 'engineering':
            return 'engineer'
        elif task_type == 'analysis':
            return 'analyst'
        else:
            return 'researcher'  # Default

    async def _process_task_with_agent(self, agent_name: str, task: Dict, task_id: str) -> Dict:
        """Process task with specific agent"""
        try:
            agent = self.agents[agent_name]
            
            # Create appropriate prompt based on task type
            prompt = self._create_prompt_for_task(task)
            
            # Generate response
            response = await agent.generate_response(prompt)
            
            return {
                'agent': agent_name,
                'task_id': task_id,
                'task_type': task.get('type'),
                'response': response,
                'status': 'completed'
            }
            
        except Exception as e:
            logging.error(f"Task execution failed for {agent_name}: {e}")
            return {
                'agent': agent_name,
                'task_id': task_id,
                'error': str(e),
                'status': 'failed'
            }

    def _create_prompt_for_task(self, task: Dict) -> str:
        """Create appropriate prompt for task type"""
        task_type = task.get('type', 'general')
        
        if task_type == 'research':
            code = task.get('code', '')
            query = task.get('query', 'Analyze this code')
            return f"""You are a code researcher. Analyze the following code and provide insights:

Code:
{code}

Query: {query}

Please provide:
1. Code structure and patterns analysis
2. Potential improvements
3. Security considerations
4. Performance implications
5. Best practices recommendations"""

        elif task_type == 'engineering':
            requirement = task.get('requirement', '')
            existing_code = task.get('existing_code', '')
            return f"""You are a code engineer. Generate high-quality code based on the requirements:

Requirement: {requirement}

Existing Code Context:
{existing_code}

Please generate code that:
1. Meets the requirements
2. Follows best practices
3. Is well-documented
4. Includes error handling
5. Is performant and secure"""

        elif task_type == 'analysis':
            code = task.get('code', '')
            metrics = task.get('metrics', ['complexity', 'maintainability'])
            return f"""You are a code analyst. Provide comprehensive analysis of the following code:

Code:
{code}

Metrics to analyze: {', '.join(metrics)}

Please provide:
1. Overall assessment
2. Key findings for each metric
3. Priority improvements
4. Risk assessment
5. Recommendations"""

        else:
            query = task.get('query', 'general assistance')
            return f"""You are an AI assistant. Please help with the following:

{query}

Provide a comprehensive and helpful response."""

    def _synthesize_results(self, results: Dict, original_request: Dict) -> Dict:
        """Synthesize results from multiple agents"""
        synthesis = {
            'summary': 'Multi-agent analysis completed',
            'key_findings': [],
            'recommendations': [],
            'generated_code': None,
            'analysis_report': None
        }
        
        for task_id, result in results.items():
            if 'response' in result:
                response = result['response']
                if 'analysis' in response.lower() or 'findings' in response.lower():
                    synthesis['key_findings'].append(response)
                if 'recommend' in response.lower() or 'suggest' in response.lower():
                    synthesis['recommendations'].append(response)
                if 'def ' in response or 'function' in response.lower():
                    synthesis['generated_code'] = response
                if 'assessment' in response.lower() or 'report' in response.lower():
                    synthesis['analysis_report'] = response
        
        return synthesis

    async def cleanup(self):
        """Clean up resources"""
        for agent in self.agents.values():
            await agent.cleanup()

# Import the rest of the components (FileSystemMonitor, FlowChartGenerator, etc.)
# These remain the same as in the original file
from quantum_code_orchestrator import (
    FileSystemMonitor, FlowChartGenerator, WebSocketManager,
    QuantumCodeOrchestratorServer, CLIInterface
)

class LMStudioQuantumCodeOrchestratorServer(QuantumCodeOrchestratorServer):
    """Server class with LM Studio integration"""
    
    async def initialize_system(self):
        """Initialize all system components with LM Studio"""
        try:
            logging.info("Initializing Quantum Code Orchestrator with LM Studio...")
            
            # Initialize LM Studio orchestrator
            self.orchestrator = LMStudioMultiAgentOrchestrator(self.config, self.tracker)
            await self.orchestrator.initialize()
            
            # Setup file system monitoring
            self.file_monitor = FileSystemMonitor(self.orchestrator, self.tracker)
            self.observer = Observer()
            
            for path in self.config["monitor_paths"]:
                if Path(path).exists():
                    self.observer.schedule(self.file_monitor, path, recursive=True)
                    logging.info(f"Monitoring path: {path}")
            
            self.observer.start()
            
            # Start background update task
            self.update_task = asyncio.create_task(self.background_updates())
            
            logging.info("LM Studio system initialization completed")
            return True
            
        except Exception as e:
            logging.error(f"LM Studio system initialization failed: {e}")
            return False

    async def shutdown(self):
        """Graceful shutdown with LM Studio cleanup"""
        logging.info("Shutting down LM Studio Quantum Code Orchestrator...")
        
        if hasattr(self, 'orchestrator') and self.orchestrator:
            await self.orchestrator.cleanup()
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        if self.update_task:
            self.update_task.cancel()
        
        # Close WebSocket connections
        for connection in self.websocket_manager.connections:
            await connection.close()

# Main entry point
async def main():
    parser = argparse.ArgumentParser(description="Quantum Code Orchestrator with LM Studio")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--cli", action="store_true", help="Run CLI interface")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize server
    server = LMStudioQuantumCodeOrchestratorServer(args.config)
    server.start_time = time.time()

    # Initialize system
    if not await server.initialize_system():
        print("❌ Failed to initialize LM Studio system")
        return

    if args.cli:
        # Run CLI interface
        cli = CLIInterface(server)
        cli.run_cli()
    else:
        # Run web server
        print(f"🚀 Starting Quantum Code Orchestrator with LM Studio")
        print(f"🌐 Dashboard: http://{args.host}:{args.port}")
        print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws")
        print(f"📡 API: http://{args.host}:{args.port}/api/")
        
        try:
            config = uvicorn.Config(
                server.app,
                host=args.host,
                port=args.port,
                log_level="info"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 