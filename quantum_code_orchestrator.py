"""
QuantumCodeOrchestrator - Real-time Multi-Agent Code Intelligence Server

A quantum-enhanced multi-agent system for code intelligence with real-time web visualization,
CLI integration, and async workflow monitoring.

Features:
- Real-time agent activity visualization
- Flow chart generation for code workflows  
- CLI interface for direct interaction
- File system monitoring and analysis
- WebSocket-based live updates
- Quantum-optimized task orchestration
- Interactive web dashboard
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

# Import our existing quantum agent system
try:
    from python.quantum_ai_backend import (
        MultiAgentOrchestrator, CodeResearcherAgent, CodeEngineerAgent, 
        CodeAnalystAgent, CompressedVectorDB, QuantumOptimizer, AgentConfig,
        CodeContext
    )
except ImportError:
    # Fallback if the module isn't available
    print("Warning: Could not import quantum_ai_backend module")
    # Create minimal stubs for demonstration
    class MultiAgentOrchestrator:
        def __init__(self, agents, quantum_optimizer):
            self.agents = agents
            self.quantum_optimizer = quantum_optimizer
        
        async def process_request(self, request):
            return {"status": "demo", "message": "Demo mode - no real processing"}
    
    class CodeResearcherAgent:
        def __init__(self, config, vector_db):
            self.config = config
            self.vector_db = vector_db
        
        async def process_task(self, task):
            return {"agent": "researcher", "analysis": "Demo analysis"}
    
    class CodeEngineerAgent:
        def __init__(self, config, vector_db):
            self.config = config
            self.vector_db = vector_db
        
        async def process_task(self, task):
            return {"agent": "engineer", "generated_code": "# Demo code"}
    
    class CodeAnalystAgent:
        def __init__(self, config, vector_db):
            self.config = config
            self.vector_db = vector_db
        
        async def process_task(self, task):
            return {"agent": "analyst", "report": "Demo report"}
    
    class CompressedVectorDB:
        def __init__(self, db_path):
            self.db_path = db_path
        
        def add_code_context(self, context):
            return 1
    
    class QuantumOptimizer:
        def __init__(self):
            pass
        
        def optimize_agent_allocation(self, tasks, agent_capabilities):
            return {f"task_{i}": list(agent_capabilities.keys())[i % len(agent_capabilities)] 
                   for i in range(len(tasks))}
    
    class AgentConfig:
        def __init__(self, name, model_path, max_context_length, temperature, top_p, quantum_enabled=False):
            self.name = name
            self.model_path = model_path
            self.max_context_length = max_context_length
            self.temperature = temperature
            self.top_p = top_p
            self.quantum_enabled = quantum_enabled
    
    class CodeContext:
        def __init__(self, file_path, content, ast_structure, dependencies, embeddings=None, metadata=None):
            self.file_path = file_path
            self.content = content
            self.ast_structure = ast_structure
            self.dependencies = dependencies
            self.embeddings = embeddings
            self.metadata = metadata or {}

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
        
    def update_agent_focus(self, agent_id: str, task: str, focus_area: str, 
                          progress: float, metadata: Dict = None):
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

class FileSystemMonitor(FileSystemEventHandler):
    """Monitor file system changes and trigger agent analysis"""
    
    def __init__(self, orchestrator: MultiAgentOrchestrator, tracker: RealTimeTracker):
        self.orchestrator = orchestrator
        self.tracker = tracker
        self.monitored_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h'}
        self.debounce_delay = 1.0  # seconds
        self.pending_files = {}
        
    def on_modified(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix in self.monitored_extensions:
                self._schedule_analysis(str(file_path))
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix in self.monitored_extensions:
                self._schedule_analysis(str(file_path))
    
    def _schedule_analysis(self, file_path: str):
        """Schedule file analysis with debouncing"""
        current_time = time.time()
        
        # Cancel previous timer if exists
        if file_path in self.pending_files:
            self.pending_files[file_path].cancel()
        
        # Schedule new analysis
        timer = threading.Timer(self.debounce_delay, self._analyze_file, [file_path])
        self.pending_files[file_path] = timer
        timer.start()
        
        self.tracker.log_event("file_change_detected", {"file_path": file_path})
    
    def _analyze_file(self, file_path: str):
        """Analyze changed file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create analysis task
            task_id = f"file_analysis_{uuid.uuid4().hex[:8]}"
            
            # Track the analysis workflow
            self.tracker.add_workflow_node(WorkflowNode(
                id=task_id,
                type='task',
                label=f"Analyze {Path(file_path).name}",
                status='pending',
                metadata={'file_path': file_path}
            ))
            
            # Schedule async analysis
            asyncio.create_task(self._async_analyze_file(task_id, file_path, content))
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {e}")
            self.tracker.log_event("file_analysis_error", {
                "file_path": file_path,
                "error": str(e)
            })
        finally:
            # Clean up
            if file_path in self.pending_files:
                del self.pending_files[file_path]
    
    async def _async_analyze_file(self, task_id: str, file_path: str, content: str):
        """Asynchronously analyze file with orchestrator"""
        try:
            self.tracker.update_task_status(task_id, 'active')
            
            request = {
                'type': 'code_review',
                'code': content,
                'file_path': file_path,
                'id': task_id
            }
            
            # Process with orchestrator
            result = await self.orchestrator.process_request(request)
            
            self.tracker.update_task_status(task_id, 'completed', {
                'result': result,
                'file_path': file_path
            })
            
        except Exception as e:
            logging.error(f"Async file analysis failed for {file_path}: {e}")
            self.tracker.update_task_status(task_id, 'failed', {'error': str(e)})

class FlowChartGenerator:
    """Generate flow charts for agent workflows"""
    
    def __init__(self, tracker: RealTimeTracker):
        self.tracker = tracker
    
    def generate_workflow_graph(self) -> str:
        """Generate workflow graph as base64 encoded PNG"""
        try:
            # Create networkx graph
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in self.tracker.workflow_nodes.items():
                G.add_node(node_id, **asdict(node))
            
            # Add edges
            for edge in self.tracker.workflow_edges:
                G.add_edge(edge.source, edge.target, **asdict(edge))
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Color nodes by status
            node_colors = []
            for node_id in G.nodes():
                node = self.tracker.workflow_nodes.get(node_id)
                if node:
                    if node.status == 'completed':
                        node_colors.append('lightgreen')
                    elif node.status == 'active':
                        node_colors.append('yellow')
                    elif node.status == 'failed':
                        node_colors.append('lightcoral')
                    else:
                        node_colors.append('lightblue')
                else:
                    node_colors.append('lightgray')
            
            # Draw graph
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=1000,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   linewidths=2)
            
            # Add labels
            labels = {}
            for node_id, node in self.tracker.workflow_nodes.items():
                labels[node_id] = node.label[:15] + ('...' if len(node.label) > 15 else '')
            
            nx.draw_networkx_labels(G, pos, labels, font_size=6)
            
            plt.title("Agent Workflow - Real-time View", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logging.error(f"Error generating workflow graph: {e}")
            return ""
    
    def generate_agent_focus_chart(self) -> str:
        """Generate agent focus visualization"""
        try:
            if not self.tracker.agent_focuses:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Agent progress chart
            agents = list(self.tracker.agent_focuses.keys())
            progress = [focus.progress for focus in self.tracker.agent_focuses.values()]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = ax1.barh(agents, progress, color=colors[:len(agents)])
            ax1.set_xlabel('Progress (%)')
            ax1.set_title('Agent Progress')
            ax1.set_xlim(0, 100)
            
            # Add progress text
            for i, (agent, prog) in enumerate(zip(agents, progress)):
                ax1.text(prog + 2, i, f'{prog:.1f}%', va='center')
            
            # Agent focus areas pie chart
            focus_areas = [focus.focus_area for focus in self.tracker.agent_focuses.values()]
            focus_counts = {}
            for area in focus_areas:
                focus_counts[area] = focus_counts.get(area, 0) + 1
            
            if focus_counts:
                ax2.pie(focus_counts.values(), labels=focus_counts.keys(), autopct='%1.1f%%')
                ax2.set_title('Current Focus Areas')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logging.error(f"Error generating agent focus chart: {e}")
            return ""

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self, tracker: RealTimeTracker):
        self.tracker = tracker
        self.connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.connections.add(websocket)
        
        # Send initial state
        await self.send_to_client(websocket, {
            "type": "initial_state",
            "data": self.tracker.get_realtime_state()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.connections.discard(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
            
        dead_connections = set()
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.add(connection)
        
        # Clean up dead connections
        self.connections -= dead_connections
    
    async def send_to_client(self, websocket: WebSocket, message: Dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except:
            self.connections.discard(websocket)

class QuantumCodeOrchestratorServer:
    """Main server class combining all components"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.tracker = RealTimeTracker()
        self.app = FastAPI(title="Quantum Code Orchestrator", version="1.0.0")
        self.setup_middleware()
        
        # Core components
        self.vector_db = None
        self.quantum_optimizer = None
        self.orchestrator = None
        self.file_monitor = None
        self.observer = None
        self.websocket_manager = WebSocketManager(self.tracker)
        self.flowchart_generator = FlowChartGenerator(self.tracker)
        
        # Setup routes
        self.setup_routes()
        
        # Background tasks
        self.update_task = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "model_path": "path/to/your/10b/model",
            "db_path": "./vector_db",
            "monitor_paths": ["./"],
            "port": 8000,
            "host": "127.0.0.1",
            "log_level": "INFO"
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            return default_config
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(self.get_dashboard_html())
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/api/state")
        async def get_state():
            return self.tracker.get_realtime_state()
        
        @self.app.get("/api/agents")
        async def get_agents():
            return {
                "agents": list(self.tracker.agent_focuses.keys()),
                "focuses": {k: asdict(v) for k, v in self.tracker.agent_focuses.items()}
            }
        
        @self.app.get("/api/workflow/graph")
        async def get_workflow_graph():
            graph_image = self.flowchart_generator.generate_workflow_graph()
            return {"image": graph_image, "format": "png"}
        
        @self.app.get("/api/agents/focus-chart")
        async def get_agent_focus_chart():
            chart_image = self.flowchart_generator.generate_agent_focus_chart()
            return {"image": chart_image, "format": "png"}
        
        @self.app.post("/api/analyze")
        async def analyze_code(request: Dict):
            """Analyze code via API"""
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            code = request.get("code", "")
            file_path = request.get("file_path", "unknown")
            
            # Create analysis request
            analysis_request = {
                'type': 'code_review',
                'code': code,
                'file_path': file_path,
                'id': f"api_analysis_{uuid.uuid4().hex[:8]}"
            }
            
            result = await self.orchestrator.process_request(analysis_request)
            return result
        
        @self.app.post("/api/generate")
        async def generate_feature(request: Dict):
            """Generate feature via API"""
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            feature = request.get("feature", "")
            context = request.get("context", "")
            
            # Create generation request
            generation_request = {
                'type': 'feature_implementation',
                'feature': feature,
                'context': context,
                'id': f"api_generation_{uuid.uuid4().hex[:8]}"
            }
            
            result = await self.orchestrator.process_request(generation_request)
            return result
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
        
        # Static files for dashboard
        static_dir = Path("static")
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            logging.info("Initializing Quantum Code Orchestrator...")
            
            # Initialize vector database
            self.vector_db = CompressedVectorDB(self.config["db_path"])
            
            # Initialize quantum optimizer
            self.quantum_optimizer = QuantumOptimizer()
            
            # Initialize agent configurations
            agent_configs = {
                'researcher': AgentConfig(
                    name='researcher',
                    model_path=self.config["model_path"],
                    max_context_length=8192,
                    temperature=0.7,
                    top_p=0.9,
                    quantum_enabled=True
                ),
                'engineer': AgentConfig(
                    name='engineer',
                    model_path=self.config["model_path"],
                    max_context_length=8192,
                    temperature=0.3,
                    top_p=0.8,
                    quantum_enabled=True
                ),
                'analyst': AgentConfig(
                    name='analyst',
                    model_path=self.config["model_path"],
                    max_context_length=8192,
                    temperature=0.5,
                    top_p=0.85,
                    quantum_enabled=True
                )
            }
            
            # Initialize agents with tracking
            agents = {}
            for name, config in agent_configs.items():
                if name == 'researcher':
                    agents[name] = TrackedCodeResearcherAgent(config, self.vector_db, self.tracker)
                elif name == 'engineer':
                    agents[name] = TrackedCodeEngineerAgent(config, self.vector_db, self.tracker)
                elif name == 'analyst':
                    agents[name] = TrackedCodeAnalystAgent(config, self.vector_db, self.tracker)
            
            # Initialize orchestrator
            self.orchestrator = TrackedMultiAgentOrchestrator(
                agents, self.quantum_optimizer, self.tracker
            )
            
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
            
            logging.info("System initialization completed")
            return True
            
        except Exception as e:
            logging.error(f"System initialization failed: {e}")
            return False
    
    async def background_updates(self):
        """Background task for real-time updates"""
        while True:
            try:
                # Update system metrics
                self.tracker.system_metrics = {
                    "active_agents": len(self.tracker.agent_focuses),
                    "active_tasks": len(self.tracker.active_tasks),
                    "workflow_nodes": len(self.tracker.workflow_nodes),
                    "connected_clients": len(self.websocket_manager.connections),
                    "uptime": time.time() - getattr(self, 'start_time', time.time())
                }
                
                # Broadcast updates to connected clients
                await self.websocket_manager.broadcast({
                    "type": "system_update",
                    "data": self.tracker.get_realtime_state()
                })
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logging.error(f"Background update error: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Code Orchestrator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .agent-status { display: flex; align-items: center; margin: 10px 0; }
        .status-indicator { width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .active { background: #4CAF50; }
        .idle { background: #FFC107; }
        .error { background: #F44336; }
        .chart-container { text-align: center; }
        .chart-container img { max-width: 100%; border-radius: 5px; }
        .event-log { max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        .event { padding: 5px; border-bottom: 1px solid #eee; }
        .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .metric { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 Quantum Code Orchestrator</h1>
        <p>Real-time Multi-Agent Code Intelligence System</p>
    </div>
    
    <div class="dashboard">
        <div class="panel">
            <h3>System Metrics</h3>
            <div class="metrics" id="metrics">
                <!-- Metrics will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="panel">
            <h3>Agent Status</h3>
            <div id="agent-status">
                <!-- Agent status will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="panel">
            <h3>Workflow Graph</h3>
            <div class="chart-container" id="workflow-chart">
                <p>Loading workflow graph...</p>
            </div>
        </div>
        
        <div class="panel">
            <h3>Agent Focus</h3>
            <div class="chart-container" id="focus-chart">
                <p>Loading focus chart...</p>
            </div>
        </div>
        
        <div class="panel" style="grid-column: 1 / -1;">
            <h3>Recent Events</h3>
            <div class="event-log" id="event-log">
                <!-- Events will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'system_update' || message.type === 'initial_state') {
                updateDashboard(message.data);
            }
        };
        
        function updateDashboard(data) {
            // Update metrics
            const metrics = data.system_metrics || {};
            document.getElementById('metrics').innerHTML = Object.entries(metrics)
                .map(([key, value]) => `
                    <div class="metric">
                        <div class="metric-value">${typeof value === 'number' ? value.toFixed(0) : value}</div>
                        <div class="metric-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                    </div>
                `).join('');
            
            // Update agent status
            const agents = data.agent_focuses || {};
            document.getElementById('agent-status').innerHTML = Object.entries(agents)
                .map(([id, focus]) => `
                    <div class="agent-status">
                        <div class="status-indicator active"></div>
                        <div>
                            <strong>${id}</strong><br>
                            <small>${focus.focus_area} (${focus.progress.toFixed(1)}%)</small>
                        </div>
                    </div>
                `).join('');
            
            // Update events
            const events = data.recent_events || [];
            document.getElementById('event-log').innerHTML = events
                .slice(-20)  // Last 20 events
                .reverse()
                .map(event => `
                    <div class="event">
                        <strong>${new Date(event.timestamp * 1000).toLocaleTimeString()}</strong>
                        ${event.type}: ${JSON.stringify(event.data)}
                    </div>
                `).join('');
        }
        
        // Load charts
        async function loadCharts() {
            try {
                const workflowResponse = await fetch('/api/workflow/graph');
                const workflowData = await workflowResponse.json();
                if (workflowData.image) {
                    document.getElementById('workflow-chart').innerHTML = 
                        `<img src="data:image/png;base64,${workflowData.image}" alt="Workflow Graph">`;
                }
                
                const focusResponse = await fetch('/api/agents/focus-chart');
                const focusData = await focusResponse.json();
                if (focusData.image) {
                    document.getElementById('focus-chart').innerHTML = 
                        `<img src="data:image/png;base64,${focusData.image}" alt="Agent Focus">`;
                }
            } catch (error) {
                console.error('Error loading charts:', error);
            }
        }
        
        // Load charts every 10 seconds
        setInterval(loadCharts, 10000);
        loadCharts();
    </script>
</body>
</html>
        """
    
    async def shutdown(self):
        """Graceful shutdown"""
        logging.info("Shutting down Quantum Code Orchestrator...")
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        if self.update_task:
            self.update_task.cancel()
        
        # Close WebSocket connections
        for connection in self.websocket_manager.connections:
            await connection.close()

# Enhanced agent classes with tracking
class TrackedCodeResearcherAgent(CodeResearcherAgent):
    """Researcher agent with real-time tracking"""
    
    def __init__(self, config, vector_db, tracker):
        super().__init__(config, vector_db)
        self.tracker = tracker
    
    async def process_task(self, task: Dict) -> Dict:
        self.tracker.update_agent_focus(
            self.config.name, 
            task.get('type', 'unknown'),
            'code_analysis',
            0.0,
            {'task_id': task.get('id')}
        )
        
        result = await super().process_task(task)
        
        self.tracker.update_agent_focus(
            self.config.name,
            task.get('type', 'unknown'),
            'completed',
            100.0,
            {'task_id': task.get('id')}
        )
        
        return result

class TrackedCodeEngineerAgent(CodeEngineerAgent):
    """Engineer agent with real-time tracking"""
    
    def __init__(self, config, vector_db, tracker):
        super().__init__(config, vector_db)
        self.tracker = tracker
    
    async def process_task(self, task: Dict) -> Dict:
        self.tracker.update_agent_focus(
            self.config.name,
            task.get('type', 'unknown'),
            'code_generation',
            0.0,
            {'task_id': task.get('id')}
        )
        
        result = await super().process_task(task)
        
        self.tracker.update_agent_focus(
            self.config.name,
            task.get('type', 'unknown'),
            'completed',
            100.0,
            {'task_id': task.get('id')}
        )
        
        return result

class TrackedCodeAnalystAgent(CodeAnalystAgent):
    """Analyst agent with real-time tracking"""
    
    def __init__(self, config, vector_db, tracker):
        super().__init__(config, vector_db)
        self.tracker = tracker
    
    async def process_task(self, task: Dict) -> Dict:
        self.tracker.update_agent_focus(
            self.config.name,
            task.get('type', 'unknown'),
            'code_evaluation',
            0.0,
            {'task_id': task.get('id')}
        )
        
        result = await super().process_task(task)
        
        self.tracker.update_agent_focus(
            self.config.name,
            task.get('type', 'unknown'),
            'completed',
            100.0,
            {'task_id': task.get('id')}
        )
        
        return result

class TrackedMultiAgentOrchestrator(MultiAgentOrchestrator):
    """Orchestrator with workflow tracking"""
    
    def __init__(self, agents, quantum_optimizer, tracker):
        super().__init__(agents, quantum_optimizer)
        self.tracker = tracker
    
    async def process_request(self, request: Dict) -> Dict:
        request_id = request.get('id', str(uuid.uuid4()))
        
        # Track request workflow
        self.tracker.add_workflow_node(WorkflowNode(
            id=request_id,
            type='request',
            label=f"Request: {request.get('type', 'unknown')}",
            status='active',
            start_time=time.time()
        ))
        
        result = await super().process_request(request)
        
        # Update completion
        self.tracker.update_task_status(request_id, 'completed')
        
        return result

# CLI Interface
class CLIInterface:
    """Command-line interface for the orchestrator"""
    
    def __init__(self, server: QuantumCodeOrchestratorServer):
        self.server = server
    
    def run_cli(self):
        """Run interactive CLI"""
        print("🌌 Quantum Code Orchestrator CLI")
        print("Commands: analyze <file>, status, agents, quit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                elif command[0] == "analyze" and len(command) > 1:
                    asyncio.run(self.analyze_file(command[1]))
                elif command[0] == "status":
                    self.show_status()
                elif command[0] == "agents":
                    self.show_agents()
                else:
                    print("Unknown command. Available: analyze <file>, status, agents, quit")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    async def analyze_file(self, file_path: str):
        """Analyze file via CLI"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            request = {
                'type': 'code_review',
                'code': content,
                'file_path': file_path,
                'id': f"cli_analysis_{uuid.uuid4().hex[:8]}"
            }
            
            result = await self.server.orchestrator.process_request(request)
            
            print(f"\n📊 Analysis Results for {file_path}:")
            print(f"Task Allocation: {result.get('task_allocation', {})}")
            
            if 'results' in result:
                synthesis = result['results']
                if synthesis.get('key_findings'):
                    print(f"\n🔍 Key Findings:")
                    for finding in synthesis['key_findings'][:3]:  # Top 3
                        print(f"  • {finding[:100]}...")
                        
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
    
    def show_status(self):
        """Show system status"""
        state = self.server.tracker.get_realtime_state()
        metrics = state.get('system_metrics', {})
        
        print("\n📈 System Status:")
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def show_agents(self):
        """Show agent status"""
        focuses = self.server.tracker.agent_focuses
        
        print("\n🤖 Agent Status:")
        for agent_id, focus in focuses.items():
            print(f"  {agent_id}: {focus.focus_area} ({focus.progress:.1f}%)")

# Main entry point
async def main():
    parser = argparse.ArgumentParser(description="Quantum Code Orchestrator")
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
    server = QuantumCodeOrchestratorServer(args.config)
    server.start_time = time.time()
    
    # Initialize system
    if not await server.initialize_system():
        print("❌ Failed to initialize system")
        return
    
    if args.cli:
        # Run CLI interface
        cli = CLIInterface(server)
        cli.run_cli()
    else:
        # Run web server
        print(f"🚀 Starting Quantum Code Orchestrator Server")
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