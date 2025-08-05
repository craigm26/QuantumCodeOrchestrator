# Quantum Code Orchestrator

## 🚀 **Repository Name: `QuantumCodeOrchestrator`**

This comprehensive system transforms your quantum-enhanced multi-agent VS Code extension into a full-featured local web server with real-time visualization and CLI capabilities.

## **🌟 Key Features Added:**

### **Real-Time Web Dashboard**
- **Live agent activity monitoring** with WebSocket updates
- **Interactive workflow visualization** showing task flows
- **Agent focus tracking** with progress indicators  
- **System metrics dashboard** with performance data
- **Event logging** with real-time updates

### **Advanced Flow Chart Generation**
- **NetworkX-powered workflow graphs** showing agent relationships
- **Agent focus pie charts** displaying current activities
- **Status-based node coloring** (active, completed, failed)
- **Dynamic graph updates** as workflows progress

### **File System Integration**
- **Watchdog-based monitoring** of code file changes
- **Automatic analysis triggering** on file saves
- **Debounced processing** to prevent spam
- **Multi-language support** (.py, .js, .ts, .cpp, etc.)

### **CLI Interface**
```bash
# Interactive commands
> analyze myfile.py     # Analyze specific file
> status               # Show system metrics  
> agents               # Display agent status
> quit                 # Exit CLI
```

### **WebSocket Real-Time Updates**
- **Bi-directional communication** with web clients
- **Live agent progress tracking**
- **Instant workflow updates**
- **Connection management** with automatic cleanup

### **Enhanced Agent Tracking**
- **Focus area monitoring** (what each agent is working on)
- **Progress percentage tracking** 
- **Task allocation visualization**
- **Performance metrics collection**

## **🛠️ Usage Examples:**

### **Web Server Mode:**
```bash
python quantum_code_orchestrator.py --port 8000
# Dashboard: http://localhost:8000
# API: http://localhost:8000/api/
# WebSocket: ws://localhost:8000/ws
```

### **CLI Mode:**
```bash
python quantum_code_orchestrator.py --cli
> analyze src/main.py
> status
> agents
```

### **API Integration:**
```bash
# Analyze code via REST API
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): pass", "file_path": "test.py"}'

# Generate features
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"feature": "Add error handling", "context": "existing_code"}'
```

## **📊 Real-Time Visualizations:**

1. **Agent Status Panel** - Live progress bars and focus areas
2. **Workflow Graph** - Dynamic network diagram of task flows  
3. **System Metrics** - Performance indicators and statistics
4. **Event Stream** - Real-time activity log
5. **Focus Charts** - Agent activity distribution

## **🔧 Configuration:**

Create `config.json`:
```json
{
  "model_path": "/path/to/10b/model",
  "db_path": "./vector_db", 
  "monitor_paths": ["./src", "./lib"],
  "port": 8000,
  "host": "0.0.0.0",
  "log_level": "INFO"
}
```

## **🏗️ Architecture Benefits:**

- **Unified Interface**: Single system for VS Code extension, web dashboard, and CLI
- **Real-Time Monitoring**: See exactly what agents are focusing on
- **Scalable Design**: WebSocket architecture supports multiple clients
- **Quantum Optimization**: Task allocation visible in real-time graphs
- **File System Integration**: Automatic analysis on code changes
- **RESTful API**: Easy integration with other tools

The system creates a comprehensive development environment where you can monitor your quantum-enhanced AI agents in real-time, see their decision-making processes through interactive visualizations, and interact with them through multiple interfaces simultaneously.

Perfect for understanding how your multi-agent system operates and optimizing its performance! 🌌
