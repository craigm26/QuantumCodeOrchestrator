# QuantumCodeOrchestrator - Real-time Multi-Agent Web Server

A comprehensive web server that transforms your quantum-enhanced multi-agent VS Code extension into a full-featured local web server with real-time visualization and CLI capabilities.

## 🌟 Key Features

### Real-Time Web Dashboard
- **Live agent activity monitoring** with WebSocket updates
- **Interactive workflow visualization** showing task flows
- **Agent focus tracking** with progress indicators
- **System metrics dashboard** with performance data
- **Event logging** with real-time updates

### Advanced Flow Chart Generation
- **NetworkX-powered workflow graphs** showing agent relationships
- **Agent focus pie charts** displaying current activities
- **Status-based node coloring** (active, completed, failed)
- **Dynamic graph updates** as workflows progress

### File System Integration
- **Watchdog-based monitoring** of code file changes
- **Automatic analysis triggering** on file saves
- **Debounced processing** to prevent spam
- **Multi-language support** (.py, .js, .ts, .cpp, etc.)

### CLI Interface
```bash
# Interactive commands
> analyze myfile.py     # Analyze specific file
> status               # Show system metrics  
> agents               # Display agent status
> quit                 # Exit CLI
```

### WebSocket Real-Time Updates
- **Bi-directional communication** with web clients
- **Live agent progress tracking**
- **Instant workflow updates**
- **Connection management** with automatic cleanup

### Enhanced Agent Tracking
- **Focus area monitoring** (what each agent is working on)
- **Progress percentage tracking**
- **Task allocation visualization**
- **Performance metrics collection**

## 🛠️ Usage Examples

### Web Server Mode:
```bash
python quantum_code_orchestrator.py --port 8000
# Dashboard: http://localhost:8000
# API: http://localhost:8000/api/
# WebSocket: ws://localhost:8000/ws
```

### CLI Mode:
```bash
python quantum_code_orchestrator.py --cli
> analyze src/main.py
> status
> agents
```

### API Integration:
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

## 📊 Real-Time Visualizations

1. **Agent Status Panel** - Live progress bars and focus areas
2. **Workflow Graph** - Dynamic network diagram of task flows  
3. **System Metrics** - Performance indicators and statistics
4. **Event Stream** - Real-time activity log
5. **Focus Charts** - Agent activity distribution

## 🔧 Configuration

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

## 🏗️ Architecture Benefits

- **Unified Interface**: Single system for VS Code extension, web dashboard, and CLI
- **Real-Time Monitoring**: See exactly what agents are focusing on
- **Scalable Design**: WebSocket architecture supports multiple clients
- **Quantum Optimization**: Task allocation visible in real-time graphs
- **File System Integration**: Automatic analysis on code changes
- **RESTful API**: Easy integration with other tools

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -r python/requirements.txt

# Install web server dependencies
pip install -r requirements_web.txt
```

### 2. Start the Web Server
```bash
python quantum_code_orchestrator.py
```

### 3. Access the Dashboard
Open your browser to `http://localhost:8000`

### 4. Monitor Real-Time Activity
- Watch agents work in real-time
- See workflow graphs update dynamically
- Monitor file system changes
- Track system performance

## 📡 API Endpoints

### REST API
- `GET /health` - System health check
- `GET /api/state` - Get real-time system state
- `GET /api/agents` - Get agent status
- `GET /api/workflow/graph` - Get workflow graph image
- `GET /api/agents/focus-chart` - Get agent focus chart
- `POST /api/analyze` - Analyze code
- `POST /api/generate` - Generate features

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates

## 🔍 System Components

### RealTimeTracker
Tracks agent activities, workflow nodes, and system metrics in real-time.

### FileSystemMonitor
Monitors file system changes and automatically triggers analysis.

### FlowChartGenerator
Generates dynamic workflow graphs and agent focus charts.

### WebSocketManager
Manages real-time communication with web clients.

### CLIInterface
Provides interactive command-line interface.

## 📈 Performance Monitoring

The system provides comprehensive monitoring:
- **Agent Activity**: Real-time focus tracking
- **Task Progress**: Workflow completion status
- **System Metrics**: Performance indicators
- **Event Logging**: Activity history
- **Resource Usage**: Memory and CPU tracking

## 🔮 Advanced Features

### Quantum Optimization Visualization
- See quantum circuit execution in real-time
- Monitor quantum-classical hybrid optimization
- Track quantum resource usage

### Multi-Agent Coordination
- Visualize agent communication patterns
- Monitor task allocation decisions
- Track collaborative problem-solving

### File System Intelligence
- Automatic code analysis on changes
- Pattern recognition across files
- Dependency tracking and visualization

## 🛡️ Security Features

- **Local-only operation** by default
- **Configurable CORS** for web access
- **Input validation** on all endpoints
- **Rate limiting** for API calls
- **Secure WebSocket connections**

## 🔧 Development

### Running in Development Mode
```bash
# Enable debug logging
python quantum_code_orchestrator.py --host 0.0.0.0 --port 8000

# Monitor specific directories
# Edit config.json to add monitor_paths
```

### Customizing Visualizations
- Modify `FlowChartGenerator` for custom charts
- Extend `RealTimeTracker` for additional metrics
- Customize dashboard HTML in `get_dashboard_html()`

### Adding New Agents
1. Create agent class inheriting from base agent
2. Add tracking capabilities
3. Register in orchestrator
4. Update dashboard visualization

## 📊 Metrics and Analytics

The system collects comprehensive metrics:
- **Agent Performance**: Task completion rates
- **System Health**: Resource utilization
- **Workflow Efficiency**: Processing times
- **Quantum Usage**: Optimization effectiveness
- **File Analysis**: Code change patterns

## 🌐 Integration Examples

### VS Code Extension Integration
```typescript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateExtensionUI(data);
};
```

### External Tool Integration
```python
import requests

# Analyze code via API
response = requests.post('http://localhost:8000/api/analyze', json={
    'code': 'def hello(): pass',
    'file_path': 'test.py'
})
result = response.json()
```

## 🎯 Use Cases

### Code Review Automation
- Monitor code changes in real-time
- Automatic analysis on file saves
- Collaborative review workflows

### Development Team Coordination
- Visualize team member activities
- Track project progress
- Monitor code quality metrics

### Research and Development
- Study multi-agent behavior patterns
- Optimize quantum algorithms
- Analyze code intelligence systems

## 🔄 Continuous Improvement

The system supports:
- **Plugin architecture** for new features
- **Custom agent types** for specialized tasks
- **Extensible visualization** components
- **Configurable monitoring** parameters

---

**QuantumCodeOrchestrator Web Server** - Where quantum computing meets real-time multi-agent visualization and monitoring. 