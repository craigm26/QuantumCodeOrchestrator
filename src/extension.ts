import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { QuantumAIClient } from './quantum-ai-client';
import { AgentStatusProvider } from './agent-status-provider';
import { QuantumStatusProvider } from './quantum-status-provider';

let quantumAIClient: QuantumAIClient;
let agentStatusProvider: AgentStatusProvider;
let quantumStatusProvider: QuantumStatusProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('Quantum AI Assistant extension is now active!');

    // Initialize the Quantum AI client
    quantumAIClient = new QuantumAIClient(context);

    // Register status providers
    agentStatusProvider = new AgentStatusProvider(quantumAIClient);
    quantumStatusProvider = new QuantumStatusProvider(quantumAIClient);

    context.subscriptions.push(
        vscode.window.registerTreeDataProvider('quantumAI.agentStatus', agentStatusProvider),
        vscode.window.registerTreeDataProvider('quantumAI.quantumStatus', quantumStatusProvider)
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('quantumAI.analyzeCode', analyzeCode),
        vscode.commands.registerCommand('quantumAI.generateFeature', generateFeature),
        vscode.commands.registerCommand('quantumAI.optimizeCode', optimizeCode),
        vscode.commands.registerCommand('quantumAI.setupQuantumEnvironment', setupQuantumEnvironment),
        vscode.commands.registerCommand('quantumAI.viewAgentStatus', viewAgentStatus)
    );

    // Initialize the Python backend
    initializePythonBackend(context);
}

async function analyzeCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const document = editor.document;
    const selection = editor.selection;
    const code = selection.isEmpty ? document.getText() : document.getText(selection);

    try {
        vscode.window.showInformationMessage('🔬 Analyzing code with Quantum AI...');
        
        const result = await quantumAIClient.analyzeCode(document.fileName, code);
        
        // Display results in a new webview
        const panel = vscode.window.createWebviewPanel(
            'quantumAnalysis',
            'Quantum AI Analysis Results',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = generateAnalysisHTML(result);
        
        // Update status
        agentStatusProvider.refresh();
        quantumStatusProvider.refresh();

    } catch (error) {
        vscode.window.showErrorMessage(`Analysis failed: ${error}`);
    }
}

async function generateFeature() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const featureDescription = await vscode.window.showInputBox({
        prompt: 'Describe the feature you want to generate:',
        placeHolder: 'e.g., Add error handling to this function'
    });

    if (!featureDescription) {
        return;
    }

    const document = editor.document;
    const contextCode = document.getText();

    try {
        vscode.window.showInformationMessage('⚡ Generating feature with AI...');
        
        const result = await quantumAIClient.generateFeature(featureDescription, contextCode);
        
        if (result.generated_code) {
            // Insert the generated code at cursor position
            const position = editor.selection.active;
            await editor.edit(editBuilder => {
                editBuilder.insert(position, '\n' + result.generated_code);
            });
            
            vscode.window.showInformationMessage('Feature generated and inserted!');
        }

    } catch (error) {
        vscode.window.showErrorMessage(`Feature generation failed: ${error}`);
    }
}

async function optimizeCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const document = editor.document;
    const code = document.getText();

    try {
        vscode.window.showInformationMessage('🚀 Optimizing code performance...');
        
        const result = await quantumAIClient.optimizeCode(document.fileName, code);
        
        // Show optimization suggestions
        const panel = vscode.window.createWebviewPanel(
            'quantumOptimization',
            'Quantum AI Optimization Results',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = generateOptimizationHTML(result);

    } catch (error) {
        vscode.window.showErrorMessage(`Optimization failed: ${error}`);
    }
}

async function setupQuantumEnvironment() {
    try {
        vscode.window.showInformationMessage('🌌 Setting up Quantum Environment...');
        
        const result = await quantumAIClient.setupEnvironment();
        
        if (result.success) {
            vscode.window.showInformationMessage('Quantum environment setup complete!');
            quantumStatusProvider.refresh();
        } else {
            vscode.window.showWarningMessage('Quantum environment setup completed with warnings');
        }

    } catch (error) {
        vscode.window.showErrorMessage(`Environment setup failed: ${error}`);
    }
}

async function viewAgentStatus() {
    try {
        const status = await quantumAIClient.getAgentStatus();
        
        const panel = vscode.window.createWebviewPanel(
            'agentStatus',
            'Quantum AI Agent Status',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = generateStatusHTML(status);

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to get agent status: ${error}`);
    }
}

function initializePythonBackend(context: vscode.ExtensionContext) {
    const pythonPath = getPythonPath();
    const backendPath = path.join(context.extensionPath, 'python', 'quantum_ai_backend.py');
    
    if (!fs.existsSync(backendPath)) {
        vscode.window.showWarningMessage('Python backend not found. Please ensure the backend is properly installed.');
        return;
    }

    // Start the Python backend process
    const pythonProcess = spawn(pythonPath, [backendPath], {
        stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log('Python Backend:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error('Python Backend Error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
        console.log('Python Backend closed with code:', code);
    });

    // Store the process for cleanup
    context.subscriptions.push({
        dispose: () => {
            pythonProcess.kill();
        }
    });
}

function getPythonPath(): string {
    // Try to get Python path from VS Code settings or environment
    const config = vscode.workspace.getConfiguration('python');
    return config.get('pythonPath', 'python') || 'python';
}

function generateAnalysisHTML(result: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Quantum AI Analysis Results</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #e1e5e9; border-radius: 6px; }
                .agent-contribution { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 4px; }
                .recommendation { background: #e3f2fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196f3; }
                .finding { background: #fff3e0; padding: 10px; margin: 5px 0; border-left: 4px solid #ff9800; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Quantum AI Code Analysis</h1>
                <p>Multi-agent analysis completed with quantum optimization</p>
            </div>
            
            <div class="section">
                <h2>📊 Task Allocation</h2>
                <p><strong>Quantum-optimized agent distribution:</strong></p>
                <ul>
                    ${Object.entries(result.task_allocation || {}).map(([task, agent]) => 
                        `<li><strong>${task}:</strong> ${agent}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="section">
                <h2>👥 Agent Contributions</h2>
                ${Object.entries(result.agent_contributions || {}).map(([agent, count]) => 
                    `<div class="agent-contribution">
                        <strong>${agent}:</strong> ${count} tasks completed
                    </div>`
                ).join('')}
            </div>
            
            <div class="section">
                <h2>🔍 Key Findings</h2>
                ${(result.results?.key_findings || []).map(finding => 
                    `<div class="finding">${finding}</div>`
                ).join('')}
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                ${(result.results?.recommendations || []).map(rec => 
                    `<div class="recommendation">${rec}</div>`
                ).join('')}
            </div>
            
            ${result.results?.generated_code ? `
            <div class="section">
                <h2>⚡ Generated Code</h2>
                <pre><code>${result.results.generated_code}</code></pre>
            </div>
            ` : ''}
        </body>
        </html>
    `;
}

function generateOptimizationHTML(result: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Quantum AI Optimization Results</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
                .header { background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #e1e5e9; border-radius: 6px; }
                .optimization { background: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #4caf50; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Quantum AI Code Optimization</h1>
                <p>Performance optimization with quantum-classical hybrid algorithms</p>
            </div>
            
            <div class="section">
                <h2>📈 Optimization Results</h2>
                ${result.optimizations ? result.optimizations.map(opt => 
                    `<div class="optimization">${opt}</div>`
                ).join('') : '<p>No optimizations found</p>'}
            </div>
        </body>
        </html>
    `;
}

function generateStatusHTML(status: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Quantum AI Agent Status</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
                .header { background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #e1e5e9; border-radius: 6px; }
                .agent { background: #f3e5f5; padding: 15px; margin: 10px 0; border-radius: 6px; }
                .status-online { color: #4caf50; }
                .status-offline { color: #f44336; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>👥 Quantum AI Agent Status</h1>
                <p>Real-time status of multi-agent system</p>
            </div>
            
            <div class="section">
                <h2>🤖 Agent Status</h2>
                ${Object.entries(status.agents || {}).map(([name, agent]: [string, any]) => 
                    `<div class="agent">
                        <h3>${name}</h3>
                        <p><strong>Status:</strong> <span class="status-${agent.online ? 'online' : 'offline'}">${agent.online ? '🟢 Online' : '🔴 Offline'}</span></p>
                        <p><strong>Model:</strong> ${agent.model || 'Not loaded'}</p>
                        <p><strong>Tasks Completed:</strong> ${agent.tasks_completed || 0}</p>
                    </div>`
                ).join('')}
            </div>
            
            <div class="section">
                <h2>🌌 Quantum Status</h2>
                <p><strong>Quantum Optimization:</strong> <span class="status-${status.quantum_enabled ? 'online' : 'offline'}">${status.quantum_enabled ? '🟢 Enabled' : '🔴 Disabled'}</span></p>
                <p><strong>Quantum Backend:</strong> ${status.quantum_backend || 'Not available'}</p>
            </div>
        </body>
        </html>
    `;
}

export function deactivate() {
    console.log('Quantum AI Assistant extension deactivated');
} 