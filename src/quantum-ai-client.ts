import * as vscode from 'vscode';
import * as axios from 'axios';

export class QuantumAIClient {
    private context: vscode.ExtensionContext;
    private baseUrl: string = 'http://localhost:8000'; // Python backend API
    private isConnected: boolean = false;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.initializeConnection();
    }

    private async initializeConnection() {
        try {
            // Wait for Python backend to start
            await this.waitForBackend();
            this.isConnected = true;
            console.log('Connected to Quantum AI backend');
        } catch (error) {
            console.error('Failed to connect to Quantum AI backend:', error);
            this.isConnected = false;
        }
    }

    private async waitForBackend(timeout: number = 30000): Promise<void> {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            try {
                await axios.default.get(`${this.baseUrl}/health`);
                return;
            } catch (error) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        throw new Error('Backend connection timeout');
    }

    async analyzeCode(filePath: string, code: string): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.post(`${this.baseUrl}/analyze`, {
                file_path: filePath,
                code: code,
                type: 'code_review'
            });

            return response.data;
        } catch (error) {
            throw new Error(`Analysis failed: ${error}`);
        }
    }

    async generateFeature(featureDescription: string, contextCode: string): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.post(`${this.baseUrl}/generate`, {
                feature: featureDescription,
                context: contextCode,
                type: 'feature_implementation'
            });

            return response.data;
        } catch (error) {
            throw new Error(`Feature generation failed: ${error}`);
        }
    }

    async optimizeCode(filePath: string, code: string): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.post(`${this.baseUrl}/optimize`, {
                file_path: filePath,
                code: code,
                type: 'performance_optimization'
            });

            return response.data;
        } catch (error) {
            throw new Error(`Optimization failed: ${error}`);
        }
    }

    async setupEnvironment(): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.post(`${this.baseUrl}/setup`);
            return response.data;
        } catch (error) {
            throw new Error(`Environment setup failed: ${error}`);
        }
    }

    async getAgentStatus(): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.get(`${this.baseUrl}/status`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get agent status: ${error}`);
        }
    }

    async getQuantumStatus(): Promise<any> {
        if (!this.isConnected) {
            throw new Error('Not connected to Quantum AI backend');
        }

        try {
            const response = await axios.default.get(`${this.baseUrl}/quantum-status`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get quantum status: ${error}`);
        }
    }

    isBackendConnected(): boolean {
        return this.isConnected;
    }
} 