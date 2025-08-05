import * as vscode from 'vscode';
import { QuantumAIClient } from './quantum-ai-client';

export class QuantumStatusProvider implements vscode.TreeDataProvider<QuantumStatusItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<QuantumStatusItem | undefined | null | void> = new vscode.EventEmitter<QuantumStatusItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private quantumAIClient: QuantumAIClient) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: QuantumStatusItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: QuantumStatusItem): Promise<QuantumStatusItem[]> {
        if (element) {
            return element.getChildren();
        }

        try {
            const status = await this.quantumAIClient.getQuantumStatus();
            
            return [
                new QuantumStatusItem(
                    'Quantum Optimization',
                    status.quantum_enabled ? '🟢 Enabled' : '🔴 Disabled',
                    vscode.TreeItemCollapsibleState.None
                ),
                new QuantumStatusItem(
                    'Quantum Backend',
                    status.quantum_backend || 'Not available',
                    vscode.TreeItemCollapsibleState.None
                ),
                new QuantumStatusItem(
                    'Qubits Available',
                    status.qubits_available?.toString() || '0',
                    vscode.TreeItemCollapsibleState.None
                ),
                new QuantumStatusItem(
                    'Optimization Tasks',
                    status.optimization_tasks?.toString() || '0',
                    vscode.TreeItemCollapsibleState.None
                )
            ];
        } catch (error) {
            return [
                new QuantumStatusItem(
                    'Connection Error',
                    '🔴 Backend not available',
                    vscode.TreeItemCollapsibleState.None
                )
            ];
        }
    }
}

export class QuantumStatusItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly status: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);

        this.tooltip = `${this.label} - ${this.status}`;
        this.description = this.status;
    }

    async getChildren(): Promise<QuantumStatusItem[]> {
        return [];
    }
} 