import * as vscode from 'vscode';
import { QuantumAIClient } from './quantum-ai-client';

export class AgentStatusProvider implements vscode.TreeDataProvider<AgentStatusItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentStatusItem | undefined | null | void> = new vscode.EventEmitter<AgentStatusItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private quantumAIClient: QuantumAIClient) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: AgentStatusItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: AgentStatusItem): Promise<AgentStatusItem[]> {
        if (element) {
            return element.getChildren();
        }

        try {
            const status = await this.quantumAIClient.getAgentStatus();
            const agents = status.agents || {};

            return Object.entries(agents).map(([name, agent]: [string, any]) => {
                return new AgentStatusItem(
                    name,
                    agent.online ? '🟢 Online' : '🔴 Offline',
                    agent.online ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None,
                    agent
                );
            });
        } catch (error) {
            return [
                new AgentStatusItem(
                    'Connection Error',
                    '🔴 Backend not available',
                    vscode.TreeItemCollapsibleState.None
                )
            ];
        }
    }
}

export class AgentStatusItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly status: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly agentData?: any
    ) {
        super(label, collapsibleState);

        this.tooltip = `${this.label} - ${this.status}`;
        this.description = this.status;

        if (this.agentData) {
            this.contextValue = 'agent';
        }
    }

    async getChildren(): Promise<AgentStatusItem[]> {
        if (!this.agentData) {
            return [];
        }

        return [
            new AgentStatusItem(
                'Model',
                this.agentData.model || 'Not loaded',
                vscode.TreeItemCollapsibleState.None
            ),
            new AgentStatusItem(
                'Tasks Completed',
                this.agentData.tasks_completed?.toString() || '0',
                vscode.TreeItemCollapsibleState.None
            ),
            new AgentStatusItem(
                'Last Activity',
                this.agentData.last_activity || 'Unknown',
                vscode.TreeItemCollapsibleState.None
            )
        ];
    }
} 