import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('VeZ extension activated');

    const config = vscode.workspace.getConfiguration('vez');
    
    if (config.get<boolean>('enableLsp', true)) {
        startLanguageServer(context);
    }

    registerCommands(context);
    registerEventListeners(context);
}

function startLanguageServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('vez');
    const lspPath = config.get<string>('lspPath', 'vez-lsp');

    const serverOptions: ServerOptions = {
        command: lspPath,
        transport: TransportKind.stdio,
        args: ['--stdio']
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'vez' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.zari')
        }
    };

    client = new LanguageClient(
        'vez',
        'VeZ Language Server',
        serverOptions,
        clientOptions
    );

    client.start().catch(err => {
        vscode.window.showErrorMessage(`Failed to start VeZ language server: ${err}`);
    });
}

function registerCommands(context: vscode.ExtensionContext) {
    const compileCmd = vscode.commands.registerCommand('vez.compile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const config = vscode.workspace.getConfiguration('vez');
        const compilerPath = config.get<string>('compilerPath', 'vezc');
        const optLevel = config.get<string>('optimizationLevel', 'O2');

        const terminal = vscode.window.createTerminal('VeZ Compile');
        terminal.sendText(`${compilerPath} "${editor.document.fileName}" -o "${getOutputPath(editor.document.fileName)}" -${optLevel}`);
        terminal.show();
    });

    const runCmd = vscode.commands.registerCommand('vez.run', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const config = vscode.workspace.getConfiguration('vez');
        const compilerPath = config.get<string>('compilerPath', 'vezc');
        const outputPath = getOutputPath(editor.document.fileName);

        const terminal = vscode.window.createTerminal('VeZ Run');
        terminal.sendText(`${compilerPath} "${editor.document.fileName}" -o "${outputPath}" && "${outputPath}"`);
        terminal.show();
    });

    const newProjectCmd = vscode.commands.registerCommand('vez.newProject', async () => {
        const folderUri = await vscode.window.showOpenDialog({
            canSelectFiles: false,
            canSelectFolders: true,
            canSelectMany: false,
            openLabel: 'Select folder for new project'
        });

        if (!folderUri || folderUri.length === 0) {
            return;
        }

        const projectName = await vscode.window.showInputBox({
            prompt: 'Project name',
            placeHolder: 'my-project'
        });

        if (!projectName) {
            return;
        }

        const config = vscode.workspace.getConfiguration('vez');
        const vpmPath = config.get<string>('packageManagerPath', 'vpm');
        const projectPath = vscode.Uri.joinPath(folderUri[0], projectName);

        const terminal = vscode.window.createTerminal('VeZ New Project');
        terminal.sendText(`cd "${folderUri[0].fsPath}" && ${vpmPath} new ${projectName}`);
        terminal.show();

        vscode.commands.executeCommand('vscode.openFolder', projectPath);
    });

    const showAstCmd = vscode.commands.registerCommand('vez.showAst', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const config = vscode.workspace.getConfiguration('vez');
        const compilerPath = config.get<string>('compilerPath', 'vezc');

        const terminal = vscode.window.createTerminal('VeZ AST');
        terminal.sendText(`${compilerPath} "${editor.document.fileName}" --emit=ast`);
        terminal.show();
    });

    const formatCmd = vscode.commands.registerCommand('vez.format', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const text = document.getText();
        const formatted = formatVeZCode(text);

        const edit = new vscode.WorkspaceEdit();
        const fullRange = new vscode.Range(
            document.positionAt(0),
            document.positionAt(text.length)
        );
        edit.replace(document.uri, fullRange, formatted);
        await vscode.workspace.applyEdit(edit);
    });

    const restartLspCmd = vscode.commands.registerCommand('vez.restartLsp', async () => {
        if (client) {
            await client.stop();
            client.start();
            vscode.window.showInformationMessage('VeZ language server restarted');
        }
    });

    const aiContextCmd = vscode.commands.registerCommand('vez.aiContext', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const config = vscode.workspace.getConfiguration('vez');
        if (!config.get<boolean>('aiContextEnabled', true)) {
            vscode.window.showErrorMessage('AI context is disabled in settings');
            return;
        }

        const position = editor.selection.active;
        const document = editor.document;
        const text = document.getText();
        const offset = document.offsetAt(position);

        const lines = text.substring(0, offset).split('\n');
        const line = lines.length;
        const column = lines[lines.length - 1].length;

        const context = extractAIContext(text, line, column);
        
        const panel = vscode.window.createWebviewPanel(
            'vezAicontext',
            'VeZ AI Context',
            vscode.ViewColumn.Beside,
            {}
        );
        
        panel.webview.html = getAIContextWebviewContent(context);
    });

    context.subscriptions.push(
        compileCmd, runCmd, newProjectCmd, showAstCmd, 
        formatCmd, restartLspCmd, aiContextCmd
    );
}

function registerEventListeners(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('vez');

    if (config.get<boolean>('formatOnSave', true)) {
        context.subscriptions.push(
            vscode.workspace.onWillSaveTextDocument(event => {
                if (event.document.languageId === 'vez') {
                    event.waitUntil(
                        vscode.commands.executeCommand('vez.format')
                    );
                }
            })
        );
    }
}

function getOutputPath(sourcePath: string): string {
    const ext = process.platform === 'win32' ? '.exe' : '';
    return sourcePath.replace(/\.[^.]+$/, ext);
}

function formatVeZCode(code: string): string {
    let result = code;
    result = result.replace(/\s+$/gm, '');
    result = result.replace(/\n{3,}/g, '\n\n');
    let lines = result.split('\n');
    lines = lines.map(line => {
        const trimmed = line.trimEnd();
        return trimmed;
    });
    return lines.join('\n');
}

interface AIContext {
    functions: string[];
    types: string[];
    imports: string[];
    variables: string[];
}

function extractAIContext(code: string, line: number, column: number): AIContext {
    const context: AIContext = {
        functions: [],
        types: [],
        imports: [],
        variables: []
    };

    const lines = code.split('\n').slice(0, line);
    
    for (const l of lines) {
        const fnMatch = l.match(/(?:fn|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (fnMatch) {
            context.functions.push(fnMatch[1]);
        }

        const structMatch = l.match(/struct\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (structMatch) {
            context.types.push(structMatch[1]);
        }

        const enumMatch = l.match(/enum\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (enumMatch) {
            context.types.push(enumMatch[1]);
        }

        const importMatch = l.match(/(?:use|import)\s+([a-zA-Z_:][a-zA-Z0-9_:]*)/);
        if (importMatch) {
            context.imports.push(importMatch[1]);
        }

        const varMatch = l.match(/(?:let|var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (varMatch) {
            context.variables.push(varMatch[1]);
        }
    }

    return context;
}

function getAIContextWebviewContent(context: AIContext): string {
    return `<!DOCTYPE html>
<html>
<head>
    <title>VeZ AI Context</title>
    <style>
        body { font-family: -apple-system, sans-serif; padding: 20px; }
        h2 { color: #e06c75; }
        ul { list-style-type: none; padding: 0; }
        li { padding: 4px 0; font-family: monospace; }
        code { background: #282c34; padding: 2px 6px; border-radius: 3px; color: #61afef; }
    </style>
</head>
<body>
    <h1>🤖 AI Context Extraction</h1>
    
    <h2>📦 Imports (${context.imports.length})</h2>
    <ul>${context.imports.map(i => `<li><code>${i}</code></li>`).join('')}</ul>
    
    <h2>🔧 Functions (${context.functions.length})</h2>
    <ul>${context.functions.map(f => `<li><code>${f}</code></li>`).join('')}</ul>
    
    <h2>📐 Types (${context.types.length})</h2>
    <ul>${context.types.map(t => `<li><code>${t}</code></li>`).join('')}</ul>
    
    <h2>📝 Variables (${context.variables.length})</h2>
    <ul>${context.variables.map(v => `<li><code>${v}</code></li>`).join('')}</ul>
</body>
</html>`;
}

export function deactivate(): Thenable<void> | undefined {
    if (client) {
        return client.stop();
    }
    return undefined;
}
