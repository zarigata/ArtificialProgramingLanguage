# VeZ VS Code Extension

Language support for the VeZ programming language in Visual Studio Code.

## Features

- **Syntax Highlighting**: Full support for VeZ syntax
- **IntelliSense**: Auto-completion for functions, types, and variables
- **Go to Definition**: Navigate to symbol definitions
- **Find References**: Find all usages of a symbol
- **Hover Information**: Show type info and documentation
- **Error Diagnostics**: Real-time error checking
- **Code Formatting**: Format VeZ code
- **Snippets**: Common code patterns
- **AI Context Extraction**: Extract semantic context for AI code generation

## Installation

### From VSIX

```bash
code --install-extension vez-x.x.x.vsix
```

### From Source

```bash
cd tools/vscode
npm install
npm run compile
npm run package
code --install-extension vez-x.x.x.vsix
```

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| `VeZ: Compile Current File` | `Ctrl+Shift+B` | Compile the current file |
| `VeZ: Run Current File` | `F5` | Compile and run the current file |
| `VeZ: Format Document` | `Shift+Alt+F` | Format the current document |
| `VeZ: Show AST` | - | Display the AST for the current file |
| `VeZ: Create New Project` | - | Create a new VeZ project |
| `VeZ: Restart Language Server` | - | Restart the LSP server |
| `VeZ: Extract AI Context` | - | Extract context for AI code generation |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `vez.compilerPath` | `vezc` | Path to the VeZ compiler |
| `vez.lspPath` | `vez-lsp` | Path to the VeZ language server |
| `vez.packageManagerPath` | `vpm` | Path to the VeZ package manager |
| `vez.enableLsp` | `true` | Enable the language server |
| `vez.formatOnSave` | `true` | Format files on save |
| `vez.showInlineErrors` | `true` | Show inline error messages |
| `vez.aiContextEnabled` | `true` | Enable AI context features |
| `vez.optimizationLevel` | `O2` | Default optimization level |
| `vez.target` | `native` | Default compilation target |

## Snippets

Type any of the following prefixes and press Tab to expand:

- `fn` - Function definition
- `pub fn` - Public function
- `async fn` - Async function
- `@gpu` - GPU kernel
- `struct` - Struct definition
- `enum` - Enum definition
- `trait` - Trait definition
- `impl` - Trait implementation
- `match` - Match expression
- `for` - For loop
- `while` - While loop
- `test` - Test function
- `main` - Entry point

## Requirements

- VS Code 1.80+
- VeZ compiler (`vezc`) in PATH
- VeZ language server (`vez-lsp`) for full features

## License

MIT
