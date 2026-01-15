# VeZ One-Liner Installer for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "╔═══════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   VeZ Programming Language Installer  ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Detect architecture
$Arch = [System.Environment]::Is64BitOperatingSystem
if (-not $Arch) {
    Write-Host "Error: VeZ requires a 64-bit Windows system" -ForegroundColor Red
    exit 1
}

Write-Host "Detected OS: Windows x64" -ForegroundColor Yellow
Write-Host ""

# Determine installation directory
$InstallDir = "$env:LOCALAPPDATA\VeZ\bin"
$Target = "x86_64-pc-windows-msvc"

# Get latest release version
Write-Host "Fetching latest release..." -ForegroundColor Yellow
try {
    $LatestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest"
    $LatestVersion = $LatestRelease.tag_name
} catch {
    Write-Host "Error: Could not determine latest version" -ForegroundColor Red
    exit 1
}

Write-Host "Latest version: $LatestVersion" -ForegroundColor Green
Write-Host ""

# Construct download URL
$ArchiveName = "vez-$LatestVersion-$Target.zip"
$DownloadUrl = "https://github.com/YOUR_USERNAME/VeZ/releases/download/$LatestVersion/$ArchiveName"

Write-Host "Downloading VeZ..." -ForegroundColor Yellow
Write-Host "URL: $DownloadUrl" -ForegroundColor Yellow

# Create temporary directory
$TmpDir = New-Item -ItemType Directory -Path "$env:TEMP\vez-install-$(Get-Random)"

try {
    # Download archive
    $ArchivePath = Join-Path $TmpDir $ArchiveName
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath -UseBasicParsing
    
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host ""
    
    # Extract archive
    Write-Host "Extracting archive..." -ForegroundColor Yellow
    Expand-Archive -Path $ArchivePath -DestinationPath $TmpDir -Force
    
    # Create installation directory
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    
    Write-Host "Installing VeZ to $InstallDir..." -ForegroundColor Yellow
    
    # Install binaries
    $BinDir = Join-Path $TmpDir $Target
    
    if (Test-Path "$BinDir\vezc.exe") {
        Copy-Item "$BinDir\vezc.exe" -Destination "$InstallDir\vezc.exe" -Force
        Write-Host "✓ Installed vezc.exe (compiler)" -ForegroundColor Green
    }
    
    if (Test-Path "$BinDir\vpm.exe") {
        Copy-Item "$BinDir\vpm.exe" -Destination "$InstallDir\vpm.exe" -Force
        Write-Host "✓ Installed vpm.exe (package manager)" -ForegroundColor Green
    }
    
    if (Test-Path "$BinDir\vez-lsp.exe") {
        Copy-Item "$BinDir\vez-lsp.exe" -Destination "$InstallDir\vez-lsp.exe" -Force
        Write-Host "✓ Installed vez-lsp.exe (language server)" -ForegroundColor Green
    }
    
    # Add to PATH if not already present
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$InstallDir*") {
        Write-Host ""
        Write-Host "Adding VeZ to PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
        $env:Path = "$env:Path;$InstallDir"
        Write-Host "✓ Added to PATH" -ForegroundColor Green
    }
    
} finally {
    # Cleanup
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   VeZ Installation Complete! 🎉       ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Installed binaries:" -ForegroundColor Yellow
Write-Host "  • vezc.exe     - VeZ compiler"
Write-Host "  • vpm.exe      - VeZ package manager"
Write-Host "  • vez-lsp.exe  - VeZ language server"
Write-Host ""
Write-Host "Installation directory: $InstallDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "IMPORTANT: Restart your terminal for PATH changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Verify installation:" -ForegroundColor Yellow
Write-Host "  vezc --version"
Write-Host ""
Write-Host "Get started:" -ForegroundColor Yellow
Write-Host "  vezc --help"
Write-Host "  vpm --help"
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "  https://github.com/YOUR_USERNAME/VeZ"
Write-Host ""
Write-Host "Happy coding with VeZ! 🚀" -ForegroundColor Green
