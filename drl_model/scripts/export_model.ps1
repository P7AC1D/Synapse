# Export script for DRL Model
param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,
    
    [string]$OutputDir = "output",
    
    [switch]$RunTests
)

# Function to log messages with timestamp
function Write-Log {
    param([string]$Message)
    Write-Host "$(Get-Date -Format 'HH:mm:ss'): $Message"
}

# Function to check if a path exists and throw if not
function Assert-PathExists {
    param(
        [string]$Path,
        [string]$PathType,
        [string]$Message
    )
    
    if (-not (Test-Path $Path)) {
        Write-Log "Error: $PathType not found at: $Path"
        if ($Message) {
            Write-Log $Message
        }
        exit 1
    }
}

# Convert to absolute paths
try {
    $ModelPath = (Resolve-Path $ModelPath).Path
}
catch {
    Write-Log "Error: Invalid model path"
    exit 1
}

# Create output directory if it doesn't exist
$OutputDir = Join-Path (Get-Location) $OutputDir
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Verify model file exists
Assert-PathExists -Path $ModelPath -PathType "Model file" -Message "Please provide a valid model file path"

# Run model export
Write-Log "Exporting model..."
try {
    $ErrorActionPreference = "Stop"
    & python export_torchscript.py `
        --model-path "$ModelPath" `
        --output-dir "$OutputDir"
        
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Export failed! Check the error messages above."
        exit 1
    }
}
catch {
    Write-Log "Error running export: $_"
    exit 1
}

# Verify output files
Assert-PathExists -Path (Join-Path $OutputDir "model.pt") -PathType "TorchScript model" -Message "Export failed to create model.pt"
Assert-PathExists -Path (Join-Path $OutputDir "model_config.json") -PathType "Model config" -Message "Export failed to create model_config.json"

# Run tests if requested
if ($RunTests) {
    Write-Log "Running tests..."
    try {
        & .\run_tests.ps1 -ModelPath $ModelPath -OutputDir $OutputDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Tests failed! Check the error messages above."
            exit 1
        }
    }
    catch {
        Write-Log "Error running tests: $_"
        exit 1
    }
}

Write-Log "Export completed successfully!"

# Print next steps
Write-Host @"

Next steps:
1. Copy the exported files to MT5:
   - Copy $OutputDir\model.pt to C:\MT5\
   - Copy $OutputDir\model_config.json to C:\MT5\

2. Configure the EA:
   - Open DRLTrader.mq5 in MetaEditor
   - Verify model paths in EA inputs
   - Compile and attach to chart

3. Verify operation:
   - Check MT5 Experts log for initialization messages
   - Monitor feature values and predictions
   - Verify position management
"@
