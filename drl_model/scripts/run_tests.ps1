# Run tests script for DRL Model
param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,
    
    [string]$OutputDir = "output",
    
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Release"
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
$ModelPath = (Resolve-Path $ModelPath).Path
$OutputDir = (Resolve-Path $OutputDir -ErrorAction SilentlyContinue).Path

# Verify model file exists
Assert-PathExists -Path $ModelPath -PathType "Model file" -Message "Please provide a valid model file path"

# Verify output directory exists
Assert-PathExists -Path $OutputDir -PathType "Output directory" -Message "Please run export_torchscript.py first"

# Verify required files
Assert-PathExists -Path (Join-Path $OutputDir "model.pt") -PathType "TorchScript model" -Message "Please run export_torchscript.py first"
Assert-PathExists -Path (Join-Path $OutputDir "model_config.json") -PathType "Model config" -Message "Please run export_torchscript.py first"

# Run model export
Write-Log "Running model comparison test..."
try {
    $ErrorActionPreference = "Stop"
    & python test_model.py `
        --original-model "$ModelPath" `
        --torchscript-model "$OutputDir\model.pt" `
        --config "$OutputDir\model_config.json"
        
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Test failed! Check the error messages above."
        exit 1
    }
    
    Write-Log "Test completed successfully!"
}
catch {
    Write-Log "Error running test: $_"
    exit 1
}
