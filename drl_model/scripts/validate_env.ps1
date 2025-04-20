# Validate development environment for DRL Model
param(
    [string]$LibTorchPath = $env:LIBTORCH_PATH,
    [string]$VcpkgPath = $env:VCPKG_PATH,
    [string]$MT5Path = $env:MT5_PATH,
    [switch]$Fix
)

# Function to log messages with timestamp
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("Info", "Warning", "Error")]
        [string]$Level = "Info"
    )
    
    $color = switch ($Level) {
        "Info" { "White" }
        "Warning" { "Yellow" }
        "Error" { "Red" }
    }
    
    Write-Host "$(Get-Date -Format 'HH:mm:ss'): " -NoNewline
    Write-Host $Message -ForegroundColor $color
}

# Function to check if a command exists
function Test-Command {
    param([string]$Name)
    
    # Try standard command check first
    if ($null -ne (Get-Command $Name -ErrorAction SilentlyContinue)) {
        return $true
    }
    
    # For CMake, try additional checks since it might have been just installed
    if ($Name -eq "cmake") {
        $cmakePaths = @(
            "C:\Program Files\CMake\bin\cmake.exe",
            "C:\Program Files (x86)\CMake\bin\cmake.exe"
        )
        
        foreach ($path in $cmakePaths) {
            if (Test-Path $path) {
                # Add to current session PATH if found
                $env:PATH += ";$(Split-Path $path -Parent)"
                return $true
            }
        }
    }
    
    return $false
}

# Function to check Python package
function Test-PythonPackage {
    param([string]$Package)
    
    $result = python -c "import $Package" 2>$null
    return $LASTEXITCODE -eq 0
}

# Function to install a Python package
function Install-PythonPackage {
    param([string]$Package)
    
    Write-Log "Installing $Package..."
    pip install $Package
}

# Check Python installation
Write-Log "Checking Python installation..."
if (-not (Test-Command "python")) {
    Write-Log "Python not found!" "Error"
    if ($Fix) {
        Write-Log "Please install Python 3.8 or later from https://www.python.org/"
    }
    exit 1
}

$pythonVersion = (python --version) -replace "Python "
Write-Log "Found Python $pythonVersion"

# Check pip installation
Write-Log "Checking pip installation..."
if (-not (Test-Command "pip")) {
    Write-Log "pip not found!" "Error"
    if ($Fix) {
        Write-Log "Installing pip..."
        python -m ensurepip --upgrade
    }
    else {
        exit 1
    }
}

# Check required Python packages
$packages = @(
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "numpy",
    "pandas"
)

foreach ($package in $packages) {
    Write-Log "Checking $package..."
    if (-not (Test-PythonPackage $package)) {
        Write-Log "$package not found!" "Warning"
        if ($Fix) {
            Install-PythonPackage $package
        }
        else {
            Write-Log "Run with -Fix to install missing packages" "Warning"
        }
    }
}

# Check Visual Studio installation
Write-Log "Checking Visual Studio installation..."
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Log "Found Visual Studio at: $vsPath"
    }
    else {
        Write-Log "Visual Studio not found!" "Error"
        if ($Fix) {
            Write-Log "Please install Visual Studio with C++ workload"
            Start-Process "https://visualstudio.microsoft.com/downloads/"
        }
        exit 1
    }
}

# Check CMake installation
Write-Log "Checking CMake installation..."
if (-not (Test-Command "cmake")) {
    Write-Log "CMake not found!" "Error"
    if ($Fix) {
        Write-Log "Please install CMake from https://cmake.org/download/"
    }
    exit 1
}

# Check LibTorch path
Write-Log "Checking LibTorch installation..."
if (-not $LibTorchPath) {
    Write-Log "LIBTORCH_PATH not set!" "Warning"
    Write-Log "Please set LIBTORCH_PATH environment variable or provide -LibTorchPath parameter"
}
elseif (-not (Test-Path $LibTorchPath)) {
    Write-Log "LibTorch not found at: $LibTorchPath" "Error"
    if ($Fix) {
        Write-Log "Please download LibTorch from https://pytorch.org/"
    }
}
else {
    Write-Log "Found LibTorch at: $LibTorchPath"
}

# Check vcpkg installation
Write-Log "Checking vcpkg installation..."
if (-not $VcpkgPath) {
    Write-Log "VCPKG_PATH not set!" "Warning"
    Write-Log "Please set VCPKG_PATH environment variable or provide -VcpkgPath parameter"
}
elseif (-not (Test-Path $VcpkgPath)) {
    Write-Log "vcpkg not found at: $VcpkgPath" "Error"
    if ($Fix) {
        Write-Log "Please clone vcpkg from https://github.com/Microsoft/vcpkg"
    }
}
else {
    Write-Log "Found vcpkg at: $VcpkgPath"
}

# Check MT5 installation
Write-Log "Checking MT5 installation..."
if (-not $MT5Path) {
    Write-Log "MT5_PATH not set!" "Warning"
    Write-Log "Please set MT5_PATH environment variable or provide -MT5Path parameter"
}
elseif (-not (Test-Path $MT5Path)) {
    Write-Log "MT5 not found at: $MT5Path" "Error"
    if ($Fix) {
        Write-Log "Please install MetaTrader 5"
    }
}
else {
    Write-Log "Found MT5 at: $MT5Path"
}

# Print summary
Write-Host "`nEnvironment Status:"
Write-Host "==================="
Write-Host "Python: $pythonVersion"
Write-Host "CMake: $((cmake --version).Split("`n")[0])"
Write-Host "LibTorch: $(if ($LibTorchPath -and (Test-Path $LibTorchPath)) { 'Found' } else { 'Missing' })"
Write-Host "vcpkg: $(if ($VcpkgPath -and (Test-Path $VcpkgPath)) { 'Found' } else { 'Missing' })"
Write-Host "MT5: $(if ($MT5Path -and (Test-Path $MT5Path)) { 'Found' } else { 'Missing' })"
Write-Host "Visual Studio: $(if ($vsPath) { 'Found' } else { 'Missing' })"

if ($Fix) {
    Write-Host "`nSome components were automatically installed/fixed."
    Write-Host "Please re-run this script without -Fix to verify the environment."
}
