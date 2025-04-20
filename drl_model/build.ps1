# Build script for DRL Model DLL
param (
    [string]$LibTorchPath = "",
    [string]$VcpkgPath = "",
    [string]$MT5Path = "",
    [string]$BuildType = "Release"
)

# Try to read parameters from .env file if not provided
if (-not $LibTorchPath -or -not $VcpkgPath -or -not $MT5Path) {
    Write-Host "Some parameters are missing, attempting to load from .env file..."
    
    # Find .env file in the script's directory
    $scriptDir = $PSScriptRoot
    $envFile = Join-Path $scriptDir ".env"
    
    if (Test-Path $envFile) {
        Write-Host "Found .env file at: $envFile"
        $envContent = Get-Content $envFile
        
        foreach ($line in $envContent) {
            # Skip comments and empty lines
            if ($line -match '^\s*#' -or $line -match '^\s*$') {
                continue
            }
            
            # Parse key=value pairs
            if ($line -match '^\s*([^=]+)=(.+)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                
                # Map environment variables to script parameters
                switch ($key) {
                    "LIBTORCH_ROOT" { 
                        if (-not $LibTorchPath) { 
                            $LibTorchPath = $value 
                            Write-Host "Using LibTorchPath from .env: $LibTorchPath"
                        }
                    }
                    "VCPKG_ROOT" { 
                        if (-not $VcpkgPath) { 
                            $VcpkgPath = $value 
                            Write-Host "Using VcpkgPath from .env: $VcpkgPath"
                        }
                    }
                    "MT5_PATH" { 
                        if (-not $MT5Path) { 
                            $MT5Path = $value 
                            Write-Host "Using MT5Path from .env: $MT5Path"
                        }
                    }
                    "BUILD_TYPE" { 
                        if ($BuildType -eq "Release") { # Only override default value
                            $BuildType = $value 
                            Write-Host "Using BuildType from .env: $BuildType"
                        }
                    }
                }
            }
        }
    } else {
        Write-Host "No .env file found at: $envFile"
    }
}

# Check parameters
if (-not $LibTorchPath) {
    Write-Host "Please provide LibTorch path with -LibTorchPath parameter"
    exit 1
}

if (-not $VcpkgPath) {
    Write-Host "Please provide vcpkg path with -VcpkgPath parameter"
    exit 1
}

if (-not $MT5Path) {
    Write-Host "Please provide MT5 installation path with -MT5Path parameter"
    exit 1
}

# Verify paths exist
if (-not (Test-Path $LibTorchPath)) {
    Write-Host "LibTorch path not found: $LibTorchPath"
    exit 1
}

if (-not (Test-Path $VcpkgPath)) {
    Write-Host "vcpkg path not found: $VcpkgPath"
    exit 1
}

if (-not (Test-Path $MT5Path)) {
    Write-Host "MT5 path not found: $MT5Path"
    exit 1
}

# Create build directory
$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Force -Path $buildDir
}
Set-Location $buildDir

# Run CMake configuration
Write-Host "Configuring CMake build..."
$cmakeCmd = @(
    "cmake",
    "..",
    "-DCMAKE_PREFIX_PATH=$LibTorchPath",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgPath\scripts\buildsystems\vcpkg.cmake",
    "-DMT5_PATH=$MT5Path",
    "-DCMAKE_BUILD_TYPE=$BuildType"
)
& $cmakeCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed"
    exit 1
}

# Build the project
Write-Host "Building project..."
cmake --build . --config $BuildType

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed"
    exit 1
}

# Copy DLL and dependencies to MT5
Write-Host "Copying files to MT5..."
$librariesPath = Join-Path $MT5Path "MQL5\Libraries"

# Create Libraries directory if it doesn't exist
if (-not (Test-Path $librariesPath)) {
    New-Item -ItemType Directory -Force -Path $librariesPath
}

# Copy DLL
Copy-Item "bin\$BuildType\DRLModel.dll" $librariesPath

# Copy LibTorch DLLs
$libTorchDlls = @(
    "c10.dll",
    "torch_cpu.dll",
    "torch.dll"
)

foreach ($dll in $libTorchDlls) {
    $sourcePath = Join-Path $LibTorchPath "lib\$dll"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $librariesPath
    } else {
        Write-Host "Warning: Could not find $dll in LibTorch path"
    }
}

Write-Host "Build completed successfully!"
Write-Host "DLL and dependencies copied to: $librariesPath"

# Additional instructions
Write-Host @"

Next steps:
1. Export your model:
   python scripts/export_torchscript.py --model-path path/to/your/model.zip --output-dir output

2. Copy model files to MT5:
   - Copy output/model.pt to C:/MT5/
   - Copy output/model_config.json to C:/MT5/

3. Configure the EA:
   - Open DRLTrader.mq5 in MetaEditor
   - Compile the EA
   - Add to chart and configure inputs

For any issues:
- Check MT5 Experts log for specific error messages
- Verify all required DLLs are in the Libraries folder
- Ensure model paths are correctly set in EA inputs
"@
