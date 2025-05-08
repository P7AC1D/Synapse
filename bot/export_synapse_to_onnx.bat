@echo off
echo Exporting synapse.zip model to ONNX format...

REM Set paths
set MODEL_PATH=c:\Code\drl\bot\model\synapse.zip
set OUTPUT_PATH=c:\Code\drl\ctrader\Models\synapse.onnx

REM Run the export script - note the removal of --is_recurrent flag since it's a standard PPO model
python c:\Code\drl\bot\src\export_model_to_onnx.py --model_path "%MODEL_PATH%" --output_path "%OUTPUT_PATH%"

echo Done!
if exist "%OUTPUT_PATH%" (
    echo Model exported successfully to: %OUTPUT_PATH%
) else (
    echo Export failed! Please check for errors.
)

pause