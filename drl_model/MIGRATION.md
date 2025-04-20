# Migrating from MQL5 to LibTorch Implementation

This guide explains how to migrate from the pure MQL5 implementation to the LibTorch DLL approach.

## Key Differences

1. **Model Loading**:
   - **Before**: Model weights loaded from `weights.mqh`
   - **Now**: Model loaded directly from TorchScript file

2. **Feature Processing**:
   - **Before**: Normalization parameters hardcoded in MQL5
   - **Now**: Parameters loaded from `model_config.json`

3. **LSTM Implementation**:
   - **Before**: Custom LSTM implementation in MQL5
   - **Now**: Using PyTorch's native LSTM through LibTorch

## Migration Steps

1. **Export Model**:
```bash
# Instead of using export_model.py, use:
python scripts/export_torchscript.py --model-path bot/model/XAUUSDm.zip --output-dir output
```

2. **Update MT5 Files**:
```diff
- Remove: mql5/Include/DRL/weights.mqh
- Remove: mql5/Include/DRL/matrix.mqh
- Remove: mql5/Include/DRL/model.mqh
+ Add: Libraries/DRLModel.dll
+ Add: Libraries/c10.dll
+ Add: Libraries/torch_cpu.dll
+ Add: Libraries/torch.dll
```

3. **Update EA Configuration**:
```diff
- input string WeightsPath = "Include\\DRL\\weights.mqh";
+ input string ModelPath = "C:\\MT5\\model.pt";
+ input string ConfigPath = "C:\\MT5\\model_config.json";
```

## Feature Changes

The feature processing now exactly matches the Python implementation:

1. Returns: `[-0.1, 0.1]`
2. RSI: `[-1, 1]`
3. ATR: `[-1, 1]`
4. Volume Change: `[-1, 1]`
5. BB Position: `[0, 1]`
6. Trend Strength: `[-1, 1]`
7. Candle Pattern: `[-1, 1]`
8. Time Features: `[-1, 1]`
9. Position Features: `[-1, 1]`

## Performance Improvements

1. **Accuracy**:
   - No more floating-point discrepancies
   - Exact match with Python behavior

2. **Speed**:
   - Native LSTM implementation
   - Optimized matrix operations
   - Zero-copy tensor conversions

3. **Memory**:
   - More efficient state management
   - Better memory allocation

## Testing and Verification

1. **Compare Outputs**:
```bash
cd drl_model/scripts
run_tests.bat path/to/bot/directory
```

2. **Verify Features**:
   - EA logs feature values
   - Compare with Python values
   - Check normalization

## Troubleshooting

1. **DLL Load Issues**:
   - Verify DLL architecture (x64)
   - Check MT5 logs
   - Install VC++ Redistributable

2. **Feature Mismatch**:
   - Compare raw feature values
   - Check normalization parameters
   - Verify config loading

3. **State Management**:
   - Check LSTM state sizes
   - Verify state reset conditions
   - Monitor sequence handling

## Benefits

1. **Maintainability**:
   - Single source of truth (Python model)
   - Simpler EA code
   - Easier updates

2. **Reliability**:
   - Production-tested LSTM implementation
   - Better error handling
   - Consistent behavior

3. **Performance**:
   - Optimized native code
   - Better memory management
   - Faster inference

## Rollback Plan

If issues arise, you can temporarily revert to the pure MQL5 implementation:

1. Keep both implementations in source control
2. Use feature flags in EA
3. Monitor performance differences

## Future Improvements

1. GPU Support (if needed)
2. Batch prediction optimization
3. Extended feature support
4. Real-time performance monitoring
