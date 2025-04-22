//+------------------------------------------------------------------+
//|                                                       Numpy.mqh |
//|                             Simplified NumPy functions for MQL5 |
//+------------------------------------------------------------------+
#property copyright "DRL Trader"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Numpy-like functions for array operations                        |
//+------------------------------------------------------------------+
class Numpy
{
public:
    // Clip values to a specified range [min_value, max_value]
    static void clip(double &arr[], double min_value, double max_value) {
        int size = ArraySize(arr);
        for(int i=0; i<size; i++) {
            arr[i] = MathMin(MathMax(arr[i], min_value), max_value);
        }
    }
    
    // Calculate difference between consecutive elements
    static void diff(const double &arr[], double &result[]) {
        int size = ArraySize(arr);
        if(size <= 1) return;
        
        ArrayResize(result, size-1);
        for(int i=0; i<size-1; i++) {
            result[i] = arr[i] - arr[i+1];  // Note: Assuming ArraySetAsSeries(true)
        }
    }
    
    // Element-wise division with handling for division by zero
    static void divide(const double &a[], const double &b[], double &result[], double fill_value=0) {
        int size = MathMin(ArraySize(a), ArraySize(b));
        ArrayResize(result, size);
        
        for(int i=0; i<size; i++) {
            if(MathAbs(b[i]) > 1e-8) {
                result[i] = a[i] / b[i];
            } else {
                result[i] = fill_value;
            }
        }
    }
    
    // Insert a value at the beginning of an array
    static void insert_at_beginning(double &arr[], double value) {
        int size = ArraySize(arr);
        double temp[];
        ArrayResize(temp, size);
        ArrayCopy(temp, arr);
        
        ArrayResize(arr, size+1);
        arr[0] = value;
        for(int i=0; i<size; i++) {
            arr[i+1] = temp[i];
        }
    }
    
    // Find the maximum of two arrays element-wise
    static void maximum(const double &a[], const double &b[], double &result[]) {
        int size = MathMin(ArraySize(a), ArraySize(b));
        ArrayResize(result, size);
        
        for(int i=0; i<size; i++) {
            result[i] = MathMax(a[i], b[i]);
        }
    }
    
    // Find the minimum of two arrays element-wise
    static void minimum(const double &a[], const double &b[], double &result[]) {
        int size = MathMin(ArraySize(a), ArraySize(b));
        ArrayResize(result, size);
        
        for(int i=0; i<size; i++) {
            result[i] = MathMin(a[i], b[i]);
        }
    }
    
    // Calculate sine values for an array
    static void sin(const double &arr[], double &result[]) {
        int size = ArraySize(arr);
        ArrayResize(result, size);
        
        for(int i=0; i<size; i++) {
            result[i] = MathSin(arr[i]);
        }
    }
    
    // Calculate cosine values for an array
    static void cos(const double &arr[], double &result[]) {
        int size = ArraySize(arr);
        ArrayResize(result, size);
        
        for(int i=0; i<size; i++) {
            result[i] = MathCos(arr[i]);
        }
    }
    
    // Calculate rolling mean (SMA)
    static void rolling_mean(const double &arr[], int window, double &result[]) {
        int size = ArraySize(arr);
        ArrayResize(result, size);
        ArrayInitialize(result, 0);
        
        for(int i=0; i<size; i++) {
            double sum = 0;
            int count = 0;
            
            for(int j=MathMax(0, i-window+1); j<=i; j++) {
                sum += arr[j];
                count++;
            }
            
            if(count > 0) {
                result[i] = sum / count;
            }
        }
    }
    
    // Calculate percentile of an array
    static double percentile(const double &arr[], double p) {
        int size = ArraySize(arr);
        if(size == 0) return 0;
        
        double sorted[];
        ArrayResize(sorted, size);
        ArrayCopy(sorted, arr);
        ArraySort(sorted);
        
        double index = (size-1) * p;
        int lower = (int)MathFloor(index);
        int upper = (int)MathCeil(index);
        
        if(lower == upper) {
            return sorted[lower];
        } else {
            double d = index - lower;
            return (1-d) * sorted[lower] + d * sorted[upper];
        }
    }
};