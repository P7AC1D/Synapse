using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using DRLTrader.Models;

namespace DRLTrader.Services
{
    /// <summary>
    /// Client for communicating with the prediction API
    /// </summary>
    public class ApiClient : IDisposable
    {
        private readonly HttpClient _client;
        private readonly string _baseUrl;
        private readonly JsonSerializerOptions _jsonOptions;
        private bool _disposed;

        public ApiClient(string baseUrl)
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _client = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(10)
            };
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
        }

        /// <summary>
        /// Get prediction from the API
        /// </summary>
        public async Task<PredictionResponse> GetPredictionAsync(MarketData data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data, _jsonOptions);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await _client.PostAsync($"{_baseUrl}/predict", content);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<PredictionResponse>(responseJson, _jsonOptions);
            }
            catch (HttpRequestException ex)
            {
                throw new Exception($"API request failed: {ex.Message}", ex);
            }
            catch (JsonException ex)
            {
                throw new Exception($"Failed to process API response: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Check API health status
        /// </summary>
        public async Task<bool> CheckHealthAsync()
        {
            try
            {
                var response = await _client.GetAsync($"{_baseUrl}/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _client?.Dispose();
                }
                _disposed = true;
            }
        }

        ~ApiClient()
        {
            Dispose(false);
        }
    }
}
