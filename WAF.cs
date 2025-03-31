using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace YourNamespace.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class LLMController : ControllerBase
    {
        private readonly HttpClient _httpClient;

        public LLMController(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient();
        }

        [HttpPost]
        public async Task<IActionResult> ProcessLLMInput([FromBody] InputModel input)
        {
            try
            {
                // Step 1: Send input to the first OpenAI assistant
                var openAiResponse = await SendToOpenAiAssistant(input.UserInput, "FirstAssistantEndpoint");

                // Step 2: Push the OpenAI response JSON to the Python script
                var pythonResponse = await PushToPythonScript(openAiResponse);

                if (pythonResponse == 0)
                {
                    // Step 3A: Send original input to the second OpenAI assistant
                    var secondAssistantResponse = await SendToOpenAiAssistant(input.UserInput, "SecondAssistantEndpoint");
                    return Ok(new { Response = secondAssistantResponse });
                }
                else if (pythonResponse == 1)
                {
                    // Step 3B: Send "canned response"
                    return Ok(new { Response = "Canned response" });
                }
                else
                {
                    return BadRequest("Unexpected response from Python script.");
                }
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal Server Error: {ex.Message}");
            }
        }

        private async Task<string> SendToOpenAiAssistant(string userInput, string assistantEndpoint)
        {
            var requestBody = new { Input = userInput };
            var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(assistantEndpoint, content);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync();
            return responseString;
        }

        private async Task<int> PushToPythonScript(string jsonResponse)
        {
            var content = new StringContent(jsonResponse, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync("http://localhost:5000/mla_model_run", content);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync();
            return int.Parse(responseString); // Assuming the Python API returns a simple numeric value
        }
    }

    public class InputModel
    {
        public string UserInput { get; set; }
    }
}
