using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;

namespace YourNamespace.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class LLMController : ControllerBase
    {
        private readonly HttpClient _httpClient;
        private readonly IConfiguration _configuration;

        public LLMController(IHttpClientFactory httpClientFactory, IConfiguration configuration)
        {
            _httpClient = httpClientFactory.CreateClient();
            _configuration = configuration;
        }

        [HttpPost]
        public async Task<IActionResult> ProcessLLMInput([FromBody] InputModel input)
        {
            try
            {
       
                var openAiResponse = await SendToOpenAiAssistant(input.UserInput, _configuration["OpenAI:FirstAssistantId"]);

          
                var pythonResponse = await PushToPythonScript(openAiResponse);

                if (pythonResponse == 0)
                {
        
                    var secondAssistantResponse = await SendToOpenAiAssistant(input.UserInput, _configuration["OpenAI:SecondAssistantId"]);
                    return Ok(new { Response = secondAssistantResponse });
                }
                else if (pythonResponse == 1)
                {
                 
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

        private async Task<string> SendToOpenAiAssistant(string userInput, string assistantId)
        {
            var requestBody = new
            {
                Input = userInput,
                AssistantId = assistantId
            };

            var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

            _httpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _configuration["OpenAI:ApiKey"]);

            var response = await _httpClient.PostAsync("https://api.openai.com/v1/assistants", content);
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
