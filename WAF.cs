<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Input Webpage</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
        }
        input, button {
            padding: 10px;
            margin: 10px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <h1>LLM Input Interface</h1>
    <p>Enter your query below:</p>
    <input type="text" id="userInput" placeholder="Type your input here">
    <button onclick="sendInput()">Submit</button>
    <p id="response"></p>

    <script>
        async function sendInput() {
            const userInput = document.getElementById("userInput").value;
            const responseElement = document.getElementById("response");

            if (!userInput) {
                responseElement.textContent = "Please enter some input.";
                return;
            }

            try {
                // Send input to the API
                const response = await fetch('/api/LLM', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ UserInput: userInput })
                });

                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }

                const data = await response.json();
                responseElement.textContent = `Response: ${data.Response}`;
            } catch (error) {
                responseElement.textContent = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
