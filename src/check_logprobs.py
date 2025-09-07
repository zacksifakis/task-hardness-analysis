import asyncio
import httpx
import json

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
URL = "http://127.0.0.1:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

async def run_diagnostic():
    print("\n--- Starting Final Logprobs Diagnostic ---")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 50,
        "temperature": 0.1,
        "logprobs": True,
        "top_logprobs": 5
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(URL, headers=HEADERS, json=payload)

        print(f"\n--- Server responded with status code: {response.status_code} ---")

        if response.status_code == 200:
            print("\n--- SUCCESS! Here is the raw JSON response from the server: ---")
            parsed_json = response.json()
            print(json.dumps(parsed_json, indent=2))
        else:
            print("\n--- ERROR: The server responded with an error. ---")
            print(response.text)

    except Exception as e:
        print(f"\n--- FATAL SCRIPT ERROR ---")
        print(e)

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
