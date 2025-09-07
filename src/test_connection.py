import litellm
import asyncio

# This is the most important line: it will show us the exact URL being called.
litellm.set_verbose = True

async def run_test():
    try:
        print("\n--- Starting Final Connection Test ---")
        response = await litellm.acompletion(
            # The 'openai/' prefix is the correct way to tell litellm the API format.
            model="openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="http://127.0.0.1:8000/v1", # We now include the /v1 path explicitly.
            api_key="EMPTY"
        )
        print("\n--- SUCCESS! CONNECTION ESTABLISHED! ---")
        print(response)
    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(e)

# Run the test
asyncio.run(run_test())
