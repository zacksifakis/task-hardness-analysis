# import numpy as np
# import json
# from tqdm import tqdm
# import torch
# from data_utils import ensure_dir
# import os
# import math
# import litellm # Use the core litellm library
# import re
# import asyncio

# # This is the most important line: it will show us the exact URL being called.
# # litellm.set_verbose = True

# def get_problem_info(problem, dataset_name):
#     # This function is correct, no changes needed
#     if (dataset_name.lower() == "aime"):
#         return {
#             "problem_text": problem["Question"], "problem_number": problem["Problem Number"],
#             "year": problem["Year"], "x_value": problem["Problem Number"],
#             "answer": problem["Answer"] if "Answer" in problem else None,
#             "part": problem["Part"] if "Part" in problem else None
#         }
#     # ... (rest of the function is the same)
#     elif (dataset_name.lower() == "webinstruct"):
#         return {"problem_text": problem["question"], "id": problem["id"], "difficulty": problem["difficulty"], "category": problem["category"], "x_value": problem["difficulty"], "answer": problem["answer"] if "answer" in problem else None}
#     elif (dataset_name.lower() == "hmmt"):
#         return {"problem_text": problem["problem"], "problem_number": problem["problem_idx"], "year": problem["year"], "x_value": problem["problem_idx"], "answer": problem["answer"] if "answer" in problem else None}
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

# def extract_answer_from_response(response_text):
#     # This function is correct, no changes needed
#     boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', response_text)]
#     if boxed_starts:
#         last_start = boxed_starts[-1]
#         count = 0
#         for i in range(last_start + 7, len(response_text)):
#             if response_text[i] == '{': count += 1
#             elif response_text[i] == '}':
#                 if count == 0: return response_text[last_start + 7:i].strip()
#                 count -= 1
#     return None

# # async def calculate_multiple_responses_litellm(model_name, problems, dataset_name, temperature, n_gens=5, batch_size=4,dtype="bfloat16"):
# #     results = []
# #     model_name_safe = model_name.replace('/', '_')
# #     problems = list(problems)

# #     if dataset_name.lower() == "aime":
# #         problems.sort(key=lambda x: int(x["Year"]), reverse=True)
# #     total_problems = len(problems)
    
# #     for problem_idx in tqdm(range(total_problems), desc=f"Processing {total_problems} problems with {model_name}"):
# #         print(f"Year being processed: {problems[problem_idx]['Year']}")
# #         problem = problems[problem_idx]
# #         info = get_problem_info(problem, dataset_name)
# #         problem_text = info["problem_text"]
# #         messages = [{"role": "user", "content": problem_text}]
        
# #         sub_batch_size = min(batch_size, n_gens)
# #         n_sub_batches = math.ceil(n_gens / sub_batch_size)
        
# #         all_outputs = []
# #         for sub_batch_idx in range(n_sub_batches):
# #             tasks = []
# #             for _ in range(sub_batch_size):
# #                 task = litellm.acompletion(
# #                     model=f"openai/{model_name}",
# #                     messages=messages,
# #                     api_base="http://127.0.0.1:8000/v1",
# #                     api_key="EMPTY",
# #                     temperature=temperature,
# #                     max_tokens=4096,
# #                     top_p=1.0,
# #                     logprobs=True,
# #                     top_logprobs=1,
# #                     repetition_penalty=1.2,
# #                 )
# #                 tasks.append(task)
            
# #             responses = await asyncio.gather(*tasks)
# #             all_outputs.extend(responses)
        
# #         # --- THIS IS THE FINAL, ROBUST LOGIC ---
# #         for gen_idx, output in enumerate(all_outputs):
# #             year, problem_number = info["year"], info["problem_number"]
# #             result_dir = f"results/self_classification/{dataset_name}/{model_name_safe}/temp_{temperature}/{year}"
# #             if dataset_name.lower() == "aime":
# #                 part = info["part"]
# #                 result_dir += f"/{part}"
# #             result_dir += f"/{problem_number}/gen_{gen_idx}"
# #             ensure_dir(result_dir)
            
# #             response_text = output.choices[0].message.content
# #             extracted_answer = extract_answer_from_response(response_text)
            
# #             # FIX 1: Get response length from the reliable 'usage' object
# #             response_len = output.usage.completion_tokens if output.usage else len(response_text.split())

# #             # FIX 2: Gracefully handle empty or malformed logprobs
# #             logprobs = output.choices[0].logprobs
# #             token_logprobs = []
# #             if logprobs and logprobs.content:
# #                 token_logprobs = [lp.logprob for lp in logprobs.content if lp.logprob is not None]

# #             cumulative_logprob = sum(token_logprobs)
# #             avg_neg_logprob = -cumulative_logprob / response_len if response_len > 0 else 0
            
# #             result = {
# #                 'problem_text': problem_text, 'response_text': response_text, 'response_length': response_len,
# #                 'avg_neg_logprob': avg_neg_logprob, 'total_neg_logprob': -cumulative_logprob,
# #                 'token_neg_logprobs': [-lp for lp in token_logprobs], 'generation_id': gen_idx,
# #                 'x_value': info["x_value"], 'answer': info["answer"], 'problem_number': info["problem_number"],
# #                 'year': info["year"], 'extracted_answer': extracted_answer
# #             }
            
# #             with open(f"{result_dir}/result.json", 'w') as f:
# #                 json.dump({k: (v if k != 'token_neg_logprobs' else [float(e) for e in v]) for k, v in result.items()}, f, indent=2)
# #             with open(f"{result_dir}/response.txt", 'w') as f:
# #                 f.write(response_text)
# #             results.append({k: v for k, v in result.items() if k != 'token_neg_logprobs'})
    
# #     torch.cuda.empty_cache()
# #     return results

# async def calculate_multiple_responses_litellm(model_name, problems, dataset_name, temperature, n_gens=5, batch_size=4,dtype="bfloat16"):
#     results = []
#     model_name_safe = model_name.replace('/', '_')
#     problems = list(problems)

#     if dataset_name.lower() == "aime":
#         problems.sort(key=lambda x: int(x["Year"]), reverse=True)
#     total_problems = len(problems)
    
#     for problem_idx in tqdm(range(total_problems), desc=f"Processing {total_problems} problems with {model_name}"):
#         print(f"Year being processed: {problems[problem_idx]['Year']}")
#         problem = problems[problem_idx]
#         info = get_problem_info(problem, dataset_name)
#         problem_text = info["problem_text"]
#         messages = [{"role": "user", "content": problem_text}]
        
#         sub_batch_size = min(batch_size, n_gens)
#         n_sub_batches = math.ceil(n_gens / sub_batch_size)
        
#         all_outputs = []
#         for sub_batch_idx in range(n_sub_batches):
#             tasks = []
#             for _ in range(sub_batch_size):
#                 task = litellm.acompletion(
#                     model=f"openai/{model_name}",
#                     messages=messages,
#                     api_base="http://127.0.0.1:8000/v1",
#                     api_key="EMPTY",
#                     temperature=temperature,
#                     max_tokens=4096,
#                     top_p=1.0,
#                     logprobs=True,
#                     top_logprobs=5, # Request top 5 to be safe
#                 )
#                 tasks.append(task)
            
#             responses = await asyncio.gather(*tasks)
#             all_outputs.extend(responses)
        
#         # --- THIS IS THE FINAL, ROBUST LOGIC ---
#         for gen_idx, output in enumerate(all_outputs):
#             year, problem_number = info["year"], info["problem_number"]
#             result_dir = f"results/self_classification/{dataset_name}/{model_name_safe}/temp_{temperature}/{year}"
#             if dataset_name.lower() == "aime":
#                 part = info["part"]
#                 result_dir += f"/{part}"
#             result_dir += f"/{problem_number}/gen_{gen_idx}"
#             ensure_dir(result_dir)
            
#             response_text = output.choices[0].message.content
#             extracted_answer = extract_answer_from_response(response_text)
            
#             # FIX 1: Get response length from the reliable 'usage' object
#             response_len = output.usage.completion_tokens if output.usage else len(response_text.split())

#             # FIX 2: Gracefully and robustly handle empty or malformed logprobs
#             logprobs = output.choices[0].logprobs
#             token_logprobs = []
#             if logprobs and hasattr(logprobs, 'content') and logprobs.content:
#                 # This is the expected format
#                 token_logprobs = [lp.logprob for lp in logprobs.content if hasattr(lp, 'logprob') and lp.logprob is not None]
            
#             if not token_logprobs:
#                 print(f"Warning: Logprobs were requested but not found for problem {problem_number}, gen {gen_idx}. Metrics will be 0.")

#             cumulative_logprob = sum(token_logprobs)
#             avg_neg_logprob = -cumulative_logprob / response_len if response_len > 0 else 0
            
#             result = {
#                 'problem_text': problem_text, 'response_text': response_text, 'response_length': response_len,
#                 'avg_neg_logprob': avg_neg_logprob, 'total_neg_logprob': -cumulative_logprob,
#                 'token_neg_logprobs': [-lp for lp in token_logprobs], 'generation_id': gen_idx,
#                 'x_value': info["x_value"], 'answer': info["answer"], 'problem_number': info["problem_number"],
#                 'year': info["year"], 'extracted_answer': extracted_answer
#             }
            
#             with open(f"{result_dir}/result.json", 'w') as f:
#                 json.dump({k: (v if k != 'token_neg_logprobs' else [float(e) for e in v]) for k, v in result.items()}, f, indent=2)
#             with open(f"{result_dir}/response.txt", 'w') as f:
#                 f.write(response_text)
#             results.append({k: v for k, v in result.items() if k != 'token_neg_logprobs'})
    
#     torch.cuda.empty_cache()
#     return results

import os
import re
import math
import json
import asyncio
import httpx
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from data_utils import ensure_dir

# ----------------------------
# Server + output configuration
# ----------------------------
BASE_URL = os.getenv("VLLM_SERVER_URL", "http://127.0.0.1:8000")
CHAT_URL = f"{BASE_URL}/v1/chat/completions"   # vLLM chat endpoint
COMP_URL = f"{BASE_URL}/v1/completions"        # not used here; kept for reference
OUTPUT_BASE = os.getenv("OUTPUT_DIR_BASE", "results")

print(f"[BOOT] Using VLLM_SERVER_URL={BASE_URL}", flush=True)
print(f"[BOOT] OUTPUT_DIR_BASE={OUTPUT_BASE}", flush=True)

# ----------------------------
# Tokenizer (loaded once)
# ----------------------------
TOKENIZER = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# ----------------------------
# Helpers
# ----------------------------
def get_problem_info(problem, dataset_name):
    ds = dataset_name.lower()
    if ds == "aime":
        return {
            "problem_text": problem["Question"],
            "problem_number": problem["Problem Number"],
            "year": problem["Year"],
            "x_value": problem["Problem Number"],
            "answer": problem["Answer"] if "Answer" in problem else None,
            "part": problem["Part"] if "Part" in problem else None,
        }
    elif ds == "webinstruct":
        return {
            "problem_text": problem["question"],
            "id": problem["id"],
            "difficulty": problem["difficulty"],
            "category": problem["category"],
            "x_value": problem["difficulty"],
            "answer": problem["answer"] if "answer" in problem else None,
        }
    elif ds == "hmmt":
        return {
            "problem_text": problem["problem"],
            "problem_number": problem["problem_idx"],
            "year": problem["year"],
            "x_value": problem["problem_idx"],
            "answer": problem["answer"] if "answer" in problem else None,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def extract_answer_from_response(response_text):
    # matches \boxed{...} and returns the last one
    boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', response_text)]
    if boxed_starts:
        last_start = boxed_starts[-1]
        count = 0
        for i in range(last_start + 7, len(response_text)):
            if response_text[i] == '{':
                count += 1
            elif response_text[i] == '}':
                if count == 0:
                    return response_text[last_start + 7:i].strip()
                count -= 1
    return None

def _parse_year(v):
    m = re.search(r"\b(19|20)\d{2}\b", str(v))
    return int(m.group(0)) if m else -1

# ----------------------------
# Main entry called by generate_multiple_responses.py
# ----------------------------
async def calculate_multiple_responses_litellm(
    model_name,
    problems,
    dataset_name,
    temperature,
    n_gens,
    server_max_context=16384,
    max_tokens_to_generate=16000,
    logprobs=False,
    top_logprobs=0,
    batch_size=4,
    dtype="bfloat16",
):
    results = []
    model_name_safe = model_name.replace("/", "_")
    problems = list(problems)

    # Sort (AIME) newest first; be robust to weird year strings
    if dataset_name.lower() == "aime":
        problems.sort(key=lambda x: _parse_year(x.get("Year")), reverse=True)

    total_problems = len(problems)
    url = CHAT_URL
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=1800.0) as client:
        for problem_idx in tqdm(
            range(total_problems),
            desc=f"Processing {total_problems} problems with {model_name}"
        ):
            tqdm.write(f"Year being processed: {problems[problem_idx]['Year']}")
            problem = problems[problem_idx]
            info = get_problem_info(problem, dataset_name)
            problem_text = info["problem_text"]

            # -------- token counting + dynamic cap --------
            try:
                prompt_tokens = len(
                    TOKENIZER.apply_chat_template(
                        [{"role": "user", "content": problem_text}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                # fallback rough estimate if chat template fails
                prompt_tokens = int(len(problem_text.split()) * 1.3)

            # reserve some buffer for EOS/etc
            available_space = max(1, server_max_context - prompt_tokens - 64)
            dynamic_max_tokens = min(max_tokens_to_generate, available_space)

            print(
                f"[DEBUG] {info.get('year')}-{info.get('problem_number')}: "
                f"prompt={prompt_tokens}, dyn_max={dynamic_max_tokens}"
            )

            if dynamic_max_tokens <= 0:
                print(
                    f"[WARN] Skipping problem {info.get('problem_number')} "
                    f"(prompt={prompt_tokens}, context={server_max_context})"
                )
                continue

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": problem_text}],
                "max_tokens": dynamic_max_tokens,
                "temperature": temperature,
                "repetition_penalty": 1.2,
            }

            if logprobs:
                payload["logprobs"] = True
                if top_logprobs and int(top_logprobs) > 0:
                    payload["top_logprobs"] = int(top_logprobs)

            print(
                f"[POST] {info.get('year')}-{info.get('problem_number')}: "
                f"sending {n_gens} reqs with max_tokens={dynamic_max_tokens}, temp={temperature}",
                flush=True,
            )

            # fire off N requests concurrently
            tasks = [client.post(url, headers=headers, json=payload) for _ in range(n_gens)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            all_outputs = []
            for i, res in enumerate(responses, start=1):
                if isinstance(res, Exception):
                    print(f"[HTTP EXC] gen#{i}: {repr(res)}", flush=True)
                    continue

                if res.status_code != 200:
                    body = res.text
                    print(f"[HTTP {res.status_code}] gen#{i}: {body[:600]}", flush=True)
                    continue

                try:
                    out = res.json()
                    all_outputs.append(out)

                    used = (out.get('usage') or {}).get('completion_tokens')
                    lp = ((out.get('choices') or [{}])[0].get('logprobs') or {})
                    has_lp = bool(lp.get('token_logprobs')) or bool(lp.get('content'))
                    print(f"[OK] gen#{i}: completion_tokens={used}  logprobs={'yes' if has_lp else 'no'}", flush=True)
                except Exception as e:
                    print(f"[PARSE ERROR] gen#{i}: {e}", flush=True)

            if not all_outputs:
                print("[WARN] No successful responses for this problem; moving on.", flush=True)
                continue

            # -------- persist results --------
            for gen_idx, output in enumerate(all_outputs):
                year, problem_number = info["year"], info["problem_number"]
                part = info.get("part")

                # build base dir once per run; it’s controlled by OUTPUT_BASE env
                result_dir = os.path.join(
                    OUTPUT_BASE,
                    "self_classification",
                    dataset_name,
                    model_name_safe,
                    f"temp_{temperature}",
                    str(year),
                )
                if dataset_name.lower() == "aime" and part:
                    result_dir = os.path.join(result_dir, str(part))
                result_dir = os.path.join(result_dir, str(problem_number), f"gen_{gen_idx}")
                ensure_dir(result_dir)

                # text + metrics
                response_text = output["choices"][0]["message"]["content"]
                extracted_answer = extract_answer_from_response(response_text)

                response_len = int((output.get("usage") or {}).get("completion_tokens") or 0)

                # parse vLLM logprobs format if present
                token_logprobs = []
                logprobs_data = ((output.get('choices') or [{}])[0].get('logprobs') or {})

                # Preferred (vLLM chat/completions current shape)
                if isinstance(logprobs_data, dict) and isinstance(logprobs_data.get('token_logprobs'), list):
                    token_logprobs = [float(x) for x in logprobs_data['token_logprobs'] if x is not None]

                # Back-compat (older/alternative shape you originally coded for)
                elif isinstance(logprobs_data, dict) and isinstance(logprobs_data.get('content'), list):
                    for t in logprobs_data['content']:
                        if t and ('logprob' in t) and (t['logprob'] is not None):
                            token_logprobs.append(float(t['logprob']))

                cumulative_logprob = sum(token_logprobs)
                avg_neg_logprob = -cumulative_logprob / response_len if response_len > 0 else 0

                result = {
                    "problem_text": problem_text,
                    "response_text": response_text,
                    "response_length": response_len,
                    "avg_neg_logprob": avg_neg_logprob,
                    "total_neg_logprob": -cumulative_logprob,
                    "token_neg_logprobs": [-lp for lp in token_logprobs],
                    "generation_id": gen_idx,
                    "x_value": info.get("x_value"),
                    "answer": info.get("answer"),
                    "problem_number": problem_number,
                    "year": year,
                    "extracted_answer": extracted_answer,
                }

                with open(os.path.join(result_dir, "result.json"), "w") as f:
                    json.dump(
                        {k: (v if k != "token_neg_logprobs" else [float(e) for e in v]) for k, v in result.items()},
                        f,
                        indent=2,
                    )
                with open(os.path.join(result_dir, "response.txt"), "w") as f:
                    f.write(response_text)

                results.append({k: v for k, v in result.items() if k != "token_neg_logprobs"})

    torch.cuda.empty_cache()
    return results
