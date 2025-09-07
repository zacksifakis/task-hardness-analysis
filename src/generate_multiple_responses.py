import argparse
import pandas as pd
import json
import os
from inference import calculate_multiple_responses_litellm
from data_utils import load_dataset_by_name, ensure_dir
import numpy as np
import re
import asyncio

print("=== PY ENTRYPOINT REACHED: generate_multiple_responses.py ===", flush=True)

# async def main():
#     parser = argparse.ArgumentParser(description='Generate multiple responses for problems from different datasets using vLLM')
#     parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help='Model name/path')
#     parser.add_argument('--temp', type=float, default=0.6, help='Temperature for sampling')
#     parser.add_argument('--n_gens', type=int, default=8, help='Number of generations per problem')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
#     parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
#     parser.add_argument('--dataset', type=str, default='aime', help='Dataset name (e.g., aime, webinstruct)')
#     year_filter_string = ",".join(map(str, range(2010, 2025)))
#     parser.add_argument('--year_filter', type=str, default = year_filter_string, help='Filter by specific years for AIME dataset, comma-separated')
#     parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model weights')
#     # 1. ADD the new command-line argument
#     parser.add_argument('--max_tokens', type=int, default=8000, help='Maximum number of tokens to generate')
#     args = parser.parse_args()

#     dataset_name = args.dataset
#     print(f"Loading {dataset_name} problems...")
#     problems = load_dataset_by_name(dataset_name=dataset_name)
    
#     if dataset_name == "aime" and args.year_filter:
#         print(f"Filtering problems by years: {args.year_filter}")
#         years = [int(y.strip()) for y in args.year_filter.split(',')]
#         problems = [p for p in problems if int(p['Year']) in years]
#         print(f"Filtered to {len(problems)} problems from years {years}")
    
#     print(f"Using model: {args.model}")
#     print(f"Temperature: {args.temp}")
#     print(f"Generations per problem: {args.n_gens}")
    
#     results = await calculate_multiple_responses_litellm(
#         model_name=args.model,
#         problems=problems,
#         dataset_name=dataset_name,
#         temperature=args.temp,
#         n_gens=args.n_gens,
#         batch_size=args.batch_size,
#         dtype=args.dtype,
#         max_tokens=args.max_tokens # 2. PASS the new argument to the function
#     )
    
#     results_df = pd.DataFrame(results)
    
#     model_name_safe = args.model.replace('/', '_')
    
#     base_results_dir = f"{args.output_dir}/self_classification/{dataset_name}/{model_name_safe}/temp_{args.temp}"
#     ensure_dir(base_results_dir)
    
#     output_path_base = f"{args.output_dir}/multiple_responses_{model_name_safe}_{dataset_name}_temp_{args.temp}_n_{args.n_gens}"
#     results_df.to_json(f"{output_path_base}.jsonl", orient='records', lines=True)
#     results_df.to_pickle(f"{output_path_base}.pkl")
    
#     print(f"Results saved to {output_path_base}.jsonl and {output_path_base}.pkl")

# if __name__ == "__main__":
#     asyncio.run(main())

async def main():
    print("[DEBUG] main() started", flush=True)
    parser = argparse.ArgumentParser(description='Generate multiple responses for problems from different datasets using vLLM')

    # core
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help='Model name/path')
    parser.add_argument('--max_tokens', type=int, default=16000, help='Maximum number of tokens to generate')

    # keep these optional with safe defaults
    parser.add_argument('--temp', type=float, default=0.6, help='Temperature for sampling')
    parser.add_argument('--n_gens', type=int, default=8, help='Number of generations per problem')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')

    # dataset + output
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--dataset', type=str, default='aime', help='Dataset name (e.g., aime, webinstruct)')
    year_filter_string = ",".join(map(str, range(2010, 2025)))
    parser.add_argument('--year_filter', type=str, default=year_filter_string, help='Filter by specific years for AIME dataset, comma-separated')

    # unused / optional knobs (do NOT require them)
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model weights')
    parser.add_argument('--server_max_context', type=int, default=16384, help='Hard limit of the server context window (optional)')
    parser.add_argument('--logprobs', type=lambda x: str(x).lower() == 'true', default=False, help='Whether to request logprobs (optional)')
    parser.add_argument('--top_logprobs', type=int, default=0, help='Number of top logprobs to return if logprobs=True')
    parser.add_argument('--vllm_url', type=str, default=None,
                    help='Override VLLM server URL (e.g., http://127.0.0.1:8001)')

    args = parser.parse_args()
    print("[DEBUG] Parsed args:", vars(args), flush=True)

    # Keep inference.py’s config consistent with CLI:
    if args.vllm_url:
        os.environ["VLLM_SERVER_URL"] = args.vllm_url

    # Ensure OUTPUT_DIR_BASE (used by inference.py) matches --output_dir unless already set from outside
    os.environ.setdefault("OUTPUT_DIR_BASE", args.output_dir)

    print(f"[DEBUG] VLLM_SERVER_URL={os.getenv('VLLM_SERVER_URL', 'http://127.0.0.1:8000')}", flush=True)
    print(f"[DEBUG] OUTPUT_DIR_BASE={os.getenv('OUTPUT_DIR_BASE')}", flush=True)


    dataset_name = args.dataset
    print(f"Loading {dataset_name} problems...")
    problems = load_dataset_by_name(dataset_name=dataset_name)
    
    print(f"Loading {dataset_name} problems... done.", flush=True)
    print(f"[DEBUG] Loaded total items: {len(problems)}", flush=True)

    # after problems = load_dataset_by_name(...)
    print(f"[DEBUG] Loaded {len(problems)} total items")

    def parse_year(v):
        # extract first 4-digit year anywhere in the string; fallback None
        m = re.search(r"\b(19|20)\d{2}\b", str(v))
        return int(m.group(0)) if m else None

    if dataset_name.lower() == "aime":
        if args.year_filter:
            years = [int(y.strip()) for y in args.year_filter.split(",")]
        else:
            years = list(range(2010, 2026))
        before = len(problems)
        problems = [p for p in problems if (parse_year(p.get("Year")) in years)]
        print(f"[DEBUG] Year-filtered: {before} -> {len(problems)} using {years[:3]}...")

    if not problems:
        print("[ERROR] No problems after filtering. Exiting.")
        return

    print(f"Using model: {args.model}")
    print(f"Temperature: {args.temp}")
    print(f"Generations per problem: {args.n_gens}")

    # DEBUG: confirm we're not about to run an empty loop
    print(f"[DEBUG] total_problems={len(problems)}  n_gens={args.n_gens}", flush=True)

    
    # We pass all the orders down to the engine.
    results = await calculate_multiple_responses_litellm(
        model_name=args.model,
        problems=problems,
        dataset_name=dataset_name,
        temperature=args.temp,
        n_gens=args.n_gens,
        server_max_context=args.server_max_context,
        max_tokens_to_generate=args.max_tokens,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs
    )
    
    results_df = pd.DataFrame(results)
    
    model_name_safe = args.model.replace('/', '_')
    
    base_results_dir = f"{args.output_dir}/self_classification/{dataset_name}/{model_name_safe}/temp_{args.temp}"
    ensure_dir(base_results_dir)
    
    output_path_base = f"{args.output_dir}/multiple_responses_{model_name_safe}_{dataset_name}_temp_{args.temp}_n_{args.n_gens}"
    results_df.to_json(f"{output_path_base}.jsonl", orient='records', lines=True)
    results_df.to_pickle(f"{output_path_base}.pkl")
    
    print(f"Results saved to {output_path_base}.jsonl and {output_path_base}.pkl")

if __name__ == "__main__":
    import traceback, sys
    try:
        print("[DEBUG] __main__ guard reached; starting asyncio.run(main())", flush=True)
        asyncio.run(main())
        print("[DEBUG] main() completed", flush=True)
    except SystemExit as e:
        # argparse or normal exits still get printed
        print(f"[DEBUG] SystemExit: code={e.code}", flush=True)
        raise
    except Exception:
        print("[FATAL] Unhandled exception in generate_multiple_responses.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)
