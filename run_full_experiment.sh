# # #!/bin/bash
# # set -e

# # # --- MASTER CONFIGURATION PANEL ---
# # DOCKER_IMAGE="zack-vllm-project:final"
# # CONTAINER_NAME="vllm_server"
# # MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # We use the 1.5B model that we know works
# # MAX_TOKENS_TO_GENERATE=7992
# # MAX_MODEL_LEN=8192

# # # --- Do not change anything below this line ---

# # PROJECT_DIR="/hdd1/zack/task-hardness"
# # SCRIPT_PATH="/workspace/src/generate_multiple_responses.py"
# # CACHE_DIR="${PROJECT_DIR}/hf_cache"

# # echo "--- Step 0: Wiping previous results and ensuring cache directory exists... ---"
# # sudo rm -rf results
# # mkdir -p results "${CACHE_DIR}"
# # sudo chown -R $(id -u):$(id -g) results "${CACHE_DIR}"

# # echo "--- Step 1: Cleaning up any old server containers... ---"
# # docker rm -f ${CONTAINER_NAME} || true

# # echo "--- Step 2: Starting the new vLLM server in the background... ---"
# # docker run \
# #   --gpus all \
# #   --ipc=host \
# #   --init \
# #   --user $(id -u):$(id -g) \
# #   -d \
# #   --name ${CONTAINER_NAME} \
# #   -p 8000:8000 \
# #   -v ${PROJECT_DIR}:/workspace \
# #   -e HF_HOME="/workspace/hf_cache" \
# #   ${DOCKER_IMAGE} \
# #   python3 -m vllm.entrypoints.openai.api_server \
# #   --model "${MODEL_NAME}" \
# #   --tensor-parallel-size 1 \
# #   --gpu-memory-utilization 0.9 \
# #   --max-model-len ${MAX_MODEL_LEN} \
# #   --dtype half

# # echo "--- Step 3: Waiting for the server to be ready... ---"
# # while ! curl -s http://localhost:8000/health > /dev/null; do
# #   echo "Server is not ready yet. Waiting 5 seconds..."
# #   if ! docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
# #     echo "ERROR: The vLLM server container failed to start. Please check the logs:"
# #     docker logs ${CONTAINER_NAME}
# #     exit 1
# #   fi
# #   sleep 5
# # done

# # echo "--- Server is ready! ---"

# # echo "--- Step 4: Running the inference script... ---"
# # docker exec -it ${CONTAINER_NAME} \
# #   python3 ${SCRIPT_PATH} \
# #   --model "${MODEL_NAME}" \
# #   --max_tokens ${MAX_TOKENS_TO_GENERATE}

# # echo "--- Step 5: Inference script finished. Stopping and removing the server. ---"
# # docker rm -f ${CONTAINER_NAME}

# # echo "--- Experiment complete! ---"

# #!/bin/bash
# set -e

# # --- MASTER CONFIGURATION PANEL ---
# DOCKER_IMAGE="zack-vllm-project:final"
# CONTAINER_NAME="vllm_server"
# # This is the final, working configuration for this hardware.
# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MAX_TOKENS_TO_GENERATE=7892
# MAX_MODEL_LEN=8192

# # --- Do not change anything below this line ---

# PROJECT_DIR="/hdd1/zack/task-hardness"
# SCRIPT_PATH="/workspace/src/generate_multiple_responses.py"
# CACHE_DIR="${PROJECT_DIR}/hf_cache"

# echo "--- Step 0: Wiping previous results and ensuring cache directory exists... ---"
# sudo rm -rf results
# mkdir -p results "${CACHE_DIR}"
# sudo chown -R $(id -u):$(id -g) results "${CACHE_DIR}"

# echo "--- Step 1: Cleaning up any old server containers... ---"
# docker rm -f ${CONTAINER_NAME} || true

# echo "--- Step 2: Starting the new vLLM server in the background... ---"
# docker run \
#   --gpus all \
#   --ipc=host \
#   --init \
#   --user $(id -u):$(id -g) \
#   -d \
#   --name ${CONTAINER_NAME} \
#   -p 8000:8000 \
#   -v ${PROJECT_DIR}:/workspace \
#   -e HF_HOME="/workspace/hf_cache" \
#   ${DOCKER_IMAGE} \
#   python3 -m vllm.entrypoints.openai.api_server \
#   --model "${MODEL_NAME}" \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len ${MAX_MODEL_LEN} \
#   --dtype half

# echo "--- Step 3: Waiting for the server to be ready... ---"
# while ! curl -s http://localhost:8000/health > /dev/null; do
#   echo "Server is not ready yet. Waiting 5 seconds..."
#   if ! docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
#     echo "ERROR: The vLLM server container failed to start. Please check the logs:"
#     docker logs ${CONTAINER_NAME}
#     exit 1
#   fi
#   sleep 5
# done

# echo "--- Server is ready! ---"

# echo "--- Step 4: Running the inference script... ---"
# docker exec -it ${CONTAINER_NAME} \
#   python3 ${SCRIPT_PATH} \
#   --model "${MODEL_NAME}" \
#   --max_tokens ${MAX_TOKENS_TO_GENERATE}

# echo "--- Step 5: Inference script finished. Stopping the server. ---"
# docker rm -f ${CONTAINER_NAME}

# echo "--- Experiment complete! ---"

#!/bin/bash
set -e

# --- MASTER CONFIGURATION PANEL ---
DOCKER_IMAGE="zack-vllm-project:final"
CONTAINER_NAME="vllm_server"

# --- Server Configuration ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SERVER_MAX_CONTEXT=16384

# --- Experiment Configuration ---
N_GENS=8
TEMPERATURE=0.6
MAX_TOKENS_TO_GENERATE=16000
YEAR_FILTER="2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024"

LOGPROBS=True
TOP_LOGPROBS=5

# --- Paths ---
PROJECT_DIR="/hdd1/zack/task-hardness"
SCRIPT_PATH="/workspace/src/generate_multiple_responses.py"
CACHE_DIR="${PROJECT_DIR}/hf_cache"

echo "--- Step 0: Preparing archive directories... ---"
MASTER_RESULTS_DIR="experiment_archive"
RUN_TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
MODEL_NAME_SAFE=$(echo "${MODEL_NAME}" | tr '/' '_')
CURRENT_RUN_DIR="${MASTER_RESULTS_DIR}/${RUN_TIMESTAMP}_${MODEL_NAME_SAFE}_gpu0"
mkdir -p "${PROJECT_DIR}/${CURRENT_RUN_DIR}"
ln -sfn "${CURRENT_RUN_DIR}" "${PROJECT_DIR}/results"
sudo chown -R $(id -u):$(id -g) "${PROJECT_DIR}/${MASTER_RESULTS_DIR}"

echo "--- Step 1: Cleaning up any old server containers... ---"
docker rm -f ${CONTAINER_NAME} || true

echo "--- Step 2: Starting the new vLLM server (GPU 0) in the background... ---"
docker run \
  --gpus "device=0" \
  --ipc=host --init --user $(id -u):$(id -g) -d \
  --name ${CONTAINER_NAME} -p 8000:8000 \
  -v ${PROJECT_DIR}:/workspace -e HF_HOME="/workspace/hf_cache" \
  ${DOCKER_IMAGE} \
  python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len ${SERVER_MAX_CONTEXT} \
  --dtype half

echo "--- Step 3: Waiting for the server to be ready on http://localhost:8000 ... ---"
while ! curl -s http://localhost:8000/health > /dev/null; do
  echo "Server is not ready yet. Waiting 5 seconds..."
  if ! docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    echo "ERROR: The vLLM server container failed to start. Please check the logs:"
    docker logs ${CONTAINER_NAME}
    exit 1
  fi
  sleep 5
done
echo "--- Server is ready! ---"

echo "--- Step 4: Running the inference script... ---"
docker exec -it ${CONTAINER_NAME} bash -lc "set -x; \
  VLLM_SERVER_URL='http://127.0.0.1:8000' \
  OUTPUT_DIR_BASE='/workspace/results' \
  PYTHONUNBUFFERED=1 \
  python3 -u ${SCRIPT_PATH} \
    --output_dir '/workspace/results' \
    --model '${MODEL_NAME}' \
    --temp ${TEMPERATURE} \
    --n_gens ${N_GENS} \
    --max_tokens ${MAX_TOKENS_TO_GENERATE} \
    --server_max_context ${SERVER_MAX_CONTEXT} \
    --year_filter '${YEAR_FILTER}' \
    --logprobs ${LOGPROBS} \
    --top_logprobs ${TOP_LOGPROBS}"

echo "--- Step 5: Inference script finished. Stopping and removing the server. ---"
docker rm -f ${CONTAINER_NAME}

echo "--- Experiment complete! Results archived in ${CURRENT_RUN_DIR} ---"
