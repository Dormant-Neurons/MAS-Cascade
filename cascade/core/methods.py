import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import time
import random

load_dotenv()

# Set up logging for Gemini adapter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeminiAdapter')


class GeminiAdapter:
    """
    Adapter that wraps Google Gemini API to mimic OpenAI's client interface.
    This allows using Gemini without modifying experimental logic.
    """

    def __init__(self, project_id: str, location: str = "global"):
        from google import genai
        from google.genai import types

        # Increase timeout to 300 seconds (5 minutes) to handle large prompts
        http_options = types.HttpOptions(timeout=300_000)  # timeout in milliseconds

        self.gemini_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=http_options
        )
        # Create nested structure to match OpenAI's client.chat.completions.create()
        self.chat = self._ChatNamespace(self.gemini_client)

    class _ChatNamespace:
        def __init__(self, gemini_client):
            self.gemini_client = gemini_client
            self.completions = self._CompletionsNamespace(gemini_client)

        class _CompletionsNamespace:
            def __init__(self, gemini_client):
                self.gemini_client = gemini_client

            def create(self, model: str, messages: list, **kwargs):
                """
                Mimics OpenAI's chat.completions.create() interface.

                Args:
                    model: Model name (e.g., "gemini-2.5-pro", "gemini-3-flash-preview")
                    messages: List of message dicts with 'role' and 'content'
                    **kwargs: Additional parameters (max_tokens, temperature, etc.)

                Returns:
                    OpenAI-compatible response object
                """
                # IMPROVEMENT 1: Separate system instruction from user contents
                # This ensures Gemini treats system prompts as instructions, not data
                system_instruction = None
                user_contents = []

                for msg in messages:
                    if msg["role"] == "system":
                        # Use dedicated system_instruction parameter for better adherence
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        user_contents.append(msg["content"])

                # Join user messages (typically there's only one)
                final_user_prompt = "\n\n".join(user_contents)

                # Map OpenAI parameters to Gemini parameters
                from google.genai import types

                # Build config with proper system_instruction parameter
                config_params = {}
                temp = kwargs.get("temperature", None)
                if temp is not None:
                    config_params["temperature"] = temp

                # Add system_instruction if present
                if system_instruction:
                    config_params["system_instruction"] = system_instruction

                # IMPROVEMENT 2: Use correct thinking parameters for each generation
                # Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget
                if "gemini-3" in model:
                    # Gemini 3 Flash supports "minimal" for near-zero reasoning
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_level="minimal"
                    )
                elif "gemini-2.5" in model:
                    # Gemini 2.5 does NOT support "minimal"/"thinking_level"
                    # Use thinking_budget=0 to disable thinking for fair comparison
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=0
                    )

                # IMPROVEMENT 3: Add small jitter delay to prevent thundering herd
                # This spreads out concurrent API calls to avoid rate limits
                jitter = random.uniform(0.1, 0.5)
                time.sleep(jitter)

                logger.info(f"Calling Gemini API with model={model}, prompt_length={len(final_user_prompt)}, system_instruction_length={len(system_instruction) if system_instruction else 0}")

                # Call Gemini API with error handling
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=final_user_prompt,
                        config=types.GenerateContentConfig(**config_params)
                    )

                    logger.info(f"Gemini API call succeeded")

                    # Convert Gemini response to OpenAI format
                    return self._convert_response(response, final_user_prompt)

                except Exception as e:
                    logger.error(f"Gemini API call failed: {type(e).__name__}: {e}")
                    logger.error(f"User prompt (first 500 chars): {final_user_prompt[:500]}")
                    if system_instruction:
                        logger.error(f"System instruction (first 200 chars): {system_instruction[:200]}")
                    raise

            def _convert_response(self, gemini_response, original_prompt):
                """Convert Gemini response to OpenAI-compatible format with error handling."""
                class Message:
                    def __init__(self, content):
                        self.content = content

                class Choice:
                    def __init__(self, message):
                        self.message = message

                class Response:
                    def __init__(self, text):
                        self.choices = [Choice(Message(text))]

                # Check if response has text attribute
                if not hasattr(gemini_response, 'text'):
                    logger.error("Gemini response has no .text attribute")
                    logger.error(f"Response type: {type(gemini_response)}")
                    logger.error(f"Response dir: {dir(gemini_response)}")

                    # Check for candidates (common Gemini response structure)
                    if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
                        candidate = gemini_response.candidates[0]
                        logger.error(f"Candidate finish_reason: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")

                        # Check if content was blocked by safety filters
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = str(candidate.finish_reason)
                            if 'SAFETY' in finish_reason or 'BLOCKED' in finish_reason:
                                logger.error(f"Content blocked by Gemini safety filters: {finish_reason}")
                                logger.error(f"Prompt (first 1000 chars): {original_prompt[:1000]}")

                                # Check for safety ratings
                                if hasattr(candidate, 'safety_ratings'):
                                    logger.error(f"Safety ratings: {candidate.safety_ratings}")

                        # Try to extract content from candidate
                        if hasattr(candidate, 'content'):
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                text = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else ""
                                if text:
                                    logger.warning(f"Extracted text from candidate.content.parts: length={len(text)}")
                                    return Response(text)

                    # If we still can't get text, return empty string
                    logger.error("Could not extract any text from Gemini response, returning empty string")
                    return Response("")

                # Normal case: response has .text
                text = gemini_response.text
                logger.info(f"Successfully extracted text from Gemini response: length={len(text)}")

                # Log first 500 chars of response for debugging
                logger.debug(f"Response preview: {text[:500]}")

                return Response(text)


def get_client(openai_api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'), backend='openai', base_url=None):
    """
    Get a client for LLM inference.

    Args:
        openai_api_key: API key for OpenAI (default from env)
        backend: 'openai', 'vllm', 'gemini', or 'blablador' (default: 'openai')
        base_url: Base URL for vLLM/Blablador server

    Returns:
        OpenAI-compatible client (or adapter for Gemini)
    """
    if backend == 'vllm':
        if base_url is None:
            base_url = os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')
        client = OpenAI(
            api_key=openai_api_key,  # vLLM doesn't require a real key but needs something
            base_url=base_url
        )
    elif backend == 'blablador':
        # Blablador API (OpenAI-compatible)
        blablador_api_key = os.environ.get('BLABLADOR_API_KEY', openai_api_key)
        if base_url is None:
            base_url = os.environ.get('BLABLADOR_BASE_URL', 'https://api.helmholtz-blablador.fz-juelich.de/v1')
        client = OpenAI(
            api_key=blablador_api_key,
            base_url=base_url
        )
    elif backend == 'gemini':
        project_id = os.environ.get('GEMINI_PROJECT_ID')
        location = os.environ.get('GEMINI_LOCATION', 'global')
        if not project_id:
            raise ValueError("GEMINI_PROJECT_ID must be set in .env file for Gemini backend")
        client = GeminiAdapter(project_id=project_id, location=location)
    else:
        client = OpenAI(api_key=openai_api_key)
    return client


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('')


def generate_adj(n, graph_type):
    """
    Return an n x n adjacency matrix (0/1) for the requested graph_type.

    Supported types:
      - "complete"   : complete graph K_n
      - "tree"       : binary tree in array layout (0-based)
      - "chain"      : path graph P_n
      - "circle"     : cycle graph C_n
      - "star"       : (your original) hub at 0 PLUS a ring among leaves (a wheel W_n)
      - "pure_star"  : hub at 0, leaves connect ONLY to hub (true star S_n)
    """
    if n < 2:
        # trivial graph
        return np.zeros((n, n), dtype=int)

    if graph_type == "complete":
        adj_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj_matrix, 0)

    elif graph_type == "tree":
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            if left_child < n:
                adj_matrix[i, left_child] = 1
                adj_matrix[left_child, i] = 1
            if right_child < n:
                adj_matrix[i, right_child] = 1
                adj_matrix[right_child, i] = 1

    elif graph_type == "chain":
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1

    elif graph_type == "circle":
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            j = (i + 1) % n
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    elif graph_type == "pure_star":
        # True star S_n: hub=0, leaves 1..n-1, NO leaf–leaf edges
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj_matrix[0, i] = 1
            adj_matrix[i, 0] = 1

    elif graph_type == "star":
        # Your original implementation: hub at 0, plus a ring among leaves → wheel W_n
        adj_matrix = np.zeros((n, n), dtype=int)
        # hub connections
        for i in range(1, n):
            adj_matrix[0, i] = 1
            adj_matrix[i, 0] = 1
        # ring over leaves (1..n-1)
        for i in range(1, n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
        adj_matrix[1, n - 1] = 1
        adj_matrix[n - 1, 1] = 1

    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'. "
                         "Valid: complete, tree, chain, circle, star, pure_star")

    return adj_matrix


def model_api_kwargs(model: str, backend: str, max_tokens: int, temperature: float | None) -> dict:
    """Return model-specific API kwargs for the given model/backend.

    Newer OpenAI models (gpt-5-*, o1-*, o3-*) require
    ``max_completion_tokens`` instead of ``max_tokens`` and only
    support ``temperature=1``.

    When *temperature* is ``None``, the key is omitted entirely so the
    model's own API default is used.
    """
    _NEW_OPENAI = ("gpt-5", "o1-", "o3-")
    if backend == "openai" and any(t in model for t in _NEW_OPENAI):
        return {"max_completion_tokens": max_tokens, "temperature": 1}
    kwargs: dict = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def get_dataset(ds_path):
    dataset = []
    with open(ds_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                import ast
                item = ast.literal_eval(line.strip())
            dataset.append(item)
    return dataset
