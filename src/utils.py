from together import Together
from dotenv import load_dotenv
from typing import Dict, Optional, Any, Union

# Load environment variables from .env file
load_dotenv()

# Define list of important parameters for method identification
IMPORTANT_PARAMETERS = [
    "n",
    "num_candidates",
    "num_rounds",
    "branching_factor",
    "max_depth",
    "beam_width",
]


def create_method_identifier(
    method_name: str,
    params_dict: Optional[Dict[str, Any]] = None,
    include_seed: bool = False,
    seed_value: Optional[Union[int, str]] = None,
) -> str:
    """
    Creates a consistent method identifier based on method name and important parameters.

    Args:
        method_name: The base method name
        params_dict: Dictionary of parameters (can include param_ prefix or not)
        include_seed: Whether to include seed in the identifier
        seed_value: Seed value to include if include_seed is True

    Returns:
        A standardized method identifier string
    """
    # Start with the method name
    method_id = method_name

    # Add parameters if provided
    if params_dict:
        param_list = []

        # Process each parameter and check if it's important
        for key, value in params_dict.items():
            # Remove param_ prefix if present
            param_name = key.replace("param_", "") if key.startswith("param_") else key

            # Only include important parameters
            if param_name in IMPORTANT_PARAMETERS and value is not None:
                param_list.append(f"{param_name}={value}")

        # Add parameters to method id if any exist
        if param_list:
            param_str = ", ".join(sorted(param_list))  # Sort for consistency
            method_id = f"{method_id} ({param_str})"

    # Add seed if requested
    if include_seed and seed_value is not None:
        method_id = f"{method_id} [seed={seed_value}]"

    return method_id


# Initialize the Together client
# This automatically reads the TOGETHER_API_KEY from your environment variables.
# Make sure the key is set in your environment (e.g., in a .env file loaded earlier,
# or exported in your shell: export TOGETHER_API_KEY='your_key_here')
try:
    client = Together()
except Exception as e:
    print(f"Failed to initialize Together client: {e}")
    print("Please ensure the TOGETHER_API_KEY environment variable is set correctly.")
    client = None  # Indicate client initialization failed


def generate_text(
    model,
    user_prompt,
    system_prompt=None,
    max_tokens=4096,
    temperature=1,
    terminators=(),
    seed=None,
    bias_against_tokens=None,
    bias_value=-1000000,
    use_chat_completions=True,
    repetition_penalty=1.0,
):
    """
    Generates text using the Together API. Defaults to the chat completions
    endpoint, but can use the standard completions endpoint if specified.
    System prompt is optional.

    Args:
        model: The model identifier.
        user_prompt: The user prompt string.
        system_prompt: Optional system prompt string.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        terminators: Tuple of stop sequences.
        seed: Random seed.
        bias_against_tokens: List of tokens to bias against in generation (e.g., ["...", "Quote:"]).
        bias_value: The logit bias value to apply (negative to discourage).
        use_chat_completions: Whether to use chat completions API vs standard completions.
    """
    if not client:
        print("Error: Together client is not initialized.")
        return "[ERROR: CLIENT NOT INITIALIZED]"

    # Convert terminators tuple to list for the API's 'stop' parameter
    stop_sequences = list(terminators) if terminators else None

    # Prepare list of tokens to bias against
    tokens_to_bias = []
    if bias_against_tokens:
        if isinstance(bias_against_tokens, str):
            tokens_to_bias = [bias_against_tokens]  # Convert single string to list
        else:
            tokens_to_bias = list(bias_against_tokens)  # Convert any iterable to list

    # Create logit bias dictionary to discourage specified tokens
    logit_bias = None
    if tokens_to_bias:
        logit_bias = {}
        for token in tokens_to_bias:
            token_ids = get_token_ids(model, token)
            if token_ids:
                # Find token(s) containing the specified string
                matching_tokens = {t: id for t, id in token_ids.items() if token in t}
                if matching_tokens:
                    # Apply a bias to discourage the model from generating these tokens
                    for token_id in matching_tokens.values():
                        logit_bias[str(token_id)] = bias_value
                    # print(
                    #     f"Applied bias {bias_value} to token '{token}': {matching_tokens}"
                    # )

    try:
        if use_chat_completions:
            # Use the standard chat completions endpoint
            messages = []
            # Only add system prompt if it's provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                seed=seed,
                logit_bias=logit_bias,
                repetition_penalty=repetition_penalty,
                stream=False,
            )
            # Extract the generated text from the chat completions response structure
            if response.choices:
                generated_text = response.choices[0].message.content
                return generated_text
            else:
                print("Warning: No choices returned from API.")
                return "[ERROR: NO RESPONSE]"

        else:
            # Use the standard completions endpoint
            # Construct prompt based on whether system_prompt exists
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                # Format without system prompt if it's None or empty
                full_prompt = f"{user_prompt}"

            response = client.completions.create(
                model=model,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                seed=seed,
                logit_bias=logit_bias,
                repetition_penalty=repetition_penalty,
                stream=False,
            )
            # Extract the generated text from the completions response structure
            if response.choices:
                generated_text = response.choices[0].text
                return generated_text
            else:
                print("Warning: No choices returned from API.")
                return "[ERROR: NO RESPONSE]"

    except Exception as e:
        print(f"Error during Together API call in generate_text: {e}")
        # You could add specific handling for RateLimitError, AuthenticationError etc. here
        return f"[ERROR: {type(e).__name__}]"


def get_prompt_logprobs(
    model, system_prompt, user_prompt, temperature=1.0, terminators=(), seed=None
):
    """
    Gets log probabilities for the tokens corresponding ONLY to the user_prompt.

    Args:
        model: The model identifier.
        system_prompt: The system prompt string.
        user_prompt: The user prompt string.
        temperature: Sampling temperature.
        terminators: Tuple of stop sequences.
        seed: Random seed.

    Returns:
        A tuple (user_tokens, user_logprobs) containing the lists of tokens
        and their logprobs corresponding to the user_prompt, or ([], []) if
        an error occurs or logprobs cannot be extracted.
    """
    if not client:
        print("Error: Together client is not initialized.")
        return [], []  # Return empty lists on client error

    marker = "\u200b"
    api_user_prompt = user_prompt
    append_marker = (
        user_prompt.endswith("\n")
        or user_prompt.endswith("\n\n")
        or user_prompt.endswith(" ")
    )
    if append_marker:
        # print("APPENDING MARKER")
        api_user_prompt += marker

    # Construct the messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": api_user_prompt})

    # Convert terminators tuple to list for the API's 'stop' parameter
    stop_sequences = list(terminators) if terminators else None

    # print()
    # print(f"IN get_prompt_logprobs DEBUG: api_user_prompt: {api_user_prompt}")
    # print()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,  # We don't need to generate tokens, just get prompt logprobs
            temperature=temperature,
            stop=stop_sequences,
            seed=seed,
            logprobs=True,  # Request log probabilities
            echo=True,  # Include the prompt tokens in the response object
            stream=False,
        )

        # Extract logprobs structure for the entire prompt
        if response.prompt and len(response.prompt) > 0 and response.prompt[0].logprobs:
            full_logprobs_data = response.prompt[0].logprobs
            # Now, extract only the user prompt portion
            user_tokens, user_logprobs = extract_user_prompt_logprobs(
                full_logprobs_data, user_prompt
            )
            # print()
            # print(f"IN get_prompt_logprobs DEBUG: user_tokens: {user_tokens}")
            # print()
            # print(f"IN get_prompt_logprobs DEBUG: user_logprobs: {user_logprobs}")
            # print()
            return user_tokens, user_logprobs
        else:
            print("Warning: Logprobs not found in the API response.")
            return [], []  # Return empty lists if logprobs structure is missing

    except Exception as e:
        print(f"Error during Together API call in get_prompt_logprobs: {e}")
        # You could add specific handling for RateLimitError, AuthenticationError etc. here
        return [], []  # Return empty lists on API error


def extract_user_prompt_logprobs(logprobs_data, user_prompt):
    """
    Extracts tokens and log probabilities corresponding only to the user_prompt.

    Args:
        logprobs_data: The LogprobsPart object returned by the API
                       (e.g., from response.prompt[0].logprobs).
                       Expected to have .tokens and .token_logprobs attributes.
        user_prompt: The original user prompt string to find within the tokens.

    Returns:
        A tuple (user_tokens, user_logprobs) containing the lists of tokens
        and their logprobs corresponding to the user_prompt, or ([], []) if
        the user_prompt cannot be located or an error occurs.
    """
    if (
        not logprobs_data
        or not hasattr(logprobs_data, "tokens")
        or not hasattr(logprobs_data, "token_logprobs")
    ):
        print("Error: Invalid logprobs_data object received.")
        return [], []

    all_tokens = logprobs_data.tokens
    all_logprobs = logprobs_data.token_logprobs

    # print()
    # print(f"logprobs_data: {logprobs_data}")
    # print()
    # print(f"IN extract_user_prompt_logprobs: user_prompt: {user_prompt}")
    # print()

    if len(all_tokens) != len(all_logprobs):
        print("Error: Mismatch between token count and logprob count.")
        return [], []

    # Reconstruct the full prompt string from tokens
    reconstructed_prompt = "".join(all_tokens)

    # Find the start character index of the user prompt within the reconstructed string
    # Use find() instead of index() for robustness against extra tokens appended by the API
    start_char_index = reconstructed_prompt.find(user_prompt)

    # Check if find() failed (returned -1)
    if start_char_index == -1:
        print(f"Error: Could not find user_prompt in reconstructed prompt.")
        print(f"  User Prompt: {repr(user_prompt)}")
        print(f"  Reconstructed: {repr(reconstructed_prompt)}")
        # This might happen due to unexpected tokenization differences or API formatting.
        return [], []

    end_char_index = start_char_index + len(user_prompt)

    # Find the token indices that overlap with the user prompt's character span
    user_token_indices = []
    current_char_index = 0
    for i, token in enumerate(all_tokens):
        token_start_char = current_char_index
        token_end_char = current_char_index + len(token)

        # Check for overlap: max(token_start, user_start) < min(token_end, user_end)
        if max(token_start_char, start_char_index) < min(
            token_end_char, end_char_index
        ):
            user_token_indices.append(i)

        current_char_index = token_end_char

        if token_start_char >= end_char_index:
            break

    if not user_token_indices:
        print(
            "Warning: User prompt found in string, but no overlapping tokens identified."
        )
        return [], []

    # Extract the tokens and logprobs using the identified indices
    user_tokens = [all_tokens[i] for i in user_token_indices]
    user_logprobs = [all_logprobs[i] for i in user_token_indices]

    reconstructed_user_prompt = "".join(user_tokens)
    if reconstructed_user_prompt != user_prompt:
        print(
            f"Info:Reconstructed user tokens ({repr(reconstructed_user_prompt)}) "
            f"differ slightly from original user prompt ({repr(user_prompt)}). "
            "(This is often expected due to tokenization)"
        )

    return user_tokens, user_logprobs


def get_embedding(input_text, model="BAAI/bge-large-en-v1.5"):
    """
    Generates an embedding for the given text using the Together API.

    Args:
        input_text: The text to embed.
        model: The embedding model identifier (e.g., "BAAI/bge-large-en-v1.5").

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    if not client:
        print("Error: Together client is not initialized.")
        return None  # Return None on client error

    try:
        response = client.embeddings.create(
            model=model,
            input=input_text,
        )
        # Check if the response has data and at least one embedding
        if response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        else:
            print("Warning: Embedding data not found in the API response.")
            print(f"Response object: {response}")  # Log the response for debugging
            return None  # Return None if embedding structure is missing

    except Exception as e:
        print(f"Error during Together API call in get_embedding: {e}")
        # You could add specific handling for RateLimitError, AuthenticationError etc. here
        return None  # Return None on API error


def brushup_statement_ending(
    text, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
):
    """
    Fixes only the ending of a statement without modifying any other part.
    Only makes changes if the ending is clearly problematic.

    Args:
        text: The statement to fix the ending for
        model: The model to use (defaults to Llama-3.3-70B-Instruct-Turbo-Free)

    Returns:
        The statement with the ending fixed (or unchanged if already well-formed)
    """
    if not client:
        print("Error: Together client is not initialized.")
        return text

    system_prompt = """You are helping to fix ONLY the ending of a generated statement.

VERY IMPORTANT: If the statement ending is already complete and well-formed, DO NOT modify it at all.

Your task is to:
1. DO NOT change any part of the statement except the last few sentences if they have issues
2. Look for and fix ONLY these issues at the end of the statement:
   - Remove repetition in the final sentences
   - Complete any unfinished final sentence that can be completed easily BUT DO NOT ADD ANY SUBSTANTIVE NEW CONTENT
   - Remove any incomplete final sentence that cannot be meaningfully finished
   - Remove any tokens that are not part of the statement
3. Keep the changes minimal
4. DO NOT add any new information or opinions
5. DO NOT modify anything except problematic text at the end
6. Return ONLY the statement without explanations or comments.
7. If the statement is already complete and well-formed, return it EXACTLY as provided."""

    user_prompt = f"""Here is the statement to fix:

{text}
"""

    try:
        brushed_text = generate_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.2,  # Low temperature for more predictable results
            use_chat_completions=True,
        )

        return brushed_text
    except Exception as e:
        print(f"Error during statement ending brushup: {e}")
        return text  # Return original text if brushup fails


def get_token_ids(model, text):
    """
    Gets the token IDs for a given text string using the specified model.
    NOTE: This function still uses the chat.completions endpoint with echo=True
          to retrieve token IDs, as the standard completions endpoint might not
          provide this information directly in the same way.
    Args:
        model: The model identifier.
        text: The text to tokenize.

    Returns:
        A dictionary mapping text fragments to their token IDs,
        or an empty dict if an error occurs.
    """
    if not client:
        print("Error: Together client is not initialized.")
        return {}

    try:
        # This function continues to use chat.completions as it needs echo and logprobs
        # to easily get token IDs for arbitrary text.
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,  # Only need prompt tokens
            logprobs=True,
            echo=True,  # Need echo to get prompt tokens/IDs
            stream=False,
        )

        # Extract token information
        # Check specifically for the logprobs structure within the prompt part
        if (
            response.prompt
            and len(response.prompt) > 0
            and response.prompt[0].logprobs
            and hasattr(response.prompt[0].logprobs, "tokens")
            and hasattr(response.prompt[0].logprobs, "token_ids")
        ):
            tokens = response.prompt[0].logprobs.tokens
            token_ids = response.prompt[0].logprobs.token_ids

            if len(tokens) == len(token_ids):
                # Create a mapping of token text to token ID
                token_map = {tokens[i]: token_ids[i] for i in range(len(tokens))}
                return token_map
            else:
                print("Warning: Mismatch between token count and token ID count.")
                return {}
        else:
            print(
                "Warning: Token information (logprobs.tokens/token_ids) not found in the API response prompt."
            )
            # Log the response structure for debugging if needed
            # print(f"Debug: Response structure: {response}")
            return {}

    except Exception as e:
        print(f"Error during Together API call in get_token_ids: {e}")
        return {}


if __name__ == "__main__":
    print("Running basic tests...")

    # Test generate_text (with system prompt)
    print("\nTesting generate_text (with system prompt)...")
    test_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    test_system = "You are a helpful assistant."
    test_user = "What are enchiladas?"
    # Using default (chat completions)
    generated_with_system = generate_text(
        test_model, test_user, test_system, max_tokens=20
    )
    print(f"Generated text (chat, with system): {generated_with_system}")
    if "[ERROR:" in generated_with_system:
        print("generate_text (with system) test FAILED.")
    else:
        print("generate_text (with system) test PASSED (check output).")

    # Test generate_text (without system prompt)
    print("\nTesting generate_text (without system prompt)...")
    # Using default (chat completions), system_prompt=None
    generated_no_system = generate_text(test_model, test_user, max_tokens=20)
    print(f"Generated text (chat, no system): {generated_no_system}")
    if "[ERROR:" in generated_no_system:
        print("generate_text (no system) test FAILED.")
    else:
        print("generate_text (no system) test PASSED (check output).")

    # Test generate_text (completions endpoint, without system prompt)
    prompt = "What are ench"

    print("\nTesting generate_text (completions endpoint, no system prompt)...")
    generated_comp_no_system = generate_text(
        test_model, prompt, max_tokens=50, use_chat_completions=False
    )
    print(
        f"Generated text (completions, no system): {prompt}{generated_comp_no_system}"
    )
    if "[ERROR:" in generated_comp_no_system:
        print("generate_text (completions, no system) test FAILED.")
    else:
        print("generate_text (completions, no system) test PASSED (check output).")

    # Test get_prompt_logprobs (which now includes extraction)
    print("\nTesting get_prompt_logprobs (user prompt extraction)...")
    # The function now directly returns the tokens and logprobs for the user prompt
    user_tokens, user_logprobs = get_prompt_logprobs(test_model, test_system, test_user)

    # Check if the extraction was successful (returned non-empty lists)
    if user_tokens:
        print(f"Successfully extracted {len(user_tokens)} tokens for the user prompt:")
        # Print the first few extracted tokens/logprobs
        for i, (token, logprob) in enumerate(
            list(zip(user_tokens, user_logprobs))[:10]
        ):  # Show up to 10
            logprob_str = f"{logprob:.4f}" if logprob is not None else "None"
            print(f"    {i}: {repr(token)} -> {logprob_str}")
        print("get_prompt_logprobs test PASSED (extraction successful).")
    else:
        # This means either the API call failed, logprobs weren't returned,
        # or the extraction within get_prompt_logprobs failed. Check logs above.
        print("\nExtraction of user prompt logprobs FAILED (see errors above).")
        print("get_prompt_logprobs test FAILED.")

    # Test get_embedding
    print("\nTesting get_embedding...")
    test_text = "This is a test sentence for embedding."
    embedding = get_embedding(test_text)
    if embedding is not None:
        print(f"Successfully got embedding of length {len(embedding)}")
        print(f"First few values: {embedding[:5]}")
        print("get_embedding test PASSED.")
    else:
        print("get_embedding test FAILED (see errors above).")

    # Test get_token_ids for finding specific tokens
    print("\nTesting get_token_ids...")
    test_model_token_ids = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Use a specific model for consistency
    test_text_token_ids = '"'

    # Get token mappings
    token_map = get_token_ids(test_model_token_ids, test_text_token_ids)

    if not token_map:
        print(
            f"get_token_ids test FAILED - could not get token mappings for {repr(test_text_token_ids)}"
        )
    else:
        # Print just the test text token and ID if found
        if test_text_token_ids in token_map:
            print(
                f"Token ID for {repr(test_text_token_ids)}: {token_map[test_text_token_ids]}"
            )
        else:
            # The exact test_text might not be a single token, so print all mappings
            print(
                f"Test text '{test_text_token_ids}' not found as a single token. All mappings:"
            )
            for token, token_id in token_map.items():
                print(f"    {repr(token)} -> {token_id}")
        print("get_token_ids test PASSED (check output).")

    # Test Llama 3 Special Tokens
    print("\nTesting Llama 3 Special Token IDs...")
    llama3_model = (
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Or another Llama 3 model
    )
    special_tokens = [
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|end_of_text|>",
        "DONE",
        # You can add role tokens too if needed, e.g., "system", "user", "assistant"
        # Note: Role names themselves might not be special *tokens* but part of the structure
    ]

    all_tokens_found = True
    for token_text in special_tokens:
        print(f"  Checking token: {repr(token_text)}")
        token_map = get_token_ids(llama3_model, token_text)
        if not token_map:
            print(f"    -> FAILED to get token map for {repr(token_text)}")
            all_tokens_found = False
        elif token_text in token_map:
            print(f"    -> Found ID: {token_map[token_text]}")
        else:
            # Check if the token was split into multiple parts
            reconstructed = "".join(token_map.keys())
            if reconstructed == token_text:
                print(f"    -> Found as multiple tokens:")
                for part, part_id in token_map.items():
                    print(f"        {repr(part)} -> {part_id}")
            else:
                print(
                    f"    -> WARNING: Token {repr(token_text)} not found as a single token or exact reconstruction."
                )
                print(f"       Returned map: {token_map}")
                # It's possible the API doesn't return special tokens this way,
                # or they are handled differently internally.
                # We won't mark this as a hard failure for now.

    if all_tokens_found:
        print(
            "Llama 3 Special Token ID test PASSED (or partially passed, check warnings)."
        )
    else:
        print(
            "Llama 3 Special Token ID test FAILED (some tokens could not be processed)."
        )
