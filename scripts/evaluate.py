"""
Evaluate and Compare SAGE vs Neuronpedia Feature Explanations

This script implements TWO evaluation methods:

METHOD 1: Generation Evaluation
    1. Extract feature explanation (SAGE or Neuronpedia)
    2. Use GPT-5 to generate test examples from explanation
    3. Test if generated examples actually activate the feature
    4. Measure success rate and activation levels

METHOD 2: Prediction Evaluation
    1. Generate diverse test examples
    2. Get their actual activation values
    3. Split into training (5) and test (5) sets
    4. Ask GPT-5 to predict test set activations from training examples + description
    5. Calculate Pearson correlation between predicted and actual

Both methods test if the explanation truly captures feature behavior.

Requirements:
    pip install openai requests scipy numpy
"""

import os
import sys
import json
import argparse
import requests
import time
import numpy as np
import re
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from openai import OpenAI

# Import required libraries
from scipy.stats import pearsonr
from dataclasses import dataclass

# Lazy initialization for OpenAI client (to avoid requiring API key at import time)
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client (lazy initialization)"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


@dataclass
class TextWithActivation:
    """Text example with activation data"""
    text: str
    true_activation: float
    predicted_activation: float = 0.0


# ============================================================================
# LogProbs-based Prediction Helper Functions
# (From test_logprobs_evaluation.py)
# ============================================================================

def get_activation_logprobs(explanation: str, token: str, context_before: str = "") -> Tuple[dict, Dict[str, int]]:
    """
    Get activation value prediction logprobs for a single token.

    Uses GPT-4 with logprobs to predict activation values (0-10) for a token
    based on the feature explanation and context.

    Args:
        explanation: Natural language explanation of neuron behavior
        token: Token to predict activation value for
        context_before: Context before the token

    Returns:
        dict: {activation_value: probability}
    """
    # Construct prompt for LLM to predict activation value (0-10)
    prompt = f"""You are a neuron simulator.

Neuron behavior rules: {explanation}

Task: Predict the activation value (0-10 integer, 0 means no activation, 10 means strongest activation) for a token.

Context: {context_before if context_before else "None"}
Current token: {token}

Please output only a number between 0-10, representing the activation value:"""

    # Call Chat API with logprobs
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=10,  # Get top 20 most likely tokens
        max_tokens=1,      # Only need first token
        temperature=1.0    # Use default temperature for true probability distribution
    )

    # Get token usage
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
        'total_tokens': response.usage.total_tokens if response.usage else 0
    }
    
    # Calculate cost for GPT-4o
    # Prices: Input $1.25/1M, Output $5.00/1M
    cost_input = (token_usage['prompt_tokens'] / 1_000_000) * 1.25
    cost_output = (token_usage['completion_tokens'] / 1_000_000) * 5.00
    cost_total = cost_input + cost_output
    token_usage['cost_input'] = cost_input
    token_usage['cost_output'] = cost_output
    token_usage['cost_total'] = cost_total
    
    # Parse logprobs
    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        print(f"Warning: No logprobs obtained for token '{token}'")
        return {}, token_usage

    # Get top logprobs for first generated token
    first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs

    # Extract 0-10 probability distribution
    activation_probs = {}

    for logprob_item in first_token_logprobs:
        token_str = logprob_item.token.strip()
        logprob_value = logprob_item.logprob

        # Check if it's a digit 0-10
        if token_str.isdigit():
            activation_value = int(token_str)
            if 0 <= activation_value <= 10:
                # Convert from log probability to probability: P = e^(log P)
                probability = math.exp(logprob_value)
                activation_probs[activation_value] = probability

    return activation_probs, token_usage


def compute_expected_activation(activation_probs: dict) -> float:
    """
    Calculate expected activation value from probability distribution.

    Args:
        activation_probs: {activation_value: probability}

    Returns:
        Expected activation value E[X] = Σ k * P(X=k)
    """
    if not activation_probs:
        return 0.0

    # Normalize probabilities (since we only have top-k, not full distribution)
    total_prob = sum(activation_probs.values())
    normalized_probs = {k: v / total_prob for k, v in activation_probs.items()}

    # Calculate expected value
    expected_value = sum(k * prob for k, prob in normalized_probs.items())

    return expected_value


def predict_activations_with_logprobs(
    explanation: str,
    tokens: List[str],
    show_details: bool = False
) -> Tuple[List[float], Dict[str, int]]:
    """
    Use logprobs to predict activation values for a series of tokens.

    Args:
        explanation: Neuron behavior explanation
        tokens: List of tokens
        show_details: Whether to print detailed information

    Returns:
        Tuple of (predicted activation values, token usage dict)
    """
    predicted_activations = []
    context = ""
    total_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    for i, token in enumerate(tokens):
        # Get logprobs probability distribution
        activation_probs, token_usage = get_activation_logprobs(explanation, token, context)
        
        # Accumulate token usage
        total_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
        total_token_usage['completion_tokens'] += token_usage['completion_tokens']
        total_token_usage['total_tokens'] += token_usage['total_tokens']

        # Calculate expected activation value
        expected_activation = compute_expected_activation(activation_probs)
        predicted_activations.append(expected_activation)

        if show_details:
            print(f"\nToken {i+1}: '{token}'")
            print(f"  Context: {context if context else '(start)'}")

            # Display probability distribution
            if activation_probs:
                print(f"  LogProbs distribution:")
                total_prob = sum(activation_probs.values())
                sorted_probs = sorted(activation_probs.items(), key=lambda x: x[1], reverse=True)

                for act_val, prob in sorted_probs[:5]:  # Show top 5
                    normalized_prob = prob / total_prob
                    print(f"    Activation {act_val}: probability {normalized_prob:.3f}")

                print(f"  → Expected activation: {expected_activation:.2f}")
            else:
                print(f"  → Expected activation: {expected_activation:.2f} (no logprobs)")

        # Update context
        context += (" " if context else "") + token

    return predicted_activations, total_token_usage


# ============================================================================


def delete_neuronpedia_explanation(
    explanation_id: str,
    api_key: str = None
) -> bool:
    """
    Delete a Neuronpedia explanation by ID.
    
    API Documentation: https://www.neuronpedia.org/api-doc#tag/explanations/post/api/explanation/{explanationId}/delete
    
    Args:
        explanation_id: Explanation ID to delete
        api_key: Neuronpedia API key
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    url = f"https://www.neuronpedia.org/api/explanation/{explanation_id}/delete"
    
    headers = {
        "x-api-key": api_key
    }
    
    try:
        print(f"   🗑️  Deleting existing explanation (ID: {explanation_id})...")
        response = requests.post(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print(f"   ✅ Explanation successfully deleted")
            return True
        else:
            print(f"   ⚠️  Failed to delete explanation: status {response.status_code}")
            try:
                error_data = response.json()
                print(f"      Message: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"      Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error deleting explanation: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error deleting explanation: {e}")
        return False


def call_neuronpedia_api(
    model_id: str,
    layer: str,
    feature_index: int,
    explanation_model_name: str = "gpt-5",
    explanation_type: str = "oai_token-act-pair",
    api_key: str = None,
    output_dir: str = None,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Call Neuronpedia API to generate feature explanation.

    API Documentation: https://www.neuronpedia.org/api-doc#tag/explanations/post/api/explanation/generate

    Args:
        model_id: Model identifier (e.g., "llama3.1-8b-it", "gemma-2-2b")
        layer: Layer identifier as string (e.g., "11-resid-post-aa", "0-gemmascope-mlp-16k")
        feature_index: Feature index (integer)
        explanation_model_name: Explanation model name (default: "gpt-5")
        explanation_type: Explanation type (default: "oai_token-act-pair")
        api_key: Neuronpedia API key
        output_dir: Output directory to check if explanation file exists locally
        force_regenerate: If True, delete existing explanation and regenerate (default: False)

    Returns:
        dict containing explanation and metadata, or None if failed
    """
    url = "https://www.neuronpedia.org/api/explanation/generate"

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key
    }

    payload = {
        "modelId": model_id,
        "layer": layer,
        "index": feature_index,
        "explanationModelName": explanation_model_name,
        "explanationType": explanation_type
    }

    print(f"\n📡 Calling Neuronpedia API...")
    print(f"   Model: {model_id}, Layer: {layer}, Feature: {feature_index}")
    print(f"   Explanation Model: {explanation_model_name}")
    print(f"   Endpoint: {url}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        print(f"   Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Neuronpedia API call successful (new explanation created)")
            # API returns: {"explanation": {"id": "...", "description": "...", ...}}
            # Mark as newly created
            if isinstance(result, dict):
                result["source"] = "new"
            return result
        elif response.status_code == 400:
            # Check if explanation already exists (this is normal and expected)
            try:
                error_data = response.json()
                error_message = error_data.get("message", "").lower()
                
                if "already exists" in error_message:
                    print(f"   ⚠️  Explanation already exists (this is normal)")
                    print(f"   Message: {error_data.get('message')}")
                    print(f"   💡 Trying to fetch existing explanation from API...")
                    
                    # Try to get the existing explanation using GET API
                    get_url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{feature_index}"
                    try:
                        get_response = requests.get(get_url, timeout=30)
                        if get_response.status_code == 200:
                            feature_data = get_response.json()
                            explanations = feature_data.get("explanations", [])
                            
                            print(f"   📋 Found {len(explanations)} explanation(s) in API response")
                            
                            # Show all available explanations
                            for i, exp in enumerate(explanations, 1):
                                exp_model = exp.get("explanationModelName", "unknown")
                                exp_type = exp.get("typeName", "unknown")
                                exp_desc = exp.get("description", "")[:80] + "..." if len(exp.get("description", "")) > 80 else exp.get("description", "")
                                print(f"      {i}. Model: {exp_model}, Type: {exp_type}")
                                print(f"         Description: {exp_desc}")
                            
                            # Find the matching explanation
                            matching_exp = None
                            for exp in explanations:
                                if (exp.get("explanationModelName") == explanation_model_name and
                                    exp.get("typeName") == explanation_type):
                                    matching_exp = exp
                                    break
                            
                            if matching_exp:
                                explanation_id = matching_exp.get('id', 'N/A')
                                print(f"   ✅ Found matching explanation (Model: {explanation_model_name}, Type: {explanation_type})")
                                print(f"      Description: {matching_exp.get('description', '')[:100]}...")
                                print(f"      ID: {explanation_id}")
                                if matching_exp.get('scores'):
                                    print(f"      Scores: {matching_exp.get('scores')}")
                                if matching_exp.get('triggeredByUser'):
                                    user_info = matching_exp.get('triggeredByUser', {})
                                    print(f"      Triggered by: {user_info.get('name', 'N/A')} (ID: {user_info.get('id', 'N/A')})")
                                
                                # Check if explanation file exists locally
                                explanation_file_exists = False
                                if output_dir:
                                    explanation_file = os.path.join(output_dir, f'feature_{feature_index}_neuronpedia_explanation.json')
                                    explanation_file_exists = os.path.exists(explanation_file)
                                
                                # If force_regenerate is True, or if explanation doesn't exist locally, delete and regenerate
                                if force_regenerate or not explanation_file_exists:
                                    if not explanation_file_exists:
                                        print(f"   💡 Explanation file not found locally: {explanation_file}")
                                        print(f"      Will delete existing explanation and regenerate...")
                                    else:
                                        print(f"   💡 Force regenerate requested, will delete existing explanation and regenerate...")
                                    
                                    # Delete the existing explanation
                                    if delete_neuronpedia_explanation(explanation_id, api_key):
                                        # Wait a bit for deletion to propagate
                                        import time
                                        time.sleep(2)
                                        
                                        # Retry generating the explanation
                                        print(f"   🔄 Retrying explanation generation...")
                                        retry_response = requests.post(url, headers=headers, json=payload, timeout=120)
                                        
                                        if retry_response.status_code == 200:
                                            result = retry_response.json()
                                            print(f"   ✅ Neuronpedia API call successful (new explanation created after deletion)")
                                            if isinstance(result, dict):
                                                result["source"] = "new"
                                            return result
                                        else:
                                            # Deletion succeeded but regeneration failed
                                            print(f"   ❌ Retry failed with status {retry_response.status_code}")
                                            try:
                                                error_data = retry_response.json()
                                                error_msg = error_data.get('message', 'Unknown error')
                                                print(f"      Error: {error_msg}")
                                            except:
                                                print(f"      Response: {retry_response.text[:200]}")
                                            print(f"   ⚠️  Explanation was deleted but regeneration failed. This may be a temporary API issue.")
                                            return None  # Return None to indicate failure
                                    else:
                                        print(f"   ⚠️  Failed to delete explanation, will use existing one")
                                
                                # Return complete data for export
                                return {
                                    "explanation": matching_exp,
                                    "source": "existing",
                                    "all_explanations": explanations,  # All explanations for reference
                                    "feature_data": feature_data  # Complete feature data (optional, for additional info)
                                }
                            else:
                                # If no exact match, show available options and use first one
                                print(f"   ⚠️  No exact match found for Model: {explanation_model_name}, Type: {explanation_type}")
                                if explanations:
                                    print(f"   💡 Using first available explanation instead")
                                    first_exp = explanations[0]
                                    print(f"      Model: {first_exp.get('explanationModelName')}, Type: {first_exp.get('typeName')}")
                                    return {
                                        "explanation": first_exp,
                                        "source": "existing_fallback",
                                        "all_explanations": explanations,
                                        "feature_data": feature_data,
                                        "warning": f"Requested {explanation_model_name}/{explanation_type} not found, using first available"
                                    }
                                else:
                                    print(f"   ❌ No explanations available in API response")
                                    return {"error": "explanation_exists_but_not_found", "message": error_data.get("message")}
                    except Exception as e:
                        print(f"   ⚠️  Could not fetch existing explanation: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Return error info if we can't get existing explanation
                    return {"error": "explanation_exists", "message": error_data.get("message")}
                elif "unsupported explanation model" in error_message:
                    print(f"   ❌ Unsupported explanation model: {explanation_model_name}")
                    print(f"   Message: {error_data.get('message')}")
                    print(f"   💡 Try using: gpt-5, gpt-4o, or o4-mini")
                    return None
                else:
                    print(f"   ❌ API call failed with status 400")
                    print(f"   Message: {error_data.get('message', 'Unknown error')}")
                    return None
            except json.JSONDecodeError:
                print(f"   ❌ Failed to parse error response")
                print(f"   Response: {response.text}")
                return None
        else:
            print(f"   ❌ API call failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"   ❌ Neuronpedia API call timed out after 120 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Neuronpedia API call failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_text = e.response.text
                print(f"   Response: {error_text}")
            except:
                pass
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None


def extract_sage_conclusion(results_path: str) -> Dict[str, Any]:
    """
    Extract conclusion from SAGE structured results (supports multiple labels).

    Args:
        results_path: Path to SAGE's structured_results.json

    Returns:
        dict with keys: description, evidence, labels, label1, label2
    """
    print(f"\n📖 Loading SAGE results from {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Extract final conclusion from analysis_history
    analysis_history = results.get('analysis_history', [])

    if not analysis_history:
        raise ValueError("No analysis history found in SAGE results")

    # Get the last analysis (should be FINAL_CONCLUSION)
    final_analysis = analysis_history[-1]

    # Parse the conclusion using regex
    import re

    # Extract description
    desc_match = re.search(r'\[DESCRIPTION\]:\s*(.*?)(?:\n\n\[EVIDENCE\]:|$)', final_analysis, re.DOTALL)

    # Extract evidence
    evidence_match = re.search(r'\[EVIDENCE\]:\s*(.*?)(?:\n\n\[LABEL)', final_analysis, re.DOTALL)

    # Extract all labels (supports LABEL 1, LABEL 2, LABEL 3, ...)
    labels = []
    label_pattern = r'\[LABEL\s+(\d+)\]:\s*(.*?)(?=\n\[LABEL\s+\d+\]:|$)'
    for match in re.finditer(label_pattern, final_analysis, re.DOTALL):
        label_num = int(match.group(1))
        label_text = match.group(2).strip()
        labels.append({
            'number': label_num,
            'text': label_text
        })

    conclusion = {
        'description': desc_match.group(1).strip() if desc_match else "",
        'evidence': evidence_match.group(1).strip() if evidence_match else "",
        'labels': labels,  # Now a list of all labels
        # Backward compatibility
        'label1': labels[0]['text'] if len(labels) > 0 else "",
        'label2': labels[1]['text'] if len(labels) > 1 else "",
    }

    print(f"   ✅ Extracted SAGE conclusion:")
    print(f"      Description: {conclusion['description'][:100]}...")

    # Print all labels
    if labels:
        for label in labels:
            print(f"      Label {label['number']}: {label['text'][:80]}...")
    else:
        print(f"      No labels found")

    return conclusion


def extract_layer_number(layer: str) -> int:
    """
    Extract layer number from layer identifier string.
    
    Args:
        layer: Layer identifier string (e.g., "11-resid-post-aa", "0-gemmascope-mlp-16k", "0")
    
    Returns:
        Layer number as integer
    """
    # Try to extract the first number from the string
    import re
    match = re.search(r'^(\d+)', layer)
    if match:
        return int(match.group(1))
    # If no number found, try to parse the entire string as int
    try:
        return int(layer)
    except ValueError:
        # Default to 0 if cannot parse
        return 0


def format_neuronpedia_explanation(neuronpedia_result: Dict[str, Any]) -> Dict[str, str]:
    """
    Format Neuronpedia explanation to match SAGE structure.

    Args:
        neuronpedia_result: Raw result from Neuronpedia API
            Expected format: {"explanation": {"description": "...", ...}}
            Or error format: {"error": "...", "message": "..."}

    Returns:
        dict with keys: description, evidence, label1, label2

    Raises:
        ValueError: If neuronpedia_result is None or contains an error
        KeyError: If required fields are missing from the response
    """
    # Check for error cases - raise exceptions instead of returning fake data
    if neuronpedia_result is None:
        raise ValueError("Neuronpedia API returned None - failed to get explanation")
    
    if isinstance(neuronpedia_result, dict) and 'error' in neuronpedia_result:
        error_msg = neuronpedia_result.get('message', 'Unknown error')
        raise ValueError(f"Neuronpedia API returned error: {error_msg}")

    # Extract explanation text from API response
    # API returns: {"explanation": {"description": "...", "explanationModelName": "...", ...}}
    explanation_text = ""

    if isinstance(neuronpedia_result, dict):
        # Try to get explanation object
        explanation_obj = neuronpedia_result.get('explanation')
        
        if explanation_obj and isinstance(explanation_obj, dict):
            # Extract description from explanation object
            explanation_text = explanation_obj.get('description', '')
            explanation_model = explanation_obj.get('explanationModelName', 'unknown')
            explanation_type = explanation_obj.get('typeName', 'unknown')
        else:
            # Fallback: try direct access (for backward compatibility)
            explanation_text = (
                neuronpedia_result.get('description', '') or
                neuronpedia_result.get('text', '') or
                None
            )
    else:
        raise ValueError(f"Invalid neuronpedia_result type: {type(neuronpedia_result)}, expected dict")

    if not explanation_text:
        raise ValueError("Neuronpedia API response missing description field - cannot extract explanation")

    return {
        'description': explanation_text,
        'label1': "",
        'label2': ""
    }


def get_activation_exemplars_from_api(
    model_id: str,
    source: str,
    feature_index: int,
    top_k: int = None,
    buffer_size: int = 5,
    return_all: bool = False
) -> List[Dict[str, Any]]:
    """
    Get activation exemplars from Neuronpedia API.
    
    Args:
        model_id: Model ID (e.g., "gpt2-small", "gemma-2-2b")
        source: SAE ID/source (e.g., "9-res-jb", "6-gemmascope-mlp-16k")
        feature_index: Feature index
        top_k: Number of top examples to retrieve (if None and return_all=False, returns top 10)
        buffer_size: Number of tokens to show before/after max activation token (default: 50)
                     This determines the context window around the max activation token
        return_all: If True, return all exemplars from API (sorted by activation descending)
                    If False, return top_k exemplars (default: top 10 if top_k is None)
    
    Returns:
        List of dicts with keys: {
            'text': str (buffer text around max activation token),
            'activation': float (max activation value),
            'max_token': str (token with max activation),
            'max_token_index': int (index of max token in buffer),
            'tokens': List[str] (tokens in buffer region),
            'values': List[float] (activation values for tokens in buffer),
            'full_text': str (original full text from API, no filtering)
        }, sorted by activation (descending)
    
    Raises:
        requests.exceptions.RequestException: If API call fails
        ValueError: If API returns invalid response
    """
    url = "https://www.neuronpedia.org/api/activation/get"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "modelId": model_id,
        "source": source,
        "index": str(feature_index)
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # API returns a list of activation objects
        if not isinstance(result, list):
            raise ValueError(f"Expected list from API, got {type(result)}")
        
        # Process each activation object
        exemplars = []
        
        for activation_obj in result:
            if not isinstance(activation_obj, dict):
                continue
            
            # Extract tokens and values from API (original, no filtering)
            tokens = activation_obj.get('tokens', [])
            values = activation_obj.get('values', [])
            max_value = activation_obj.get('maxValue', 0.0)
            max_value_token_index = activation_obj.get('maxValueTokenIndex', 0)
            
            # Skip if no tokens or values
            if not tokens or not values:
                continue
            
            # Validate index
            if max_value_token_index >= len(tokens) or max_value_token_index >= len(values):
                # Invalid index, skip this exemplar
                continue
            
            # Get the original full text (no filtering)
            full_text = ''.join(tokens)
            
            # Get max token from original tokens
            max_token = tokens[max_value_token_index] if max_value_token_index < len(tokens) else ""
            
            # Build buffer region around max activation token
            # buffer_size tokens before and after the max token
            start_idx = max(0, max_value_token_index - buffer_size)
            end_idx = min(len(tokens), max_value_token_index + buffer_size + 1)
            
            # Extract buffer tokens and values
            buffer_tokens = tokens[start_idx:end_idx]
            buffer_values = values[start_idx:end_idx]
            
            # Calculate relative index of max token in buffer
            buffer_max_idx = max_value_token_index - start_idx
            
            # Build buffer text (concatenate buffer tokens)
            buffer_text = ''.join(buffer_tokens)
            
            # Use maxValue as the activation value
            activation = float(max_value)
            
            exemplars.append({
                'text': buffer_text,  # Buffer text around max activation token
                'activation': activation,  # Max activation value
                'max_token': max_token,  # Token with max activation
                'max_token_index': buffer_max_idx,  # Index of max token in buffer
                'tokens': buffer_tokens,  # Tokens in buffer region
                'values': buffer_values,  # Activation values for buffer tokens
                'full_text': full_text  # Original full text (no filtering)
            })
        
        # Sort by activation (descending)
        exemplars.sort(key=lambda x: x['activation'], reverse=True)
        
        if len(exemplars) == 0:
            print(f"   ⚠️  Warning: No valid exemplars found")
            print(f"      This may indicate that the API returned no activation data")
            # Raise exception instead of returning empty list
            raise ValueError("No valid exemplars found from API. API may have returned empty results.")
        
        # Determine how many to return
        if return_all:
            # Return all exemplars
            result_exemplars = exemplars
            print(f"   ✅ Retrieved all {len(result_exemplars)} exemplars from Neuronpedia API")
        else:
            # Return top_k (default: 10 if top_k is None)
            if top_k is None:
                top_k = 10
            result_exemplars = exemplars[:top_k]
            print(f"   ✅ Retrieved {len(result_exemplars)} exemplars from Neuronpedia API (top {top_k} of {len(exemplars)} total)")
        
        print(f"      Activation range: [{exemplars[-1]['activation']:.4f}, {exemplars[0]['activation']:.4f}]")
        
        # Show sample exemplars with buffer text and token activations
        if len(result_exemplars) > 0:
            num_to_show = min(5, len(result_exemplars))
            print(f"\n   📋 Sample {num_to_show} exemplars:")
            for i, exemplar in enumerate(result_exemplars[:num_to_show], 1):
                buffer_text = exemplar.get('text', '')
                activation = exemplar.get('activation', 0.0)
                max_token = exemplar.get('max_token', 'N/A')
                buffer_tokens = exemplar.get('tokens', [])
                buffer_values = exemplar.get('values', [])
                buffer_max_idx = exemplar.get('max_token_index', 0)
                
                print(f"      {i}. text: {buffer_text}")
                
                # Show all tokens in buffer with their activation values
                if buffer_tokens and buffer_values and len(buffer_tokens) == len(buffer_values):
                    token_parts = []
                    for j, (token, value) in enumerate(zip(buffer_tokens, buffer_values)):
                        if j == buffer_max_idx:
                            # Mark max token
                            token_parts.append(f"{token} {value:.2f} (max)")
                        else:
                            token_parts.append(f"{token} {value:.2f}")
                    tokens_str = ", ".join(token_parts)
                    print(f"         Tokens: {tokens_str}")
                
                print(f"         Max token: {max_token} {activation:.2f}")
                print()
        
        return result_exemplars
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"Neuronpedia API HTTP error: {e.response.status_code}"
        if e.response.text:
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('message', e.response.text[:200])}"
            except:
                error_msg += f" - {e.response.text[:200]}"
        raise requests.exceptions.RequestException(error_msg) from e
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Neuronpedia API request failed: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Invalid response from Neuronpedia API: {e}") from e


def categorize_exemplars_by_activation(
    all_exemplars: List[Dict[str, Any]],
    num_high: int = 10,
    num_medium: int = 5,
    num_low: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize exemplars into high, medium, and low activation groups.
    
    Args:
        all_exemplars: List of all exemplars (sorted by activation descending)
        num_high: Number of high-activating exemplars to return (default: 10)
        num_medium: Number of medium-activating exemplars to return (default: 5)
        num_low: Number of low-activating exemplars to return (default: 5)
    
    Returns:
        dict with keys: 'high', 'medium', 'low', each containing a list of exemplars
    """
    if not all_exemplars:
        return {'high': [], 'medium': [], 'low': []}
    
    total = len(all_exemplars)
    
    # High-activating: top num_high
    high_exemplars = all_exemplars[:min(num_high, total)]
    
    # Medium-activating: middle section
    # Calculate indices for medium-activating exemplars
    # Use exemplars from approximately 30% to 60% of the sorted list
    medium_start = max(num_high, int(total * 0.3))
    medium_end = min(medium_start + num_medium, int(total * 0.6))
    
    if medium_end > medium_start and total > num_high:
        medium_exemplars = all_exemplars[medium_start:medium_end]
    else:
        # If we don't have enough exemplars, use middle section
        medium_start = max(num_high, total // 3)
        medium_end = min(medium_start + num_medium, total - num_low) if total > num_low else medium_start + num_medium
        medium_exemplars = all_exemplars[medium_start:medium_end] if medium_end > medium_start and total > num_high else []
    
    # Low-activating: bottom num_low
    if total > num_low:
        low_exemplars = all_exemplars[-num_low:]
    else:
        # If we don't have enough exemplars, try to get some from lower-middle section
        low_start = max(medium_end if len(medium_exemplars) > 0 else num_high + num_medium, int(total * 0.6))
        low_end = min(low_start + num_low, total)
        low_exemplars = all_exemplars[low_start:low_end] if low_end > low_start else []
    
    return {
        'high': high_exemplars,
        'medium': medium_exemplars,
        'low': low_exemplars
    }


def select_exemplars_for_prediction_evaluation(
    all_exemplars: List[Dict[str, Any]],
    exclude_top_n: int = 10,
    num_high: int = 4,
    num_medium: int = 3,
    num_low: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Select exemplars for Prediction Evaluation from exemplars (excluding top N).
    
    This function:
    1. Excludes the top N exemplars (used for Generation Evaluation)
    2. Uses position-based intervals to categorize remaining exemplars into high, medium, low groups
    3. Randomly selects specified number from each interval
    4. Returns buffer_text (truncated text) and their corresponding max activation values
    
    Selection intervals (based on total length, excluding top N):
    - High-activating: indices [exclude_top_n, total_length * 0.4] (top 40% after excluding top N)
    - Medium-activating: indices [total_length * 0.4, total_length * 0.7] (40% to 70%)
    - Low-activating: indices [total_length * 0.7, total_length] (bottom 30%)
    
    Args:
        all_exemplars: List of all exemplars (sorted by activation descending)
        exclude_top_n: Number of top exemplars to exclude (default: 10, used for Generation Evaluation)
        num_high: Number of high-activating exemplars to select (default: 4)
        num_medium: Number of medium-activating exemplars to select (default: 3)
        num_low: Number of low-activating exemplars to select (default: 3)
    
    Returns:
        Tuple of (selected_texts, selected_activations) where:
        - selected_texts: List of buffer_text strings (truncated text around max activation token, 
                         using buffer_size tokens before/after max token)
        - selected_activations: List of max activation values (from exemplar's 'activation' field,
                               which is the max activation value from the full text, not just buffer region)
    """
    import random
    
    total_length = len(all_exemplars) if all_exemplars else 0
    
    if not all_exemplars or total_length <= exclude_top_n:
        print(f"   ⚠️  Not enough exemplars to select from (need more than {exclude_top_n}, got {total_length})")
        return [], []
    
    # Calculate position-based intervals (based on total length)
    # User's requirement: Use position-based intervals from sorted exemplars (by activation descending)
    # - High-activating: indices [10, total_length * 0.4] (from index 10 to 40% of total)
    # - Medium-activating: indices [total_length * 0.4, total_length * 0.7] (40% to 70% of total)
    # - Low-activating: indices [total_length * 0.7, total_length] (70% to 100% of total)
    # 
    # Note: Since exemplars are sorted by activation descending (from API), these position-based
    # intervals correspond to activation levels:
    # - High: Relatively high activation (top 40% after excluding top 10)
    # - Medium: Medium activation (middle 30%)
    # - Low: Lower activation (bottom 30%)
    
    # Calculate interval boundaries based on total length percentages
    # High-activating: indices [10, total_length * 0.4]
    #   - If total_length * 0.4 <= 10, then high interval is empty, need to adjust
    high_start = exclude_top_n  # Start from index 10 (after excluding top 10)
    high_end_idx = int(total_length * 0.4)  # 40% of total length
    
    # Medium-activating: indices [total_length * 0.4, total_length * 0.7]
    medium_start_idx = int(total_length * 0.4)  # Start at 40% of total
    medium_end_idx = int(total_length * 0.7)  # Up to 70% of total
    
    # Low-activating: indices [total_length * 0.7, total_length]
    low_start_idx = int(total_length * 0.7)  # Start at 70% of total
    
    # Determine actual interval boundaries
    # High interval: from index 10 to 40% of total (or adjust if 40% <= 10)
    if high_end_idx > high_start:
        # Normal case: 40% of total > 10, so high interval is [10, 40%]
        high_end = high_end_idx
    else:
        # Edge case: 40% of total <= 10, need to extend high interval
        # Extend to include at least num_high candidates
        high_end = min(high_start + max(num_high, int(total_length * 0.2)), total_length)
    
    # Medium interval: from 40% to 70% of total (must start after high interval ends)
    medium_start = max(medium_start_idx, high_end)  # Start after high interval ends
    if medium_end_idx > medium_start:
        medium_end = medium_end_idx
    else:
        # Edge case: 70% <= medium_start, need to extend
        medium_end = min(medium_start + max(num_medium, int(total_length * 0.2)), total_length)
    
    # Low interval: from 70% to 100% of total (must start after medium interval ends)
    low_start = max(low_start_idx, medium_end)  # Start after medium interval ends
    low_end = total_length  # End of list
    
    # Final validation: ensure intervals are non-empty and non-overlapping
    high_end = min(max(high_end, high_start + 1), total_length)
    medium_start = max(medium_start, high_end)
    medium_end = min(max(medium_end, medium_start + 1), total_length)
    low_start = max(low_start, medium_end)
    low_end = min(low_end, total_length)
    
    # Get candidates from each interval (using Python range, which is exclusive of end)
    high_candidates_indices = list(range(high_start, high_end)) if high_end > high_start else []
    medium_candidates_indices = list(range(medium_start, medium_end)) if medium_end > medium_start else []
    low_candidates_indices = list(range(low_start, low_end)) if low_end > low_start else []
    
    # Validate we have enough candidates
    if len(high_candidates_indices) < num_high:
        print(f"   ⚠️  Not enough high-activating candidates: need {num_high}, got {len(high_candidates_indices)}")
        print(f"      Interval: [{high_start}, {high_end}), Total length: {total_length}")
    if len(medium_candidates_indices) < num_medium:
        print(f"   ⚠️  Not enough medium-activating candidates: need {num_medium}, got {len(medium_candidates_indices)}")
        print(f"      Interval: [{medium_start}, {medium_end}), Total length: {total_length}")
    if len(low_candidates_indices) < num_low:
        print(f"   ⚠️  Not enough low-activating candidates: need {num_low}, got {len(low_candidates_indices)}")
        print(f"      Interval: [{low_start}, {low_end}), Total length: {total_length}")
    
    print(f"   📊 Position-based intervals (total length: {total_length}, excluding top {exclude_top_n}):")
    print(f"      High-activating: indices [{high_start}, {high_end}) = {len(high_candidates_indices)} candidates")
    print(f"      Medium-activating: indices [{medium_start}, {medium_end}) = {len(medium_candidates_indices)} candidates")
    print(f"      Low-activating: indices [{low_start}, {low_end}) = {len(low_candidates_indices)} candidates")
    
    # Randomly select indices from each interval
    selected_texts = []
    selected_activations = []
    selected_indices = []  # For tracking which exemplars were selected
    
    # Select high-activating exemplars
    if len(high_candidates_indices) >= num_high:
        selected_high_indices = random.sample(high_candidates_indices, num_high)
        for idx in selected_high_indices:
            exemplar = all_exemplars[idx]
            # Use buffer_text (already truncated around max activation token)
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            # Use max activation value from exemplar (not buffer region)
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ✅ Selected {num_high} high-activating exemplars from indices {selected_high_indices}")
    else:
        # If not enough, use all available
        for idx in high_candidates_indices:
            exemplar = all_exemplars[idx]
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ⚠️  Selected {len(high_candidates_indices)} high-activating exemplars (less than requested {num_high})")
    
    # Select medium-activating exemplars
    if len(medium_candidates_indices) >= num_medium:
        selected_medium_indices = random.sample(medium_candidates_indices, num_medium)
        for idx in selected_medium_indices:
            exemplar = all_exemplars[idx]
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ✅ Selected {num_medium} medium-activating exemplars from indices {selected_medium_indices}")
    else:
        # If not enough, use all available
        for idx in medium_candidates_indices:
            exemplar = all_exemplars[idx]
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ⚠️  Selected {len(medium_candidates_indices)} medium-activating exemplars (less than requested {num_medium})")
    
    # Select low-activating exemplars
    if len(low_candidates_indices) >= num_low:
        selected_low_indices = random.sample(low_candidates_indices, num_low)
        for idx in selected_low_indices:
            exemplar = all_exemplars[idx]
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ✅ Selected {num_low} low-activating exemplars from indices {selected_low_indices}")
    else:
        # If not enough, use all available
        for idx in low_candidates_indices:
            exemplar = all_exemplars[idx]
            buffer_text = exemplar.get('text', exemplar.get('full_text', ''))
            activation = exemplar.get('activation', 0.0)
            selected_texts.append(buffer_text)
            selected_activations.append(activation)
            selected_indices.append(idx)
        print(f"   ⚠️  Selected {len(low_candidates_indices)} low-activating exemplars (less than requested {num_low})")
    
    print(f"   📊 Total selected: {len(selected_texts)} exemplars with activations")
    if selected_activations:
        print(f"      Activation range: [{min(selected_activations):.4f}, {max(selected_activations):.4f}]")
        print(f"      Selected indices: {selected_indices}")
    
    # Show selected exemplars with buffer text and max activation
    print(f"\n   📋 Selected Exemplars for Prediction Evaluation (using buffer_text, truncated around max activation token):")
    for i, (text, activation, idx) in enumerate(zip(selected_texts, selected_activations, selected_indices), 1):
        # Get max token info from exemplar for display
        exemplar = all_exemplars[idx]
        max_token = exemplar.get('max_token', 'N/A')
        text_preview = text[:80] + "..." if len(text) > 80 else text
        print(f"      {i}. Index: {idx}, Max Activation: {activation:.4f}, Max Token: '{max_token}', Text: {text_preview}")
    
    return selected_texts, selected_activations




def generate_examples_from_explanation(
    explanation: Dict[str, Any],
    source: str,
    num_examples: int = 10,
    llm_model: str = "gpt-5"
) -> Tuple[List[str], Dict[str, int]]:
    """
    Use GPT-5 to generate test examples based on feature explanation.

    Args:
        explanation: Feature explanation dict
        source: Source of explanation ("SAGE" or "Neuronpedia")
        num_examples: Number of examples to generate
        llm_model: LLM model to use for generation

    Returns:
        List of generated test sentences
    """
    print(f"\n🤖 Generating {num_examples} examples from {source} explanation using {llm_model}...")

    # Build labels text (support multiple labels)
    labels_text = ""
    if 'labels' in explanation and explanation['labels']:
        for label in explanation['labels']:
            labels_text += f"Label {label['number']}: {label['text']}\n"
    else:
        # Backward compatibility
        if explanation.get('label1'):
            labels_text += f"Label 1: {explanation['label1']}\n"
        if explanation.get('label2'):
            labels_text += f"Label 2: {explanation['label2']}\n"

    # Format labels section if available
    labels_section = ""
    if labels_text:
        labels_section = f"\nLabels:\n{labels_text}"
    
    print(f"explanation: {explanation['description']}{labels_section}")
    if source == "SAGE":
         explanations =labels_section

    else:
         explanations = explanation['description']

    # Construct optimized prompt for GPT-5
    prompt = f"""You are an expert test case generator for Sparse Autoencoder (SAE) features. Your task is to generate diverse test sentences that strongly activate the given feature.

FEATURE DESCRIPTION:
{explanations}

TASK:
Generate exactly {num_examples} diverse test sentences that should activate this feature HIGHLY. Each sentence must clearly demonstrate the pattern or behavior described in the feature description.

REQUIREMENTS:
1. Sentence length: 5-15 words each
2. Diversity: Vary contexts, phrasings, subjects, and scenarios
3. Clarity: Each sentence should unambiguously match the feature description
4. Naturalness: Use natural, grammatically correct English
5. Uniqueness: Avoid repetitive or overly similar sentences
6. Coverage: Test different aspects and variations of the described pattern

OUTPUT FORMAT:
Output each sentence on a separate line, numbered from 1 to {num_examples}. Format:
1. [first sentence]
2. [second sentence]
3. [third sentence]
...
{num_examples}. [last sentence]

IMPORTANT:
- Output ONLY the numbered sentences, no additional commentary
- Do not include quotes around sentences
- Ensure each sentence is distinct and tests different variations of the pattern
- Focus on sentences that would produce strong activation values for this feature
"""

    # # Print complete prompt for debugging
    # print(f"\n   📝 Complete Prompt for {source} Explanation:")
    # print("   " + "="*76)
    # print(f"   {prompt}")
    # print("   " + "="*76)

    # Call GPT-5 via OpenAI API directly to get token usage
    try:
        conversation_log = [
            {"role": "system", "content": "You are an expert test case generator for Sparse Autoencoder features. You generate diverse, high-quality test sentences that strongly activate specific features."},
            {"role": "user", "content": prompt}
        ]

        # Use OpenAI client directly to get token usage
        client = get_openai_client()
        
        # Map model names
        model_name = llm_model
        if model_name == 'gpt-5':
            model_name = 'gpt-5'
        
        # GPT-5 models use max_completion_tokens instead of max_tokens
        if model_name.startswith('gpt-5'):
            params = {
                "model": model_name,
                "messages": conversation_log,
                "max_completion_tokens": 4096,
            }
        else:
            params = {
                "model": model_name,
                "messages": conversation_log,
                "max_tokens": 4096,
            }
        
        api_response = client.chat.completions.create(**params)
        response = api_response.choices[0].message.content
        
        # Get token usage
        token_usage = {
            'prompt_tokens': api_response.usage.prompt_tokens if api_response.usage else 0,
            'completion_tokens': api_response.usage.completion_tokens if api_response.usage else 0,
            'total_tokens': api_response.usage.total_tokens if api_response.usage else 0
        }

        # Parse generated examples
        import re
        lines = response.strip().split('\n')
        examples = []

        for line in lines:
            # Match numbered lines: "1. sentence" or "1) sentence"
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                sentence = match.group(1).strip()
                # Remove quotes if present
                sentence = sentence.strip('"\'')
                if sentence:
                    examples.append(sentence)

        print(f"   ✅ Generated {len(examples)} examples")
        for i, ex in enumerate(examples[:3], 1):
            print(f"      {i}. {ex}")
        if len(examples) > 3:
            print(f"      ... and {len(examples)-3} more")
        
        # Calculate cost for GPT-5
        # Prices: Input $0.625/1M, Output $5.00/1M
        cost_input = (token_usage['prompt_tokens'] / 1_000_000) * 0.625
        cost_output = (token_usage['completion_tokens'] / 1_000_000) * 5.00
        cost_total = cost_input + cost_output
        
        print(f"   📊 Token usage: {token_usage['prompt_tokens']} prompt + {token_usage['completion_tokens']} completion = {token_usage['total_tokens']} total")
        print(f"   💰 Cost: ${cost_total:.6f} (input: ${cost_input:.6f}, output: ${cost_output:.6f})")
        
        # Add cost to token_usage dict
        token_usage['cost_input'] = cost_input
        token_usage['cost_output'] = cost_output
        token_usage['cost_total'] = cost_total

        return examples[:num_examples], token_usage

    except Exception as e:
        print(f"   ❌ Failed to generate examples: {e}")
        import traceback
        traceback.print_exc()
        return [], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0}


def evaluate_examples(
    examples: List[str],
    system = None,
    activation_threshold: float = 8.0,
    use_api: bool = False,
    model_id: str = None,
    layer: str = None,
    feature_index: int = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Evaluate generated examples against actual SAE activations.

    Args:
        examples: List of test sentences
        system: SAGE System object (model + SAE) - only used if use_api=False
        activation_threshold: Threshold for "high activation"
        use_api: If True, use Neuronpedia API to get activations instead of system
        model_id: Model ID for API (required if use_api=True)
        layer: Layer identifier for API (required if use_api=True)
        feature_index: Feature index for API (required if use_api=True)

    Returns:
        dict with evaluation metrics and detailed results
    """
    print(f"\n📊 Evaluating {len(examples)} examples (threshold: {activation_threshold})...")
    
    if use_api:
        if not all([model_id, layer, feature_index is not None]):
            raise ValueError("API mode requires model_id, layer, and feature_index")
        print(f"   📡 Using Neuronpedia API: Model={model_id}, Layer={layer}, Feature={feature_index}")
    else:
        if system is None:
            raise ValueError("System object required when use_api=False")

    detailed_results = []
    successful_count = 0
    total_max_activation = 0.0
    total_mean_activation = 0.0

    for i, example in enumerate(examples, 1):
        try:
            if use_api:
                # Show progress for API calls
                if i % 2 == 0 or i == len(examples):
                    print(f"   Getting activation {i}/{len(examples)}...", end='\r')
                
                # Get activation from Neuronpedia API
                # API will raise exception on failure
                activation_data = get_activation_data_from_api(
                    model_id=model_id,
                    source=layer,
                    feature_index=feature_index,
                    custom_text=example,
                    api_key=api_key
                )
                
                max_activation = activation_data['maxValue']
                mean_activation = activation_data.get('meanValue', 0.0)
                tokens = activation_data.get('tokens', [])
                per_token_activations = activation_data.get('values', [])
                max_token_idx = activation_data.get('maxValueTokenIndex', 0)
            else:
                # Get activation trace from system
                trace = system.get_activation_trace(example)

                max_activation = trace.get('summary_activation', 0.0)
                mean_activation = trace.get('summary_activation_mean', 0.0)
                tokens = trace.get('tokens', [])
                per_token_activations = trace.get('per_token_activation', [])
                max_token_idx = trace.get('max_token_index', 0)

            # Determine success
            success = max_activation >= activation_threshold
            if success:
                successful_count += 1

            total_max_activation += max_activation
            total_mean_activation += mean_activation

            # Find highest activating token
            max_token = tokens[max_token_idx] if max_token_idx < len(tokens) else "N/A"

            result = {
                'example': example,
                'max_activation': max_activation,
                'mean_activation': mean_activation,
                'success': success,
                'max_token': max_token,
                'max_token_idx': max_token_idx,
                'tokens': tokens,
                'per_token_activations': per_token_activations
            }

            detailed_results.append(result)

        except Exception as e:
            print()  # New line to clear progress display
            print(f"   ❌ Error evaluating example {i}: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise exception instead of silently failing
            raise RuntimeError(f"Failed to evaluate example {i}: {e}") from e

    # Clear progress display and print results
    if use_api:
        print()  # New line to clear progress display
    
    # Print evaluation results
    for i, result in enumerate(detailed_results, 1):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"   Example {i}: {status} - max_act: {result['max_activation']:.4f}")
        print(f"      Text: \"{result['example']}\"")
        print(f"      Max token: '{result['max_token']}' at idx {result['max_token_idx']}")

    # Calculate metrics
    num_examples = len(examples)
    success_rate = successful_count / num_examples if num_examples > 0 else 0.0
    avg_max_activation = total_max_activation / num_examples if num_examples > 0 else 0.0
    avg_mean_activation = total_mean_activation / num_examples if num_examples > 0 else 0.0

    # Calculate metrics for successful examples only
    successful_examples = [r for r in detailed_results if r.get('success', False)]
    avg_max_activation_successful = (
        sum(r['max_activation'] for r in successful_examples) / len(successful_examples)
        if successful_examples else 0.0
    )
    avg_mean_activation_successful = (
        sum(r['mean_activation'] for r in successful_examples) / len(successful_examples)
        if successful_examples else 0.0
    )

    metrics = {
        'total_examples': num_examples,
        'successful_examples': successful_count,
        'success_rate': success_rate,
        'avg_max_activation_all': avg_max_activation,
        'avg_mean_activation_all': avg_mean_activation,
        'avg_max_activation_successful': avg_max_activation_successful,
        'avg_mean_activation_successful': avg_mean_activation_successful
    }

    print(f"\n   📈 Metrics:")
    print(f"      Success Rate: {success_rate*100:.1f}% ({successful_count}/{num_examples})")
    print(f"      Avg Max Activation (all): {avg_max_activation:.4f}")
    print(f"      Avg Max Activation (successful): {avg_max_activation_successful:.4f}")

    return {
        'metrics': metrics,
        'detailed_results': detailed_results
    }


def get_activation_data_from_api(
    model_id: str,
    source: str,
    feature_index: int,
    custom_text: str,
    normalize_to_0_10: bool = False,
    global_max_activation: float = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Get full activation data for custom text using Neuronpedia API.

    Note: This function automatically filters out special tokens like <|begin_of_text|>,
    <|endoftext|>, and other common special tokens (e.g., <bos>, <eos>, <pad>, <unk>)
    and recalculates statistics (maxValue, minValue, maxValueTokenIndex, meanValue)
    from the filtered data. This ensures that special tokens don't interfere with
    activation analysis.

    Args:
        model_id: Model ID (e.g., "gpt2-small", "gemma-2-2b")
        source: SAE ID/source (e.g., "9-res-jb", "6-gemmascope-mlp-16k")
        feature_index: Feature index
        custom_text: Custom text to get activation for
        normalize_to_0_10: If True, normalize activation values to 0-10 range
                          (for Method 2 logprobs-based prediction).
                          If False (default), return raw activation values
                          (for Method 1 Generation Comparison).
        global_max_activation: If provided and normalize_to_0_10=True, use this value
                              as the normalization baseline instead of the sample's max.
                              This ensures all samples use the same normalization baseline.

    Returns:
        dict with activation data: {
            'maxValue': float (raw or normalized to 0-10 based on normalize_to_0_10),
            'minValue': float (raw or normalized to 0-10 based on normalize_to_0_10),
            'maxValueTokenIndex': int (index in filtered arrays),
            'tokens': List[str] (filtered, special tokens removed),
            'values': List[float] (raw or normalized, corresponding to filtered tokens),
            'meanValue': float (calculated from values)
        }

    Raises:
        requests.exceptions.RequestException: If API call fails
        ValueError: If API returns invalid response
    """
    url = "https://www.neuronpedia.org/api/activation/new"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided
    if api_key:
        headers["x-api-key"] = api_key
    
    payload = {
        "feature": {
            "modelId": model_id,
            "source": source,
            "index": str(feature_index)
        },
        "customText": custom_text
    }
    
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # If 5xx error, retry
            if 500 <= response.status_code < 600:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"   ⚠️  API {response.status_code} Error. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    # Last attempt failed, raise error
                    response.raise_for_status()
            
            # Raise for other error codes (4xx)
            response.raise_for_status()
            
            # If successful, break loop
            break
            
        except requests.exceptions.RequestException as e:
            # Check if it's a 5xx error caught by raise_for_status or connection error
            is_5xx = False
            if hasattr(e, 'response') and e.response is not None:
                if 500 <= e.response.status_code < 600:
                    is_5xx = True
            
            # Retry on 5xx or connection errors (which might not have response)
            if (is_5xx or not hasattr(e, 'response') or e.response is None) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"   ⚠️  API Request Failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                # Re-raise the last exception if no more retries
                raise e
        
    # Process response after successful retry loop
    try:
        result = response.json()
        
        # Check if response contains error field
        if 'error' in result and result['error']:
            raise ValueError(f"API returned error: {result['error']}")
        
        # Validate required fields
        if 'maxValue' not in result:
            raise ValueError(f"API response missing 'maxValue' field: {result}")
        
        # Extract activation data
        tokens = result.get('tokens', [])
        values = result.get('values', [])
        
        # Filter out special tokens and their activation values
        # These tokens are added by the API/tokenizer but shouldn't be considered in evaluation
        # Include both beginning-of-text and end-of-text tokens, as well as other common special tokens
        special_tokens = {
            # End-of-text tokens
            '<|endoftext|>', '<|eot_id|>', '<|eot|>', '<eos>', '</s>',
            # Beginning-of-text tokens
            '<|begin_of_text|>', '<|beginoftext|>', '<|begin|>', '<|startoftext|>', 
            '<|start_of_text|>', '<|start|>', '<bos>', '<s>',
            # Other common special tokens
            '<pad>', '<unk>', '<mask>', '<sep>', '<cls>'
        }
        filtered_tokens = []
        filtered_values = []
        
        for i, token in enumerate(tokens):
            # Skip special tokens (case-insensitive comparison after stripping)
            token_clean = token.strip()
            # Check if token matches any special token (case-insensitive)
            if any(token_clean.lower() == st.lower() for st in special_tokens):
                continue
            filtered_tokens.append(token)
            if i < len(values):
                filtered_values.append(values[i])
        
        # Recalculate statistics from filtered data
        if not filtered_values:
            # All tokens were filtered out, set default values
            max_value = 0.0
            min_value = 0.0
            max_value_token_index = 0
            mean_value = 0.0
            final_values = []
        else:
            # Recalculate max, min from filtered values
            max_value_raw = max(filtered_values)
            min_value_raw = min(filtered_values)

            # Decide whether to normalize based on parameter
            if normalize_to_0_10:
                # Normalize activation values to 0-10 range (for Method 2: logprobs-based prediction)
                # This ensures consistency with get_activation_logprobs output range
                if global_max_activation is not None and global_max_activation > 0:
                    # Use global max activation as normalization baseline (all samples use same baseline)
                    final_values = [(val / global_max_activation) * 10.0 for val in filtered_values]
                elif max_value_raw > 0:
                    # Fallback: use sample's max activation as normalization baseline
                    final_values = [(val / max_value_raw) * 10.0 for val in filtered_values]
                else:
                    final_values = [0.0] * len(filtered_values)
            else:
                # Use raw values (for Method 1: Generation Comparison)
                final_values = filtered_values

            # Recalculate statistics from final values
            max_value = max(final_values) if final_values else 0.0
            min_value = min(final_values) if final_values else 0.0
            max_value_token_index = final_values.index(max_value) if final_values else 0
            mean_value = sum(final_values) / len(final_values) if final_values else 0.0

        return {
            'maxValue': max_value,
            'minValue': min_value,
            'maxValueTokenIndex': max_value_token_index,
            'tokens': filtered_tokens,
            'values': final_values,  # Return raw or normalized values based on parameter
            'meanValue': mean_value
        }

    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"Invalid response from Neuronpedia API for text '{custom_text[:50]}...': {e}"
        ) from e



def get_activation_from_neuronpedia_api(
    model_id: str,
    source: str,
    feature_index: int,
    custom_text: str
) -> float:
    """
    Get activation value for custom text using Neuronpedia API.
    Convenience wrapper that returns only the maxValue.
    
    Args:
        model_id: Model ID (e.g., "gpt2-small", "gemma-2-2b")
        source: SAE ID/source (e.g., "9-res-jb", "6-gemmascope-mlp-16k")
        feature_index: Feature index
        custom_text: Custom text to get activation for
    
    Returns:
        Maximum activation value (can be 0.0 if feature doesn't activate)
    
    Raises:
        requests.exceptions.RequestException: If API call fails
        ValueError: If API returns invalid response
    """
    data = get_activation_data_from_api(model_id, source, feature_index, custom_text)
    return data['maxValue']


def evaluate_prediction_ability(
    explanation: Dict[str, Any],
    examples: List[str],
    activations: List[float] = None,
    source: str = "SAGE",
    llm_model: str = "gpt-5",
    use_api: bool = False,
    model_id: str = None,
    layer: str = None,
    feature_index: int = None,
    global_max_activation: float = None
) -> Dict[str, Any]:
    """
    Evaluate LLM's ability to predict activations using logprobs-based method (Method 2).

    This method now uses the logprobs-based prediction approach from test_logprobs_evaluation.py:
    1. Get token-level activation data from API (normalized to 0-10 range)
    2. Use get_activation_logprobs to predict activation for each token
    3. Calculate correlation between predicted and actual token-level activations

    Args:
        explanation: Feature explanation
        examples: List of test examples
        activations: Not used (kept for backward compatibility)
        source: Source of explanation
        llm_model: LLM model for prediction (not used, hardcoded to gpt-4o for logprobs)
        use_api: If True, use Neuronpedia API to get token-level activations (required)
        model_id: Model ID for API (required if use_api=True)
        layer: Layer identifier for API (required if use_api=True)
        feature_index: Feature index for API (required if use_api=True)
        global_max_activation: Global maximum activation value from all exemplars.
                              If provided, all samples will use this as normalization baseline.

    Returns:
        dict with correlation and prediction results
    """
    print(f"\n🔮 Evaluating {source} prediction ability using logprobs-based method...")

    if not use_api:
        print(f"   ⚠️  LogProbs-based prediction requires API mode. Skipping prediction evaluation.")
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'predictions': [],
            'skipped': True,
            'error': 'LogProbs-based prediction requires use_api=True'
        }

    if len(examples) < 5:
        print(f"   ⚠️  Need at least 5 examples, got {len(examples)}. Skipping prediction evaluation.")
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'predictions': [],
            'skipped': True
        }

    # Validate API parameters
    if not all([model_id, layer, feature_index is not None]):
        raise ValueError("API mode requires model_id, layer, and feature_index")

    print(f"   📡 Getting token-level activations from Neuronpedia API...")
    print(f"      Model: {model_id}, Layer: {layer}, Feature: {feature_index}")

    # Extract feature description for logprobs prediction
    # Extract feature description for logprobs prediction
    labels_text = ""
    if 'labels' in explanation and explanation['labels']:
        for label in explanation['labels']:
            labels_text += f"Label {label['number']}: {label['text']}\n"
    else:
        # Backward compatibility
        if explanation.get('label1'):
            labels_text += f"Label 1: {explanation['label1']}\n"
        if explanation.get('label2'):
            labels_text += f"Label 2: {explanation['label2']}\n"

    # Set feature description based on source
    if source == "SAGE" and labels_text:
        feature_description = f"\nLabels:\n{labels_text}"
    else:
        feature_description = explanation.get('description', '')
    
    if not feature_description:
        print(f"   ⚠️  No feature description found. Cannot use logprobs prediction.")
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'predictions': [],
            'skipped': True,
            'error': 'No feature description available'
        }
    print(f"   ***Feature description: {feature_description}")
    # Collect all token-level predictions and actual activations across all examples
    all_predicted_activations = []
    all_actual_activations = []
    example_results = []
    total_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0}

    try:
        for i, example_text in enumerate(examples, 1):
            # Show progress
            print(f"      Processing example {i}/{len(examples)}: '{example_text[:50]}...'", end='\r')

            # Get token-level activation data from API (normalized to 0-10 range for logprobs comparison)
            activation_data = get_activation_data_from_api(
                model_id=model_id,
                source=layer,
                feature_index=feature_index,
                custom_text=example_text,
                normalize_to_0_10=True,  # Enable normalization for Method 2 (logprobs-based prediction)
                global_max_activation=global_max_activation  # Use global max activation as normalization baseline
            )

            tokens = activation_data['tokens']
            actual_values = activation_data['values']  # Already normalized to 0-10 range

            if not tokens or not actual_values:
                print(f"\n   ⚠️  No tokens/activations for example {i}. Skipping.")
                continue

            # Use logprobs to predict activation for each token
            predicted_values, token_usage = predict_activations_with_logprobs(
                explanation=feature_description,
                tokens=tokens,
                show_details=False
            )
            
            # Accumulate token usage and cost
            total_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
            total_token_usage['completion_tokens'] += token_usage['completion_tokens']
            total_token_usage['total_tokens'] += token_usage['total_tokens']
            total_token_usage['cost_input'] += token_usage.get('cost_input', 0.0)
            total_token_usage['cost_output'] += token_usage.get('cost_output', 0.0)
            total_token_usage['cost_total'] += token_usage.get('cost_total', 0.0)

            # Ensure we have the same number of predictions as actual values
            if len(predicted_values) != len(actual_values):
                print(f"\n   ⚠️  Token count mismatch for example {i}: "
                      f"predicted={len(predicted_values)}, actual={len(actual_values)}")
                # Pad or truncate to match
                if len(predicted_values) < len(actual_values):
                    predicted_values.extend([0.0] * (len(actual_values) - len(predicted_values)))
                else:
                    predicted_values = predicted_values[:len(actual_values)]

            # Collect all token-level activations
            all_predicted_activations.extend(predicted_values)
            all_actual_activations.extend(actual_values)

            # Store per-example results
            example_results.append({
                'text': example_text,
                'tokens': tokens,
                'predicted_values': predicted_values,
                'actual_values': actual_values,
                'num_tokens': len(tokens)
            })

        print()  # New line after progress

        if not all_predicted_activations or not all_actual_activations:
            print(f"   ⚠️  No token-level activations collected. Cannot calculate correlation.")
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'predictions': [],
                'skipped': True,
                'error': 'No token-level activations collected'
            }

        print(f"      ✅ Collected {len(all_predicted_activations)} token-level predictions across {len(examples)} examples")

        # Calculate statistics
        pred_array = np.array(all_predicted_activations)
        actual_array = np.array(all_actual_activations)

        print(f"         Predicted range: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
        print(f"         Actual range: [{actual_array.min():.4f}, {actual_array.max():.4f}]")

        # Check for conditions that would cause nan correlation
        pred_variance = np.var(pred_array)
        actual_variance = np.var(actual_array)

        if len(all_predicted_activations) < 2:
            print(f"   ⚠️  Cannot calculate correlation: need at least 2 data points")
            correlation = np.nan
            p_value = np.nan
        elif pred_variance == 0:
            print(f"   ⚠️  Cannot calculate correlation: all predictions are identical (variance=0)")
            correlation = np.nan
            p_value = np.nan
        elif actual_variance == 0:
            print(f"   ⚠️  Cannot calculate correlation: all actual values are identical (variance=0)")
            correlation = np.nan
            p_value = np.nan
        else:
            # Calculate Pearson correlation
            correlation, p_value = pearsonr(all_predicted_activations, all_actual_activations)

            if np.isnan(correlation) or np.isnan(p_value):
                print(f"   ⚠️  Correlation calculation returned NaN")

        if not np.isnan(correlation) and not np.isnan(p_value):
            print(f"   📊 Token-Level Pearson Correlation: {correlation:.4f} (p={p_value:.4f})")
            # Interpret significance
            if p_value < 0.001:
                significance = "*** (highly significant, p<0.001)"
            elif p_value < 0.01:
                significance = "** (very significant, p<0.01)"
            elif p_value < 0.05:
                significance = "* (significant, p<0.05)"
            elif p_value < 0.1:
                significance = "~ (marginally significant, p<0.1)"
            else:
                significance = " (not significant, p>=0.1)"
            print(f"      Significance: {significance}")
            print(f"      Interpretation: {'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'} "
                  f"{'positive' if correlation > 0 else 'negative'} correlation")
        else:
            print(f"   📊 Token-Level Pearson Correlation: NaN (p=NaN)")
            print(f"      ⚠️  Cannot interpret correlation due to data issues")

        # Print sample predictions vs actual for first few examples
        print(f"\n   Sample Predictions (first 3 examples):")
        for i, ex_result in enumerate(example_results[:3], 1):
            print(f"      Example {i}: '{ex_result['text']}'")
            print(f"         Tokens: {ex_result['num_tokens']}")
            # Show first 5 tokens
            for j, (token, pred, actual) in enumerate(zip(
                ex_result['tokens'],
                ex_result['predicted_values'],
                ex_result['actual_values']
            )):
                print(f"           Token '{token}': Pred={pred:.2f}, Actual={actual:.2f}, Error={abs(pred-actual):.2f}")


        # Handle NaN values for JSON serialization
        correlation_for_json = correlation if not np.isnan(correlation) else None
        p_value_for_json = p_value if not np.isnan(p_value) else None

        result = {
            'correlation': correlation_for_json,
            'p_value': p_value_for_json,
            'predictions': all_predicted_activations,
            'true_values': all_actual_activations,
            'num_tokens': len(all_predicted_activations),
            'num_examples': len(example_results),
            'example_results': example_results,
            'skipped': False,
            'correlation_valid': not (np.isnan(correlation) or np.isnan(p_value)),
            'prediction_variance': float(pred_variance) if len(all_predicted_activations) >= 2 else None,
            'actual_variance': float(actual_variance) if len(all_actual_activations) >= 2 else None,
            'method': 'logprobs_token_level',
            'token_usage': total_token_usage  # Add token usage to result
        }
        
        # Calculate total cost for GPT-4o (if not already calculated)
        # Prices: Input $1.25/1M, Output $5.00/1M
        if total_token_usage.get('cost_total', 0.0) == 0.0 and total_token_usage['total_tokens'] > 0:
            total_cost_input = (total_token_usage['prompt_tokens'] / 1_000_000) * 1.25
            total_cost_output = (total_token_usage['completion_tokens'] / 1_000_000) * 5.00
            total_cost = total_cost_input + total_cost_output
            total_token_usage['cost_input'] = total_cost_input
            total_token_usage['cost_output'] = total_cost_output
            total_token_usage['cost_total'] = total_cost
        else:
            # Cost already accumulated from individual calls
            total_cost = total_token_usage.get('cost_total', 0.0)
            total_cost_input = total_token_usage.get('cost_input', 0.0)
            total_cost_output = total_token_usage.get('cost_output', 0.0)
        
        # Print token usage
        print(f"\n   📊 Token usage (GPT-4o for logprobs): {total_token_usage['prompt_tokens']} prompt + {total_token_usage['completion_tokens']} completion = {total_token_usage['total_tokens']} total")
        print(f"   💰 Cost: ${total_cost:.6f} (input: ${total_cost_input:.6f}, output: ${total_cost_output:.6f})")

        # Add API usage information
        result['activation_source'] = 'neuronpedia_api'
        result['api_config'] = {
            'model_id': model_id,
            'layer': layer,
            'feature_index': feature_index
        }

        return result

    except Exception as e:
        print(f"\n   ❌ Prediction evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'predictions': [],
            'error': str(e),
            'skipped': True
        }


def compare_results(
    sage_gen_eval: Dict,
    neuronpedia_gen_eval: Dict,
    sage_pred_eval: Dict = None,
    neuronpedia_pred_eval: Dict = None
) -> Dict[str, Any]:
    """
    Compare evaluation results from SAGE vs Neuronpedia (both methods).

    Args:
        sage_gen_eval: SAGE generation evaluation results
        neuronpedia_gen_eval: Neuronpedia generation evaluation results
        sage_pred_eval: SAGE prediction evaluation results (optional)
        neuronpedia_pred_eval: Neuronpedia prediction evaluation results (optional)

    Returns:
        dict with comparison metrics
    """
    print("\n" + "="*80)
    print("📊 COMPARISON: SAGE vs Neuronpedia")
    print("="*80)

    sage_metrics = sage_gen_eval['metrics']
    neuro_metrics = neuronpedia_gen_eval['metrics']

    comparison = {
        'generation': {
            'sage': sage_metrics,
            'neuronpedia': neuro_metrics,
            'differences': {}
        }
    }

    # Calculate generation differences
    comparison['generation']['differences'] = {
        'success_rate_diff': sage_metrics['success_rate'] - neuro_metrics['success_rate'],
        'avg_max_activation_diff': sage_metrics['avg_max_activation_all'] - neuro_metrics['avg_max_activation_all'],
        'avg_max_activation_successful_diff': sage_metrics['avg_max_activation_successful'] - neuro_metrics['avg_max_activation_successful']
    }

    # Print generation comparison table
    print(f"\n📝 METHOD 1: Generation Evaluation (LLM generates text from description)")
    print(f"\n{'Metric':<40} {'SAGE':>15} {'Neuronpedia':>15} {'Difference':>15}")
    print("-" * 85)

    print(f"{'Success Rate':.<40} {sage_metrics['success_rate']*100:>14.1f}% {neuro_metrics['success_rate']*100:>14.1f}% {comparison['generation']['differences']['success_rate_diff']*100:>+14.1f}%")
    print(f"{'Avg Max Activation (all)':.<40} {sage_metrics['avg_max_activation_all']:>15.4f} {neuro_metrics['avg_max_activation_all']:>15.4f} {comparison['generation']['differences']['avg_max_activation_diff']:>+15.4f}")
    print(f"{'Avg Max Activation (successful)':.<40} {sage_metrics['avg_max_activation_successful']:>15.4f} {neuro_metrics['avg_max_activation_successful']:>15.4f} {comparison['generation']['differences']['avg_max_activation_successful_diff']:>+15.4f}")
    print(f"{'Successful Examples':.<40} {sage_metrics['successful_examples']:>15} {neuro_metrics['successful_examples']:>15} {sage_metrics['successful_examples'] - neuro_metrics['successful_examples']:>+15}")

    # Detailed per-example comparison for generation
    print(f"\n   📋 Detailed Generation Comparison:")
    print(f"\n   {'System':<12} {'#':<3} {'Generated Text':<50} {'Max Act':>10} {'Result':>8}")
    print(f"   {'-'*95}")

    sage_detailed = sage_gen_eval.get('detailed_results', [])
    neuro_detailed = neuronpedia_gen_eval.get('detailed_results', [])

    # Show SAGE examples
    for i, result in enumerate(sage_detailed, 1):
        text = result['example'][:50] + "..." if len(result['example']) > 50 else result['example']
        max_act = result['max_activation']
        success = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"   {'SAGE':<12} {i:<3} {text:<50} {max_act:>10.4f} {success:>8}")

    print(f"   {'-'*95}")

    # Show Neuronpedia examples
    for i, result in enumerate(neuro_detailed, 1):
        text = result['example'][:50] + "..." if len(result['example']) > 50 else result['example']
        max_act = result['max_activation']
        success = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"   {'Neuronpedia':<12} {i:<3} {text:<50} {max_act:>10.4f} {success:>8}")


    # Add prediction comparison if available
    if sage_pred_eval and neuronpedia_pred_eval and not sage_pred_eval.get('skipped') and not neuronpedia_pred_eval.get('skipped'):
        print(f"\n🔮 METHOD 2: Prediction Evaluation (LLM predicts activations)")
        print(f"\n{'Metric':<40} {'SAGE':>15} {'Neuronpedia':>15} {'Difference':>15}")
        print("-" * 85)

        sage_corr = sage_pred_eval.get('correlation')
        neuro_corr = neuronpedia_pred_eval.get('correlation')
        sage_p = sage_pred_eval.get('p_value')
        neuro_p = neuronpedia_pred_eval.get('p_value')
        
        # Handle NaN/None values
        sage_corr_display = f"{sage_corr:.4f}" if sage_corr is not None and not (isinstance(sage_corr, float) and np.isnan(sage_corr)) else "NaN"
        neuro_corr_display = f"{neuro_corr:.4f}" if neuro_corr is not None and not (isinstance(neuro_corr, float) and np.isnan(neuro_corr)) else "NaN"
        sage_p_display = f"{sage_p:.4f}" if sage_p is not None and not (isinstance(sage_p, float) and np.isnan(sage_p)) else "NaN"
        neuro_p_display = f"{neuro_p:.4f}" if neuro_p is not None and not (isinstance(neuro_p, float) and np.isnan(neuro_p)) else "NaN"
        
        # Calculate difference only if both are valid
        if (sage_corr is not None and not (isinstance(sage_corr, float) and np.isnan(sage_corr)) and
            neuro_corr is not None and not (isinstance(neuro_corr, float) and np.isnan(neuro_corr))):
            corr_diff = sage_corr - neuro_corr
            corr_diff_display = f"{corr_diff:>+15.4f}"
        else:
            corr_diff_display = "N/A (NaN present)"

        print(f"{'Pearson Correlation':.<40} {sage_corr_display:>15} {neuro_corr_display:>15} {corr_diff_display:>15}")
        print(f"{'P-value':.<40} {sage_p_display:>15} {neuro_p_display:>15} {'':>15}")
        
        # Add interpretation notes
        if sage_corr is not None and neuro_corr is not None:
            if not (isinstance(sage_corr, float) and np.isnan(sage_corr)) and not (isinstance(neuro_corr, float) and np.isnan(neuro_corr)):
                print(f"\n   📊 Interpretation:")
                print(f"      - Correlation ranges from -1 (perfect negative) to +1 (perfect positive)")
                print(f"      - |correlation| > 0.7: strong, 0.4-0.7: moderate, < 0.4: weak")
                print(f"      - p-value < 0.05: statistically significant correlation")
                print(f"      - Higher correlation = better prediction ability")

        # Detailed per-example comparison for predictions
        print(f"\n   📋 Detailed Prediction Comparison (Test Set):")
        print(f"\n   {'#':<3} {'Text':<45} {'SAGE':>8} {'Neuro':>8} {'Actual':>8} {'SAGE Err':>10} {'Neuro Err':>10}")
        print(f"   {'-'*95}")

        sage_example_results = sage_pred_eval.get('example_results', [])
        neuro_example_results = neuronpedia_pred_eval.get('example_results', [])

        for i, (sage_ex, neuro_ex) in enumerate(zip(sage_example_results, neuro_example_results), 1):
            text = sage_ex['text'][:45] + "..." if len(sage_ex['text']) > 45 else sage_ex['text']
            
            # Calculate average predicted and actual values for this example
            sage_pred_avg = sum(sage_ex['predicted_values']) / len(sage_ex['predicted_values']) if sage_ex['predicted_values'] else 0.0
            neuro_pred_avg = sum(neuro_ex['predicted_values']) / len(neuro_ex['predicted_values']) if neuro_ex['predicted_values'] else 0.0
            actual_avg = sum(sage_ex['actual_values']) / len(sage_ex['actual_values']) if sage_ex['actual_values'] else 0.0
            
            sage_err = abs(sage_pred_avg - actual_avg)
            neuro_err = abs(neuro_pred_avg - actual_avg)

            # Mark which prediction is better with color indicators
            sage_marker = "✓" if sage_err < neuro_err else " "
            neuro_marker = "✓" if neuro_err < sage_err else " "

            print(f"   {i:<3} {text:<45} {sage_pred_avg:>7.3f}{sage_marker} {neuro_pred_avg:>7.3f}{neuro_marker} {actual_avg:>8.3f} {sage_err:>10.3f} {neuro_err:>10.3f}")



    print("="*80 + "\n")

    return comparison


def process_single_feature(
    sage_results_path: str,
    model_name: str,
    sae_path: str,
    layer: str,
    feature_index: int,
    neuronpedia_model_id: str,
    neuronpedia_api_key: str,
    llm_model: str,
    num_examples: int,
    activation_threshold: float,
    device: str,
    output_dir: str,
    explanation_model_name: str = "gpt-5",
    explanation_type: str = "oai_token-act-pair"
) -> Dict[str, Any]:
    """
    Process a single feature comparison.
    
    Note: All activation evaluations now use Neuronpedia API instead of local system.
    The model_name, sae_path, and device parameters are kept for backward compatibility
    and record-keeping only, but are NOT used for any activation calculations.
    All activations are fetched from Neuronpedia API using neuronpedia_model_id, layer, and feature_index.

    Args:
        sage_results_path: Path to SAGE results JSON file
        model_name: Model name (optional, kept for compatibility/record-keeping only, not used)
        sae_path: Path to SAE file (optional, kept for compatibility/record-keeping only, not used)
        layer: Layer identifier as string (e.g., "11-resid-post-aa") - REQUIRED for API calls
        feature_index: Feature index - REQUIRED for API calls
        neuronpedia_model_id: Neuronpedia model ID (REQUIRED for API calls)
        neuronpedia_api_key: Neuronpedia API key
        llm_model: LLM model for evaluation
        num_examples: Number of examples to generate
        activation_threshold: Activation threshold
        device: Device to use (optional, kept for compatibility/record-keeping only, not used)
        output_dir: Output directory
        explanation_model_name: Explanation model name (default: "gpt-5")
        explanation_type: Explanation type (default: "oai_token-act-pair")

    Returns:
        dict with comparison results
    """
    print("\n" + "="*80)
    print(f"🔬 Processing Feature {feature_index}")
    print("="*80)

    # Step 0: Get Activation Exemplars from API to calculate dynamic activation threshold
    # Also select exemplars for Prediction Evaluation (excluding top 10)
    print("\n" + "="*80)
    print("STEP 0: Get Activation Exemplars from API (for dynamic threshold calculation and Prediction Evaluation)")
    print("="*80)
    dynamic_activation_threshold = activation_threshold  # Default to provided threshold
    all_api_exemplars = []  # All exemplars from API
    high_exemplars = []  # Top 10 high-activating exemplars (for Generation Evaluation and threshold)
    prediction_examples_texts = []  # Selected texts for Prediction Evaluation
    prediction_examples_activations = []  # Activation values for selected texts
    global_max_activation = None  # Global max activation from all exemplars (for normalization)
    
    try:
        # Get all exemplars from API (not just top 10)
        all_api_exemplars = get_activation_exemplars_from_api(
            model_id=neuronpedia_model_id,
            source=layer,
            feature_index=feature_index,
            return_all=True  # Get all exemplars
        )
        
        if all_api_exemplars and len(all_api_exemplars) > 0:
            # Get top 10 high-activating exemplars for Generation Evaluation and threshold calculation
            high_exemplars = all_api_exemplars[:min(10, len(all_api_exemplars))]
            
            # Calculate global max activation from all exemplars (for normalization baseline)
            # Since exemplars are sorted by activation descending, the first one has the max
            global_max_activation = all_api_exemplars[0].get('activation', 0.0) if all_api_exemplars else None
            
            print(f"   ✅ Retrieved {len(all_api_exemplars)} exemplars from API")
            print(f"      Top 10 high-activating exemplars (for Generation Evaluation): {len(high_exemplars)} exemplars")
            print(f"      Remaining exemplars (for Prediction Evaluation): {len(all_api_exemplars) - len(high_exemplars)} exemplars")
            if global_max_activation is not None:
                print(f"      Global max activation (for normalization): {global_max_activation:.4f}")
            
            # Calculate average of max activations from top 10 exemplars for threshold
            if high_exemplars and len(high_exemplars) > 0:
                max_activations = [exemplar.get('activation', 0.0) for exemplar in high_exemplars]
                if max_activations:
                    avg_max_activation = sum(max_activations) / len(max_activations)
                    dynamic_activation_threshold = avg_max_activation / 2
                    print(f"   ✅ Calculated dynamic activation threshold from {len(max_activations)} high-activating exemplars")
                    print(f"      Original threshold: {activation_threshold:.4f}")
                    print(f"      Dynamic threshold (avg of top {len(max_activations)} max activations / 2): {dynamic_activation_threshold:.4f}")
                    print(f"      Max activation range: [{min(max_activations):.4f}, {max(max_activations):.4f}]")
                else:
                    print(f"   ⚠️  No activation values found in high exemplars, using original threshold: {activation_threshold:.4f}")
            else:
                print(f"   ⚠️  No high-activating exemplars found, using original threshold: {activation_threshold:.4f}")
            
            # Select exemplars for Prediction Evaluation (excluding top 10)
            # From remaining exemplars, select: 4 high, 3 medium, 3 low
            print(f"\n   📋 Selecting exemplars for Prediction Evaluation (excluding top 10):")
            prediction_examples_texts, prediction_examples_activations = select_exemplars_for_prediction_evaluation(
                all_api_exemplars,
                exclude_top_n=10,  # Exclude top 10 (used for Generation Evaluation)
                num_high=4,  # Select 4 high-activating exemplars
                num_medium=3,  # Select 3 medium-activating exemplars
                num_low=3  # Select 3 low-activating exemplars
            )
            
            # Print top 10 high-activating exemplars (for Generation Evaluation)
            print(f"\n   📋 Top 10 High-Activating Exemplars (for Generation Evaluation, variable: high_exemplars):")
            print("   " + "="*76)
            if high_exemplars:
                for i, exemplar in enumerate(high_exemplars, 1):
                    buffer_text = exemplar.get('text', '')
                    activation = exemplar.get('activation', 0.0)
                    max_token = exemplar.get('max_token', 'N/A')
                    buffer_tokens = exemplar.get('tokens', [])
                    buffer_values = exemplar.get('values', [])
                    buffer_max_idx = exemplar.get('max_token_index', 0)
                    
                    print(f"   {i}. text: {buffer_text}")
                    if buffer_tokens and buffer_values and len(buffer_tokens) == len(buffer_values):
                        token_parts = []
                        for j, (token, value) in enumerate(zip(buffer_tokens, buffer_values)):
                            if j == buffer_max_idx:
                                token_parts.append(f"{token} {value:.2f} (max)")
                            else:
                                token_parts.append(f"{token} {value:.2f}")
                        tokens_str = ", ".join(token_parts)
                        print(f"      Tokens: {tokens_str}")
                    print(f"      Max token: {max_token} {activation:.2f}")
                    print()
            else:
                print("   (No high-activating exemplars)")
            print("   " + "="*76)
            
            # Print summary of variable assignments
            print(f"\n   📊 Variable Assignment Summary:")
            print(f"      - high_exemplars: {len(high_exemplars)} exemplars (top 10 highest activation, for Generation Evaluation)")
            print(f"      - prediction_examples_texts: {len(prediction_examples_texts)} texts (selected from remaining exemplars, for Prediction Evaluation)")
            print(f"      - prediction_examples_activations: {len(prediction_examples_activations)} activation values (corresponding to selected texts)")
            print(f"      - all_api_exemplars: {len(all_api_exemplars)} exemplars (all exemplars from API, sorted by activation descending)")
        else:
            print(f"   ⚠️  No exemplars retrieved, using original threshold: {activation_threshold:.4f}")
    except Exception as e:
        print(f"   ⚠️  Failed to get exemplars for threshold calculation: {e}")
        print(f"      Will use original threshold: {activation_threshold:.4f}")
        import traceback
        traceback.print_exc()

    # Step 1: Extract SAGE conclusion
    print("\n" + "="*80)
    print("STEP 1: Extract SAGE Explanation")
    print("="*80)
    try:
        sage_conclusion = extract_sage_conclusion(sage_results_path)
    except Exception as e:
        print(f"   ❌ Failed to extract SAGE conclusion: {e}")
        return {'error': str(e), 'feature_index': feature_index}

    # Step 2: Get Neuronpedia explanation
    print("\n" + "="*80)
    print("STEP 2: Generate Neuronpedia Explanation")
    print("="*80)
    # Check if explanation file exists locally
    explanation_file = os.path.join(output_dir, f'feature_{feature_index}_neuronpedia_explanation.json')
    explanation_file_exists = os.path.exists(explanation_file)
    
    # If explanation doesn't exist locally, we'll delete and regenerate if it exists on API
    force_regenerate = not explanation_file_exists
    
    neuronpedia_result = call_neuronpedia_api(
        model_id=neuronpedia_model_id,
        layer=layer,
        feature_index=feature_index,
        explanation_model_name=explanation_model_name,
        explanation_type=explanation_type,
        api_key=neuronpedia_api_key,
        output_dir=output_dir,
        force_regenerate=force_regenerate
    )

    if neuronpedia_result is None:
        print("❌ Failed to get Neuronpedia explanation. Skipping this feature.")
        return {'error': 'Neuronpedia API failed', 'feature_index': feature_index}

    # Format Neuronpedia explanation - will raise exception on error
    try:
        neuronpedia_explanation = format_neuronpedia_explanation(neuronpedia_result)
    except (ValueError, KeyError) as e:
        print(f"❌ Failed to extract Neuronpedia explanation: {e}")
        print("   Skipping this feature.")
        return {'error': f'Invalid Neuronpedia explanation: {str(e)}', 'feature_index': feature_index}

    # Steps 4-9: Generation and Prediction Evaluation
    # (All using Neuronpedia API instead of local system)
    try:
        # Initialize token usage tracking
        total_token_usage_gpt5 = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0}
        
        # Generate examples from SAGE
        print("\n" + "="*80)
        print("STEP 3: Generate Examples from SAGE Explanation")
        print("="*80)
        sage_examples, sage_gen_token_usage = generate_examples_from_explanation(
            sage_conclusion, "SAGE", num_examples=num_examples, llm_model=llm_model
        )
        # Accumulate token usage and cost
        total_token_usage_gpt5['prompt_tokens'] += sage_gen_token_usage['prompt_tokens']
        total_token_usage_gpt5['completion_tokens'] += sage_gen_token_usage['completion_tokens']
        total_token_usage_gpt5['total_tokens'] += sage_gen_token_usage['total_tokens']
        total_token_usage_gpt5['cost_input'] += sage_gen_token_usage.get('cost_input', 0.0)
        total_token_usage_gpt5['cost_output'] += sage_gen_token_usage.get('cost_output', 0.0)
        total_token_usage_gpt5['cost_total'] += sage_gen_token_usage.get('cost_total', 0.0)

        # Generate examples from Neuronpedia
        print("\n" + "="*80)
        print("STEP 4: Generate Examples from Neuronpedia Explanation")
        print("="*80)
        neuronpedia_examples, neuro_gen_token_usage = generate_examples_from_explanation(
            neuronpedia_explanation, "Neuronpedia", num_examples=num_examples, llm_model=llm_model
        )
        # Accumulate token usage and cost
        total_token_usage_gpt5['prompt_tokens'] += neuro_gen_token_usage['prompt_tokens']
        total_token_usage_gpt5['completion_tokens'] += neuro_gen_token_usage['completion_tokens']
        total_token_usage_gpt5['total_tokens'] += neuro_gen_token_usage['total_tokens']
        total_token_usage_gpt5['cost_input'] += neuro_gen_token_usage.get('cost_input', 0.0)
        total_token_usage_gpt5['cost_output'] += neuro_gen_token_usage.get('cost_output', 0.0)
        total_token_usage_gpt5['cost_total'] += neuro_gen_token_usage.get('cost_total', 0.0)

        # Evaluate SAGE examples using Neuronpedia API (using dynamic threshold)
        print("\n" + "="*80)
        print("STEP 5: Evaluate SAGE-Generated Examples (using Neuronpedia API)")
        print("="*80)
        print(f"   Using activation threshold: {dynamic_activation_threshold:.4f} (calculated from API exemplars)")
        sage_evaluation = evaluate_examples(
            sage_examples, 
            system=None,
            activation_threshold=dynamic_activation_threshold,
            use_api=True,
            model_id=neuronpedia_model_id,
            layer=layer,
            feature_index=feature_index
        )

        # Evaluate Neuronpedia examples using Neuronpedia API (using dynamic threshold)
        print("\n" + "="*80)
        print("STEP 6: Evaluate Neuronpedia-Generated Examples (using Neuronpedia API)")
        print("="*80)
        print(f"   Using activation threshold: {dynamic_activation_threshold:.4f} (calculated from API exemplars)")
        neuronpedia_evaluation = evaluate_examples(
            neuronpedia_examples, 
            system=None,
            activation_threshold=dynamic_activation_threshold,
            use_api=True,
            model_id=neuronpedia_model_id,
            layer=layer,
            feature_index=feature_index
        )

        # Prediction evaluation: Use selected exemplars from STEP 0 (no generation needed)
        # These exemplars already have activation values, so we can use them directly
        print("\n" + "="*80)
        print("STEP 7: Select Exemplars for Prediction Evaluation (from API, excluding top 10)")
        print("="*80)
        
        # Check if we have selected exemplars from STEP 0
        if prediction_examples_texts and prediction_examples_activations and len(prediction_examples_texts) >= 10:
            print(f"   ✅ Using selected exemplars from STEP 0:")
            print(f"      Selected {len(prediction_examples_texts)} exemplars with activation values")
            print(f"      Activation range: [{min(prediction_examples_activations):.4f}, {max(prediction_examples_activations):.4f}]")
            print(f"      💡 Note: These exemplars already have activation values, no API calls needed for activations")
        else:
            # Fallback: select exemplars from API if not available from STEP 0
            try:
                print(f"   📡 Selected exemplars not available from STEP 0, selecting from API...")
                if all_api_exemplars and len(all_api_exemplars) > 10:
                    prediction_examples_texts, prediction_examples_activations = select_exemplars_for_prediction_evaluation(
                        all_api_exemplars,
                        exclude_top_n=10,
                        num_high=4,
                        num_medium=3,
                        num_low=3
                    )
                else:
                    # Try to fetch all exemplars again
                    all_exemplars_fallback = get_activation_exemplars_from_api(
                        model_id=neuronpedia_model_id,
                        source=layer,
                        feature_index=feature_index,
                        return_all=True
                    )
                    
                    if all_exemplars_fallback and len(all_exemplars_fallback) > 10:
                        prediction_examples_texts, prediction_examples_activations = select_exemplars_for_prediction_evaluation(
                            all_exemplars_fallback,
                            exclude_top_n=10,
                            num_high=4,
                            num_medium=3,
                            num_low=3
                        )
                    else:
                        print(f"   ❌ Not enough exemplars to select from (need more than 10, got {len(all_exemplars_fallback) if all_exemplars_fallback else 0})")
                        prediction_examples_texts = []
                        prediction_examples_activations = []
            except Exception as e:
                print(f"   ❌ Failed to select exemplars: {e}")
                print("   ⚠️  Skipping prediction evaluation")
                import traceback
                traceback.print_exc()
                prediction_examples_texts = []
                prediction_examples_activations = []
        
        if prediction_examples_texts and len(prediction_examples_texts) >= 10 and len(prediction_examples_activations) == len(prediction_examples_texts):
            # Prediction evaluation for SAGE (using logprobs-based method, requires API for token-level activations)
            print("\n" + "="*80)
            print("STEP 8: Prediction Evaluation - SAGE (using logprobs-based method with API)")
            print("="*80)
            if global_max_activation is not None:
                print(f"   Using global max activation ({global_max_activation:.4f}) as normalization baseline for all samples")
            sage_prediction_eval = evaluate_prediction_ability(
                sage_conclusion, 
                prediction_examples_texts,  # Use selected exemplars from API
                activations=None,  # Not used in logprobs-based method, will get from API
                source="SAGE", 
                llm_model=llm_model,
                use_api=True,  # Required for logprobs-based prediction to get token-level activations
                model_id=neuronpedia_model_id,
                layer=layer,
                feature_index=feature_index,
                global_max_activation=global_max_activation  # Use global max activation for normalization
            )

            # Prediction evaluation for Neuronpedia (using logprobs-based method, requires API for token-level activations)
            print("\n" + "="*80)
            print("STEP 9: Prediction Evaluation - Neuronpedia (using logprobs-based method with API)")
            print("="*80)
            if global_max_activation is not None:
                print(f"   Using global max activation ({global_max_activation:.4f}) as normalization baseline for all samples")
            neuronpedia_prediction_eval = evaluate_prediction_ability(
                neuronpedia_explanation, 
                prediction_examples_texts,  # Use selected exemplars from API
                activations=None,  # Not used in logprobs-based method, will get from API
                source="Neuronpedia", 
                llm_model=llm_model,
                use_api=True,  # Required for logprobs-based prediction to get token-level activations
                model_id=neuronpedia_model_id,
                layer=layer,
                feature_index=feature_index,
                global_max_activation=global_max_activation  # Use global max activation for normalization
            )
            
            # Accumulate token usage from prediction evaluations (GPT-4o for logprobs)
            sage_pred_token_usage = sage_prediction_eval.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0})
            neuro_pred_token_usage = neuronpedia_prediction_eval.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0})
            
            total_token_usage_gpt4o = {
                'prompt_tokens': sage_pred_token_usage.get('prompt_tokens', 0) + neuro_pred_token_usage.get('prompt_tokens', 0),
                'completion_tokens': sage_pred_token_usage.get('completion_tokens', 0) + neuro_pred_token_usage.get('completion_tokens', 0),
                'total_tokens': sage_pred_token_usage.get('total_tokens', 0) + neuro_pred_token_usage.get('total_tokens', 0),
                'cost_input': sage_pred_token_usage.get('cost_input', 0.0) + neuro_pred_token_usage.get('cost_input', 0.0),
                'cost_output': sage_pred_token_usage.get('cost_output', 0.0) + neuro_pred_token_usage.get('cost_output', 0.0),
                'cost_total': sage_pred_token_usage.get('cost_total', 0.0) + neuro_pred_token_usage.get('cost_total', 0.0)
            }
        else:
            print(f"   ⚠️  Only selected {len(prediction_examples_texts) if prediction_examples_texts else 0} exemplars, need at least 10 for prediction evaluation")
            if prediction_examples_texts and prediction_examples_activations:
                print(f"      Texts: {len(prediction_examples_texts)}, Activations: {len(prediction_examples_activations)}")
            print("   ⚠️  Skipping prediction evaluation")
            sage_prediction_eval = {
                'correlation': None,
                'p_value': None,
                'predictions': [],
                'skipped': True,
                'reason': 'Insufficient exemplars selected or activation mismatch'
            }
            neuronpedia_prediction_eval = {
                'correlation': None,
                'p_value': None,
                'predictions': [],
                'skipped': True,
                'reason': 'Insufficient exemplars selected or activation mismatch'
            }
            total_token_usage_gpt4o = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_input': 0.0, 'cost_output': 0.0, 'cost_total': 0.0}

        # Compare results
        print("\n" + "="*80)
        step_num = 10 if prediction_examples_texts and len(prediction_examples_texts) >= 10 else 8
        print(f"STEP {step_num}: Compare Results (Both Methods)")
        print("="*80)
        comparison = compare_results(
            sage_evaluation, neuronpedia_evaluation,
            sage_prediction_eval, neuronpedia_prediction_eval
        )
        
        # Print total token usage and cost summary
        print("\n" + "="*80)
        print("📊 TOKEN USAGE & COST SUMMARY")
        print("="*80)
        print(f"   GPT-5 (for generation):")
        print(f"      Prompt tokens: {total_token_usage_gpt5['prompt_tokens']:,}")
        print(f"      Completion tokens: {total_token_usage_gpt5['completion_tokens']:,}")
        print(f"      Total tokens: {total_token_usage_gpt5['total_tokens']:,}")
        print(f"      Cost: ${total_token_usage_gpt5.get('cost_total', 0.0):.6f} (input: ${total_token_usage_gpt5.get('cost_input', 0.0):.6f}, output: ${total_token_usage_gpt5.get('cost_output', 0.0):.6f})")
        if total_token_usage_gpt4o['total_tokens'] > 0:
            print(f"   GPT-4o (for logprobs prediction):")
            print(f"      Prompt tokens: {total_token_usage_gpt4o['prompt_tokens']:,}")
            print(f"      Completion tokens: {total_token_usage_gpt4o['completion_tokens']:,}")
            print(f"      Total tokens: {total_token_usage_gpt4o['total_tokens']:,}")
            print(f"      Cost: ${total_token_usage_gpt4o.get('cost_total', 0.0):.6f} (input: ${total_token_usage_gpt4o.get('cost_input', 0.0):.6f}, output: ${total_token_usage_gpt4o.get('cost_output', 0.0):.6f})")
        total_all_tokens = total_token_usage_gpt5['total_tokens'] + total_token_usage_gpt4o.get('total_tokens', 0)
        total_all_cost = total_token_usage_gpt5.get('cost_total', 0.0) + total_token_usage_gpt4o.get('cost_total', 0.0)
        print(f"   Total (all models):")
        print(f"      Tokens: {total_all_tokens:,}")
        print(f"      Cost: ${total_all_cost:.6f}")
        print("="*80)

        # Prepare output data
        output_data = {
            'configuration': {
                'model_name': model_name,  # Kept for record-keeping only
                'sae_path': sae_path,  # Kept for record-keeping only (optional)
                'layer': layer,  # REQUIRED for API calls
                'feature_index': feature_index,  # REQUIRED for API calls
                'llm_model': llm_model,
                'num_examples': num_examples,
                'activation_threshold_original': activation_threshold,  # Original threshold provided as parameter
                'activation_threshold_used': dynamic_activation_threshold,  # Dynamic threshold calculated from API exemplars
                'activation_threshold_source': 'dynamic_from_api_exemplars' if high_exemplars else 'original_parameter',
                'evaluation_method': 'neuronpedia_api',
                'generation_evaluation_uses_api': True,
                'prediction_evaluation_uses_api': True,
                'prediction_evaluation_uses_exemplars': True,  # New: indicates using exemplars, not explanations
                'prediction_evaluation_uses_selected_exemplars': True,  # New: indicates using selected exemplars directly (no generation)
                'prediction_evaluation_activations_from_exemplars': True,  # New: indicates using activation values from exemplars (no API calls needed)
                'prediction_evaluation_uses_global_normalization': True,  # New: indicates using global max activation for normalization
                'global_max_activation': global_max_activation,  # Global max activation from all exemplars (used as normalization baseline)
                'neuronpedia_model_id': neuronpedia_model_id,  # REQUIRED for API calls
                'token_usage': {
                    'gpt5': total_token_usage_gpt5,  # GPT-5 token usage for generation
                    'gpt4o': total_token_usage_gpt4o,  # GPT-4o token usage for logprobs
                    'total': {
                        'prompt_tokens': total_token_usage_gpt5['prompt_tokens'] + total_token_usage_gpt4o.get('prompt_tokens', 0),
                        'completion_tokens': total_token_usage_gpt5['completion_tokens'] + total_token_usage_gpt4o.get('completion_tokens', 0),
                        'total_tokens': total_token_usage_gpt5['total_tokens'] + total_token_usage_gpt4o.get('total_tokens', 0),
                        'cost_input': total_token_usage_gpt5.get('cost_input', 0.0) + total_token_usage_gpt4o.get('cost_input', 0.0),
                        'cost_output': total_token_usage_gpt5.get('cost_output', 0.0) + total_token_usage_gpt4o.get('cost_output', 0.0),
                        'cost_total': total_token_usage_gpt5.get('cost_total', 0.0) + total_token_usage_gpt4o.get('cost_total', 0.0)
                    },
                    'pricing': {
                        'gpt5': {
                            'input_per_1m': 0.625,
                            'cached_input_per_1m': 0.0625,
                            'output_per_1m': 5.00
                        },
                        'gpt4o': {
                            'input_per_1m': 1.25,
                            'output_per_1m': 5.00
                        }
                    }
                },
                'threshold_calculation': {
                    'exemplars_used': len(high_exemplars) if high_exemplars else 0,
                    'max_activations': [ex.get('activation', 0.0) for ex in high_exemplars[:10]] if high_exemplars else [],
                    'average_max_activation': dynamic_activation_threshold if high_exemplars else activation_threshold,
                    'min_max_activation': min([ex.get('activation', 0.0) for ex in high_exemplars[:10]]) if high_exemplars and len(high_exemplars) > 0 else None,
                    'max_max_activation': max([ex.get('activation', 0.0) for ex in high_exemplars[:10]]) if high_exemplars and len(high_exemplars) > 0 else None
                },
                'prediction_evaluation_selection': {
                    'total_exemplars': len(all_api_exemplars) if all_api_exemplars else 0,
                    'excluded_top_n': 10,  # Top 10 used for Generation Evaluation
                    'selected_count': len(prediction_examples_texts) if prediction_examples_texts else 0,
                    'selection_distribution': {
                        'num_high': 4,
                        'num_medium': 3,
                        'num_low': 3
                    },
                    'selected_texts': prediction_examples_texts if prediction_examples_texts else [],
                    'selected_activations': prediction_examples_activations if prediction_examples_activations else [],
                    'activation_range': [min(prediction_examples_activations), max(prediction_examples_activations)] if prediction_examples_activations and len(prediction_examples_activations) > 0 else None
                },
                'note': 'All activation evaluations use Neuronpedia API. Activation threshold is dynamically calculated from top 10 high-activating API exemplars (average of max activations / 2). Generation Evaluation uses top 10 high-activating exemplars. Prediction Evaluation uses selected exemplars directly from API (excluding top 10), with activation values already available (no API calls needed for activations). model_name and sae_path are kept for record-keeping only and are not used for any calculations.'
            },
            'sage': {
                'explanation': sage_conclusion,
                'generated_examples': sage_examples,
                'generation_evaluation': sage_evaluation,
                'prediction_evaluation': sage_prediction_eval
            },
            'neuronpedia': {
                'explanation': neuronpedia_explanation,
                'raw_api_response': neuronpedia_result,
                'explanation_source': neuronpedia_result.get('source', 'unknown'),  # 'new', 'existing', or 'existing_fallback'
                'explanation_id': neuronpedia_result.get('explanation', {}).get('id') if neuronpedia_result.get('explanation') else None,
                'all_available_explanations': neuronpedia_result.get('all_explanations', []),  # All explanations from API
                'generated_examples': neuronpedia_examples,
                'generation_evaluation': neuronpedia_evaluation,
                'prediction_evaluation': neuronpedia_prediction_eval
            },
            'prediction_evaluation': {
                'selected_exemplars': [
                    {
                        'text': text,
                        'activation': activation
                    } for text, activation in zip(prediction_examples_texts, prediction_examples_activations)
                ] if prediction_examples_texts and prediction_examples_activations else [],
                'selection_method': 'random_selection_from_categorized_exemplars',
                'selection_details': {
                    'excluded_top_n': 10,
                    'num_high_selected': 4,
                    'num_medium_selected': 3,
                    'num_low_selected': 3,
                    'total_selected': len(prediction_examples_texts) if prediction_examples_texts else 0
                },
                'note': 'Prediction Evaluation uses selected exemplars directly from API (excluding top 10 used for Generation Evaluation). Exemplars are categorized into high, medium, and low activation groups from remaining exemplars, and randomly selected from each group. Activation values are already available from exemplars, so no API calls are needed to get activations. This avoids bias from explanation text and ensures diverse activation levels in evaluation examples.'
            },
            'comparison': comparison
        }

        # Save results
        print("\n" + "="*80)
        step_num = step_num + 1
        print(f"STEP {step_num}: Save Results")
        print("="*80)
        output_file = os.path.join(output_dir, f'feature_{feature_index}_comparison.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Results saved to: {output_file}")
        except Exception as e:
            error_msg = f"Failed to save results to {output_file}: {e}"
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
        
        # Export Neuronpedia explanation separately for easy access
        explanation_file = os.path.join(output_dir, f'feature_{feature_index}_neuronpedia_explanation.json')
        explanation_export_data = {
            'feature_index': feature_index,
            'model_id': neuronpedia_model_id,
            'layer': layer,
            'explanation_source': neuronpedia_result.get('source', 'unknown'),  # 'new', 'existing', or 'existing_fallback'
            'explanation_model_name': explanation_model_name,
            'explanation_type': explanation_type,
            'explanation': neuronpedia_result.get('explanation', {}),  # Complete explanation data
            'all_available_explanations': neuronpedia_result.get('all_explanations', []),  # All explanations from API
            'explanation_id': neuronpedia_result.get('explanation', {}).get('id') if neuronpedia_result.get('explanation') else None,
            'formatted_explanation': neuronpedia_explanation,  # The formatted version used in evaluation
            'exported_at': datetime.now().isoformat()
        }
        
        # Add warning if fallback was used
        if neuronpedia_result.get('warning'):
            explanation_export_data['warning'] = neuronpedia_result.get('warning')
        
        # Export explanation - raise exception on failure
        try:
            with open(explanation_file, 'w', encoding='utf-8') as f:
                json.dump(explanation_export_data, f, indent=2, ensure_ascii=False)
            
            source_msg = neuronpedia_result.get('source', 'unknown')
            if source_msg == 'existing':
                print(f"   ✅ Neuronpedia explanation (existing) exported to: {explanation_file}")
            elif source_msg == 'existing_fallback':
                print(f"   ⚠️  Neuronpedia explanation (fallback) exported to: {explanation_file}")
                if neuronpedia_result.get('warning'):
                    print(f"      Warning: {neuronpedia_result.get('warning')}")
            elif source_msg == 'new':
                print(f"   ✅ Neuronpedia explanation (newly created) exported to: {explanation_file}")
            else:
                print(f"   ✅ Neuronpedia explanation exported to: {explanation_file}")
        except Exception as e:
            error_msg = f"Failed to export Neuronpedia explanation to {explanation_file}: {e}"
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg) from e

        return output_data

    except Exception as e:
        print(f"   ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'feature_index': feature_index}


def main():
    parser = argparse.ArgumentParser(
        description='Compare SAGE vs Neuronpedia feature explanations'
    )

    # SAGE configuration (supports both single file and batch mode)
    parser.add_argument('--sage_results_path', type=str, required=True,
                       help='Path to SAGE structured_results.json OR base directory (e.g., ./results/gpt-5/layer_6/)')

    # Feature specification
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument('--feature_index', type=int,
                              help='Single feature index')
    feature_group.add_argument('--features', type=str,
                              help='Multiple feature indices (comma-separated, e.g., "0,1,2,3")')

    # Shared configuration (must match between SAGE and Neuronpedia)
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (optional, kept for compatibility/record-keeping only, not used for activations. If not provided, will use neuronpedia_model_id)')
    parser.add_argument('--sae_path', type=str, default=None,
                       help='SAE path or SAE Lens URI (optional, kept for compatibility/record-keeping only, not used for activations)')
    parser.add_argument('--layer', type=str, required=True,
                       help='Layer identifier as string (e.g., "11-resid-post-aa", "0-gemmascope-mlp-16k")')

    # Neuronpedia API configuration
    parser.add_argument('--neuronpedia_model_id', type=str, default='llama3.1-8b-it',
                       help='Neuronpedia model ID (default: llama3.1-8b-it)')
    parser.add_argument('--neuronpedia_api_key', type=str,
                       default=os.environ.get('NEURONPEDIA_API_KEY'),
                       help='Neuronpedia API key (default: from NEURONPEDIA_API_KEY environment variable)')
    parser.add_argument('--explanation_model_name', type=str, default='gpt-5',
                       help='Explanation model name: gpt-5, gpt-4o, or o4-mini (default: gpt-5)')
    parser.add_argument('--explanation_type', type=str, default='oai_token-act-pair',
                       help='Explanation type (default: oai_token-act-pair)')

    # Evaluation configuration
    parser.add_argument('--llm_model', type=str, default='gpt-5',
                       help='LLM model for generating examples (default: gpt-5)')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to generate per explanation (default: 10)')
    parser.add_argument('--activation_threshold', type=float, default=8.0,
                       help='Activation threshold for success (default: 8.0)')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device (default: cuda)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as SAGE results)')

    args = parser.parse_args()

    # Parse feature indices
    if args.features:
        # Batch mode: parse comma-separated features
        feature_indices = [int(f.strip()) for f in args.features.split(',')]
        batch_mode = True
    else:
        # Single feature mode
        feature_indices = [args.feature_index]
        batch_mode = False

    # Determine if sage_results_path is a directory or file
    sage_path = Path(args.sage_results_path)
    if sage_path.is_dir():
        # Directory mode: construct paths automatically
        base_dir = sage_path
        auto_construct_paths = True
    elif sage_path.is_file():
        # File mode: single feature only
        if batch_mode:
            print("❌ Error: Cannot use --features with a single file path.")
            print("   Use a directory path (e.g., ./results/gpt-5/layer_6/) for batch processing.")
            sys.exit(1)
        base_dir = sage_path.parent
        auto_construct_paths = False
    else:
        print(f"❌ Error: Path does not exist: {args.sage_results_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(base_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("🔬 SAGE vs Neuronpedia Comparison Evaluation")
    print("="*80)
    
    # Use neuronpedia_model_id as model_name if model_name is not provided
    if args.model_name is None:
        args.model_name = args.neuronpedia_model_id
    
    print(f"\nConfiguration (for record-keeping only):")
    print(f"  Model: {args.model_name}")
    if args.sae_path:
        print(f"  SAE: {args.sae_path}")
    else:
        print(f"  SAE: (not provided, not needed for API mode)")
    print(f"\nAPI Configuration (REQUIRED for activation calculations):")
    print(f"  Layer: {args.layer}")
    print(f"  Neuronpedia Model ID: {args.neuronpedia_model_id}")
    print(f"  Explanation Model: {args.explanation_model_name}")
    print(f"  Explanation Type: {args.explanation_type}")
    print(f"\nEvaluation Configuration:")
    print(f"  Features: {feature_indices}")
    print(f"  LLM for generation: {args.llm_model}")
    print(f"  Examples per system: {args.num_examples}")
    print(f"  Activation threshold (initial): {args.activation_threshold}")
    print(f"  Note: Activation threshold will be dynamically calculated from top 10 API exemplars")
    print(f"        (average of max activations). Initial value used as fallback if API fails.")
    print(f"\nPaths:")
    print(f"  Base directory: {base_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Batch mode: {batch_mode}")
    print(f"\n⚠️  Note: All activations are fetched from Neuronpedia API.")
    print(f"   model_name and sae_path are kept for record-keeping only.")

    # Process features
    all_results = []
    success_count = 0
    error_count = 0

    for feature_idx in feature_indices:
        # Construct path to SAGE results for this feature
        if auto_construct_paths:
            # Auto-construct path: base_dir/feature_{idx}/structured_results.json
            sage_results_file = base_dir / f"feature_{feature_idx}" / "structured_results.json"
            feature_output_dir = base_dir / f"feature_{feature_idx}"
        else:
            # Use provided file path
            sage_results_file = sage_path
            feature_output_dir = base_dir

        # Check if file exists
        if not sage_results_file.exists():
            print(f"\n⚠️  Skipping Feature {feature_idx}: File not found - {sage_results_file}")
            error_count += 1
            all_results.append({'error': 'File not found', 'feature_index': feature_idx})
            continue

        os.makedirs(feature_output_dir, exist_ok=True)

        # Process this feature
        result = process_single_feature(
            sage_results_path=str(sage_results_file),
            model_name=args.model_name,
            sae_path=args.sae_path,
            layer=args.layer,
            feature_index=feature_idx,
            neuronpedia_model_id=args.neuronpedia_model_id,
            neuronpedia_api_key=args.neuronpedia_api_key,
            llm_model=args.llm_model,
            num_examples=args.num_examples,
            activation_threshold=args.activation_threshold,
            device=args.device,
            output_dir=str(feature_output_dir),
            explanation_model_name=args.explanation_model_name,
            explanation_type=args.explanation_type
        )

        all_results.append(result)

        if 'error' in result:
            error_count += 1
        else:
            success_count += 1

    # Print batch summary
    if batch_mode:
        print("\n" + "="*80)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"\nTotal features processed: {len(feature_indices)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")

        print(f"\n{'Feature':<10} {'Status':<15} {'Gen Winner':<15} {'Pred Winner':<15} {'Overall':<15}")
        print("-" * 75)

        for result in all_results:
            feat_idx = result.get('feature_index', '?')
            if 'error' in result:
                print(f"{feat_idx:<10} {'ERROR':<15} {'-':<15} {'-':<15} {'-':<15}")
            else:
                comp = result.get('comparison', {})
                gen_winner = comp.get('generation', {}).get('winner', '?')
                pred_winner = comp.get('prediction', {}).get('winner', '?') if 'prediction' in comp else 'N/A'
                overall = comp.get('overall_winner', '?')
                print(f"{feat_idx:<10} {'SUCCESS':<15} {gen_winner:<15} {pred_winner:<15} {overall:<15}")

        # Count overall wins
        sage_wins = sum(1 for r in all_results if not 'error' in r and r.get('comparison', {}).get('overall_winner') == 'SAGE')
        neuro_wins = sum(1 for r in all_results if not 'error' in r and r.get('comparison', {}).get('overall_winner') == 'Neuronpedia')
        ties = sum(1 for r in all_results if not 'error' in r and r.get('comparison', {}).get('overall_winner') == 'TIE')

        print(f"\n🏆 Overall Statistics:")
        print(f"   SAGE Wins: {sage_wins}/{success_count}")
        print(f"   Neuronpedia Wins: {neuro_wins}/{success_count}")
        print(f"   Ties: {ties}/{success_count}")

        # Save batch summary
        batch_summary_file = os.path.join(args.output_dir, 'batch_comparison_summary.json')
        batch_summary = {
            'configuration': {
                'model_name': args.model_name,  # Record-keeping only
                'sae_path': args.sae_path,  # Record-keeping only (optional)
                'layer': args.layer,  # Used for API calls
                'neuronpedia_model_id': args.neuronpedia_model_id,  # Used for API calls
                'features': feature_indices,
                'llm_model': args.llm_model,
                'num_examples': args.num_examples,
                'activation_threshold': args.activation_threshold,
                'note': 'All activations use Neuronpedia API. model_name and sae_path are record-keeping only.'
            },
            'summary_stats': {
                'total_features': len(feature_indices),
                'successful': success_count,
                'errors': error_count,
                'sage_wins': sage_wins,
                'neuronpedia_wins': neuro_wins,
                'ties': ties
            },
            'results': all_results
        }

        with open(batch_summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)

        print(f"\n✅ Batch summary saved to: {batch_summary_file}")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()