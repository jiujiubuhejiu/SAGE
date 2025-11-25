"""
sage_system.py

Core interface to interact with the analyzed model (e.g., Mistral-7B + SAE).

This module implements the System API for SAGE,
providing SAE-backed feature extraction and text analysis capabilities.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    AutoTokenizer = None
    AutoModel = None

# Optional: TransformerLens for HookedTransformer
try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None

# Optional: SAELens for loading pretrained SAEs from Hugging Face
try:
    from sae_lens import SAE as SAELensSAE  # type: ignore
    from sae_lens import HookedSAETransformer  # type: ignore
    SAELENS_AVAILABLE = True
except Exception:
    SAELENS_AVAILABLE = False
    SAELensSAE = None  # type: ignore
    HookedSAETransformer = None  # type: ignore


@dataclass
class FeatureActivationResult:
    """Result of feature activation analysis for a single text.
    
    Attributes:
        text: Original input text
        activation_max: Maximum activation value across all tokens
        activation_mean: Mean activation value across all tokens
        activation_sum: Sum of activation values
        max_token_index: Index of token with maximum activation
        tokens: List of token strings
        per_token_activations: Activation values for each token
        layer: Layer number where activation was computed
        feature_index: SAE feature index
    """
    text: str
    activation_max: float
    activation_mean: float
    activation_sum: float
    max_token_index: int
    tokens: List[str]
    per_token_activations: List[float]
    layer: int
    feature_index: int


@dataclass
class SAEConfig:
    """Configuration for SAE-backed feature extraction.

    Attributes:
        sae_checkpoint_path: Path to the SAE checkpoint (e.g., `sae_models/mistral_layer15.pt`).
        model_name: Base LM name identifier.
        target_layer: Which transformer layer the SAE corresponds to.
    """

    sae_checkpoint_path: str
    model_name: str = "mistral-7b"
    target_layer: int = 15


class System:
    """
    A Python class containing the language model and the specific SAE feature to interact with.
    
    This class provides the System API for SAGE, adapted for text/SAE analysis.
    
    Attributes
    ----------
    feature_index : int
        The serial number of the SAE feature.
    layer : int
        The layer number where the SAE feature is located.
    model_name : string
        The name of the language model.
    model : nn.Module
        The loaded PyTorch model.
    tokenizer : AutoTokenizer
        The tokenizer for the language model.
    sae : dict
        The loaded SAE checkpoint containing encoder/decoder weights.
    device : torch.device
        The device (CPU/GPU) used for computations.
    threshold : float
        Activation threshold for feature analysis.

    Methods
    -------
    load_model(model_name: str) -> nn.Module
        Loads the language model from HuggingFace.
    load_sae(sae_path: str) -> dict
        Loads the SAE checkpoint.
    call_feature(text_list: List[str]) -> Tuple[List[float], List[str]]
        Returns the feature activation for each text in the input list.
    """

    def __init__(self, llm_name: str, sae_path: str, sae_layer: int, feature_index: int, device: str = "cpu", thresholds: Optional[Dict] = None, debug: bool = False, use_api_for_activations: bool = False, neuronpedia_model_id: Optional[str] = None, neuronpedia_source: Optional[str] = None):
        """
        Initializes a SAE feature object by specifying its index and layer location
        and the language model that the feature belongs to.

        Parameters
        ----------
        llm_name : str
            The name of the language model (e.g., 'mistralai/Mistral-7B-v0.1').
        sae_path : str
            Path to the SAE checkpoint file.
        sae_layer : int
            The layer number that the SAE feature is located at.
        feature_index : int
            The serial number of the SAE feature.
        device : str
            The computational device ('cpu' or 'cuda').
        thresholds : dict, optional
            Dictionary containing threshold values for features.
        debug : bool, optional
            Enable debug output (default: False).
        use_api_for_activations : bool, optional
            If True, use Neuronpedia API for get_activation_trace (default: False).
        neuronpedia_model_id : str, optional
            Neuronpedia model ID for API calls (required if use_api_for_activations=True).
        neuronpedia_source : str, optional
            Neuronpedia source/layer identifier for API calls (required if use_api_for_activations=True).
        """
        self.feature_index = int(feature_index)  # Ensure it's always an integer
        self.layer = sae_layer
        self.debug = debug  # Store debug flag
        self.use_api_for_activations = use_api_for_activations
        self.neuronpedia_model_id = neuronpedia_model_id
        self.neuronpedia_source = neuronpedia_source
        # Robust device parsing: "cpu" forces CPU; numeric or "cuda" selects GPU if available
        if TORCH_AVAILABLE:
            parsed_device: Union[str, torch.device]
            device_str = str(device).lower()

            if device_str == "cpu":
                parsed_device = "cpu"
            elif device_str in ["gpu", "cuda"]:
                # "gpu" or "cuda" -> use cuda if available
                parsed_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                # Accept integers like 0/1 or strings like "0", "cuda:0"
                try:
                    if isinstance(device, (int,)) or (isinstance(device, str) and device.isdigit()):
                        gpu_index = int(device) if not isinstance(device, int) else device
                        parsed_device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
                    else:
                        # Try to parse as torch device
                        parsed_device = device if torch.cuda.is_available() else "cpu"
                except Exception:
                    parsed_device = "cpu"

            self.device = torch.device(parsed_device) if isinstance(parsed_device, str) else parsed_device
            print(f"🖥️  Device: {self.device} (requested: {device})")
        else:
            self.device = "cpu"
        self.model_name = llm_name
        self.sae_path = sae_path

        # Determine if we're using SAELens (which requires HookedTransformer)
        self.use_hooked_transformer = sae_path.startswith("sae-lens://") if isinstance(sae_path, str) else False

        # Load model and SAE only if not using API for activations
        if self.use_api_for_activations:
            # Skip model and SAE loading when using API
            print("📡 API mode enabled: Skipping model and SAE loading (using Neuronpedia API instead)")
            self.model = None
            self.tokenizer = None
            self.sae = None
        elif TORCH_AVAILABLE:
            self.sae = self._load_sae(sae_path)  # Load SAE first to check compatibility
            self.model = self._load_model(llm_name)
            self.tokenizer = self._load_tokenizer(llm_name)
        else:
            self.model = None
            self.tokenizer = None
            self.sae = None
            
        # Set threshold
        if thresholds is not None and self.layer in thresholds and self.feature_index in thresholds[self.layer]:
            self.threshold = thresholds[self.layer][self.feature_index]
        else:
            self.threshold = 0.0

    def _load_model(self, model_name: str) -> Any:
        """
        Loads the language model from HuggingFace.

        Parameters
        ----------
        model_name : str
            The name of the model to load.

        Returns
        -------
        nn.Module
            The loaded PyTorch language model.
        """
        try:
            if self.use_hooked_transformer:
                # Use HookedSAETransformer for SAELens compatibility
                if SAELENS_AVAILABLE and HookedSAETransformer is not None:
                    print(f"Loading HookedSAETransformer for model: {model_name}")
                    model = HookedSAETransformer.from_pretrained(
                        model_name=model_name,
                        device=str(self.device),
                        dtype="float32" if self.device.type == "cpu" else "float16"
                    )
                    return model
                elif TRANSFORMER_LENS_AVAILABLE and HookedTransformer is not None:
                    print(f"Loading HookedTransformer for model: {model_name}")
                    model = HookedTransformer.from_pretrained(
                        model_name,
                        device=str(self.device),
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                    )
                    return model
                else:
                    raise RuntimeError("Neither SAELens nor TransformerLens is available for HookedTransformer")
            else:
                # Use standard HuggingFace AutoModel
                print(f"Loading AutoModel for model: {model_name}")
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                )
                model = model.to(self.device).eval()
                return model
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            return None

    def _load_tokenizer(self, model_name: str) -> Any:
        """
        Loads the tokenizer for the language model.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load tokenizer for.
        
        Returns
        -------
        AutoTokenizer
            The loaded tokenizer.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_name}: {e}")
            return None

    def _load_sae(self, sae_path: str) -> Dict[str, Any]:
        """
        Loads the SAE checkpoint.
        
        Parameters
        ----------
        sae_path : str
            Path to the SAE checkpoint file.
        
        Returns
        -------
        dict
            Dictionary containing SAE encoder/decoder weights.
        """
        try:
            # Support SAELens URI scheme: "sae-lens://release=...;sae_id=..."
            if isinstance(sae_path, str) and sae_path.startswith("sae-lens://"):
                if not SAELENS_AVAILABLE:
                    print("Warning: sae-lens not installed; cannot load SAEs from Hugging Face.")
                    return {}
                spec = sae_path[len("sae-lens://"):]
                parts = [p.strip() for p in spec.split(";") if p.strip()]
                kv: Dict[str, str] = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        kv[k.strip()] = v.strip()
                release = kv.get("release") or kv.get("repo") or kv.get("model")
                sae_id = kv.get("sae_id") or kv.get("path")
                if not release or not sae_id:
                    print("Warning: Invalid sae-lens URI. Expected keys: release and sae_id")
                    return {}
                # Load SAE and move to correct device
                print(f"Loading SAE from {release}/{sae_id} to device {self.device}")
                sae_obj = SAELensSAE.from_pretrained(
                    release=release,
                    sae_id=sae_id,
                    device=str(self.device)  # ← 关键修复：指定设备
                )  # type: ignore
                print(f"✅ SAE loaded on {self.device}")
                return {"__sae_lens_obj__": sae_obj, "__source__": "sae-lens", "release": release, "sae_id": sae_id}

            # Local checkpoint path
            if os.path.exists(sae_path):
                sae_data = torch.load(sae_path, map_location=self.device)
                return sae_data
            else:
                print(f"Warning: SAE file not found at {sae_path}")
                return {}
        except Exception as e:
            print(f"Warning: Could not load SAE from {sae_path}: {e}")
            return {}

    def call_feature(self, text_list: List[str], batch_size: int = 8, return_full_info: bool = False) -> Union[Tuple[List[float], List[str]], List[FeatureActivationResult]]:
        """
        Returns the SAE feature's activation value for each text in the input list.
        
        Parameters
        ----------
        text_list : List[str]
            List of input texts to analyze.
        batch_size : int
            Batch size for processing multiple texts simultaneously.
        return_full_info : bool
            If True, returns FeatureActivationResult objects with complete information.
            If False, returns (activation_list, text_list) for compatibility.
        
        Returns
        -------
        Union[Tuple[List[float], List[str]], List[FeatureActivationResult]]
            If return_full_info=False: (activation_list, text_list) for compatibility
            If return_full_info=True: List[FeatureActivationResult] with complete analysis
            
        Examples
        --------
        >>> # Basic usage (compatible mode)
        >>> texts = ["import argparse", "def main():", "class MyClass:"]
        >>> activation_list, text_list = system.call_feature(texts)
        
        >>> # Full analysis mode
        >>> results = system.call_feature(texts, return_full_info=True)
        >>> for result in results:
        >>>     print(f"Text: {result.text}")
        >>>     print(f"Max activation: {result.activation_max} at token: {result.tokens[result.max_token_index]}")
        >>>     print(f"Activation pattern: {result.per_token_activations}")
        """
        # Check if API mode is enabled
        if self.use_api_for_activations:
            raise RuntimeError(
                "call_feature is not available in API mode. "
                "Use get_activation_trace() for individual texts or "
                "use Tools.find_maximally_activating_examples() which uses the API automatically."
            )
        
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            raise RuntimeError("PyTorch, model, or tokenizer not available. Cannot compute feature activations.")
            
        if not self.sae:
            raise RuntimeError("No SAE loaded. Cannot compute feature activations without SAE.")
        
        results = []
        
        # Process texts in batches
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)
        
        if return_full_info:
            return results
        else:
            # Return compatibility format
            activation_list = [result.activation_max for result in results]
            text_list = [result.text for result in results]
            return activation_list, text_list
    
    def _process_batch(self, batch_texts: List[str]) -> List[FeatureActivationResult]:
        results = []

        try:
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            input_ids = inputs['input_ids'].to(self.device)

            with torch.no_grad():
                if self.use_hooked_transformer:
                    # Use TransformerLens's hook system for MLP output
                    hook_name = f"blocks.{self.layer}.hook_mlp_out"
                    _, cache = self.model.run_with_cache(
                        input_ids,
                        names_filter=[hook_name]
                    )
                    layer_activations = cache[hook_name]  # [batch, seq, hidden_dim]
                else:
                    # Use standard HuggingFace forward pass
                    inputs_dict = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs_dict, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                    if self.layer >= len(hidden_states):
                        raise ValueError(f"Layer {self.layer} not available. Model has {len(hidden_states)} layers.")

                    layer_activations = hidden_states[self.layer]  # [batch, seq, hidden_dim]

            # Process each text
            for batch_idx, text in enumerate(batch_texts):
                text_activations = layer_activations[batch_idx]  # [seq, hidden_dim]

                # Get tokens
                text_input_ids = input_ids[batch_idx]
                tokens = self.tokenizer.convert_ids_to_tokens(text_input_ids)

                # Remove padding
                attention_mask = inputs['attention_mask'][batch_idx]
                non_pad_mask = attention_mask.bool()
                text_activations = text_activations[non_pad_mask]
                tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]

                # Apply SAE
                feature_activations = self._apply_sae(text_activations)

                # Create result
                result = FeatureActivationResult(
                    text=text,
                    activation_max=float(feature_activations.max().item()),
                    activation_mean=float(feature_activations.mean().item()),
                    activation_sum=float(feature_activations.sum().item()),
                    max_token_index=int(feature_activations.argmax().item()),
                    tokens=tokens,
                    per_token_activations=feature_activations.tolist(),
                    layer=self.layer,
                    feature_index=self.feature_index
                )
                results.append(result)

        except Exception as e:
            print(f"Error in batch processing: {e}")
            import traceback
            traceback.print_exc()
            # Create error results for all texts in batch
            for text in batch_texts:
                error_result = FeatureActivationResult(
                    text=text,
                    activation_max=0.0,
                    activation_mean=0.0,
                    activation_sum=0.0,
                    max_token_index=0,
                    tokens=[],
                    per_token_activations=[],
                    layer=self.layer,
                    feature_index=self.feature_index
                )
                results.append(error_result)

        return results
    
    def _apply_sae(self, layer_activations: torch.Tensor) -> torch.Tensor:
        if self.sae and "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            
            # Ensure correct shape: [batch, seq, hidden]
            if layer_activations.ndim == 2:
                layer_activations = layer_activations.unsqueeze(0)
            
            # Encode
            sae_features = sae_obj.encode(layer_activations)  # [batch, seq, n_features]
            
            # Extract target feature
            if self.feature_index >= sae_features.shape[-1]:
                raise ValueError(f"Feature {self.feature_index} out of range")
            
            return sae_features[0, :, self.feature_index]  # [seq]
        else:
            raise RuntimeError("SAE not properly loaded")

    def _get_activation_trace_from_api(self, text: str) -> Dict[str, Any]:
        """
        Get activation trace from Neuronpedia API (without filtering special tokens).
        
        Note: This is for get_activation_trace, which doesn't filter <|endoftext|> tokens
        because it's getting activation data from the API's stored activations, not custom text.
        
        Args:
            text: Text to get activation for (may not be used if API returns stored activations)
        
        Returns:
            dict with activation trace data compatible with get_activation_trace format
        """
        if not self.use_api_for_activations:
            raise ValueError("API mode not enabled")
        
        if not self.neuronpedia_model_id or not self.neuronpedia_source:
            raise ValueError("Neuronpedia model_id and source required for API calls")
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for API calls. Install with: pip install requests")
        
        url = "https://www.neuronpedia.org/api/activation/new"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "feature": {
                "modelId": self.neuronpedia_model_id,
                "source": self.neuronpedia_source,
                "index": str(self.feature_index)
            },
            "customText": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if response contains error field
            if 'error' in result and result['error']:
                raise ValueError(f"API returned error: {result['error']}")
            
            # Validate required fields
            if 'maxValue' not in result:
                raise ValueError(f"API response missing 'maxValue' field: {result}")
            
            # Extract activation data (NO filtering of special tokens for get_activation_trace)
            tokens = result.get('tokens', [])
            values = result.get('values', [])
            max_value = float(result.get('maxValue', 0.0))
            min_value = float(result.get('minValue', 0.0))
            max_value_token_index = result.get('maxValueTokenIndex', 0)
            
            # Calculate statistics
            mean_value = sum(values) / len(values) if values else 0.0
            sum_value = sum(values) if values else 0.0
            
            # Convert to get_activation_trace format
            trace = {
                "tokens": tokens,
                "token_ids": [],  # API doesn't return token IDs
                "per_token_activation": values,
                "summary_activation": max_value,  # max activation (primary metric)
                "summary_activation_mean": mean_value,
                "summary_activation_sum": sum_value,
                "max_token_index": max_value_token_index,
                "layer_index": self.layer,
                "shapes": {
                    "tokens_count": len(tokens),
                    "values_count": len(values)
                },
                "raw_stats": {
                    "min": min_value,
                    "max": max_value,
                    "mean": mean_value,
                    "sum": sum_value,
                    "std": (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5 if values else 0.0,
                    "count": len(values)
                }
            }
            
            return trace
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Neuronpedia API HTTP error for text '{text[:50]}...': {e.response.status_code}"
            if e.response.text:
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data.get('message', e.response.text[:200])}"
                except:
                    error_msg += f" - {e.response.text[:200]}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Neuronpedia API request failed for text '{text[:50]}...': {e}"
            ) from e
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid response from Neuronpedia API for text '{text[:50]}...': {e}"
            ) from e

    def get_activation_trace(self, text: str) -> Dict[str, Any]:
        """Return a detailed activation trace for a single text input.

        The trace includes:
        - tokens: decoded tokens
        - token_ids: input ids
        - per_token_activation: activation per token for the selected feature (if available)
        - summary_activation: max activation across tokens (primary metric for feature discovery)
        - summary_activation_mean: mean activation across tokens (additional metric)
        - summary_activation_sum: sum of activations across tokens
        - max_token_index: index of token with maximum activation
        - layer_index: which hidden state layer was used
        - shapes: basic tensor shapes to confirm real computation
        - raw_stats: comprehensive stats for transparency (min/max/mean/sum/std)
        Note: raw tensors are not dumped verbatim to avoid huge outputs; shapes and stats are provided instead.
        """
        # Use API if enabled
        if self.use_api_for_activations:
            return self._get_activation_trace_from_api(text)
        
        trace: Dict[str, Any] = {
            "tokens": [],
            "token_ids": [],
            "per_token_activation": [],
            "summary_activation": 0.0,  # max activation (primary metric)
            "summary_activation_mean": 0.0,
            "summary_activation_sum": 0.0,
            "max_token_index": 0,
            "layer_index": self.layer,
            "shapes": {},
            "raw_stats": {},
        }

        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            # Fallback stub
            ids = [ord(c) % 256 for c in text]
            fallback_activations = [0.5] * len(ids)
            trace.update({
                "tokens": list(text),
                "token_ids": ids,
                "per_token_activation": fallback_activations,
                "summary_activation": max(fallback_activations),  # max activation
                "summary_activation_mean": sum(fallback_activations) / len(fallback_activations),
                "summary_activation_sum": sum(fallback_activations),
                "max_token_index": 0,
            })
            return trace

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=512)
        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            if self.use_hooked_transformer:
                # Use TransformerLens's run_with_cache for HookedTransformer
                hook_name = f"blocks.{self.layer}.hook_mlp_out"
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=[hook_name]
                )
                layer_activations = cache[hook_name]  # [batch, seq, hidden_dim]
            else:
                # Use standard HuggingFace forward pass
                inputs_dict = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs_dict, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                assert isinstance(hidden_states, (list, tuple))

                if self.layer >= len(hidden_states):
                    return trace

                layer_activations = hidden_states[self.layer]  # [batch, seq, d_model]

            batch_size, seq_len, hidden_dim = layer_activations.shape

            # Decode tokens
            ids_cpu = input_ids[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(ids_cpu)
            trace["tokens"] = tokens
            trace["token_ids"] = ids_cpu
            trace["shapes"] = {
                "layer_activations": list(layer_activations.shape),
            }

            per_token_act: Optional[torch.Tensor] = None
            summary_activation: float = 0.0
            summary_activation_mean: float = 0.0
            summary_activation_sum: float = 0.0
            max_token_index: int = 0

            # Compute SAE feature activations if SAE present
            if self.sae and "__sae_lens_obj__" in self.sae:
                try:
                    sae_obj = self.sae["__sae_lens_obj__"]
                    # Use encode() for consistency with _process_batch
                    sae_features = sae_obj.encode(layer_activations)  # type: ignore
                    if sae_features is not None and sae_features.ndim == 3:
                        # [batch, seq, num_features]
                        if self.feature_index < sae_features.shape[-1]:
                            per_token_act = sae_features[0, :, self.feature_index]
                            summary_activation = float(per_token_act.max().item())  # Use max for feature discovery
                            summary_activation_mean = float(per_token_act.mean().item())
                            summary_activation_sum = float(per_token_act.sum().item())
                            max_token_index = int(per_token_act.argmax().item())
                            trace["shapes"]["sae_features"] = list(sae_features.shape)
                except Exception as e:
                    if hasattr(self, 'debug') and self.debug:
                        print(f"Warning: SAE encoding failed in get_activation_trace: {e}")
                    per_token_act = None

            elif self.sae and 'encoder' in self.sae and 'encoder.weight' in self.sae:
                # Manual linear encoder path
                encoder_weight = self.sae['encoder.weight'].to(self.device)
                flat = layer_activations.view(-1, hidden_dim)  # [batch*seq, d_model]
                feats = torch.matmul(flat, encoder_weight.T)  # [batch*seq, num_features]
                num_features = feats.shape[-1]
                if self.feature_index < num_features:
                    per_token_act = feats.view(batch_size, seq_len, num_features)[0, :, self.feature_index]
                    summary_activation = float(per_token_act.max().item())  # Use max for feature discovery
                    summary_activation_mean = float(per_token_act.mean().item())
                    summary_activation_sum = float(per_token_act.sum().item())
                    max_token_index = int(per_token_act.argmax().item())
                trace["shapes"]["feats"] = list(feats.shape)
            else:
                # Fallback: hidden state norm per token
                norms = layer_activations.norm(dim=-1)[0]  # [seq]
                per_token_act = norms
                summary_activation = float(norms.max().item())  # Use max for feature discovery
                summary_activation_mean = float(norms.mean().item())
                summary_activation_sum = float(norms.sum().item())
                max_token_index = int(norms.argmax().item())

            if per_token_act is not None:
                per_token_list = [float(x) for x in per_token_act.detach().cpu().tolist()]
                trace["per_token_activation"] = per_token_list
                trace["summary_activation"] = float(round(summary_activation, 4))  # max activation (primary)
                trace["summary_activation_mean"] = float(round(summary_activation_mean, 4))
                trace["summary_activation_sum"] = float(round(summary_activation_sum, 4))
                trace["max_token_index"] = max_token_index
                
                # Comprehensive stats
                if len(per_token_list) > 0:
                    mean_val = sum(per_token_list) / len(per_token_list)
                    variance = sum((x - mean_val) ** 2 for x in per_token_list) / len(per_token_list)
                    std_val = variance ** 0.5
                    
                    trace["raw_stats"] = {
                        "min": float(min(per_token_list)),
                        "max": float(max(per_token_list)),
                        "mean": float(mean_val),
                        "sum": float(sum(per_token_list)),
                        "std": float(std_val),
                        "count": len(per_token_list),
                    }

        return trace

    def is_ready(self) -> bool:
        """Return True if model, tokenizer, and SAE are loaded."""
        return (TORCH_AVAILABLE and 
                self.model is not None and 
                self.tokenizer is not None and 
                self.sae is not None)

    def encode(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=True)
        else:
            # Fallback
            return [ord(ch) % 256 for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """Detokenize token IDs into text."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # Fallback
            return "".join(chr(t) for t in token_ids if t < 256)


__all__ = ["SAEConfig", "System"]


