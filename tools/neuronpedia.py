"""
Neuronpedia integration logic for SAGE.
Handles interaction with Neuronpedia API and SAEdashboard NeuronpediaRunner.
"""

import os
import json
import time
from typing import Optional, Dict, Any, Union, List, Tuple

# Optional: SAEdashboard NeuronpediaRunner for max activation extraction
try:
    from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig
    from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
    SAEDASHBOARD_AVAILABLE = True
except ImportError:
    SAEDASHBOARD_AVAILABLE = False
    NeuronpediaRunnerConfig = None
    NeuronpediaRunner = None


class NeuronpediaManager:
    """Manages Neuronpedia API interactions and data generation."""

    def __init__(self, system: Any, use_api_for_activations: bool = False,
                 neuronpedia_model_id: Optional[str] = None,
                 neuronpedia_source: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 dataset_config: Optional[str] = None):
        self.system = system
        self.use_api_for_activations = use_api_for_activations
        self.neuronpedia_model_id = neuronpedia_model_id
        self.neuronpedia_source = neuronpedia_source
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def _extract_sae_info_from_path(self, sae_path: str) -> Dict[str, Optional[str]]:
        """Extract SAE set and path information from sae_path string.
        
        Supports both sae-lens:// URIs and standard paths.
        Returns dict with 'sae_set' and 'sae_path' keys.
        """
        sae_info = {'sae_set': None, 'sae_path': None}
        
        # Handle sae-lens:// URI format: "sae-lens://release=...;sae_id=..."
        if isinstance(sae_path, str) and sae_path.startswith("sae-lens://"):
            spec = sae_path[len("sae-lens://"):]
            parts = [p.strip() for p in spec.split(";") if p.strip()]
            kv: Dict[str, str] = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()
            
            release = kv.get("release") or kv.get("repo") or kv.get("model")
            sae_id = kv.get("sae_id") or kv.get("path")
            
            if release and sae_id:
                # Extract sae_set from release (e.g., "gemma-scope-2b-pt-mlp" from release)
                sae_info['sae_set'] = release
                # Use sae_id as sae_path (e.g., "layer_6/width_16k/average_l0_133")
                sae_info['sae_path'] = sae_id
        else:
            # For standard paths, try to infer from path structure
            # This is a fallback - may need adjustment based on actual path format
            if os.path.exists(sae_path):
                # Try to extract info from path
                path_parts = sae_path.split(os.sep)
                # Look for patterns like "gemma-scope-2b-pt-mlp" and "layer_X/width_Y/..."
                for i, part in enumerate(path_parts):
                    if 'gemma-scope' in part or 'pt' in part:
                        sae_info['sae_set'] = part
                    if 'layer_' in part and i + 2 < len(path_parts):
                        # Reconstruct relative path
                        sae_info['sae_path'] = os.sep.join(path_parts[i:])
                        break
        
        return sae_info

    def generate_neuronpedia_data(self, outputs_dir: str = "neuronpedia_outputs/", 
                                   n_prompts_total: int = 24576, 
                                   n_tokens_in_prompt: int = 128,
                                   sparsity_threshold: float = 1.0) -> Optional[str]:
        """Generate Neuronpedia data using SAEdashboard NeuronpediaRunner.
        
        Returns the path to the generated batch JSON file, or None if generation fails.
        """
        if not SAEDASHBOARD_AVAILABLE:
            print("⚠️  SAEdashboard not available, cannot use NeuronpediaRunner")
            return None
        
        try:
            # Extract SAE information from system's sae_path
            sae_path_str = getattr(self.system, 'sae_path', '')
            sae_info = self._extract_sae_info_from_path(sae_path_str)
            
            if not sae_info['sae_set'] or not sae_info['sae_path']:
                print(f"⚠️  Cannot extract SAE info from path: {sae_path_str}")
                return None
            
            # Get model ID from system
            model_id = getattr(self.system, 'model_name', '')
            if not model_id:
                print("⚠️  Cannot get model_id from system")
                return None
            
            # Determine dataset path
            huggingface_dataset_path = self.dataset_name
            if self.dataset_config:
                huggingface_dataset_path = f"{self.dataset_name}/{self.dataset_config}"
            
            if not huggingface_dataset_path:
                print("⚠️  Hugging Face dataset path not configured")
                return None
            
            # Generate np_set_name from layer
            layer = getattr(self.system, 'layer', 0)
            np_set_name = f"{layer}-gemmascope-mlp-16k"
            
            # Get device info from system
            device_str = str(getattr(self.system, 'device', 'cpu'))

            # Determine dtype to match System's configuration for consistency
            # System uses: float32 (CPU) or float16 (GPU)
            # We need to match this to ensure numerical consistency
            import torch
            if hasattr(self.system, 'model') and self.system.model is not None:
                # Get dtype from loaded model
                try:
                    model_param = next(self.system.model.parameters())
                    if model_param.dtype == torch.float32:
                        model_dtype_str = "float32"
                        sae_dtype_str = "float32"
                    elif model_param.dtype == torch.float16:
                        model_dtype_str = "float16"
                        sae_dtype_str = "float32"  # SAE stays float32 for stability
                    elif model_param.dtype == torch.bfloat16:
                        model_dtype_str = "bfloat16"
                        sae_dtype_str = "float32"  # SAE stays float32 for stability
                    else:
                        # Fallback
                        model_dtype_str = "float32" if 'cpu' in device_str else "bfloat16"
                        sae_dtype_str = "float32"
                except Exception as e:
                    print(f"   ⚠️  Could not detect model dtype: {e}, using defaults")
                    model_dtype_str = "float32" if 'cpu' in device_str else "bfloat16"
                    sae_dtype_str = "float32"
            else:
                # Fallback if model not available
                model_dtype_str = "float32" if 'cpu' in device_str else "bfloat16"
                sae_dtype_str = "float32"

            print(f"🔄 Generating Neuronpedia data using SAEdashboard...")
            print(f"   Layer: {layer} (from system.layer)")
            print(f"   Feature Index: {getattr(self.system, 'feature_index', 'N/A')}")
            print(f"   SAE Set: {sae_info['sae_set']}")
            print(f"   SAE Path: {sae_info['sae_path']}")
            print(f"   Model ID: {model_id}")
            print(f"   Dataset: {huggingface_dataset_path}")
            print(f"   Device: {device_str}")
            print(f"   Model dtype: {model_dtype_str} (matching System)")
            print(f"   SAE dtype: {sae_dtype_str}")
            print(f"   Prompts: {n_prompts_total}, Tokens: {n_tokens_in_prompt}")

            # Force single GPU usage to avoid multi-GPU device mismatch
            # Extract GPU number from device string (e.g., "cuda:0" -> "0", "cuda" -> "0")
            original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                if 'cuda' in device_str:
                    if ':' in device_str:
                        gpu_id = device_str.split(':')[1]
                    else:
                        gpu_id = '0'
                    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                    print(f"   💡 Setting CUDA_VISIBLE_DEVICES={gpu_id} to force single GPU usage")

                # Create NeuronpediaRunnerConfig
                # Note: Using same dtype as System model to ensure numerical consistency
                # This prevents activation value mismatches between System and NeuronpediaRunner
                config_kwargs = {
                    'sae_set': sae_info['sae_set'],
                    'sae_path': sae_info['sae_path'],
                    'np_set_name': np_set_name,
                    'model_id': model_id,
                    'huggingface_dataset_path': huggingface_dataset_path,
                    'n_prompts_total': n_prompts_total,
                    'n_tokens_in_prompt': n_tokens_in_prompt,
                    'n_features_at_a_time': 128,
                    'n_prompts_in_forward_pass': 128,
                    'sae_dtype': sae_dtype_str,  # Match System for consistency
                    'model_dtype': model_dtype_str,  # Match System for consistency
                    'sparsity_threshold': sparsity_threshold,
                    'outputs_dir': outputs_dir,
                    'end_batch': 1,
                    'prepend_bos': True,
                    'use_wandb': False,
                }

                # Try to add device parameters if NeuronpediaRunnerConfig supports them
                # Note: These parameters may not exist in all versions
                try:
                    # Check if model_device parameter exists (some versions may support it)
                    import inspect
                    config_params = inspect.signature(NeuronpediaRunnerConfig.__init__).parameters
                    if 'model_device' in config_params:
                        config_kwargs['model_device'] = 'cuda' if 'cuda' in device_str else 'cpu'
                    if 'sae_device' in config_params:
                        config_kwargs['sae_device'] = 'cuda' if 'cuda' in device_str else 'cpu'
                except Exception:
                    # If parameter inspection fails, just use default device handling
                    pass

                config = NeuronpediaRunnerConfig(**config_kwargs)

                # Temporarily free GPU memory to avoid OOM
                # NeuronpediaRunner will load its own copy of the model
                print(f"   💡 Temporarily clearing GPU cache to free memory...")
                import torch
                if hasattr(self.system, 'model'):
                    # Move model to CPU temporarily to free GPU memory
                    original_model_device = next(self.system.model.parameters()).device
                    print(f"   💡 Moving model to CPU to free GPU memory...")
                    self.system.model.cpu()
                    torch.cuda.empty_cache()
                    print(f"   💡 GPU memory freed")

                    try:
                        # Run NeuronpediaRunner
                        # Note: Device mismatch errors typically occur when tokens are on wrong device
                        # NeuronpediaRunner should handle this internally, but we catch and provide better error message
                        runner = NeuronpediaRunner(config)
                        runner.run()
                    finally:
                        # Restore model to original device
                        print(f"   💡 Restoring model to {original_model_device}...")
                        self.system.model.to(original_model_device)
                        torch.cuda.empty_cache()
                else:
                    # No model to move, just run normally
                    try:
                        runner = NeuronpediaRunner(config)
                        runner.run()
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if "out of memory" in error_str:
                            print(f"⚠️  GPU Out of Memory in NeuronpediaRunner:")
                            print(f"   Error: {e}")
                            print(f"   💡 Solutions:")
                            print(f"      1. Use a GPU with more memory")
                            print(f"      2. Reduce n_prompts_total or n_tokens_in_prompt")
                            print(f"      3. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                            print(f"      4. Use --use_saedashboard False to use fallback method")
                            raise
                        elif "device" in error_str or "deserialize" in error_str:
                            print(f"⚠️  Device error in NeuronpediaRunner:")
                            print(f"   System device: {device_str}")
                            print(f"   Error: {e}")
                            print(f"   💡 The cached tokens may have wrong device references.")
                            print(f"   💡 Try: rm -f neuronpedia_outputs/*/tokens_*.pt")
                            raise
                        else:
                            raise
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible_devices is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Determine the output file path
            # Format: {model_id}_{sae_set}_{hook_name}/batch-0.json
            hook_name = f"blocks.{layer}.hook_mlp_out"
            # Extract width from sae_path if possible, otherwise use default
            width = "16384"  # Default
            if "16k" in sae_info['sae_path'] or "16K" in sae_info['sae_path']:
                width = "16384"
            elif "8k" in sae_info['sae_path'] or "8K" in sae_info['sae_path']:
                width = "8192"
            
            output_dir = os.path.join(outputs_dir, f"{model_id}_{sae_info['sae_set']}_{hook_name}_{width}")
            batch_file = os.path.join(output_dir, "batch-0.json")
            
            if os.path.exists(batch_file):
                print(f"✅ Neuronpedia data generated successfully: {batch_file}")
                return batch_file
            else:
                print(f"⚠️  Neuronpedia batch file not found at expected path: {batch_file}")
                return None
                
        except Exception as e:
            print(f"⚠️  Failed to generate Neuronpedia data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_activations_from_neuronpedia_json(self, json_path: str, feature_index: int) -> Dict[str, Any]:
        """Load max activating examples from Neuronpedia batch JSON file.
        
        Similar to extract_feature_8_complete.py but integrated into Tools class.
        Returns a dictionary with activation data for the specified feature.
        """
        try:
            print(f"📖 Loading activations from {json_path}...")
            print(f"   Looking for feature index: {feature_index}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Find the target feature
            target_feature = None
            available_features = []
            for feature in data.get('features', []):
                feat_idx = feature.get('feature_index')
                available_features.append(feat_idx)
                if feat_idx == feature_index:
                    target_feature = feature
                    break
            
            if target_feature is None:
                print(f"   Available feature indices in JSON: {available_features[:10]}... (showing first 10)")
                raise ValueError(f"Feature {feature_index} not found in batch file")
            
            print(f"   ✅ Found feature {feature_index} in JSON file")
            
            # Extract all token activations from activation contexts
            max_activations = []
            
            for context_idx, context in enumerate(target_feature.get('activations', [])):
                tokens = context.get('tokens', [])
                values = context.get('values', [])
                
                for token_idx, (token, value) in enumerate(zip(tokens, values)):
                    if value > 0:  # Only non-zero activations
                        # Get context window (±10 tokens)
                        start_idx = max(0, token_idx - 10)
                        end_idx = min(len(tokens), token_idx + 11)
                        context_tokens = tokens[start_idx:end_idx]
                        context_text = ''.join(context_tokens)
                        
                        max_activations.append({
                            'activation': round(value, 3),
                            'token': token,
                            'context': context_text.strip(),
                            'text': context_text.strip(),  # Full context as text
                            'position_in_sequence': token_idx,
                            'context_id': context_idx,
                            'tokens': tokens,
                            'per_token_activations': values
                        })
            
            # Sort by activation value
            max_activations.sort(key=lambda x: x['activation'], reverse=True)
            
            return {
                'max_activations': max_activations,
                'feature_stats': {
                    'feature_index': target_feature.get('feature_index'),
                    'frac_nonzero': target_feature.get('frac_nonzero', 0.0),
                    'n_prompts_total': target_feature.get('n_prompts_total', 0),
                    'n_tokens_in_prompt': target_feature.get('n_tokens_in_prompt', 0),
                    'dataset': target_feature.get('dataset', '')
                }
            }
            
        except Exception as e:
            print(f"⚠️  Failed to load activations from JSON: {e}")
            import traceback
            traceback.print_exc()
            return {'max_activations': [], 'feature_stats': {}}

    def get_maximally_activating_examples_from_api(self, top_k: int = 10, return_detailed: bool = False) -> Union[List[Tuple[str, float]], List[Dict[str, Any]]]:
        """
        Get maximally activating examples from Neuronpedia API.
        
        Args:
            top_k: Number of top examples to return
            return_detailed: If True, return detailed format with tokens and values; if False, return (text, activation) tuples
        
        Returns:
            If return_detailed=False: List of (text, activation) tuples, sorted by activation (descending)
            If return_detailed=True: List of dicts with detailed activation data
        """
        if not self.use_api_for_activations:
            raise ValueError("API mode not enabled")
        
        if not self.neuronpedia_model_id or not self.neuronpedia_source:
            raise ValueError("Neuronpedia model_id and source required for API calls")
        
        feature_index = getattr(self.system, 'feature_index', 0)
        layer = getattr(self.system, 'layer', 0)
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for API calls. Install with: pip install requests")
        
        url = "https://www.neuronpedia.org/api/activation/get"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "modelId": self.neuronpedia_model_id,
            "source": self.neuronpedia_source,
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
            detailed_exemplars = []
            
            for activation_obj in result:
                if not isinstance(activation_obj, dict):
                    continue
                
                # Extract tokens and values
                tokens = activation_obj.get('tokens', [])
                values = activation_obj.get('values', [])
                
                # Find max activation in this example
                max_val = 0.0
                if values:
                    max_val = max(values)
                
                # Reconstruct text
                text = "".join(tokens)
                
                # Add to lists
                exemplars.append((text, max_val))
                
                if return_detailed:
                    detailed_exemplars.append({
                        'text': text,
                        'max_activation': max_val,
                        'tokens': tokens,
                        'per_token_activations': values
                    })
            
            # Sort by activation
            exemplars.sort(key=lambda x: x[1], reverse=True)
            detailed_exemplars.sort(key=lambda x: x['max_activation'], reverse=True)
            
            # Return requested format
            if return_detailed:
                return detailed_exemplars[:top_k]
            else:
                return exemplars[:top_k]
                
        except Exception as e:
            print(f"❌ Error fetching from Neuronpedia API: {e}")
            if return_detailed:
                return []
            else:
                return []
