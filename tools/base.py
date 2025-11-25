"""
sage_tools.py

Toolbox of callable utilities the agent can invoke. Exposes a registry-based
API so the experiment environment can enumerate and execute tools by name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import os

# Import managers
from .corpus import CorpusManager
from .neuronpedia import NeuronpediaManager

# Optional: SAELens ActivationsStore for efficient streaming
try:
    from sae_lens import ActivationsStore
    SAELENS_ACTIVATIONS_AVAILABLE = True
except ImportError:
    SAELENS_ACTIVATIONS_AVAILABLE = False
    ActivationsStore = None


class Tools:
    """Compatibility wrapper expected by main.py for logging + tool access."""

    def __init__(self, system: Any, agent_llm_name: str, dataset_path: str = "",
                 dataset_name: Optional[str] = None, dataset_config: Optional[str] = None,
                 dataset_split: str = "train", text_column: str = "text",
                 use_activations_store: bool = True, context_size: int = 128,
                 store_batch_size: int = 8, default_max_samples: int = 5000,
                 use_saedashboard: bool = True, use_api_for_activations: bool = False,
                 neuronpedia_model_id: Optional[str] = None, neuronpedia_source: Optional[str] = None,
                 default_top_k: int = 10) -> None:
        self.system = system
        self.agent_llm_name = agent_llm_name
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.text_column = text_column
        self.use_activations_store = use_activations_store
        self.context_size = context_size
        self.store_batch_size = store_batch_size
        self.default_max_samples = default_max_samples
        self.use_saedashboard = use_saedashboard
        self.use_api_for_activations = use_api_for_activations
        self.neuronpedia_model_id = neuronpedia_model_id
        self.neuronpedia_source = neuronpedia_source
        self.default_top_k = default_top_k
        self._log: List[Dict[str, str]] = []
        self._activations_store = None  # Lazy initialization

        # Initialize managers
        self.corpus_manager = CorpusManager(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            text_column=text_column
        )
        
        self.neuronpedia_manager = NeuronpediaManager(
            system=system,
            use_api_for_activations=use_api_for_activations,
            neuronpedia_model_id=neuronpedia_model_id,
            neuronpedia_source=neuronpedia_source,
            dataset_name=dataset_name,
            dataset_config=dataset_config
        )

        print(f"📊 Tools Configuration:")
        print(f"   Default max_samples: {default_max_samples}")
        print(f"   Context size: {context_size} tokens")
        print(f"   Batch size: {store_batch_size} prompts")
        print(f"   Default top_k: {default_top_k}")
        print(f"   Use ActivationsStore: {use_activations_store}")
        print(f"   Use SAEdashboard: {use_saedashboard}")
        print(f"   Use API for activations: {use_api_for_activations}")
        if use_api_for_activations:
            print(f"   Neuronpedia API: model_id={neuronpedia_model_id}, source={neuronpedia_source}")

        # Try to initialize ActivationsStore (if using HF dataset)
        if self.dataset_name and use_activations_store:
            self._init_activations_store()

    def init_log(self) -> None:
        """Initialize log with simplified system prompt."""
        simplified_system_prompt = self._get_simplified_system_prompt()
        self._log = [
            {"role": "system", "content": simplified_system_prompt},
        ]
    
    def _get_simplified_system_prompt(self) -> str:
        """Get simplified system prompt."""
        top_k = self.default_top_k
        return f"""You are SAGE, a scientific AI researcher analyzing SAE features.

## Core Mission
Systematically analyze SAE features using scientific methodology: OBSERVE → HYPOTHESIZE → TEST → ANALYZE → REFINE → CONCLUSION

## Available Tools
- [TOOL] text_exemplars top_k={top_k} - Get corpus exemplars
- [TOOL] model.run prompt='text' - Test text inputs

## Critical Rules
- Use ONE [TOOL] per response, then STOP
- Analyze REAL activation data only
- Follow state-specific instructions exactly
- Be scientific and evidence-based

## Output Format
- State hypotheses clearly
- Include specific activation values
- Draw evidence-based conclusions"""

    def get_log(self) -> List[Dict[str, str]]:
        return list(self._log)

    def update_log(self, role: str, content: str) -> None:
        self._log.append({"role": role, "content": content})

    def _init_activations_store(self) -> None:
        """Initialize SAELens ActivationsStore for efficient streaming."""
        if not SAELENS_ACTIVATIONS_AVAILABLE:
            print("⚠️  SAELens ActivationsStore not available, falling back to manual streaming")
            return

        try:
            # Check if system has required components
            if not hasattr(self.system, 'model') or not hasattr(self.system, 'sae'):
                print("⚠️  System model or SAE not available, skipping ActivationsStore initialization")
                return

            # Only initialize if using SAELens SAE
            if not (hasattr(self.system, 'sae') and isinstance(self.system.sae, dict) and "__sae_lens_obj__" in self.system.sae):
                print("⚠️  Not using SAELens SAE, skipping ActivationsStore initialization")
                return

            sae_obj = self.system.sae["__sae_lens_obj__"]
            model = self.system.model

            # Construct dataset path
            dataset_path = self.dataset_name
            if self.dataset_config:
                dataset_path = f"{self.dataset_name}/{self.dataset_config}"

            print(f"🔄 Initializing SAELens ActivationsStore...")
            print(f"   Dataset: {dataset_path}")
            print(f"   Split: {self.dataset_split}")
            print(f"   Context size: {self.context_size} tokens")
            print(f"   Batch size: {self.store_batch_size} prompts")

            # Calculate buffer size
            n_batches_in_buffer = max(self.store_batch_size, 32)
            
            if n_batches_in_buffer < self.store_batch_size:
                n_batches_in_buffer = self.store_batch_size

            self._activations_store = ActivationsStore.from_sae(
                model=model,
                sae=sae_obj,
                dataset=dataset_path,
                streaming=True,
                context_size=self.context_size,
                store_batch_size_prompts=self.store_batch_size,
                n_batches_in_buffer=n_batches_in_buffer,
                disable_concat_sequences=True,
                device=str(self.system.device),
            )

            print(f"   Buffer size: {n_batches_in_buffer} batches = {n_batches_in_buffer * self.store_batch_size} prompts")

            print(f"✅ ActivationsStore initialized successfully")

        except Exception as e:
            print(f"⚠️  Failed to initialize ActivationsStore: {e}")
            print(f"   Falling back to manual streaming")
            self._activations_store = None

    # --- Delegated methods ---

    def load_corpus(self, max_samples: Optional[int] = None,
                    segment_every_n_lines: Optional[int] = None,
                    preserve_newlines: bool = True,
                    split_on_three_newlines: bool = True) -> List[str]:
        return self.corpus_manager.load_corpus(
            max_samples, segment_every_n_lines, preserve_newlines, split_on_three_newlines
        )

    def _generate_cache_key(self, top_k: int, max_samples: int) -> str:
        system_info = {
            'sae_path': getattr(self.system, 'sae_path', ''),
            'feature_index': getattr(self.system, 'feature_index', 0),
            'layer': getattr(self.system, 'layer', 0)
        }
        return self.corpus_manager.generate_cache_key(system_info, top_k, max_samples)

    def _get_cache_path(self, cache_key: str) -> str:
        return self.corpus_manager.get_cache_path(cache_key)

    def _load_cached_exemplars(self, cache_key: str) -> Optional[List[Tuple[str, float]]]:
        return self.corpus_manager.load_cached_exemplars(cache_key)

    def _save_cached_exemplars(self, cache_key: str, exemplars: List[Tuple[str, float]]) -> None:
        self.corpus_manager.save_cached_exemplars(cache_key, exemplars)

    def _save_cached_detailed_exemplars(self, cache_key: str, detailed_exemplars: List[Dict[str, Any]]) -> None:
        self.corpus_manager.save_cached_detailed_exemplars(cache_key, detailed_exemplars)

    def _load_cached_detailed_exemplars(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        return self.corpus_manager.load_cached_detailed_exemplars(cache_key)

    def _generate_neuronpedia_data(self, outputs_dir: str = "neuronpedia_outputs/", 
                                   n_prompts_total: int = 24576, 
                                   n_tokens_in_prompt: int = 128,
                                   sparsity_threshold: float = 1.0) -> Optional[str]:
        return self.neuronpedia_manager.generate_neuronpedia_data(
            outputs_dir, n_prompts_total, n_tokens_in_prompt, sparsity_threshold
        )

    def _load_activations_from_neuronpedia_json(self, json_path: str, feature_index: int) -> Dict[str, Any]:
        return self.neuronpedia_manager.load_activations_from_neuronpedia_json(json_path, feature_index)

    def _get_maximally_activating_examples_from_api(self, top_k: int = 10, return_detailed: bool = False) -> Union[List[Tuple[str, float]], List[Dict[str, Any]]]:
        return self.neuronpedia_manager.get_maximally_activating_examples_from_api(top_k, return_detailed)
    
    def find_maximally_activating_examples(self, top_k: int = 10, max_samples: int = 5000) -> List[Tuple[str, float]]:
        """
        Find maximally activating examples from corpus using either API or local model.
        
        Args:
            top_k: Number of top examples to return
            max_samples: Maximum corpus samples to evaluate (only used in local mode)
        
        Returns:
            List of (text, max_activation) tuples
        """
        # Use API if configured
        if self.use_api_for_activations:
            print(f"📡 Using Neuronpedia API to fetch maximally activating examples...")
            return self._get_maximally_activating_examples_from_api(top_k, return_detailed=False)
        
        # Otherwise use local model/SAE (fallback to old implementation)
        print(f"🔍 Searching corpus locally for maximally activating examples (top_k={top_k}, max_samples={max_samples})...")
        
        # Check cache first
        cache_key = self._generate_cache_key(top_k, max_samples)
        cached = self._load_cached_exemplars(cache_key)
        if cached is not None:
            print(f"✅ Loaded {len(cached)} cached exemplars")
            return cached[:top_k]
        
        # Load corpus
        corpus = self.load_corpus(max_samples=max_samples)
        if not corpus:
            print("⚠️  Warning: Corpus is empty!")
            return []
        
        print(f"📚 Loaded {len(corpus)} corpus samples")
        
        # Use ActivationsStore if available and initialized
        if self._activations_store is not None:
            print(f"⚡ Using ActivationsStore for efficient activation computation...")
            return self._find_with_activations_store(corpus, top_k, cache_key)
        
        # Fallback: Compute activations manually
        print(f"🔧 Computing activations manually (this may be slow)...")
        return self._find_with_manual_computation(corpus, top_k, max_samples, cache_key)
    
    def find_detailed_maximally_activating_examples(self, top_k: int = 10, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        Find maximally activating examples with detailed token-level information.
        
        Args:
            top_k: Number of top examples to return
            max_samples: Maximum corpus samples to evaluate (only used in local mode)
        
        Returns:
            List of dicts with detailed activation data including tokens and per-token activations
        """
        # Use API if configured
        if self.use_api_for_activations:
            print(f"📡 Using Neuronpedia API to fetch detailed maximally activating examples...")
            return self._get_maximally_activating_examples_from_api(top_k, return_detailed=True)
        
        # Otherwise use local model/SAE
        print(f"🔍 Searching corpus locally for detailed maximally activating examples (top_k={top_k}, max_samples={max_samples})...")
        
        # Check cache first
        cache_key = self._generate_cache_key(top_k, max_samples)
        cached = self._load_cached_detailed_exemplars(cache_key)
        if cached is not None:
            print(f"✅ Loaded {len(cached)} cached detailed exemplars")
            return cached[:top_k]
        
        # Get basic exemplars first
        basic_exemplars = self.find_maximally_activating_examples(top_k, max_samples)
        
        # Enhance with detailed token-level information
        detailed_exemplars = []
        for text, max_activation in basic_exemplars:
            try:
                # Get full activation trace with token-level data
                trace = self.system.get_activation_trace(text)
                
                detailed_exemplars.append({
                    'text': text,
                    'max_activation': max_activation,
                    'mean_activation': trace.get('summary_activation_mean', 0.0),
                    'sum_activation': trace.get('summary_activation_sum', 0.0),
                    'tokens': trace.get('tokens', []),
                    'per_token_activations': trace.get('per_token_activation', []),
                    'max_token_index': trace.get('max_token_index', 0),
                    'layer': trace.get('layer_index', -1),
                    'feature_index': getattr(self.system, 'feature_index', -1)
                })
            except Exception as e:
                print(f"⚠️  Warning: Failed to get detailed trace for exemplar: {e}")
                # Fallback to basic data
                detailed_exemplars.append({
                    'text': text,
                    'max_activation': max_activation,
                    'mean_activation': 0.0,
                    'sum_activation': 0.0,
                    'tokens': [],
                    'per_token_activations': [],
                    'max_token_index': 0,
                    'layer': -1,
                    'feature_index': -1
                })
        
        # Cache the results
        if detailed_exemplars:
            self._save_cached_detailed_exemplars(cache_key, detailed_exemplars)
        
        return detailed_exemplars
    
    def _find_with_activations_store(self, corpus: List[str], top_k: int, cache_key: str) -> List[Tuple[str, float]]:
        """Find maximally activating examples using SAELens ActivationsStore (efficient)."""
        # This method would use the ActivationsStore to efficiently compute activations
        # For now, fall back to manual computation
        return self._find_with_manual_computation(corpus, top_k, len(corpus), cache_key)
    
    def _find_with_manual_computation(self, corpus: List[str], top_k: int, max_samples: int, cache_key: str) -> List[Tuple[str, float]]:
        """Find maximally activating examples by manually computing activations for each corpus sample."""
        results = []
        
        for i, text in enumerate(corpus[:max_samples]):
            if i % 100 == 0:
                print(f"   Progress: {i}/{min(len(corpus), max_samples)} samples processed...")
            
            try:
                # Get activation trace for this text
                trace = self.system.get_activation_trace(text)
                max_activation = trace.get('summary_activation', 0.0)
                results.append((text, max_activation))
            except Exception as e:
                # Skip samples that cause errors
                continue
        
        # Sort by activation (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the results
        if results:
            self._save_cached_exemplars(cache_key, results)
        
        print(f"✅ Found {len(results)} activating examples, returning top {top_k}")
        return results[:top_k]
