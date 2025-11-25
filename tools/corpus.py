"""
Corpus management logic for SAGE.
Handles loading text data from local files or Hugging Face datasets.
"""

import os
import re
import pickle
import hashlib
import html
import unicodedata
from typing import List, Optional, Dict, Tuple, Any

# Optional: Hugging Face datasets support
try:
    from datasets import load_dataset, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None
    Dataset = None

# Optional: ftfy for robust text fixing
try:
    import ftfy  # type: ignore
    FTFY_AVAILABLE = True
except Exception:
    FTFY_AVAILABLE = False


class CorpusManager:
    """Manages corpus loading and caching."""

    def __init__(self, dataset_path: str = "", dataset_name: Optional[str] = None,
                 dataset_config: Optional[str] = None, dataset_split: str = "train",
                 text_column: str = "text"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.text_column = text_column
        self._corpus_cache: Optional[List[str]] = None

    def load_corpus(self, max_samples: Optional[int] = None,
                    segment_every_n_lines: Optional[int] = None,
                    preserve_newlines: bool = True,
                    split_on_three_newlines: bool = True) -> List[str]:
        """Load the corpus from dataset_path or Hugging Face dataset as a list of non-empty texts.

        Supports both local files and Hugging Face datasets with streaming.
        Caches local file results to avoid repeated disk IO.
        
        Args:
            max_samples: Limit number of returned samples.
            segment_every_n_lines: If set (e.g., 2), segment by fixed line count instead of blank-line paragraphs.
            preserve_newlines: If True, keep "\n" within samples (no intra-sample newline-to-space collapse).
            split_on_three_newlines: If True, use sequences of three-or-more newlines as sample boundaries. Disables 3+ newline collapsing.
        """
        # Try Hugging Face dataset first if configured
        if self.dataset_name and DATASETS_AVAILABLE:
            return self._load_hf_dataset(max_samples)
        
        # Fallback to local file
        if self._corpus_cache is not None:
            return self._corpus_cache[:max_samples] if max_samples else self._corpus_cache

        path = (self.dataset_path or "").strip()
        if not path or not os.path.exists(path):
            self._corpus_cache = []
            return self._corpus_cache

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Apply robust text fixing first (handles mojibake, weird quotes, etc.)
            if FTFY_AVAILABLE:
                try:
                    raw_text = ftfy.fix_text(raw_text)
                except Exception:
                    pass

            # HTML entity unescape (e.g., &nbsp; &amp; etc.)
            raw_text = html.unescape(raw_text)

            # Normalize Unicode to NFKC to fold compatibility forms
            raw_text = unicodedata.normalize("NFKC", raw_text)


            # Replace various unicode spaces with regular spaces
            raw_text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", raw_text)

            # Only collapse 3+ newlines to 2 when NOT using three-newline splitting
            if not split_on_three_newlines:
                raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

            # Build raw segments according to segmentation strategy
            if split_on_three_newlines:
                # Force preservation of newlines inside samples for this mode
                preserve_newlines = True
                raw_segments = re.split(r"\n{3,}", raw_text)
            elif segment_every_n_lines and segment_every_n_lines > 0:
                split_lines = raw_text.splitlines()
                raw_segments: List[str] = []
                for i in range(0, len(split_lines), segment_every_n_lines):
                    chunk = split_lines[i:i + segment_every_n_lines]
                    raw_segments.append("\n".join(chunk))
            else:
                # Default: split into paragraphs by blank lines
                raw_segments = re.split(r"\n\s*\n+", raw_text)

            def _normalize_segment(s: str) -> str:
                # Remove zero-width and other control chars except common whitespace
                s = re.sub(r"[\u200C\u200D\u2060\uFEFF]", "", s)
                s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
                if preserve_newlines:
                    # Collapse spaces/tabs but keep newlines
                    s = re.sub(r"[ \t\f\v]+", " ", s)
                    s = re.sub(r" *\n *", "\n", s)
                else:
                    # Collapse all whitespace to single spaces
                    s = re.sub(r"\s+", " ", s)
                return s.strip()

            lines = [_normalize_segment(p) for p in raw_segments]
            lines = [ ln for ln in lines if ln ]
            # Deduplicate while preserving order (best-effort)
            seen: Dict[str, bool] = {}
            corpus: List[str] = []
            for ln in lines:
                if ln in seen:
                    continue
                seen[ln] = True
                corpus.append(ln)
            self._corpus_cache = corpus
        except Exception:
            self._corpus_cache = []
        
        return self._corpus_cache[:max_samples] if max_samples else self._corpus_cache

    def _load_hf_dataset(self, max_samples: Optional[int] = None) -> List[str]:
        """Load corpus from Hugging Face dataset with streaming support."""
        try:
            # Load dataset with streaming for efficiency
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.dataset_split,
                streaming=True
            )
            
            texts = []
            count = 0
            for example in dataset:
                if max_samples and count >= max_samples:
                    break
                
                # Extract text from the specified column
                text = example.get(self.text_column, "")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    count += 1
                elif isinstance(text, list):
                    # Handle cases where text_column contains a list
                    for item in text:
                        if isinstance(item, str) and item.strip():
                            texts.append(item.strip())
                            count += 1
                            if max_samples and count >= max_samples:
                                break
            
            return texts
            
        except Exception as e:
            print(f"Warning: Failed to load Hugging Face dataset '{self.dataset_name}': {e}")
            return []

    def generate_cache_key(self, system_info: Dict[str, Any], top_k: int, max_samples: int) -> str:
        """Generate a cache key based on current configuration and parameters."""
        # Get system parameters for cache key
        sae_path = system_info.get('sae_path', '')
        feature_index = system_info.get('feature_index', 0)
        layer = system_info.get('layer', 0)
        
        # Create a unique identifier for this configuration
        config_str = f"sae:{sae_path}|feature:{feature_index}|layer:{layer}|dataset:{self.dataset_name or self.dataset_path}|config:{self.dataset_config}|split:{self.dataset_split}|column:{self.text_column}|top_k:{top_k}|max_samples:{max_samples}"
        
        # Hash the configuration string for a shorter cache key
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_cache_path(self, cache_key: str) -> str:
        """Get the file path for storing cached exemplars."""
        cache_dir = "./cache/exemplars"
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}.pkl")

    def load_cached_exemplars(self, cache_key: str) -> Optional[List[Tuple[str, float]]]:
        """Load cached exemplars from disk."""
        try:
            cache_path = self.get_cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cached exemplars: {e}")
        return None

    def save_cached_exemplars(self, cache_key: str, exemplars: List[Tuple[str, float]]) -> None:
        """Save exemplars to disk cache."""
        try:
            cache_path = self.get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(exemplars, f)
        except Exception as e:
            print(f"Warning: Failed to save cached exemplars: {e}")

    def save_cached_detailed_exemplars(self, cache_key: str, detailed_exemplars: List[Dict[str, Any]]) -> None:
        """Save detailed exemplars to disk cache."""
        try:
            cache_path = self.get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(detailed_exemplars, f)
        except Exception as e:
            print(f"Warning: Failed to save cached detailed exemplars: {e}")

    def load_cached_detailed_exemplars(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached detailed exemplars from disk."""
        try:
            cache_path = self.get_cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cached detailed exemplars: {e}")
        return None
