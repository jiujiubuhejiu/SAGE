# SAGE: SAE Automated Generation of Explanations

SAGE (formerly SAIA) is an automated interpretability system for analyzing Sparse Autoencoder (SAE) features using agentic LLM workflows. The system can automatically generate natural language explanations for SAE features through iterative hypothesis testing and refinement.

## Features

- 🤖 **Automated Feature Analysis**: Agentic LLM-driven workflow for SAE feature interpretation
- 🔬 **Iterative Hypothesis Testing**: Multi-round hypothesis generation, testing, and refinement
- 📊 **Dual Activation Modes**: 
  - **API Mode**: Use Neuronpedia API (no model loading required)
  - **Local Mode**: Load models from Hugging Face and SAE Lens
- 💰 **Token Usage Tracking**: Automatic tracking of LLM API costs (GPT-5, GPT-4o)
- 📈 **Evaluation Framework**: Compare SAGE explanations against Neuronpedia baselines
- 🎯 **Multiple Evaluation Methods**:
  - Generation Evaluation (LLM generates examples from explanations)
  - Prediction Evaluation (LogProbs-based activation prediction)

## Architecture

SAGE uses a 3-layer architecture:

1. **State Machine Layer**: Hard-coded workflow control for systematic analysis
2. **Dynamic Prompt Generator**: Code-based prompt generation for each analysis state
3. **LLM Layer**: Executes current task based on generated prompts

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sga

# Install dependencies
pip install -r requirements.txt

# Set up environment variables

# Edit env and add your API keys:
# - OPENAI_API_KEY (for GPT-5/GPT-4)
# - ANTHROPIC_API_KEY (for Claude, optional)
# - GOOGLE_API_KEY (for Gemini, optional)
# - NEURONPEDIA_API_KEY (for API mode)
```

## Quick Start

### 1. Feature Analysis with SAGE

#### Option A: Using Neuronpedia API (Recommended - No Model Loading)

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer0=0" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 0-gemmascope-mlp-16k \
  --max_rounds 14 \
  --top_k 10
```

**Advantages of API Mode:**
- ✅ No need to download/load large models (saves memory and time)
- ✅ Faster activation computation
- ✅ Access to pre-computed activation data
- ✅ Automatically disables `use_saedashboard`

#### Option B: Using Local Models (Hugging Face + SAE Lens)

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --sae_path "sae-lens://release=gemma-scope-2b-pt-res-canonical;sae_id=layer_0/width_16k/canonical" \
  --features "layer0=0" \
  --use_api_for_activations false \
  --use_saedashboard true \
  --device cuda \
  --max_rounds 14 \
  --top_k 10
```

**When to use Local Mode:**
- You have custom SAE models not available on Neuronpedia
- You need full control over model inference
- You're working with private/experimental models

#### Analyzing Multiple Features

```bash
# Analyze features 0, 1, 2 from layer 0
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer0=0,1,2" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 0-gemmascope-mlp-16k
```

### 2. Evaluation: SAGE vs Neuronpedia

Compare SAGE-generated explanations against Neuronpedia baselines:

```bash
python scripts/evaluate.py \
  --sage_results_path ./results/gpt-5/google_gemma-2-2b/layer_0/feature_0/structured_results.json \
  --feature_index 0 \
  --layer 0-gemmascope-mlp-16k \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_api_key $NEURONPEDIA_API_KEY \
  --llm_model gpt-5 \
  --num_examples 10 \
  --activation_threshold 8.0 \
  --explanation_model_name gpt-5 \
  --explanation_type oai_token-act-pair
```

#### Batch Evaluation (Multiple Features)

```bash
python scripts/evaluate.py \
  --sage_results_path ./results/gpt-5/google_gemma-2-2b/layer_0/ \
  --features "0,1,2,3,4" \
  --layer 0-gemmascope-mlp-16k \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_api_key $NEURONPEDIA_API_KEY \
  --llm_model gpt-5
```

## Configuration Parameters

### Main Analysis (`main.py`)

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--agent_llm` | LLM for reasoning/analysis | `gpt-5-nano` | `gpt-5`, `claude-3-5-sonnet-latest` |
| `--target_llm` | Model to interpret | `google/gemma-2-2b` | `meta-llama/Llama-3.1-8B` |
| `--sae_path` | SAE path (local mode) | See example | SAE Lens URI or local path |
| `--features` | Features to analyze | `layer0=0` | `layer0=0,1,2` or `layer6=100` |
| `--use_api_for_activations` | Use Neuronpedia API | `false` | `true` (recommended) |
| `--neuronpedia_model_id` | Neuronpedia model ID | - | `gemma-2-2b`, `gpt2-small` |
| `--neuronpedia_source` | Neuronpedia layer source | - | `0-gemmascope-mlp-16k` |
| `--max_rounds` | Maximum analysis rounds | `14` | `10`, `20` |
| `--top_k` | Top activating examples | `10` | `5`, `15` |
| `--device` | Compute device (local mode) | `cpu` | `cuda`, `mps` |

### Evaluation (`scripts/evaluate.py`)

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--sage_results_path` | Path to SAGE results | Required | `./results/gpt-5/layer_0/` |
| `--feature_index` | Single feature to evaluate | - | `0` |
| `--features` | Multiple features (batch) | - | `"0,1,2,3"` |
| `--layer` | Layer identifier | Required | `0-gemmascope-mlp-16k` |
| `--neuronpedia_model_id` | Neuronpedia model ID | `llama3.1-8b-it` | `gemma-2-2b` |
| `--explanation_model_name` | Model for explanations | `gpt-5` | `gpt-4o`, `o4-mini` |
| `--num_examples` | Examples to generate | `10` | `15`, `20` |
| `--activation_threshold` | Success threshold | `8.0` | `5.0`, `10.0` |

## Output Structure

### SAGE Analysis Results

```
results/
└── {agent_llm}/
    └── {target_llm}/
        └── layer_{X}/
            └── feature_{Y}/
                ├── structured_results.json    # Complete analysis results
                ├── description.txt            # Final feature description
                ├── evidence.txt              # Supporting evidence
                ├── labels.txt                # Feature labels
                ├── token_usage.json          # Token usage statistics
                └── log.json                  # Detailed execution log
```

### Evaluation Results

```
results/
└── {agent_llm}/
    └── {target_llm}/
        └── layer_{X}/
            └── feature_{Y}/
                ├── feature_{Y}_comparison.json              # Full comparison results
                ├── feature_{Y}_neuronpedia_explanation.json # Neuronpedia explanation
                └── batch_comparison_summary.json            # Batch summary (if applicable)
```

## Evaluation Methods

### Method 1: Generation Evaluation
1. Extract feature explanation (SAGE or Neuronpedia)
2. Use GPT-5 to generate test examples from explanation
3. Test if generated examples actually activate the feature
4. Measure success rate and activation levels

### Method 2: Prediction Evaluation (LogProbs-based)
1. Select diverse test examples from API exemplars
2. Get token-level activation data from Neuronpedia API
3. Use GPT-4o with LogProbs to predict token-level activations
4. Calculate Pearson correlation between predicted and actual activations

## Token Usage & Cost Tracking

SAGE automatically tracks token usage and costs for all LLM API calls:

- **GPT-5**: $0.625/1M input, $5.00/1M output
- **GPT-4o**: $1.25/1M input, $5.00/1M output
- Cached tokens are tracked separately with reduced pricing

Results are saved to `token_usage.json` and displayed in terminal output.

## Supported Models

### Target Models (for interpretation)
- **Gemma-2-2B**: `google/gemma-2-2b`
- **Llama-3.1-8B**: `meta-llama/Llama-3.1-8B-Instruct`
- **GPT-2**: `gpt2` or `gpt2-small`

### Agent LLMs (for reasoning)
- **GPT-5**: `gpt-5`, `gpt-5-nano`, `gpt-5-mini`
- **GPT-4**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **Claude**: `claude-3-5-sonnet-latest`
- **Gemini**: `gemini-1.5-pro`, `gemini-2.5-flash`

### Neuronpedia Model IDs
- `gemma-2-2b`: Gemma-2-2B
- `gpt2-small`: GPT-2 Small
- `llama3.1-8b-it`: Llama-3.1-8B-Instruct

### Neuronpedia Source Formats
- Gemma: `{layer}-gemmascope-mlp-16k` (e.g., `0-gemmascope-mlp-16k`)
- GPT-2: `{layer}-res-jb` (e.g., `9-res-jb`)
- Llama: `{layer}-resid-post-aa` (e.g., `11-resid-post-aa`)

## Examples

### Example 1: Analyze Gemma-2-2B Feature 0 (API Mode)

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer0=0" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 0-gemmascope-mlp-16k
```

### Example 2: Analyze GPT-2 Feature 100 (Local Mode)

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm gpt2 \
  --sae_path "sae-lens://release=gpt2-small-res-jb;sae_id=blocks.9.hook_resid_pre" \
  --features "layer9=100" \
  --use_api_for_activations false \
  --device cuda
```

### Example 3: Batch Analysis of Multiple Features

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer0=0,1,2,3,4" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 0-gemmascope-mlp-16k \
  --max_rounds 14
```

### Example 4: Evaluate SAGE vs Neuronpedia

```bash
# Single feature evaluation
python scripts/evaluate.py \
  --sage_results_path ./results/gpt-5/google_gemma-2-2b/layer_0/feature_0/structured_results.json \
  --feature_index 0 \
  --layer 0-gemmascope-mlp-16k \
  --neuronpedia_model_id gemma-2-2b \
  --llm_model gpt-5

# Batch evaluation
python scripts/evaluate.py \
  --sage_results_path ./results/gpt-5/google_gemma-2-2b/layer_0/ \
  --features "0,1,2,3,4" \
  --layer 0-gemmascope-mlp-16k \
  --neuronpedia_model_id gemma-2-2b \
  --llm_model gpt-5
```

## Troubleshooting

### API Key Issues
- Ensure `OPENAI_API_KEY` is set in `saia_config.env`
- For Neuronpedia API mode, set `NEURONPEDIA_API_KEY`

### Memory Issues (Local Mode)
- Use `--device cpu` if GPU memory is insufficient
- Reduce `--batch_size` and `--max_samples`
- Consider using API mode instead

### Token Tracking Not Working
- Ensure `tools/token_tracker.py` exists and is importable
- Check that the module is correctly imported in `main.py` and `core/agent.py`

## Citation

If you use SAGE in your research, please cite:

```bibtex
@software{sage2024,
  title={SAGE: SAE Automated Generation of Explanations},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sage}
}
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [SAE Lens](https://github.com/jbloomAus/SAELens) for SAE infrastructure
- [Neuronpedia](https://neuronpedia.org) for activation data and API
- OpenAI, Anthropic, and Google for LLM APIs
