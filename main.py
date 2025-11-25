"""
SAGE Main v2 - New version using 3-layer architecture

Layer 1: State Machine - Hard-coded workflow control
Layer 2: Dynamic Prompt Generator - Code logic for prompt generation
Layer 3: LLM - Executes current task only
"""

import argparse
import os
import json
import time
from typing import Dict, Any, Optional

try:
    import torch  # optional
except Exception:
    torch = None

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('sage_config.env')
    print("Loaded SAGE environment variables from sage_config.env")
except ImportError:
    # If dotenv is not available, manually load environment variables
    env_file = 'sage_config.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("Loaded SAGE environment variables from sage_config.env")
except Exception as e:
    print(f"Warning: Could not load environment variables: {e}")

from core.system import System
from tools.base import Tools
from core.agent import ask_agent
from environment.experiment import ExperimentEnvironment
from utils.common import save_final_results, str2dict
from core.controller import SAGEController

# Import token tracker
try:
    from tools.token_tracker import get_tracker, reset_tracker
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False
    def get_tracker():
        return None
    def reset_tracker():
        pass


def get_neuronpedia_config(target_llm: str, sae_path: str, sae_layer: int, 
                           neuronpedia_model_id: Optional[str] = None,
                           neuronpedia_source: Optional[str] = None) -> Dict[str, str]:
    """
    Infer Neuronpedia API model_id and source parameters from model name and SAE path.
    
    Args:
        target_llm: Target model name (e.g., "google/gemma-2-2b")
        sae_path: SAE path (e.g., "sae-lens://release=gemma-scope-2b-pt-res-canonical;sae_id=layer_0/width_16k/canonical")
        sae_layer: SAE layer index (e.g., 0)
        neuronpedia_model_id: If provided, use directly; otherwise infer from target_llm
        neuronpedia_source: If provided, use directly; otherwise infer from sae_path and layer
    
    Returns:
        dict with 'model_id' and 'source' keys
    """
    # Determine model_id
    if neuronpedia_model_id:
        model_id = neuronpedia_model_id
    else:
        # Infer model_id from target_llm
        target_llm_lower = target_llm.lower()
        if 'gpt2' in target_llm_lower:
            model_id = 'gpt2-small'
        elif 'gemma' in target_llm_lower:
            if '2-2b' in target_llm_lower or '2b' in target_llm_lower:
                model_id = 'gemma-2-2b'
            else:
                model_id = 'gemma-2-2b'  # default
        elif 'llama3.1' in target_llm_lower or 'llama-3.1' in target_llm_lower:
            if '8b' in target_llm_lower:
                model_id = 'llama3.1-8b-it'
            else:
                model_id = 'llama3.1-8b-it'  # default
        else:
            # Default to gpt2-small
            model_id = 'gpt2-small'
    
    # Determine source: use provided value if available, otherwise infer
    if neuronpedia_source:
        source = neuronpedia_source
    else:
        # Infer source from sae_path and layer (fallback)
        sae_path_lower = sae_path.lower()
        
        # Check for gemmascope-related identifiers
        if 'gemmascope' in sae_path_lower:
            if '16k' in sae_path_lower or '16K' in sae_path_lower:
                source = f"{sae_layer}-gemmascope-mlp-16k"
            elif '8k' in sae_path_lower or '8K' in sae_path_lower:
                source = f"{sae_layer}-gemmascope-mlp-8k"
            else:
                source = f"{sae_layer}-gemmascope-mlp-16k"  # default
        elif 'res-jb' in sae_path_lower or 'res_jb' in sae_path_lower:
            # GPT2 format: "9-res-jb"
            source = f"{sae_layer}-res-jb"
        elif 'resid-post-aa' in sae_path_lower or 'resid_post_aa' in sae_path_lower:
            # Llama format: "11-resid-post-aa"
            source = f"{sae_layer}-resid-post-aa"
        else:
            # Default format: use layer
            if 'gemma' in model_id.lower():
                source = f"{sae_layer}-gemmascope-mlp-16k"
            elif 'gpt2' in model_id.lower():
                source = f"{sae_layer}-res-jb"
            elif 'llama3.1' in model_id.lower():
                source = f"{sae_layer}-resid-post-aa"
            else:
                source = f"{sae_layer}-res-jb"  # default
    
    return {
        'model_id': model_id,
        'source': source
    }


def call_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SAGE v2: SAE Automated Interpretability Agent with 3-Layer Architecture')
    parser.add_argument('--agent_llm', type=str, default='gpt-5-nano', help='The LLM agent for reasoning')
    parser.add_argument('--target_llm', type=str, default='google/gemma-2-2b', help='The LLM to interpret')
    parser.add_argument('--sae_path', type=str, default='sae-lens://release=gemma-scope-2b-pt-res-canonical;sae_id=layer_0/width_16k/canonical', help='Path/URI to the pretrained SAE')
    parser.add_argument('--features', type=str2dict, default='layer0=0', help='Features to interpret. Format: "layerX=feat1,feat2"')
    parser.add_argument('--path2save', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--dataset_path', type=str, default='./dataset/corpus.txt', help='Path to dataset corpus')
    parser.add_argument('--dataset_name', type=str, default=None, help='Hugging Face dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='Hugging Face dataset config name')
    parser.add_argument('--dataset_split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--text_column', type=str, default='text', help='Column name containing text data')
    parser.add_argument('--device', type=str, default='cpu', help='Compute device')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--max_rounds', type=int, default=14, help='Maximum number of rounds')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Timeout in minutes')

    # Dataset sampling parameters (match Neuronpedia: n_prompts_total=24576, n_prompts_in_forward_pass=128)
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum corpus samples to evaluate (default: 5000, Neuronpedia: 24576)')
    parser.add_argument('--context_size', type=int, default=64, help='Tokens per prompt (default: 128)')
    parser.add_argument('--batch_size', type=int, default=8, help='Prompts per forward pass (default: 8, Neuronpedia: 128)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of maximally activating corpus examples to retrieve (default: 10)')

    # SAEdashboard options
    parser.add_argument('--use_saedashboard', type=lambda x: x.lower() == 'true', default=True, help='Use SAEdashboard NeuronpediaRunner for activation extraction (default: True). Set to False to use fallback method.')

    # Neuronpedia API options
    parser.add_argument('--use_api_for_activations', type=lambda x: x.lower() == 'true', default=False, help='Use Neuronpedia API for find_maximally_activating_examples and get_activation_trace (default: False)')
    parser.add_argument('--neuronpedia_model_id', type=str, default=None, help='Neuronpedia model ID for API calls (e.g., "gpt2-small", "gemma-2-2b", "llama3.1-8b-it"). If not provided, will be inferred from target_llm.')
    parser.add_argument('--neuronpedia_source', type=str, default=None, help='Neuronpedia source/layer identifier for API calls (e.g., "0-gemmascope-mlp-16k", "9-res-jb", "11-resid-post-aa"). Required if use_api_for_activations=True. If not provided, will be inferred from sae_path and layer.')

    args = parser.parse_args()
    return args


def run_single_feature_experiment(args, sae_layer_index: int, feature_index: int) -> Dict[str, Any]:
    """Run experiment for a single feature."""
    print(f"\n{'='*20} Starting Experiment: Layer {sae_layer_index}, Feature {feature_index} {'='*20}")
    
    # Determine save path
    path2save = os.path.join(args.path2save, args.agent_llm, args.target_llm.replace('/', '_'), f"layer_{sae_layer_index}", f"feature_{feature_index}")
    
    # Check if results already exist
    if os.path.exists(os.path.join(path2save, 'description.txt')):
        print("Results already exist. Skipping.")
        return {"status": "skipped", "reason": "results_exist"}
    
    os.makedirs(path2save, exist_ok=True)
    
    # Reset token tracker (create new statistics for each experiment)
    if TOKEN_TRACKING_AVAILABLE:
        reset_tracker()
    
    try:
        # Get Neuronpedia API configuration (if using API)
        neuronpedia_config = None
        if args.use_api_for_activations:
            print("=" * 80)
            print("📡 API Mode Enabled: Using Neuronpedia API for all activation operations")
            print("=" * 80)
            print("   - Model and SAE will NOT be loaded (saves memory and time)")
            print("   - All activation data will be fetched from Neuronpedia API")
            print("   - use_saedashboard will be automatically disabled")
            print("=" * 80)
            
            # Validate that source is provided or can be inferred
            if not args.neuronpedia_source:
                print(f"⚠️  Warning: --neuronpedia_source not provided. Will attempt to infer from sae_path and layer.")
                print(f"   For better accuracy, please provide --neuronpedia_source explicitly.")
            
            neuronpedia_config = get_neuronpedia_config(
                target_llm=args.target_llm,
                sae_path=args.sae_path,
                sae_layer=sae_layer_index,
                neuronpedia_model_id=args.neuronpedia_model_id,
                neuronpedia_source=args.neuronpedia_source
            )
            print(f"📡 Neuronpedia API Config: model_id={neuronpedia_config['model_id']}, source={neuronpedia_config['source']}")
            if args.neuronpedia_source:
                print(f"   ✅ Source provided explicitly: {args.neuronpedia_source}")
            else:
                print(f"   ⚠️  Source inferred from sae_path and layer: {neuronpedia_config['source']}")
                print(f"   💡 Tip: To ensure accuracy, provide --neuronpedia_source explicitly")
            
            # Automatically disable use_saedashboard when using API
            if args.use_saedashboard:
                print(f"   ⚠️  use_saedashboard is enabled but will be ignored (API mode takes precedence)")
            args.use_saedashboard = False
        
        # Initialize system components
        system = System(
            llm_name=args.target_llm,
            sae_path=args.sae_path,
            sae_layer=sae_layer_index,
            feature_index=feature_index,
            device=args.device,
            debug=args.debug,  # Pass debug parameter
            use_api_for_activations=args.use_api_for_activations,
            neuronpedia_model_id=neuronpedia_config['model_id'] if neuronpedia_config else None,
            neuronpedia_source=neuronpedia_config['source'] if neuronpedia_config else None
        )
        
        tools = Tools(
            system=system,
            agent_llm_name=args.agent_llm,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            use_activations_store=True,
            context_size=args.context_size,
            store_batch_size=args.batch_size,
            default_max_samples=args.max_samples,
            use_saedashboard=args.use_saedashboard,
            use_api_for_activations=args.use_api_for_activations,
            neuronpedia_model_id=neuronpedia_config['model_id'] if neuronpedia_config else None,
            neuronpedia_source=neuronpedia_config['source'] if neuronpedia_config else None,
            default_top_k=args.top_k
        )

        experiment_env = ExperimentEnvironment(tools, debug=args.debug, default_top_k=args.top_k)

        # Initialize log (simplified initialization, all prompts dynamically generated by prompt_generator)
        tools.init_log()
        
        # Create SAGE controller
        controller = SAGEController(
            feature_id=feature_index,
            layer=sae_layer_index,
            llm_client=args.agent_llm,  # Pass string directly, ask_agent function will handle it
            tools=tools,
            experiment_env=experiment_env,
            debug=args.debug,
            top_k=args.top_k
        )
        
        # Run experiment
        start_time = time.time()
        results = controller.run()
        end_time = time.time()
        
        # Add execution time
        results["execution_time_seconds"] = end_time - start_time
        
        # Save results
        save_final_results(tools.get_log(), path2save)
        
        # Save token statistics (if available)
        if TOKEN_TRACKING_AVAILABLE:
            tracker = get_tracker()
            if tracker:
                token_summary = tracker.get_summary()
                results["token_usage"] = token_summary
                
                # Save detailed token statistics to separate file
                token_stats_path = os.path.join(path2save, 'token_usage.json')
                tracker.save_to_file(token_stats_path)
                
                # Print token statistics summary
                tracker.print_summary()
        
        # Save structured results
        with open(os.path.join(path2save, 'structured_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Check if there is a valid conclusion
        has_conclusion = any(
            "[DESCRIPTION]:" in analysis and "[EVIDENCE]:" in analysis and "[LABEL" in analysis
            for analysis in results.get("analysis_history", [])
        )
        
        if has_conclusion:
            print(f"✅ Experiment for Feature {feature_index} completed successfully with conclusion. Results saved to {path2save}")
            return {"status": "completed", "results": results}
        else:
            print(f"⚠️  Experiment for Feature {feature_index} completed but no valid conclusion generated. Results saved to {path2save}")
            return {"status": "incomplete", "results": results}
        
    except Exception as e:
        print(f"❌ Fatal error during experiment for feature {feature_index}: {e}")
        
        # Save error log
        try:
            save_final_results(tools.get_log(), path2save, filename="error_log.json")
        except:
            pass
        
        return {"status": "error", "error": str(e)}


def main(args):
    """Main function - orchestrate all workflows."""
    print("🚀 Starting SAGE v2 with 3-Layer Architecture...")
    print(f"📁 Project directory: {os.getcwd()}")
    print(f"🐍 Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    print(f"📄 Main file: {os.path.abspath(__file__)}")
    print(f"⚙️  Environment config: {os.path.join(os.getcwd(), 'sage_config.env')}")
    print("-" * 50)
    
    # Get list of features to analyze
    features_to_run = args.features
    total_experiments = sum(len(features) for features in features_to_run.values())
    completed_experiments = 0
    
    print(f"📊 Total experiments to run: {total_experiments}")
    
    # Outer loop: iterate through all specified layers
    for layer_str, feature_indices in features_to_run.items():
        sae_layer_index = int(layer_str.replace('layer', ''))
        
        # Inner loop: iterate through all specified features in this layer
        for feature_index in feature_indices:
            try:
                result = run_single_feature_experiment(args, sae_layer_index, feature_index)
                completed_experiments += 1
                
                if result["status"] == "completed":
                    print(f"✅ Experiment {completed_experiments}/{total_experiments} completed successfully with conclusion")
                elif result["status"] == "incomplete":
                    print(f"⚠️  Experiment {completed_experiments}/{total_experiments} completed but no valid conclusion")
                elif result["status"] == "skipped":
                    print(f"⏭️  Experiment {completed_experiments}/{total_experiments} skipped (already exists)")
                else:
                    print(f"❌ Experiment {completed_experiments}/{total_experiments} failed: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n🛑 Experiment interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error in experiment {completed_experiments + 1}/{total_experiments}: {e}")
                completed_experiments += 1
    
    print(f"\n🎉 All experiments completed! {completed_experiments}/{total_experiments} experiments processed.")


if __name__ == '__main__':
    # Parse arguments and run
    args = call_argparse()
    main(args)
