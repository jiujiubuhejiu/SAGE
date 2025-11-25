"""
SAGE Experiment Environment - v2 execution bridge

Bridges agent responses to actual tool execution for v2 (text_exemplars, model.run).
"""

from typing import Any, List


class ExperimentEnvironment:
    """Compatibility layer expected by main.py.

    It accepts an object `tools` that contains both a `system` and a `registry`.
    The agent response string is parsed very simply to decide actions.
    """

    def __init__(self, tools: Any, debug: bool = False, default_top_k: int = 10) -> None:
        self.tools = tools
        self.text_exemplars_called = False  # Track if text_exemplars has been called
        self.debug = debug  # Enable debug output
        self.last_detailed_exemplars = []  # Store detailed exemplars data for buffer extraction
        self.default_top_k = default_top_k

    def execute_experiment(self, agent_response: str) -> Any:
        # Parse agent response and execute appropriate actions
        # 1) Robustly extract any [TOOL] calls even if embedded in lists/blocks
        tool_calls: List[str] = []
        text = agent_response
        if "[TOOL]" in text:
            # split on occurrences of [TOOL] and rebuild commands
            parts = text.split("[TOOL]")
            for segment in parts[1:]:  # skip text before first
                # take until newline or end; allow entire segment if single-line
                line = segment.strip()
                # stop at first newline to avoid capturing following prose
                if "\n" in line:
                    line = line.split("\n", 1)[0].strip()
                # prepend marker back
                if line:
                    tool_calls.append("[TOOL] " + line)

        # Fallback single-line behavior if no embedded calls found
        if not tool_calls and agent_response.startswith("[TOOL]"):
            tool_calls = [agent_response]

        if tool_calls:
            outputs: List[str] = []
            for call in tool_calls:
                # Extract tool command from response
                tool_part = call.replace("[TOOL]", "").strip()
                
                # Handle model.run tool - the primary scientific tool
                if "model.run" in tool_part:
                    # Extract prompt from the response
                    if "prompt=" in tool_part:
                        prompt_raw = tool_part.split("prompt=")[1].strip()
                        # 移除开头和结尾的引号（如果存在）
                        if prompt_raw.startswith("'") and prompt_raw.endswith("'"):
                            prompt = prompt_raw[1:-1]
                        elif prompt_raw.startswith('"') and prompt_raw.endswith('"'):
                            prompt = prompt_raw[1:-1]
                        else:
                            prompt = prompt_raw
                        # Process转义的单引号：将\'还原为'
                        prompt = prompt.replace("\\'", "'")
                        try:
                            # Detailed trace output with enhanced feature discovery information
                            trace = self.tools.system.get_activation_trace(prompt)
                            tokens = trace.get("tokens", [])
                            per_token = trace.get("per_token_activation", [])
                            summary_max = trace.get("summary_activation", 0.0)  # max activation (primary)
                            summary_mean = trace.get("summary_activation_mean", 0.0)
                            summary_sum = trace.get("summary_activation_sum", 0.0)
                            max_token_idx = trace.get("max_token_index", 0)
                            layer_idx = trace.get("layer_index", -1)
                            shapes = trace.get("shapes", {})
                            raw_stats = trace.get("raw_stats", {})
                            
                            # Build enhanced output for feature discovery
                            token_pairs = list(zip(tokens, [f"{v:.4f}" for v in per_token]))
                            preview = token_pairs[:32]
                            
                            # Identify the token with maximum activation
                            max_token = tokens[max_token_idx] if max_token_idx < len(tokens) else "N/A"
                            
                            # 完整输出用于调试
                            full_output = (
                                "Real model.run trace (Feature Discovery Enhanced):\n"
                                f"- Test prompt: '{prompt}'\n"
                                f"- Layer: {layer_idx}\n"
                                f"- Shapes: {shapes}\n"
                                f"- Max activation: {summary_max:.4f} (at token: '{max_token}')\n"
                                f"- Mean activation: {summary_mean:.4f}\n"
                                f"- Sum activation: {summary_sum:.4f}\n"
                                f"- Activation stats: min={raw_stats.get('min', 0):.4f}, max={raw_stats.get('max', 0):.4f}, std={raw_stats.get('std', 0):.4f}\n"
                                f"- Token count: {len(tokens)}\n"
                                f"- Tokens/activations (first {len(preview)} of {len(token_pairs)}): {preview}\n"
                            )

                            # Debug 模式下打印完整输出
                            if self.debug:
                                print("\n" + "="*80)
                                print("🔍 DEBUG: Full Model Trace Output")
                                print("="*80)
                                print(full_output)
                                print("="*80 + "\n")

                            # 简化输出用于LLM (修复格式错误)
                            simplified_output = (
                                "Output:\n"
                                f"Test prompt: '{prompt}'\n"
                                f"Max activation: {summary_max:.4f} (at token: '{max_token}')\n"
                                f"Tokens/activations: {preview}\n"
                            )

                            outputs.append(simplified_output)
                        except Exception as e:
                            outputs.append(f"Error running model with prompt '{prompt}': {str(e)}")
                    else:
                        outputs.append("Error: model.run requires a prompt parameter")
                
                # Handle other tools with more realistic responses
                elif "text_exemplars" in tool_part:
                    # Corpus-driven maximally activating exemplars
                    # Check if text_exemplars has already been called
                    if self.text_exemplars_called:
                        outputs.append("ERROR: [TOOL] text_exemplars has already been called in this experiment. Use [TOOL] model.run for hypothesis testing instead.")
                        outputs.append("The corpus analysis has been completed. Focus on testing specific hypotheses with model.run based on the initial corpus observations.")
                        continue
                    
                    self.text_exemplars_called = True  # Mark as called
                    try:
                        # Support optional parameters: top_k and max_samples
                        top_k = self.default_top_k
                        max_samples = 5000  # Increase sample size for better coverage
                        if "top_k=" in tool_part:
                            try:
                                top_k_str = tool_part.split("top_k=")[1].split()[0].strip().strip("'\"")
                                top_k = int(top_k_str)
                            except Exception:
                                pass
                        if "max_samples=" in tool_part:
                            try:
                                ms_str = tool_part.split("max_samples=")[1].split()[0].strip().strip("'\"")
                                max_samples = int(ms_str)
                            except Exception:
                                pass
                        

                        # Get detailed exemplars with token-level information
                        detailed_exemplars = []
                        if hasattr(self.tools, "find_detailed_maximally_activating_examples"):
                            print(f"🔍 Calling find_detailed_maximally_activating_examples(top_k={top_k}, max_samples={max_samples})...")
                            detailed_exemplars = self.tools.find_detailed_maximally_activating_examples(top_k=top_k, max_samples=max_samples)
                            print(f"🔍 Received {len(detailed_exemplars)} detailed exemplars from find_detailed_maximally_activating_examples")
                        elif hasattr(self.tools, "find_maximally_activating_examples"):
                            # Fallback to basic method if detailed method not available
                            print(f"🔍 Using fallback find_maximally_activating_examples...")
                            basic_exemplars = self.tools.find_maximally_activating_examples(top_k=top_k, max_samples=max_samples)
                            detailed_exemplars = [{"text": txt, "max_activation": act, "mean_activation": 0.0, "sum_activation": 0.0, 
                                                  "tokens": [], "per_token_activations": [], "max_token_index": 0, 
                                                  "layer": -1, "feature_index": -1} for txt, act in basic_exemplars]
                            print(f"🔍 Converted {len(detailed_exemplars)} basic exemplars to detailed format")
                        
                        # Store detailed exemplars for buffer extraction
                        self.last_detailed_exemplars = detailed_exemplars
                        print(f"✅ Stored {len(detailed_exemplars)} exemplars to last_detailed_exemplars")

                        if detailed_exemplars:
                            # Check if all activations are negative (suppression rather than activation)
                            all_negative = all(ex["max_activation"] < 0 for ex in detailed_exemplars)
                            if all_negative:
                                outputs.append(f"WARNING: All {len(detailed_exemplars)} corpus samples show NEGATIVE activation (suppression). This feature may be inactive on the current corpus.")
                                outputs.append("Consider:")
                                outputs.append("1. Testing with different text types (code, mathematical expressions, etc.)")
                                outputs.append("2. Using a different corpus or dataset")
                                outputs.append("3. This feature might be specialized for very specific content")
                                outputs.append("\nTop exemplars (all negative):")
                            else:
                                outputs.append(f"=== DETAILED FEATURE ANALYSIS ===")
                                outputs.append(f"Top {len(detailed_exemplars)} maximally activating examples from corpus (top_k={top_k}, max_samples={max_samples}):")
                            
                            # Format detailed exemplars with token-level information (limit to 5 for context)
                            for i, exemplar in enumerate(detailed_exemplars[:10], 1):
                                text = exemplar["text"]
                                max_act = exemplar["max_activation"]
                                mean_act = exemplar["mean_activation"]
                                tokens = exemplar["tokens"]
                                per_token_acts = exemplar["per_token_activations"]
                                max_token_idx = exemplar["max_token_index"]
                                
                                # Basic exemplar info (ultra-simplified for context length)
                                outputs.append(f"\n{i}. max_activation={max_act:.4f}, mean_activation={mean_act:.4f}")
                                outputs.append(f"   Text: {text}")#{'...' if len(text) > 80 else ''}
                                
                                # Token-level analysis if available (ultra-simplified)
                                if tokens and per_token_acts and len(tokens) == len(per_token_acts):
                                    # Show only top 3 activating tokens to save space
                                    token_pairs = list(zip(tokens, per_token_acts))
                                    if token_pairs:
                                        sorted_pairs = sorted(token_pairs, key=lambda x: x[1], reverse=True)
                                        top_3 = sorted_pairs
                                        
                                        # Only show top 3 non-BOS tokens
                                        key_tokens = []
                                        for token, act in top_3:
                                            if token != '<bos>':
                                                key_tokens.append(f"'{token}':{act:.3f}")
                                                # if len(key_tokens) >= 3:
                                                #     break
                                        
                                        if key_tokens:
                                            outputs.append(f"   Key tokens: {', '.join(key_tokens)}")
                                else:
                                    outputs.append(f"   Token-level analysis not available")
                        else:
                            outputs.append("No corpus-based exemplars available (empty corpus or error). Consider providing --dataset_path.")
                    except Exception as e:
                        outputs.append(f"Error finding corpus exemplars: {str(e)}")
                
                else:
                    outputs.append(f"Tool command executed: {tool_part}")
            
            # Join multiple tool outputs if more than one was present
            return "\n".join(outputs)
        
        elif agent_response.startswith("[DESCRIPTION]"):
            return "Final description received - experiment concluded"
        
        else:
            return f"Agent response: {agent_response[:200]}..."