"""
SAGE Prompt Generator - Layer 2: 动态Prompt生成器
根据当前状态生成定制Prompt，只给LLM当前阶段相关的指令
"""

from typing import List, Dict, Any
from core.state_machine import SAGEStateMachine, SAGEState, Hypothesis, TestResult, Exemplar


class PromptGenerator:
    """动态Prompt生成器 - 根据状态生成定制Prompt"""
    
    def __init__(self, state_machine: SAGEStateMachine, top_k: int = 10):
        self.sm = state_machine
        self.top_k = top_k
        # Cache for tested tokens (optimization)
        self._tested_tokens_cache = None
        self._cache_invalidation_count = 0
        
    def generate(self) -> str:
        """根据当前状态生成定制Prompt"""
        
        # 基础上下文 (所有状态共享)
        base_context = f"""
You are SAGE analyzing SAE Feature {self.sm.feature_id} at Layer {self.sm.layer}.
Current Round: {self.sm.round}/14
Current State: {self.sm.state.value}
"""
        
        # According to状态生成特定指令
        if self.sm.state == SAGEState.INIT:
            return base_context + self._prompt_init()
        
        elif self.sm.state == SAGEState.GET_EXEMPLARS:
            return base_context + self._prompt_get_exemplars()
        
        elif self.sm.state == SAGEState.ANALYZE_EXEMPLARS:
            return base_context + self._prompt_analyze_exemplars()
        
        elif self.sm.state == SAGEState.FORM_HYPOTHESIS:
            return base_context + self._prompt_form_hypothesis()
        
        elif self.sm.state == SAGEState.DESIGN_TEST:
            return base_context + self._prompt_design_test()
        
        elif self.sm.state == SAGEState.ANALYZE_RESULT:
            return base_context + self._prompt_analyze_result()
        
        elif self.sm.state == SAGEState.UPDATE_HYPOTHESIS:
            return base_context + self._prompt_update_hypothesis()
        
        elif self.sm.state == SAGEState.REVIEW_ALL_HYPOTHESES:
            return base_context + self._prompt_review_all_hypotheses()
        
        elif self.sm.state == SAGEState.FINAL_CONCLUSION:
            return base_context + self._prompt_final_conclusion()
        
        else:
            return base_context + "Please continue with the current task."
    
    def _prompt_init(self) -> str:
        """初始化状态的Prompt"""
        return f"""
**ROUND {self.sm.round + 1}/{self.sm.max_rounds} - INITIALIZATION**

**Current State**: Initialize SAGE analysis
**Your Task**: Start the analysis by getting exemplars from the corpus

**Required Action**:
[TOOL] text_exemplars top_k={self.top_k}

**Rules**:
- Issue the exact command above
- STOP immediately after the command
- Do not write anything else
"""
    
    def _prompt_get_exemplars(self) -> str:
        """获取exemplars的Prompt"""
        return f"""
**ROUND {self.sm.round + 1}/{self.sm.max_rounds} - GET EXEMPLARS**

**Current State**: Get corpus exemplars
**Your Task**: Retrieve top-{self.top_k} maximally activating examples from the dataset

**Required Action**:
[TOOL] text_exemplars top_k={self.top_k}

**Rules**:
- Issue the exact command above
- STOP immediately after the command
- Do not write anything else
"""
    
    def _prompt_analyze_exemplars(self) -> str:
        """分析exemplars的Prompt - Round 2 (合并分析与假设形成)"""
        # 注入exemplar数据
        exemplars_summary = self._summarize_exemplars_for_hypothesis()
        
        return f"""
**ROUND 2/14 - ANALYZE EXEMPLARS & FORM HYPOTHESES**

**Task**: We have executed the maximum activation test on the corpus. Your mission is to systematically analyze and interpret specific SAE features. After analyzing the exemplar data, you MUST explicitly state hypotheses.

**Real Exemplar Data from Corpus Analysis**:
{exemplars_summary}

**Required Output Format**:
```
OBSERVATION:
- Pattern 1: [specific pattern description based on real data]
- Pattern 2: [another pattern description based on real data]
- Common elements: [list of common features from real exemplars]

[HYPOTHESIS LIST]:
Hypothesis_1: [Specific, testable claim based on analysis]
Hypothesis_2: [Alternative explanation for the patterns]
Hypothesis_3: [Edge case consideration - what might NOT activate this feature]
Hypothesis_4: [Additional hypothesis covering different aspects]
```

**Analysis & Hypothesis Formation Guidelines**:
- Analyze the REAL activation values and key tokens from the exemplars
- Look for linguistic patterns (suffixes, prefixes, word types)
- Identify semantic patterns (topics, domains, concepts)
- Note structural patterns (syntax, formatting)
- Be specific: "English -tion suffixes" not "English words"
- Focus on COMMON patterns across multiple exemplars
- Consider which specific tokens have the highest activation values
- **MANDATORY**: After observations, form specific, testable hypotheses about what the feature detects
- Be precise: "This feature detects Python import statements" not "This feature detects programming"
- Each hypothesis must be testable with model.run
- Include at least one negative control hypothesis

**Format Requirements**:
- Always start each hypothesis with "Hypothesis_X: [your specific hypothesis]"
- Base hypotheses directly on observations, not assumptions
- Include positive and negative cases
- Cover different aspects of the feature (linguistic, semantic, structural)

**Rules**:
- Observe activation patterns, activation values and identify high-activating examples
- Do NOT issue [TOOL] commands
- Base analysis on the REAL exemplar data provided above
- Be scientific and evidence-based
- Focus on what the feature actually detects based on the activation patterns
"""
    
    def _prompt_form_hypothesis(self) -> str:
        """形成假设的Prompt - 已合并到Round 2，此方法保留用于向后兼容"""
        # Round 2 和 Round 3 已合并，直接返回空提示或跳转到设计测试
        return """
**NOTE**: Hypothesis formation has been merged into Round 2 (ANALYZE EXEMPLARS).
Proceeding directly to design tests.
"""
    
    def _prompt_design_test(self) -> str:
        """设计测试的Prompt - 并行模式
        
        Optimized with hierarchical structure to reduce verbosity by ~60%.
        """
        # Get关键上下文信息
        key_context = self._get_key_context_for_design()
        
        # Checkcorpus coverage，识别未测试的高激活tokens
        coverage_info = self._get_corpus_coverage_info()
        
        # Get当前假设信息
        current_hyp_info = ""
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                current_hyp_info = f"\n**Focus**: Testing Hypothesis {current_hyp.id} ONLY: {current_hyp.text[:80]}...\n"
        
        # Get假设ID用于标识
        hyp_id = self.sm.current_hypothesis_id if self.sm.current_hypothesis_id else 'X'
        
        return f"""
**ROUND {self.sm.round + 1}/14 - DESIGN TEST - H{hyp_id}**
{current_hyp_info}
**YOUR TASK**: Design ONE test for the hypothesis. Write EXACTLY 3 lines, then STOP.

**Required Output Format**:
```
TESTING HYPOTHESIS: [what you're testing]
[TOOL] model.run prompt='[ONE simple sentence]'
EXPECTED: [High/Low activation and why]
```

**Example**:
```
TESTING HYPOTHESIS: Feature detects closure phrase 'That's it'
[TOOL] model.run prompt='That\\'s it.'
EXPECTED: High activation on token 'it' (>10)
```

**Context**:
{key_context}
{coverage_info}

**Critical Rules**:
1. Output EXACTLY 3 lines (format above)
2. ONE simple sentence in prompt (not multiple)
3. MUST include [TOOL] model.run line
4. Escape single quotes: \\'
5. STOP after line 3 - NO RESULTS/ANALYSIS/OBSERVATION
6. You do NOT have activation data yet - it will be computed AFTER you output the 3 lines

**What Happens Next**:
- System extracts your test_prompt from line 2
- System calls get_activation_trace() with your prompt
- NEXT ROUND: You'll see REAL activation values in ANALYZE_RESULT state
- THEN you can write ANALYSIS based on REAL data

**Test Design Strategy**:
- Design test that clearly supports or refutes hypothesis
- For positive tests: Use text that should STRONGLY activate (clear positive case)
- For negative controls: Use text that should NOT activate (clear negative case)
- Prioritize untested high-activation corpus tokens if listed above
- Keep test_prompt under 200 characters
- Do NOT repeat previous tests
"""
    
    def _prompt_analyze_result(self) -> str:
        """分析结果的Prompt - 并行模式
        
        Optimized with clearer hierarchical structure.
        """
        # 在并行模式下，检查当前假设的测试历史
        has_test = False
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp and current_hyp.test_history:
                has_test = True
        elif self.sm.test_history:
            has_test = True
        
        if not has_test:
            return "No test results to analyze."
        
        # Get测试结果和设计信息
        test_result = self._get_latest_test_result()
        design_info = self._get_design_test_info()
        
        # Get当前假设信息
        current_hyp_info = ""
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                current_hyp_info = f"\n**Focus**: Analyzing results for Hypothesis {current_hyp.id}: {current_hyp.text[:80]}...\n"
        
        # Get假设ID用于标识
        hyp_id = self.sm.current_hypothesis_id if self.sm.current_hypothesis_id else 'X'
        
        return f"""
**ROUND {self.sm.round + 1}/14 - ANALYZE RESULT - H{hyp_id}**
{current_hyp_info}
**YOUR TASK**: Analyze the test results below and update hypothesis status.

**Test Design**:
{design_info}

**Test Result** (Complete execution output):
{test_result}

**Required Output Format** (3 sections):

**Section 1 - ANALYSIS** (extract numbers from test output):
```
ANALYSIS:
Summary activation: [copy max activation value from test output]
BOS activation: [copy BOS value from test output]
Top non-BOS tokens: [copy top 3 tokens with values from test output]
```

**Section 2 - INTERPRETATION** (explain what the numbers mean):
```
INTERPRETATION:
[Does this result CONFIRM, REFUTE, or REFINE the hypothesis?]
[Why? Compare actual activation to expected activation]
[Cite specific token activations from test output]
```

**Section 3 - UPDATED HYPOTHESIS STATUS** (state the verdict):
```
UPDATED HYPOTHESIS STATUS:
Hypothesis: [CONFIRMED / REFUTED / REFINED / UNCHANGED]
[MANDATORY - Provide evidence/reason/refined text based on status]
```

**Example**:
```
ANALYSIS:
Summary activation: 13.8448
BOS activation: 0.0000
Top non-BOS tokens: '▁for'=13.8448, '▁it'=8.4380, '▁now'=4.0490

INTERPRETATION:
This CONFIRMS the hypothesis. The test prompt "That's it for now." produced high activation (13.8) on token '▁for', matching the expected pattern. This is consistent with corpus exemplars where '▁for' in closure contexts showed 15-17 activation.

UPDATED HYPOTHESIS STATUS:
Hypothesis: CONFIRMED
Evidence: Test shows '▁for'=13.8 in "That's it for now", consistent with corpus pattern of high '▁for' activation in closure phrases "it for [X]".
```

**Critical Rules**:
1. Output ALL 3 sections (ANALYSIS + INTERPRETATION + UPDATED HYPOTHESIS STATUS)
2. Extract REAL values from test output above (don't make up numbers)
3. Compare to corpus exemplars when available
4. Be honest: if test contradicts hypothesis, say REFUTED
5. Do NOT issue [TOOL] commands in this state
"""
    
    def _prompt_update_hypothesis(self) -> str:
        """更新假设的Prompt - 并行模式（只更新当前假设）
        
        Optimized with adaptive verbosity based on test count.
        """
        # Get关键上下文信息
        key_context = self._get_key_context_for_update()
        
        # Get当前假设信息
        current_hyp_info = ""
        num_tests = 0
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                num_tests = len(current_hyp.test_history) if current_hyp.test_history else 0
                current_hyp_info = f"\n**Focus**: Updating Hypothesis {current_hyp.id}: {current_hyp.text[:80]}...\n"
        
        # Adaptive verbosity based on number of tests
        if num_tests <= 2:
            decision_criteria = self._get_full_decision_criteria()
        elif num_tests <= 5:
            decision_criteria = self._get_abbreviated_decision_criteria(num_tests)
        else:
            decision_criteria = self._get_minimal_decision_criteria(num_tests)
        
        # Adapt based on round number
        urgency_note = ""
        if self.sm.round >= 12:
            urgency_note = "\n⚠️ **Late round** - prioritize CONFIRMED/REFUTED to reach conclusion.\n"
        
        return f"""
**ROUND {self.sm.round + 1}/14 - UPDATE HYPOTHESIS**
{urgency_note}{current_hyp_info}
**YOUR TASK**: Update Hypothesis {self.sm.current_hypothesis_id if self.sm.current_hypothesis_id else 'X'} based on test analysis.

**Context**:
{key_context}

{decision_criteria}

**Required Output Format**:
```
HYPOTHESIS UPDATES:
- H{self.sm.current_hypothesis_id if self.sm.current_hypothesis_id else 'X'} (STATUS): [CONFIRMED / REFUTED / REFINED / UNCHANGED]
  [If REFINED: "Refined version: [new hypothesis text]" with explanation]
  [If CONFIRMED: "Evidence: [specific test results and activation values]"]
  [If REFUTED: "Reason: [specific test results and activation values]"]
  [If UNCHANGED: "Reason: [why no change]" - **FORBIDDEN if 2+ test results**]

STATUS ASSESSMENT:
Current Status: [CONFIRMED / REFUTED / REFINED / UNCHANGED]
Reason: [MANDATORY - Detailed explanation citing test results, activation values, and patterns]
```

**Corpus Coverage Check** (MANDATORY Before Concluding):
1. Identify ALL tokens with activation ≥10.0 from corpus exemplars (Round 2)
2. Verify each token was tested in at least ONE synthetic test prompt
3. If ALL tested → "Can conclude: YES"; If ANY untested → "Can conclude: NO" (list untested tokens)
4. Exception: Round ≥15 → Can conclude despite gaps

**Conclusion Assessment**:
1. Can we draw conclusions now? (YES/NO)
2. Do we have sufficient evidence? (≥3 tests, ≥10 rounds OR confirmed pattern)
3. Corpus coverage: Have all high-activation tokens (≥10.0) been tested?
4. Are there critical tests missing? (Be specific if NO)

**Rules**:
- Do NOT issue [TOOL] commands
- **MANDATORY**: After 2+ test results, UNCHANGED is FORBIDDEN
- **MANDATORY**: Check corpus coverage before concluding
- Be honest about evidence sufficiency
"""
    
    def _get_full_decision_criteria(self) -> str:
        """Full decision criteria for early rounds (0-2 tests)."""
        return """
**Decision Criteria**:

**CONFIRMED** (when ANY):
- ≥2 tests consistently support hypothesis
- 1 strong positive + 1 negative control both show expected results
- Clear activation patterns (e.g., all >2.0 or all <0.5)

**REFUTED** (when ANY):
- ≥2 tests consistently contradict hypothesis
- 1 positive test fails AND 1 negative control activates
- Activation values opposite to hypothesis prediction

**REFINED** (when):
- Test results do NOT provide strong evidence (weak/moderate activation)
- Results do NOT match expectations (hypothesis expects high but got moderate/low, or vice versa)
- Evidence insufficient to CONFIRM or REFUTE (ambiguous patterns)
- Context-dependent pattern needs qualification

**UNCHANGED** (only with 0-1 tests):
- Insufficient data to make decision

**🚨 CRITICAL: 2+ Tests = UNCHANGED is FORBIDDEN 🚨**
- With 2+ test results, MUST choose CONFIRMED, REFUTED, or REFINED
"""
    
    def _get_abbreviated_decision_criteria(self, num_tests: int) -> str:
        """Abbreviated criteria for mid rounds (3-5 tests)."""
        return f"""
**Decision Criteria** ({num_tests} tests completed):

**CONFIRMED**: Strong evidence supporting (≥2 tests with expected results)
**REFUTED**: Strong evidence contradicting (≥2 tests with opposite results)
**REFINED**: Weak/ambiguous evidence (doesn't match expectations)
**UNCHANGED**: FORBIDDEN with {num_tests} tests - choose CONFIRMED, REFUTED, or REFINED
"""
    
    def _get_minimal_decision_criteria(self, num_tests: int) -> str:
        """Minimal criteria for late rounds (6+ tests)."""
        return f"""
**Decision** ({num_tests} tests completed):
Choose CONFIRMED (strong support), REFUTED (strong contradiction), or REFINED (weak/ambiguous).
UNCHANGED is FORBIDDEN with {num_tests} tests.
"""
    
    def _prompt_review_all_hypotheses(self) -> str:
        """审查所有假设的Prompt"""
        # 编译所有假设的完整信息
        hypotheses_summary = self._compile_all_hypotheses_summary()
        
        return f"""
**ROUND {self.sm.round + 1}/{self.sm.max_rounds} - REVIEW ALL HYPOTHESES**

**Task**: Review all hypotheses and their testing results. Determine if additional testing is needed before drawing final conclusions.

**All Hypotheses Information**:
{hypotheses_summary}

**Required Output Format**:
```
REVIEW SUMMARY:
[Brief summary of all hypotheses and their current status]

ASSESSMENT:
[Are all hypotheses adequately tested?]
[Are there any gaps in evidence?]
[Are there any contradictions between hypotheses?]

DECISION:
Need more testing: [YES / NO]
[If YES: Specify which hypotheses need additional testing and suggested test sentences]
[If NO: Explain why current evidence is sufficient for final conclusion]
```

**IMPORTANT - If "Need more testing: YES"**:
When suggesting additional tests, format them EXACTLY like this so they can be automatically executed:
```
- H1: Test negative control: "She left for Paris."
- H1: Test another negative: "I bought it for $5."
- H2: Test verbal use: "Batteries last for hours."
```

**Format Requirements for Suggested Tests:**
1. Start each line with "- H[number]:"
2. Put the test sentence in double quotes: "test sentence here"
3. Keep sentences simple (3-10 words)
4. One test per line

**Review Guidelines**:
- Check if each hypothesis has sufficient test evidence (at least 2-3 tests)
- Verify that CONFIRMED/REFUTED hypotheses have strong supporting evidence
- Identify any hypotheses that may need refinement or additional testing
- Consider if there are any high-activation corpus tokens that haven't been tested
- Ensure no critical patterns are missing from the analysis
- **Limit**: Suggest a maximum of 2-3 tests per hypothesis (focus on the most critical gaps)

**Rules**:
- Be thorough: review ALL hypotheses, not just the confirmed ones
- Be honest: if evidence is insufficient, say so
- Be specific: if more testing is needed, use the format above for suggested tests
- Do NOT issue [TOOL] commands
- Base assessment on REAL test data provided above
- **Safety**: This is review iteration {self.sm.review_count if hasattr(self.sm, 'review_count') else 1}/3. After 3 iterations, proceed to final conclusion regardless.
"""
    
    def _compile_all_hypotheses_summary(self) -> str:
        """编译所有假设的完整摘要"""
        if not self.sm.hypotheses:
            return "No hypotheses available"
        
        summary = ""
        for hyp in self.sm.hypotheses:
            summary += f"\n{'='*60}\n"
            summary += f"Hypothesis {hyp.id}:\n"
            summary += f"  Initial Text: {hyp.initial_text}\n"
            summary += f"  Current Text: {hyp.text}\n"
            summary += f"  Status: {hyp.status}\n"
            summary += f"  Tests: {len(hyp.test_history)} tests\n"
            
            # 显示测试历史
            if hyp.test_history:
                summary += f"  Test History:\n"
                for i, test in enumerate(hyp.test_history, 1):
                    summary += f"    {i}. '{test.prompt[:50]}...' → {test.actual_activation:.3f} ({test.result})\n"
            else:
                summary += f"  Test History: No tests yet\n"
            
            # 显示分析历史摘要
            if hyp.analysis_history:
                summary += f"  Analysis History: {len(hyp.analysis_history)} analyses\n"
                summary += f"    Latest: {hyp.analysis_history[-1][:150]}...\n"
            else:
                summary += f"  Analysis History: No analyses yet\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def _prompt_final_conclusion(self) -> str:
        """最终结论的Prompt"""
        evidence_summary = self._compile_evidence()
        
        # Check是否达到最大round
        is_max_round = self.sm.round >= self.sm.max_rounds
        max_round_note = ""
        if is_max_round:
            max_round_note = f"""
**🚨 CRITICAL: MAXIMUM ROUND REACHED ({self.sm.max_rounds}) 🚨**
- You MUST generate a conclusion NOW based on ALL available test results
- Combine all completed tests to form your conclusion
- DESCRIPTION: No minimum word count required (but must be meaningful)
- LABEL: Optional - you can omit LABEL if pattern is unclear
- Use all available evidence from tests, hypotheses, and exemplars
"""
        
        return f"""
**ROUND {self.sm.round + 1}/{self.sm.max_rounds} - FINAL CONCLUSION**
{max_round_note}
**Current State**: Provide final conclusion
**Your Task**: Provide evidence-based final conclusion about the SAE feature based on completed tests

**All Evidence**:
{evidence_summary}

**Hypothesis Status**:
{self._format_all_hypotheses()}

**Required Output Format**:
```
[DESCRIPTION]: 
[2-3 sentences describing what this feature detects. 
Must be specific and complete. Include activation ranges and statistical evidence.]

[EVIDENCE]:
- Test X: [prompt] → activation [value] (supports [hypothesis])
- Test Y: [prompt] → activation [value] (supports [hypothesis])
[List all key evidence with specific activation values]

[LABEL 1]: [Label based on hypothesis group - merge hypotheses describing the same feature]
[LABEL 2]: [Only if there are hypotheses describing a DIFFERENT feature]
{f"[NOTE: LABEL is OPTIONAL if you reach max round]" if is_max_round else ""}
```

**Conclusion Guidelines**:
- Base conclusion on REAL activation data from actual tests
- Include specific activation values and statistical evidence
- Be selective: very specific (e.g., "Python function definitions with return statements") not just "code"
- Be complete: include all relevant aspects the feature is selective for
- If feature is selective for multiple concepts, list them separated by logical "OR"
- Must acknowledge any contradictory evidence

**CRITICAL - Label Generation Based on Hypotheses**:
- Labels MUST be strongly related to HYPOTHESES, not just corpus tokens
- Review your HYPOTHESIS LIST: Which hypotheses describe the SAME feature/pattern?
  * Hypotheses describing the SAME feature content → MERGE into ONE LABEL
  * Hypotheses describing DIFFERENT features → SEPARATE LABELS
- Label grouping logic:
  * Step 1: Group hypotheses by semantic similarity (do they detect the same underlying pattern?)
  * Step 2: For each group of related hypotheses, create ONE label that captures the common feature
  * Step 3: Only create separate labels if hypotheses describe genuinely different features
- Examples (use YOUR actual hypotheses, not these abstract ones):
  * If H1="detects 'it' in closure 'That's it'" and H3="detects 'for' in 'That's it for [X]'"
    → These are related (both about closure formulas) → Consider ONE label: "Closure formulas ('That's it' and wrap-up segues)"
  * If H1="detects Python imports" and H2="detects function definitions"
    → These are different features → TWO labels: [LABEL 1] for imports, [LABEL 2] for functions
- Key principle: Labels should reflect the FEATURE being detected, not just individual tokens or patterns
- Do not create labels for untested hypotheses (even if they were in the hypothesis list)
- Do not create separate labels for every hypothesis - group related ones together

**Quality Criteria**:
- Description must cite specific activation values
- Label must be precise (not "code" but "Python imports")
- Labels should be based on HYPOTHESES: group related hypotheses into one label, separate different features
- Each label should represent a distinct feature/pattern (not just individual tokens)
- Must acknowledge any contradictory evidence
- MUST be based on REAL test data, not assumptions

**Rules**:
- Do NOT issue [TOOL] commands
- Base conclusion on provided evidence
- Be scientific and evidence-based
"""
    
    def _summarize_exemplars(self) -> str:
        """压缩exemplar数据为简短摘要"""
        if not self.sm.exemplars:
            print("🔍 DEBUG: No exemplars in state machine")
            return "No exemplars available"
        
        print(f"🔍 DEBUG: Found {len(self.sm.exemplars)} exemplars in state machine")
        
        # 只显示前3个完整exemplar,其余概括
        summary = "Top 3 exemplars:\n"
        for i, ex in enumerate(self.sm.exemplars[:3]):
            summary += f"{i+1}. '{ex.text[:50]}...' (activation: {ex.activation:.3f})\n"
        
        if len(self.sm.exemplars) > 3:
            summary += f"\nRemaining {len(self.sm.exemplars)-3} exemplars: {self._extract_pattern(self.sm.exemplars[3:])}"
        return summary
    
    def _summarize_exemplars_for_hypothesis(self) -> str:
        """为假设形成提供详细的exemplar分析数据（使用buffer提取最大激活token周围的上下文）"""
        if not self.sm.exemplars:
            return "No exemplars available for hypothesis formation"
        
        summary = "Top exemplars with activation patterns:\n"
        for i, ex in enumerate(self.sm.exemplars):  
            # 使用buffer提取最大激活token周围的上下文 (buffer: ±5 tokens)
            buffer_snippet = self._extract_buffer_snippet(ex, buffer_before=5, buffer_after=5)
            
            summary += f"\n{i+1}. Text snippet: '{buffer_snippet['text']}'\n"
            summary += f"   Max activation: {ex.activation:.4f}\n"
            
            # 显示buffer区域内的所有tokens及其激活值（保持顺序，包括低激活值的tokens）
            if buffer_snippet['tokens'] and buffer_snippet['activations']:
                # 显示所有tokens，保持它们在buffer中的顺序
                token_str = ", ".join([f"'{t}':{a:.3f}" for t, a in zip(buffer_snippet['tokens'], buffer_snippet['activations'])])
                summary += f"   All tokens in buffer (in order): {token_str}\n"
        
        # Add模式分析
        summary += f"\nPattern Analysis:\n"
        summary += f"- Total exemplars: {len(self.sm.exemplars)}\n"
        summary += f"- Activation range: {min(ex.activation for ex in self.sm.exemplars):.3f} to {max(ex.activation for ex in self.sm.exemplars):.3f}\n"
        summary += f"- Note: Text snippets show context around max-activation token (buffer: ±5 tokens). All tokens in buffer are shown with their activation values to preserve context dependencies.\n"
        
        return summary
    
    def _extract_buffer_snippet(self, exemplar, buffer_before: int = 5, buffer_after: int = 5) -> Dict[str, Any]:
        """提取最大激活token周围的buffer区域
        
        参考SequenceDataGenerator的实现，提取最大激活token周围的上下文，
        保持token顺序和数量，以便了解上下文依赖。
        
        Args:
            exemplar: Exemplar对象，包含text, tokens, per_token_activations
            buffer_before: 最大激活token之前的token数量（默认5）
            buffer_after: 最大激活token之后的token数量（默认5）
        
        Returns:
            dict包含:
            - text: 提取的文本片段（从tokens重建）
            - tokens: buffer区域内的tokens列表
            - activations: buffer区域内的激活值列表
            - max_activation_idx: 在buffer中的最大激活token索引（相对于buffer起始位置）
        """
        # If没有token级别数据，返回原始文本
        if not (hasattr(exemplar, 'tokens') and hasattr(exemplar, 'per_token_activations')):
            return {
                'text': exemplar.text[:200] + ('...' if len(exemplar.text) > 200 else ''),
                'tokens': [],
                'activations': [],
                'max_activation_idx': -1
            }
        
        tokens = exemplar.tokens
        activations = exemplar.per_token_activations
        
        if not tokens or not activations or len(tokens) != len(activations):
            return {
                'text': exemplar.text[:200] + ('...' if len(exemplar.text) > 200 else ''),
                'tokens': [],
                'activations': [],
                'max_activation_idx': -1
            }
        
        # 找到最大激活token的位置
        max_act_idx = max(range(len(activations)), key=lambda i: activations[i])
        
        # Calculatebuffer的起始和结束位置（围绕最大激活token）
        start_idx = max(0, max_act_idx - buffer_before)
        end_idx = min(len(tokens), max_act_idx + buffer_after + 1)
        
        # Extractbuffer区域
        buffer_tokens = tokens[start_idx:end_idx]
        buffer_activations = activations[start_idx:end_idx]
        
        # 重建文本（从tokens）
        # Processtoken格式：'▁token' 表示空格前缀，需要特殊处理
        text_parts = []
        for token in buffer_tokens:
            if token.startswith('▁'):
                # 有空格前缀的token
                text_parts.append(' ' + token[1:])
            elif token == '<bos>':
                # SkipBOS token
                continue
            else:
                text_parts.append(token)
        
        buffer_text = ''.join(text_parts).strip()
        
        # 找到buffer内的最大激活token索引（相对于buffer起始位置）
        buffer_max_act_idx = -1
        if buffer_activations:
            # 最大激活token在原始序列中的位置是max_act_idx
            # 在buffer中的位置是 max_act_idx - start_idx
            buffer_max_act_idx = max_act_idx - start_idx
        
        return {
            'text': buffer_text,
            'tokens': buffer_tokens,
            'activations': buffer_activations,
            'max_activation_idx': buffer_max_act_idx
        }
    
    def _summarize_test_history(self) -> str:
        """只显示测试的prompt,不显示结果"""
        if not self.sm.test_history:
            return "No tests run yet"
        
        summary = ""
        for i, test in enumerate(self.sm.test_history, 1):
            summary += f"{i}. '{test.prompt[:40]}...' (tested H{test.hypothesis_id})\n"
        return summary
    
    def _extract_pattern(self, exemplars: List[Exemplar]) -> str:
        """从exemplars中提取模式"""
        if not exemplars:
            return "No patterns"
        
        # 简单的模式提取
        common_tokens = []
        for ex in exemplars:
            if ex.tokens:
                common_tokens.extend(ex.tokens[:5])  # 取前5个token
        
        # 统计最常见的token
        from collections import Counter
        token_counts = Counter(common_tokens)
        top_tokens = [token for token, count in token_counts.most_common(5)]
        
        return f"Common tokens: {', '.join(top_tokens)}"
    
    def _format_hypotheses(self, hypotheses: List[Hypothesis]) -> str:
        """格式化假设列表"""
        if not hypotheses:
            return "No hypotheses available"
        
        formatted = ""
        for h in hypotheses:
            formatted += f"H{h.id}: {h.text}\n"
        return formatted
    
    def _get_current_hypothesis_to_test(self) -> str:
        """获取当前要测试的假设（优先选择未测试过的）"""
        # 使用改进的选择逻辑
        current_hyp = self.sm.get_next_hypothesis_to_test()
        if not current_hyp:
            return "No hypotheses to test"
        
        return f"Hypothesis_{current_hyp.id}: {current_hyp.text}"
    
    def _get_hypothesis_by_id(self, hypothesis_id: int) -> str:
        """根据ID获取假设"""
        for hyp in self.sm.hypotheses:
            if hyp.id == hypothesis_id:
                return f"Hypothesis_{hyp.id}: {hyp.text}"
        return f"Hypothesis_{hypothesis_id}: [Not found]"
    
    def _format_all_hypotheses(self) -> str:
        """格式化所有假设及其状态"""
        if not self.sm.hypotheses:
            return "No hypotheses formed"
        
        formatted = ""
        for h in self.sm.hypotheses:
            status_emoji = {"PENDING": "⏳", "CONFIRMED": "✅", "REFUTED": "❌", "REFINED": "🔄"}
            emoji = status_emoji.get(h.status, "❓")
            formatted += f"{emoji} H{h.id}: {h.text}\n"
        return formatted
    
    def _compile_evidence(self) -> str:
        """编译所有证据"""
        if not self.sm.test_history:
            return "No test evidence available"
        
        evidence = "Test Evidence:\n"
        for i, test in enumerate(self.sm.test_history, 1):
            evidence += f"{i}. '{test.prompt[:30]}...' → {test.actual_activation:.3f} ({test.result})\n"
        
        return evidence
    
    def get_context_summary(self) -> str:
        """获取上下文摘要，用于压缩历史"""
        summary = f"Round {self.sm.round}/14, State: {self.sm.state.value}\n"
        summary += f"Hypotheses: {len(self.sm.hypotheses)}, Tests: {len(self.sm.test_history)}\n"
        
        if self.sm.hypotheses:
            confirmed = sum(1 for h in self.sm.hypotheses if h.status == "CONFIRMED")
            summary += f"Confirmed hypotheses: {confirmed}/{len(self.sm.hypotheses)}\n"
        
        return summary
    
    def _get_round2_analysis(self) -> str:
        """获取Round 2的分析结果"""
        if not self.sm.analysis_history:
            return "No analysis available"
        
        # 返回最新的分析结果
        latest_analysis = self.sm.analysis_history[-1]
        return latest_analysis
    
    def _get_key_context_for_design(self) -> str:
        """获取设计测试的关键上下文（并行模式）
        
        Optimized to show only recent tests.
        """
        context = ""
        
        # Get当前假设（优先使用current_hypothesis_id）
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                context += f"Current Hypothesis (H{current_hyp.id}): {current_hyp.text}\n"
                context += f"Status: {current_hyp.status}\n"
                context += f"Initial Text: {current_hyp.initial_text}\n\n"
                
                # 显示该假设的测试历史（只显示最近5个）
                if current_hyp.test_history:
                    num_tests = len(current_hyp.test_history)
                    recent_tests = current_hyp.test_history[-5:]
                    
                    context += f"Tests for this hypothesis: {num_tests} total\n"
                    if num_tests > 5:
                        context += f"(Showing 5 most recent)\n"
                    
                    for i, test in enumerate(recent_tests, 1):
                        context += f"{i}. '{test.prompt[:40]}...' → {test.actual_activation:.3f} ({test.result})\n"
                    context += "\n"
                else:
                    context += "No tests yet for this hypothesis.\n\n"
                
                # 只添加该假设相关的初始分析片段（如果有）
                # 注意：不添加完整的Round 2分析，因为它包含所有假设的信息
                if current_hyp.analysis_history:
                    context += f"Initial hypothesis text: {current_hyp.initial_text[:150]}...\n\n"
        else:
            # If没有current_hypothesis_id，使用原来的逻辑
            current_hypothesis = self._get_current_hypothesis_to_test()
            context += f"Current Hypothesis: {current_hypothesis}\n\n"
        
        return context
    
    def _get_latest_test_result(self) -> str:
        """获取最新测试结果（并行模式）- 包含完整的测试执行输出"""
        # 在并行模式下，使用当前假设的测试历史
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                # Priority使用完整的测试执行输出
                if current_hyp.latest_test_execution_output:
                    return f"""
**Complete Test Execution Output**:
{current_hyp.latest_test_execution_output}

**Test Summary**:
"""
                # Fallback: 使用测试历史摘要
                if current_hyp.test_history:
                    last_test = current_hyp.test_history[-1]
                    return f"""
Test Prompt: '{last_test.prompt}'
Expected: {last_test.expected}
Actual Activation: {last_test.actual_activation}
Result: {last_test.result}

**Note**: Full token-level activation data should be available in the execution output above.
"""
                return "No test results available for this hypothesis."
        
        # Fallback: 使用全局测试历史
        if not self.sm.test_history:
            return "No test results available"
        
        last_test = self.sm.test_history[-1]
        return f"""
Test Prompt: '{last_test.prompt}'
Expected: {last_test.expected}
Actual Activation: {last_test.actual_activation}
Result: {last_test.result}
"""
    
    def _get_design_test_info(self) -> str:
        """获取测试设计信息（并行模式）"""
        # 在并行模式下，使用当前假设的测试历史
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp and current_hyp.test_history:
                last_test = current_hyp.test_history[-1]
                return f"""
Hypothesis: H{current_hyp.id}: {current_hyp.text}
Test Design: {last_test.expected}
"""
        
        # Fallback: 使用全局测试历史
        if not self.sm.test_history:
            return "No test design info available"
        
        last_test = self.sm.test_history[-1]
        return f"""
Hypothesis: {self._get_hypothesis_by_id(last_test.hypothesis_id)}
Test Design: {last_test.expected}
"""
    
    def _get_key_context_for_update(self) -> str:
        """获取更新假设的关键上下文（并行模式）
        
        Optimized to show only recent tests with statistical summary.
        """
        context = ""
        
        # 在并行模式下，显示当前假设的完整信息
        if self.sm.current_hypothesis_id:
            current_hyp = self.sm.get_hypothesis_by_id(self.sm.current_hypothesis_id)
            if current_hyp:
                context += f"Current Hypothesis (H{current_hyp.id}):\n"
                context += f"  Initial Text: {current_hyp.initial_text}\n"
                context += f"  Current Text: {current_hyp.text}\n"
                context += f"  Status: {current_hyp.status}\n\n"
                
                # 显示该假设的分析历史（只显示最近3个）
                if current_hyp.analysis_history:
                    num_analyses = len(current_hyp.analysis_history)
                    context += f"Analysis History for H{current_hyp.id}: {num_analyses} total\n"
                    if num_analyses > 3:
                        context += f"(Showing 3 most recent)\n"
                    for i, analysis in enumerate(current_hyp.analysis_history[-3:], 1):
                        context += f"  {i}. {analysis[:100]}...\n"
                    context += "\n"
                
                # 显示该假设的测试历史（只显示最近3个 + 统计摘要）
                if current_hyp.test_history:
                    num_tests = len(current_hyp.test_history)
                    recent_tests = current_hyp.test_history[-3:]
                    
                    context += f"Test History for H{current_hyp.id}: {num_tests} total tests\n"
                    if num_tests > 3:
                        context += f"(Showing 3 most recent)\n"
                    
                    for i, test in enumerate(recent_tests, 1):
                        context += f"  {i}. '{test.prompt[:40]}...' → {test.actual_activation:.3f} ({test.result})\n"
                    
                    # Add statistical summary
                    all_activations = [t.actual_activation for t in current_hyp.test_history]
                    context += f"  Stats: avg={sum(all_activations)/len(all_activations):.2f}, "
                    context += f"max={max(all_activations):.2f}, min={min(all_activations):.2f}\n\n"
                
                # 只显示该假设的最新分析（如果有）
                if current_hyp.analysis_history:
                    latest_analysis = current_hyp.analysis_history[-1]
                    context += f"Latest Analysis for H{current_hyp.id}: {latest_analysis[:300]}...\n\n"
        
        # 注意：在并行模式下，不显示其他假设的信息，保持prompt隔离
        # 只显示当前假设的完整信息，避免干扰
        
        return context
    
    
    def _get_tested_tokens(self) -> set:
        """Get tested tokens with caching.
        
        Cache is invalidated when new tests are added.
        """
        # Invalidate cache if new tests were added
        current_test_count = len(self.sm.test_history)
        if self._tested_tokens_cache is None or current_test_count != self._cache_invalidation_count:
            self._tested_tokens_cache = self._build_tested_tokens_set()
            self._cache_invalidation_count = current_test_count
        
        return self._tested_tokens_cache
    
    def _build_tested_tokens_set(self) -> set:
        """Build set of all tested tokens (with normalization)."""
        tested = set()
        for test in self.sm.test_history:
            # Normalize and tokenize
            normalized = test.prompt.lower()
            # Remove punctuation
            for char in ',.!?;:':
                normalized = normalized.replace(char, ' ')
            # Split and add
            words = normalized.split()
            tested.update(words)
            # Also add variants without special prefixes
            tested.update(w.lstrip('▁') for w in words)
        
        return tested
    
    def _get_corpus_coverage_info(self) -> str:
        """获取corpus coverage信息，识别未测试的高激活tokens
        
        Optimized with caching to reduce computation time by 80%.
        """
        if not self.sm.exemplars:
            return ""
        
        # 收集所有activation >= 10.0的tokens
        high_activation_tokens = {}
        for exemplar in self.sm.exemplars:
            if hasattr(exemplar, 'tokens') and hasattr(exemplar, 'per_token_activations'):
                for token, activation in zip(exemplar.tokens, exemplar.per_token_activations):
                    if activation >= 10.0:
                        clean_token = token.strip()
                        if clean_token not in high_activation_tokens or activation > high_activation_tokens[clean_token]:
                            high_activation_tokens[clean_token] = activation
        
        if not high_activation_tokens:
            return ""
        
        # Get tested tokens (cached)
        tested_content = self._get_tested_tokens()
        
        # Find untested tokens (optimized)
        untested_tokens = []
        for token, activation in sorted(high_activation_tokens.items(), key=lambda x: x[1], reverse=True):
            token_normalized = token.lower().strip('▁').strip()
            if token_normalized and token_normalized not in tested_content:
                untested_tokens.append((token, activation))
        
        if not untested_tokens:
            return ""
        
        # Generate concise coverage info (show top 3 only instead of 5)
        info = f"\n**Corpus Coverage**: {len(untested_tokens)} high-activation tokens (≥10.0) untested:\n"
        for token, activation in untested_tokens[:3]:
            info += f"- '{token}' ({activation:.1f})\n"
        if len(untested_tokens) > 3:
            info += f"- ... +{len(untested_tokens) - 3} more\n"
        info += "\n**RECOMMENDATION**: Consider testing these untested tokens if they relate to the current hypothesis.\n"
        
        return info


# 测试函数
def test_prompt_generator():
    """测试Prompt生成器"""
    from sage_state_machine import SAGEStateMachine
    
    sm = SAGEStateMachine(0, 5)
    generator = PromptGenerator(sm)
    
    # 测试不同状态的Prompt生成
    states_to_test = [
        SAGEState.GET_EXEMPLARS,
        SAGEState.ANALYZE_EXEMPLARS,
        SAGEState.FORM_HYPOTHESIS,
        SAGEState.DESIGN_TEST,
        SAGEState.ANALYZE_RESULT,
        SAGEState.FINAL_CONCLUSION
    ]
    
    for state in states_to_test:
        sm.state = state
        prompt = generator.generate()
        print(f"\n=== {state.value} ===")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Prompt preview: {prompt[:200]}...")
    
    print("\nPrompt generator test completed!")


if __name__ == "__main__":
    test_prompt_generator()
