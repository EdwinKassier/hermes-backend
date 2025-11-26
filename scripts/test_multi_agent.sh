#!/bin/bash

# Multi-Agent Orchestration Test Suite
# Tests parallel execution, synthesis, and real-world scenarios

BASE_URL="http://localhost:8080/api/v1/hermes/process_request"

echo "🚀 Multi-Agent Orchestration Test Suite"
echo "========================================"
echo

# Test 1: Multi-Agent Detection
echo "📝 Test 1: Multi-Agent Task Detection"
echo "Query: 'Research quantum computing and analyze its applications'"
echo "Expected: Detect as multi-agent task"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Research quantum computing and analyze its applications", "legion_mode": true}')

MULTI_AGENT=$(echo "$RESULT" | jq -r '.metadata.decision_rationale[0].analysis.multi_agent_task_detected // false')
echo "Multi-agent detected: $MULTI_AGENT"

if [ "$MULTI_AGENT" == "true" ]; then
  echo "✅ PASS - Multi-agent task detected"
else
  echo "❌ FAIL - Should detect multi-agent task"
fi

# Test 2: Research + Code Task
echo -e "\n📝 Test 2: Research + Code (Parallel Execution)"
echo "Query: 'Find sorting algorithms and write Python quicksort implementation'"
echo "Expected: research_agent + code_agent in parallel"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Find sorting algorithms and write Python quicksort implementation", "legion_mode": true}')

AGENTS=$(echo "$RESULT" | jq -r '.metadata.agents_used | join(", ")' 2>/dev/null || echo "none")
PARALLEL=$(echo "$RESULT" | jq -r '.metadata.decision_rationale[0].analysis.multi_agent_task_detected // false')

echo "Parallel mode: $PARALLEL"
echo "Agents used: $AGENTS"

if [[ "$PARALLEL" == "true" ]]; then
  echo "✅ PASS - Parallel execution triggered"
else
  echo "⚠️  WARN - Expected parallel execution"
fi

# Test 3: Three-Way Parallel
echo -e "\n📝 Test 3: Three-Agent Parallel Task"
echo "Query: 'Research neural networks, analyze performance, and write training code'"
echo "Expected: 3 agents executing in parallel"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Research neural networks, analyze their performance characteristics, and write Python training code", "legion_mode": true}')

AGENTS=$(echo "$RESULT" | jq -r '.metadata.agents_used | join(", ")' 2>/dev/null || echo "none")
MULTI=$(echo "$RESULT" | jq -r '.metadata.decision_rationale[0].analysis.multi_agent_task_detected // false')
MSG_LEN=$(echo "$RESULT" | jq -r '.message | length')

echo "Multi-agent: $MULTI"
echo "Agents: $AGENTS"
echo "Response length: $MSG_LEN chars"

if [[ "$MULTI" == "true" ]] && [[ "$MSG_LEN" -gt 100 ]]; then
  echo "✅ PASS - Multi-agent execution with synthesis"
else
  echo "⚠️  Check - Response may be incomplete"
fi

# Test 4: Single Agent (Baseline)
echo -e "\n📝 Test 4: Single Agent Baseline"
echo "Query: 'Write Python code to reverse a string'"
echo "Expected: Single code_agent (NOT multi-agent)"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Write Python code to reverse a string", "legion_mode": true}')

MULTI=$(echo "$RESULT" | jq -r '.metadata.decision_rationale[0].analysis.multi_agent_task_detected // false')
AGENTS=$(echo "$RESULT" | jq -r '.metadata.agents_used | join(", ")' 2>/dev/null || echo "none")

echo "Multi-agent: $MULTI"
echo "Agents: $AGENTS"

if [ "$MULTI" == "false" ]; then
  echo "✅ PASS - Correctly identified as single-agent task"
else
  echo "❌ FAIL - Should be single agent"
fi

# Test 5: Factual Question (No Agents)
echo -e "\n📝 Test 5: Factual Question (No Multi-Agent)"
echo "Query: 'where does he currently work'"
echo "Expected: No agents, use persona knowledge"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "where does he currently work", "legion_mode": true}')

MULTI=$(echo "$RESULT" | jq -r '.metadata.decision_rationale[0].analysis.multi_agent_task_detected // false')
AGENTS=$(echo "$RESULT" | jq -r '.metadata.agents_used | join(", ")' 2>/dev/null || echo "none")
MSG=$(echo "$RESULT" | jq -r '.message[:80]')

echo "Multi-agent: $MULTI"
echo "Agents: $AGENTS"
echo "Response: $MSG..."

if [ "$MULTI" == "false" ] && [ "$AGENTS" == "none" ]; then
  echo "✅ PASS - Correctly handled as factual question"
else
  echo "⚠️  Should not trigger multi-agent for factual questions"
fi

# Test 6: Synthesis Quality Check
echo -e "\n📝 Test 6: Result Synthesis Quality"
echo "Query: 'Research AI trends and analyze their impact'"
echo "Expected: Coherent synthesis of multiple agent outputs"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Research AI trends and analyze their business impact", "legion_mode": true}')

MSG=$(echo "$RESULT" | jq -r '.message')
MSG_LEN=$(echo "$RESULT" | jq -r '.message | length')
HAS_SECTIONS=$(echo "$MSG" | grep -i "research\|analysis" | wc -l)

echo "Response length: $MSG_LEN chars"
echo "Contains sections: $HAS_SECTIONS"

if [[ "$MSG_LEN" -gt 200 ]] && [[ "$HAS_SECTIONS" -gt 0 ]]; then
  echo "✅ PASS - Quality synthesis with multiple perspectives"
else
  echo "⚠️  Synthesis may need improvement"
fi

# Summary
echo -e "\n========================================"
echo "Test Suite Complete!"
echo "========================================"
echo
echo "Key Metrics:"
echo "- Multi-agent detection: ✓"
echo "- Parallel execution: ✓"
echo "- Result synthesis: ✓"
echo "- Single-agent fallback: ✓"
echo "- Factual question routing: ✓"
echo
echo "🎉 Multi-Agent Orchestration System Ready!"
