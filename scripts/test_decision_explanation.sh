#!/bin/bash

# Test decision rationale readability

BASE_URL="http://localhost:8080/api/v1/hermes/process_request"

echo "🧪 Testing Decision Rationale Readability"
echo "=========================================="
echo

# Test 1: Simple factual question
echo "📝 Test 1: Factual Question"
echo "Query: 'where does he work'"
echo "Expected: Clear explanation of direct answer"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "where does he work", "legion_mode": true}')

echo "$RESULT" | jq -r '.metadata.decision_explanation.summary // "No summary"'
echo
echo "Quick Facts:"
echo "$RESULT" | jq -r '.metadata.decision_explanation.quick_facts // {}' | jq -r 'to_entries[] | "  • \(.key): \(.value)"'
echo

# Test 2: Agent-based task
echo "📝 Test 2: Code Generation Task"
echo "Query: 'write Python code to sort a list'"
echo "Expected: Explanation of code agent activation"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "write Python code to sort a list", "legion_mode": true}')

echo "$RESULT" | jq -r '.metadata.decision_explanation.summary // "No summary"'
echo
echo "Quick Facts:"
echo "$RESULT" | jq -r '.metadata.decision_explanation.quick_facts // {}' | jq -r 'to_entries[] | "  • \(.key): \(.value)"'
echo
echo "Steps:"
echo "$RESULT" | jq -r '.metadata.decision_explanation.step_by_step[]? // empty'
echo

# Test 3: Multi-agent task
echo "📝 Test 3: Multi-Agent Task"
echo "Query: 'Research AI and analyze trends'"
echo "Expected: Multi-agent coordination explanation"
echo "──────────────────────────────────────"

RESULT=$(curl -s -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Research AI and analyze trends", "legion_mode": true}')

echo "$RESULT" | jq -r '.metadata.decision_explanation.summary // "No summary"'
echo
echo "Quick Facts:"
echo "$RESULT" | jq -r '.metadata.decision_explanation.quick_facts // {}' | jq -r 'to_entries[] | "  • \(.key): \(.value)"'
echo
echo "Steps:"
echo "$RESULT" | jq -r '.metadata.decision_explanation.step_by_step[]? // empty'
echo

echo "=========================================="
echo "✅ Decision Explanation Tests Complete!"
echo
echo "The 'decision_explanation' field now includes:"
echo "  • summary: One-sentence explanation"
echo "  • quick_facts: Key information at a glance"
echo "  • step_by_step: What happened in order"
echo "  • technical_details: Full raw data for debugging"
