#!/bin/bash

echo "🧪 Testing Orchestration Rationale Metadata"
echo "============================================"
echo ""

# Test 1: Simple question
echo "📝 Test 1: Simple Question"
echo "Query: 'Hello'"
echo "Expected: orchestration_rationale with direct response explanation"
echo "──────────────────────────────────────"
curl -s -X POST "http://localhost:8080/api/v1/hermes/process_request" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Hello", "legion_mode": true}' | jq '{
  has_rationale: (.metadata.orchestration_rationale != null),
  structure: .metadata.orchestration_rationale.orchestration_structure,
  execution_mode: .metadata.orchestration_rationale.execution_mode,
  agents: .metadata.orchestration_rationale.agents
}'
echo ""

# Test 2: Code generation
echo "📝 Test 2: Code Generation Task"
echo "Query: 'Write Python code to reverse a string'"
echo "Expected: Single-agent with code agent and tools"
echo "──────────────────────────────────────"
curl -s -X POST "http://localhost:8080/api/v1/hermes/process_request" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Write Python code to reverse a string", "legion_mode": true}' | jq '{
  structure: .metadata.orchestration_rationale.orchestration_structure,
  execution_mode: .metadata.orchestration_rationale.execution_mode,
  agents: .metadata.orchestration_rationale.agents,
  toolsets: .metadata.orchestration_rationale.toolsets
}'
echo ""

# Test 3: Multi-agent task
echo "📝 Test 3: Multi-Agent Task"
echo "Query: 'Research quantum computing and analyze applications'"
echo "Expected: Parallel execution with multiple agents"
echo "──────────────────────────────────────"
curl -s -X POST "http://localhost:8080/api/v1/hermes/process_request" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "Research quantum computing and analyze applications", "legion_mode": true}' | jq '{
  structure: .metadata.orchestration_rationale.orchestration_structure,
  execution_mode: .metadata.orchestration_rationale.execution_mode,
  agents: .metadata.orchestration_rationale.agents,
  performance: .metadata.orchestration_rationale.performance
}'
echo ""

# Test 4: Verify no old fields
echo "📝 Test 4: Verify Old Fields Removed"
echo "Checking for deprecated metadata fields..."
echo "──────────────────────────────────────"
curl -s -X POST "http://localhost:8080/api/v1/hermes/process_request" \
  -H "Content-Type: application/json" \
  -d '{"request_text": "test", "legion_mode": true}' | jq '{
  has_decision_explanation: (.metadata.decision_explanation != null),
  has_decision_rationale: (.metadata.decision_rationale != null),
  has_orchestration_reasoning: (.metadata.orchestration_reasoning != null),
  has_orchestration_rationale: (.metadata.orchestration_rationale != null),
  metadata_keys: (.metadata | keys)
}'
echo ""

echo "============================================"
echo "✅ Orchestration Rationale Tests Complete!"
echo ""
echo "Expected metadata structure:"
echo "  • orchestration_rationale.orchestration_structure"
echo "  • orchestration_rationale.agents"
echo "  • orchestration_rationale.toolsets"
echo "  • orchestration_rationale.execution_mode"
echo "  • orchestration_rationale.performance (if multi-agent)"
