#!/bin/bash

# Legion Mode Audit Script
# Compares responses between legion_mode=false and legion_mode=true

BASE_URL="http://localhost:8080/api/v1/hermes/process_request"

echo "🔍 Legion Mode Audit - Comparing Responses"
echo "=" | tr ' ' '=' | head -c 70 && echo
echo

# Test queries
QUERIES=(
  "Hello, how are you?"
  "where does he currently work"
  "Research quantum computing"
  "Write Python code to sort a list"
  "Analyze sales trends for Q4 2024"
)

for i in "${!QUERIES[@]}"; do
  QUERY="${QUERIES[$i]}"
  echo -e "\n📝 Test $((i+1)): \"$QUERY\""
  echo "─" | tr ' ' '─' | head -c 70 && echo

  # Test with legion_mode=false
  echo -e "\n🔵 legion_mode=FALSE (Normal Mode)"
  RESULT_FALSE=$(curl -s -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -d "{\"request_text\": \"$QUERY\", \"legion_mode\": false}")

  echo "Response preview: $(echo "$RESULT_FALSE" | jq -r '.message[:100]' 2>/dev/null || echo "Error parsing")"
  echo "Response length: $(echo "$RESULT_FALSE" | jq -r '.message | length' 2>/dev/null || echo "N/A") chars"
  echo "Metadata: $(echo "$RESULT_FALSE" | jq -c '.metadata | {model, prompt_length, response_length}' 2>/dev/null || echo "N/A")"

  # Test with legion_mode=true
  echo -e "\n🟢 legion_mode=TRUE (LangGraph Mode)"
  RESULT_TRUE=$(curl -s -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -d "{\"request_text\": \"$QUERY\", \"legion_mode\": true}")

  echo "Response preview: $(echo "$RESULT_TRUE" | jq -r '.message[:100]' 2>/dev/null || echo "Error parsing")"
  echo "Response length: $(echo "$RESULT_TRUE" | jq -r '.message | length' 2>/dev/null || echo "N/A") chars"

  # Check for errors
  ERROR=$(echo "$RESULT_TRUE" | jq -r '.error // "none"')
  if [ "$ERROR" != "none" ]; then
    echo "❌ ERROR: $ERROR"
    echo "Details: $(echo "$RESULT_TRUE" | jq -r '.details.error // "N/A"')"
  else
    echo "✅ Success"
  fi

  # LangGraph specific metadata
  echo "LangGraph enabled: $(echo "$RESULT_TRUE" | jq -r '.metadata.langgraph_enabled // "N/A"')"
  echo "Agent needed: $(echo "$RESULT_TRUE" | jq -r '.metadata.decision_rationale[0].decisions.agent_needed // "N/A"')"
  echo "Task type: $(echo "$RESULT_TRUE" | jq -r '.metadata.decision_rationale[0].analysis.identified_task_type // "none"')"
  echo "Agents used: $(echo "$RESULT_TRUE" | jq -r '.metadata.agents_used // [] | join(", ")' || echo "none")"

  # Comparison
  echo -e "\n📊 Comparison:"
  LEN_FALSE=$(echo "$RESULT_FALSE" | jq -r '.message | length' 2>/dev/null || echo "0")
  LEN_TRUE=$(echo "$RESULT_TRUE" | jq -r '.message | length' 2>/dev/null || echo "0")

  if [ "$LEN_FALSE" -gt 0 ] && [ "$LEN_TRUE" -gt 0 ]; then
    DIFF=$((LEN_TRUE - LEN_FALSE))
    echo "Length difference: $DIFF chars"

    if [ "$ERROR" == "none" ]; then
      echo "✅ Both modes working"
    else
      echo "⚠️  LangGraph mode has errors"
    fi
  else
    echo "⚠️  Unable to compare (parsing error)"
  fi

  echo
  sleep 1  # Rate limiting
done

echo -e "\n" && echo "=" | tr ' ' '=' | head -c 70 && echo
echo "Audit Complete!"
echo "=" | tr ' ' '=' | head -c 70 && echo
