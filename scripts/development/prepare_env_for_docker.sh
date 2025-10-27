#!/bin/bash
# Script to prepare .env file for Docker by removing quotes

INPUT_FILE="${1:-.env}"
OUTPUT_FILE="${2:-.env.docker}"

# Remove .env.docker if it exists
rm -f "$OUTPUT_FILE"

# Read the .env file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Split on first = sign
    if [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"

        # Remove surrounding quotes (both single and double)
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

        # Write to output file
        echo "${key}=${value}" >> "$OUTPUT_FILE"
    fi
done < "$INPUT_FILE"

echo "Created $OUTPUT_FILE with unquoted values"
