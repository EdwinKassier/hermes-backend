"""Standardized output formatting instructions for Legion responses."""

OUTPUT_FORMAT_GUIDELINES = """
**CRITICAL OUTPUT FORMATTING REQUIREMENTS**:

Your response MUST use proper markdown formatting that a frontend can parse:

1. **Code Blocks**: ALL code MUST be wrapped in fenced code blocks with language identifiers:
   - Use triple backticks with language: ```python, ```javascript, ```bash, etc.
   - NEVER output raw code without proper fencing
   - Example:
     ```python
     def example():
         return "Hello"
     ```

2. **Section Spacing**:
   - Separate ALL major sections with double newlines (\\n\\n)
   - Use horizontal rules (---) surrounded by blank lines to separate distinct topics
   - Ensure paragraphs are separated by blank lines

3. **Section Headers**: Use ## or ### for major sections, always preceded and followed by a blank line

4. **Lists**: Use - or * for unordered lists, 1. 2. 3. for ordered lists, with blank lines before and after the list

5. **Emphasis**: Use **bold** for important terms, `backticks` for inline code/filenames

6. **Block Quotes**: Use > for important notes or warnings

7. **Separators**: Use --- for thematic breaks between major sections

8. **General Text Structure**:
   - Break long text into short, readable paragraphs
   - Avoid dense blocks of text
   - Use bullet points for complex lists/steps
"""

CODE_SPECIFIC_GUIDELINES = """
**CODE OUTPUT REQUIREMENTS**:
- ALWAYS wrap code in fenced code blocks with the correct language identifier
- Use ```python for Python, ```javascript for JavaScript, etc.
- Include the language identifier on the opening fence
- Ensure closing ``` is on its own line
- Separate code blocks from explanatory text with blank lines
"""
