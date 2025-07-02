import os
import re
import argparse

def remove_docstrings_from_file(file_path):
    """
    Removes docstrings (triple-quoted strings at the start of modules,
    functions, classes, or methods) from a Python file.
    It handles both single-line and multi-line docstrings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex to find docstrings:
        # r'("""|\'\'\')' - Matches either triple double quotes or triple single quotes
        # r'(?:(?!\1).)*?' - Non-greedy match for any characters that are not the closing quote
        # r'\1' - Matches the same opening quote (e.g., if it started with """, it ends with """)
        # re.DOTALL - Allows '.' to match newline characters
        # re.MULTILINE - Allows '^' and '$' to match start/end of lines (not strictly needed here but good practice)

        # Pattern to match docstrings at the start of module/function/class/method
        # This pattern specifically looks for triple quotes that are the first non-whitespace
        # content on a line or after a definition.
        # It's challenging to perfectly remove ONLY docstrings without affecting
        # regular triple-quoted strings within code. This attempts to be reasonably safe.
        # A more robust solution might involve parsing the AST.

        # This regex targets docstrings that appear immediately after a 'def', 'class'
        # or at the very beginning of the file (module docstring).
        # It specifically looks for indentation followed by triple quotes.
        # It's a heuristic and might not catch all edge cases or might remove
        # intended string literals if they look like docstrings.

        # Simplified approach: find all triple-quoted strings and then apply a heuristic
        # to decide if it's a docstring. For this script, we'll remove any
        # triple-quoted string that appears at the start of a logical block.

        # This regex specifically targets docstrings.
        # It looks for optional whitespace, then 'def' or 'class' or start of file,
        # then optional whitespace, then triple quotes.
        # This is still a heuristic. For truly robust docstring removal,
        # parsing the AST (Abstract Syntax Tree) is recommended, but that's
        # beyond a simple regex script.

        # A common regex for docstrings, assuming they are at the start of a line
        # after optional whitespace, or immediately after def/class.
        # This version attempts to capture docstrings by looking for triple quotes
        # that are not preceded by an assignment or a function call.
        # This is still a heuristic.
        docstring_pattern = re.compile(r'^\s*(?:def|class)?\s*"""(.*?)"""', re.DOTALL | re.MULTILINE)
        docstring_pattern_single = re.compile(r'^\s*(?:def|class)?\s*\'\'\'(.*)\'\'\'', re.DOTALL | re.MULTILINE)


        # Remove module-level docstring (at the very beginning of the file)
        # Check for triple quotes at the very beginning of the file, possibly with leading whitespace
        content = re.sub(r'^\s*"""[^"]*"""', '', content, count=1, flags=re.DOTALL)
        content = re.sub(r"^\s*'''[^']*'''", '', content, count=1, flags=re.DOTALL)


        # Remove function/class/method docstrings
        # This regex looks for 'def' or 'class' followed by a name and parentheses/colon,
        # then optional whitespace, then a docstring.
        # It's still a heuristic, but a common one.
        # It tries to avoid removing multi-line strings that are not docstrings.
        # The `re.sub` will replace the first match in each block.
        def_class_docstring_pattern = re.compile(
            r'^(?P<indent>\s*)(?:def|class)\s+[\w_]+\s*\(.*?\):\s*\n' # Matches def/class line
            r'(?P=indent)\s*("""|\'\'\')(?:(?!\1).)*?\1', # Matches docstring indented at the same level
            re.DOTALL | re.MULTILINE
        )

        # Replace docstrings found by the pattern
        def replace_docstring_match(match):
            # Keep the 'def' or 'class' line, but remove the docstring part
            indent = match.group('indent')
            return match.group(0).replace(match.group(2), '') # Remove the matched docstring string

        # This is a simpler, more aggressive approach: remove any triple-quoted string
        # that appears at the beginning of a line (after optional whitespace).
        # This might remove some non-docstring triple-quoted strings if they are
        # formatted similarly.
        new_content = re.sub(r'^\s*("""|\'\'\')(?:(?!\1).)*?\1', '', content, flags=re.DOTALL | re.MULTILINE)


        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove docstrings from Python files.")
    parser.add_argument("path", type=str, default=".", nargs='?',
                        help="The root directory or specific Python file to process (default: current directory).")

    args = parser.parse_args()

    target_path = args.path

    if os.path.isfile(target_path) and target_path.endswith(".py"):
        print(f"Processing single file: {target_path}")
        if remove_docstrings_from_file(target_path):
            print(f"Docstrings removed from {target_path}")
        else:
            print(f"Failed to remove docstrings from {target_path}")
    elif os.path.isdir(target_path):
        print(f"Processing directory: {target_path}")
        processed_count = 0
        for root, _, files in os.walk(target_path):
            for file_name in files:
                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    print(f"  Processing {file_path}...")
                    if remove_docstrings_from_file(file_path):
                        processed_count += 1
        print(f"\nFinished processing. Docstrings removed from {processed_count} Python files.")
    else:
        print(f"Error: '{target_path}' is not a valid Python file or directory.")

    print("\nScript finished.")
