import os
import argparse

def rename_string_in_files(directory, old_string, new_string):
    """
    Recursively finds and replaces an old string with a new string in all
    HTML, Python, JSON, and YAML files within the specified directory.
    """
    renamed_files_count = 0
    for root, _, files in os.walk(directory):
        for file_name in files:
            # Process .html, .py, .json, and .yaml files
            if file_name.endswith((".html", ".py", ".json", ".yaml")):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if old_string in content:
                        new_content = content.replace(old_string, new_string)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Renamed '{old_string}' to '{new_string}' in: {file_path}")
                        renamed_files_count += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return renamed_files_count

if __name__ == "__main__":
    print("Welcome to YourNewModel! Please select a name!")

    parser = argparse.ArgumentParser(description="Rename a string in HTML, Python, JSON, and YAML files across your project.")
    parser.add_argument("new_model_name", type=str,
                        help="The new name for your model (e.g., 'MyAwesomeLLM').")
    parser.add_argument("--path", type=str, default=".",
                        help="The root directory of your project (default: current directory).")

    args = parser.parse_args()

    # The string to be replaced (case-sensitive)
    old_model_name = "YourNewModel"
    new_model_name = args.new_model_name
    project_directory = args.path

    print(f"\nAttempting to rename '{old_model_name}' to '{new_model_name}' in files under '{project_directory}'...")

    total_renamed = rename_string_in_files(project_directory, old_model_name, new_model_name)

    if total_renamed > 0:
        print(f"\nSuccessfully renamed '{old_model_name}' to '{new_model_name}' in {total_renamed} files.")
    else:
        print(f"\nNo occurrences of '{old_model_name}' found in specified file types, or no such files processed.")

    print("\nScript finished.")
