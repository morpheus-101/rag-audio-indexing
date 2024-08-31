import subprocess
import sys


def run_command(command, display_output=False):
    try:
        if display_output:
            # Run the command and display output in real-time
            subprocess.run(command, check=True)
        else:
            # Run the command and capture output
            result = subprocess.run(command, check=True, capture_output=True,
                                    text=True)
            print(f"Successfully ran {' '.join(command)}")
            if result.stdout:
                print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(command)}:")
        if display_output:
            # Error output already displayed in real-time
            pass
        else:
            print(e.stderr)


def format_and_lint(file_path):
    commands = [
        # Initial autopep8 pass
        ["autopep8", "--in-place", "--aggressive", "--aggressive", file_path],
        # Additional autopep8 pass to fix long lines
        ["autopep8", "--in-place", "--max-line-length", "79", file_path],
        # Black for consistent styling
        ["black", "--line-length", "79", file_path],
        # Flake8 for final linting
        ["flake8", file_path]
    ]

    for command in commands:
        # Display output for flake8, but not for autopep8 and black
        display_output = (command[0] == "flake8")
        run_command(command, display_output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_file.py <path_to_python_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    format_and_lint(file_path)
