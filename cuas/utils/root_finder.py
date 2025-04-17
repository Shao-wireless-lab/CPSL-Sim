from pathlib import Path

def find_root(target_name="CPSL-Sim"):
    """
    Recursively searches upward from this file to find the CPSL-Sim project root.

    Args:
        target_name (str): Folder name to search for (default: "CPSL-Sim").

    Returns:
        Path: Absolute Path to the CPSL-Sim root directory.

    Raises:
        FileNotFoundError: If the folder is not found.
    """
    try:
        current = Path(__file__).resolve()
    except NameError:
        # In Jupyter or REPL
        current = Path.cwd()

    while current != current.parent:
        if current.name == target_name:
            return current
        current = current.parent

    raise FileNotFoundError(f"❌ Could not find '{target_name}' in any parent directories.")
