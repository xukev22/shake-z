import random


def genz_sonnet_to_arr(file_path):
    try:
        # Read the entire file content
        with open(file_path, "r") as file:
            content = file.read()

        # Split content into paragraphs assuming paragraphs are separated by two newlines
        paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]

        sonnets = []
        for paragraph in paragraphs:
            # Split each paragraph into lines
            lines = paragraph.splitlines()

            # If the first line starts with "Sonnet", remove it
            if lines and lines[0].startswith("Sonnet"):
                cleaned_paragraph = "\n".join(lines[1:]).strip()
            else:
                cleaned_paragraph = paragraph

            sonnets.append(cleaned_paragraph)

        return sonnets
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def shake_sonnet_to_arr(file_path):
    try:
        # Read the entire file content
        with open(file_path, "r") as file:
            content = file.read()

        # Split content into paragraphs (each paragraph is assumed to be separated by two newlines)
        paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]

        sonnets = []
        for paragraph in paragraphs:
            # Split each paragraph into lines
            lines = paragraph.splitlines()

            # Remove the first line if it is solely a number (the sonnet number)
            if lines and lines[0].strip().isdigit():
                cleaned_paragraph = "\n".join(lines[1:]).strip()
            else:
                cleaned_paragraph = paragraph

            # Only add non-empty sonnet content
            if cleaned_paragraph:
                sonnets.append(cleaned_paragraph)

        return sonnets
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def train_test_split_data(shakespeare, genz, test_ratio=0.2, seed=None):
    """
    Splits the paired data of Shakespeare sonnets and their Gen Z translations
    into training and testing sets.

    Parameters:
        shakespeare (list): List of Shakespeare sonnets.
        genz (list): List of Gen Z translations corresponding to the sonnets.
        test_ratio (float): Fraction of data to allocate to the test set (default 0.2).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_data, test_data) where each element is a tuple (sonnet, translation).
    """
    if len(shakespeare) != len(genz):
        raise ValueError("Both arrays must have the same length.")

    # Pair the corresponding items from both lists
    combined = list(zip(shakespeare, genz))

    # Set seed if provided for reproducible shuffling
    if seed is not None:
        random.seed(seed)

    # Randomly shuffle the paired data
    random.shuffle(combined)

    # Determine the split index
    test_size = int(len(combined) * test_ratio)
    test_data = combined[:test_size]
    train_data = combined[test_size:]

    return train_data, test_data
