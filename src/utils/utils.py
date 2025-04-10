def genz_sonnet_to_arr():
    file_path = "data/processed/shake_sonnets_genz.txt"
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


def shake_sonnet_to_arr():
    file_path = "data/raw/shake_sonnets.txt"  # Adjust this path if needed
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


# Get the array of sonnets from the shake sonnets file
sonnet_array = shake_sonnet_to_arr()

# Display the resulting array; each element represents a sonnet block without the number at the beginning
for i in range(len(sonnet_array)):
    if i < 160:
        print(sonnet_array[i])
        print("-" * 40)  # Separator for clarity in the output

print(f"Total sonnets: {len(sonnet_array)}")


# # Get the array of sonnets
# sonnet_array = genz_sonnet_to_arr()

# # Display the resulting array; each element represents a sonnet block without the "Sonnet X" title
# for sonnet in sonnet_array:
#     print(sonnet)
#     print("-" * 40)  # Separator for clarity in the output

# print(len(sonnet_array))
