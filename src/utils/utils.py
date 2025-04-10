def sonnets_to_array():
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


# Get the array of sonnets
sonnet_array = sonnets_to_array()

# Display the resulting array; each element represents a sonnet block without the "Sonnet X" title
for sonnet in sonnet_array:
    print(sonnet)
    print("-" * 40)  # Separator for clarity in the output

print(len(sonnet_array))
