# Shakespeare
From: [Project Gutenberg](https://www.gutenberg.org/ebooks/100)

## Processing:
### Removed:
- Headers/Footers
- Capitalization
- TOC/Character lists
- Enter (Character)
- [...Actions...]
- Lines less than 3 tokens (o!, Ay!, ...)
- Special characters [\', \", \(, ,\)]

### o3 Query:
_"You are a Gen Z Slang Translator.  Your job is to take a line of Shakespearean English and render it into natural, punchy “Gen Z” style—complete with slang, and abbreviations where appropriate—while preserving the original meaning."_


# Sonnets
From: idk
## Processing:
```py
# Define the function to clean and format the sonnets
def clean_sonnets(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        content = infile.read()
        # Remove numeric headings, "Sonnet #" headers, and excess whitespace
        sonnets = re.split(r"(?:\n\s*\d+\s*\n|Sonnet\s*\d+\s*\n)", content.strip())

        # Process each sonnet to a single line and keep basic punctuation
        for sonnet in sonnets:
            if sonnet.strip():
                # Replace internal newlines and excess spaces with a single space
                single_line_sonnet = " ".join(sonnet.strip().split())
                # Remove special characters, but keep basic punctuation necessary for NLP
                single_line_sonnet = single_line_sonnet.replace("-", " ")
                single_line_sonnet = re.sub(
                    r"[^a-zA-Z0-9\s,.?!;:]", "", single_line_sonnet
                )
                outfile.write(single_line_sonnet.lower() + "\n")
```
### Removed:
- Sonnet #
- Capitalization
- Special characters [\', \", \(, ,\)]