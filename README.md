# approach_explanation.md

## Approach Explanation
PDF Heading Detection (Round 1A)

We built this project to read PDF files and figure out which lines are headings like titles, subtitles, and section names. The goal was to extract the structure of the document, for example, what the main sections and sub-sections are,  just like a table of contents.

So why this idea was used?

PDF files don’t tell us directly which line is a heading. There’s no tag like "h1" or "h2" and "h3" as in HTML. So, the only way to guess if a line is a heading is by looking at how it appears, like if it’s in a bigger font, bold, or placed at the top of the page.

To do this smartly, a small trained machine learning model was used. This model looks at features like font size, boldness, and line position to decide:
1. Is this line a heading?
2. If yes, is it a main heading (H1), subheading (H2), or something else (H3)?

Next, how the system works?
1. The program reads all PDF files from the input folder.
2. It breaks down each page into lines and collects information like font size, boldness, and page number.
3. It removes repeated lines that appear on every page, like footers or headers.
4. It uses a trained ML model (stored in a `.pkl` file) to detect which lines are headings and what level they are.
5. It saves the result in a JSON file that shows the title of the document and the list of headings with their levels and page numbers.

How it is checked?

To check if the system is working correctly, another script compares the output with a correct answer file. It checks:
- How many headings were found correctly
- How many headings were missed
- If the heading levels (H1, H2, etc.) are also correct

It gives scores like precision, recall, F1-score, and hierarchy accuracy to show how good the results are. 

The final output, that is, what has been achieved?

- The system works completely offline and fast.
- It handles different types of PDFs and layouts.
- The output is clean and easy to understand.
- It meets all hackathon requirements (small size, fast speed, no internet).

This solution uses a mix of simple rules and machine learning to read documents the way a human would and build a table of contents automatically.