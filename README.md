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

# Execution instructions

Overview
This document explains how to prepare your input data, build the Docker image, run your containerized solution, and verify the output for Round 1A — PDF Heading Detection and Outline Generation.

## 1. Prepare Input Directory

Place all PDF files you want to process into the input directory.

For example:
```console
text
input/
├── document1.pdf
├── document2.pdf
├── sample.pdf
└── ...
```

## 2. Output Directory
This will be used to store the resulting JSON outline files.

## 3. Docker Build
From your project root, build the Docker image with the following command:

```
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

Replace mysolutionname:somerandomidentifier with your preferred image name and tag.

The --platform linux/amd64 flag ensures compatibility with the judging environment.

This build process will install all dependencies and pre-download models as required offline.

## 4. Run the Docker Container
Once your Docker image is built, run the container with the following command:

```console

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```
Command explanation:

--rm: Automatically remove container after execution.

-v $(pwd)/input:/app/input: Mounts your local input directory as /app/input inside the container.

-v $(pwd)/output:/app/output: Mounts your local output directory as /app/output where your results will be saved.

--network none: Disables network access during execution to comply with hackathon constraints.

mysolutionname:somerandomidentifier: Your Docker image name and tag.

## 5. Expected Output
For every PDF file in the input folder (e.g., document1.pdf), your program will generate a corresponding JSON file document1.json in the output folder.

Additionally, a consolidated output.json file may be created if specified by your solution.

Each JSON output file will follow the format:

``` json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Heading Text", "page": 1 },
    { "level": "H2", "text": "Subheading Text", "page": 2 }
  ]
}
```
## 6. Verification
After running the container:

List the contents of the output directory:

```console
ls -l output/
```

## 7. Important Notes
File Naming: Ensure PDF filenames do not contain problematic characters. The output JSON files will be named exactly as corresponding PDFs, but with .json extension.

Performance: Your solution must process a 50-page PDF within 10 seconds on an 8 CPU / 16GB RAM system (this is a runtime constraint to consider during development).

Offline Compliance: The Docker image must run offline (--network none), and all models and dependencies are pre-fetched and bundled during the Docker build.

Architecture: Ensure the solution runs on linux/amd64 CPU architecture, no GPU dependencies.

## 8. Example Full Workflow
bash
Step 1: Prepare your directories and place PDFs
```console
cp path_to_your_pdfs/*.pdf input/
```

Step 2: Build Docker image
```console
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

Step 3: Run the container
```console
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

Step 4: Inspect output
```console
ls -l output/
cat output/your_pdf_filename.json
```

## 9. Troubleshooting
If you encounter No PDF files found in input directory! error, ensure that your input directory contains valid PDF files and that volume mounting is correct.

If the output files are not generated, verify that your main.py and Dockerfile paths are configured properly.

To debug inside the container, you can run an interactive shell:

```console
docker run --rm -it -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier /bin/bash
```
Then manually run your processing script inside the container.

