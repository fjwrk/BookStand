# BookStand

Small CLI to generate daily reading plans and Obsidian-compatible outputs for PDFs.

## Features

- Extracts text from a PDF and estimates reading time per page.
- Generates a daily reading plan targeting N minutes/day.
- Optional translation (local MarianMT or remote translation server via `--translate-url`).
- Exports per-page HTML viewer (left: embedded PDF, right: English above / Japanese below) with TTS controls.
- Produces Obsidian Tasks-style `reading_plan.md`, `md/translations.md`, `md/reading_segments.md`, `pages/pageN.html`, and `metadata.json`.
- Writes to a timestamped per-PDF output folder. Generation is performed in a staging folder and moved atomically to the final folder when complete to avoid incomplete outputs.

## Install

Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(If you don't have `requirements.txt`, install: `PyPDF2 langdetect transformers torch requests`.)

## Usage

Basic quick run (no translation, no TTS measurement):

```bash
python pdf_reading_plan.py /path/to/file.pdf --no-translate --no-measure --export-html
```

Use remote translation server (POST /translate) instead of local model:

```bash
python pdf_reading_plan.py /path/to/file.pdf --translate-url http://<host>:8000/translate --no-measure --export-html
```

Use local MarianMT model (default if no `--translate-url`):

```bash
python pdf_reading_plan.py /path/to/file.pdf --model-id staka/fugumt-en-ja --export-html
```

TTS speed options: `1.0`, `1.5`, `2.0` (use `--tts-speed 1.5`)

Output is written under the `--out-root` (defaults to iCloud Obsidian vault set in the script). Each run creates a timestamped folder.

## Notes

- By default the script performs generation in a `.staging` directory and atomically renames it to the final folder when complete.
- If using local MarianMT, the model will be downloaded and may require sufficient disk space and time.
- The `export_html` module is used to generate per-page HTML; ensure it is present in the same workspace.

## Version

Tool version: 0.2

## License

MIT
