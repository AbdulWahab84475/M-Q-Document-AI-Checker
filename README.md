# M&Q AI Document Checker

A simple Streamlit MVP for checking labeled manufacturing characteristics between an engineering drawing and a Process Control/M&Q document.

## Core rule

Only **labeled characteristics** are compared.

For example:

- `L1 = 25 mm` on the drawing → check
- `L1 = 25 mm` in Process Control → check
- an unlabeled `50 mm` dimension on the drawing → **ignore**

Labels are **case-sensitive**:

- `L1` ≠ `l1`
- `A` ≠ `a`

Dimensions and tolerances are compared separately.

## MVP workflow

1. Upload PDF, image, or DOCX.
2. AI reads the document pages/images.
3. AI extracts labeled characteristics.
4. Python performs deterministic label/dimension/tolerance comparison.
5. Streamlit displays:
   - MATCH
   - DIMENSION MISMATCH
   - TOLERANCE MISMATCH
   - DIMENSION + TOLERANCE MISMATCH
   - MISSING
   - EXTRA / NOT FOUND ON DRAWING
6. User can rate the overall AI decision as Good or Bad.
7. Feedback is saved to `feedback.csv`.

## Recommended input

PDF is recommended for engineering drawings because it preserves page layout.

DOCX support in this MVP extracts DOCX text/tables and embedded images. For visually complex drawings, export the Word document to PDF before uploading.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit

1. Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Create a Streamlit app from the GitHub repository.
3. Select `app.py` as the main file.
4. Add the OpenAI API key through Streamlit secrets or enter it in the app sidebar.
5. Deploy.

## API key

For production deployment, prefer Streamlit Secrets rather than asking users to paste an API key.

Example secret:

```toml
OPENAI_API_KEY = "your-key"
```

The app also accepts `OPENAI_API_KEY` as an environment variable.

## Future improvements

- Better leader-line/label association
- More robust engineering-symbol normalization
- GD&T support
- Drawing view/section association
- Characteristic-level Good/Bad feedback
- Persistent database instead of CSV
- Evaluation dataset and regression testing
- Human-confirmed corrections used as training/evaluation data
