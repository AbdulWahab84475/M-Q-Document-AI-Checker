import base64
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from docx import Document
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "M&Q AI Document Checker"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FEEDBACK_FILE = Path("feedback.csv")

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def pdf_to_images(data: bytes, max_pages: int = 12):
    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for i in range(min(len(doc), max_pages)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append((i + 1, img))
    return images


def docx_to_images_and_text(data: bytes):
    """
    DOCX support in this MVP:
    - extracts text/tables
    - extracts embedded images
    A DOCX does not have a reliable page renderer in a minimal Streamlit
    environment, so visual page layout is best handled by PDF/image input.
    """
    doc = Document(io.BytesIO(data))
    text_parts = []

    for p in doc.paragraphs:
        if p.text.strip():
            text_parts.append(p.text.strip())

    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        text_parts.append("\n".join(rows))

    images = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            blob = rel.target_part.blob
            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
                images.append((len(images) + 1, img))
            except Exception:
                pass

    return images, "\n\n".join(text_parts)


def load_upload(uploaded_file):
    suffix = uploaded_file.name.lower().split(".")[-1]
    data = uploaded_file.getvalue()

    if suffix == "pdf":
        return pdf_to_images(data), ""
    if suffix in {"png", "jpg", "jpeg"}:
        return [(1, Image.open(io.BytesIO(data)).convert("RGB"))], ""
    if suffix == "docx":
        return docx_to_images_and_text(data)

    raise ValueError("Unsupported file type.")


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("AI did not return JSON.")
    return json.loads(match.group(0))


def normalize_label(label):
    if label is None:
        return ""
    # Labels are case-sensitive by requirement. Do NOT lowercase them.
    return re.sub(r"\s+", "", str(label).strip())


def normalize_dimension(value):
    if value is None:
        return ""
    s = str(value).strip()
    s = s.replace(" ", "")
    s = s.replace("×", "x")
    s = s.replace("Ø", "DIA")
    return s


def compare_characteristics(drawing_items, control_items):
    drawing = {normalize_label(x.get("label")): x for x in drawing_items
               if normalize_label(x.get("label"))}
    control = {normalize_label(x.get("label")): x for x in control_items
               if normalize_label(x.get("label"))}

    results = []

    # Only labeled drawing characteristics are in scope.
    for label, d in drawing.items():
        if label not in control:
            results.append({
                "label": label,
                "drawing": d,
                "process_control": None,
                "result": "MISSING",
                "reason": "Labeled characteristic exists on drawing but was not found in Process Control."
            })
            continue

        c = control[label]

        d_dim = normalize_dimension(d.get("dimension"))
        c_dim = normalize_dimension(c.get("dimension"))
        d_tol = normalize_dimension(d.get("tolerance"))
        c_tol = normalize_dimension(c.get("tolerance"))

        dimension_match = d_dim == c_dim
        tolerance_match = d_tol == c_tol

        if dimension_match and tolerance_match:
            result = "MATCH"
            reason = "Label, dimension and tolerance match."
        elif not dimension_match and not tolerance_match:
            result = "DIMENSION + TOLERANCE MISMATCH"
            reason = "Dimension and tolerance differ."
        elif not dimension_match:
            result = "DIMENSION MISMATCH"
            reason = "Dimension differs."
        else:
            result = "TOLERANCE MISMATCH"
            reason = "Tolerance differs."

        results.append({
            "label": label,
            "drawing": d,
            "process_control": c,
            "result": result,
            "reason": reason
        })

    # Control-only labels are also useful to flag.
    for label, c in control.items():
        if label not in drawing:
            results.append({
                "label": label,
                "drawing": None,
                "process_control": c,
                "result": "EXTRA / NOT FOUND ON DRAWING",
                "reason": "Process Control characteristic could not be found among labeled drawing characteristics."
            })

    return results


def build_prompt():
    return r"""
You are an expert manufacturing quality document analyst with experience reading
engineering drawings, machining dimensions, tolerances and process-control documents.

You will receive one or more page images from an M&Q (Manufacturing & Quality /
Process Control) document.

Your task is EXTRACTION, not final quality approval.

Important rules:
1. A drawing may contain many dimensions that have NO characteristic label.
   IGNORE all unlabeled dimensions. They are NOT missing characteristics.
2. Only dimensions/features explicitly associated with a characteristic label such
   as L1, L2, L3, D1, A1, etc. are in scope.
3. Labels are CASE-SENSITIVE. L1 and l1 are DIFFERENT labels. A and a are
   DIFFERENT labels.
4. Preserve the label exactly as shown.
5. Preserve dimension and tolerance text faithfully.
6. Do not invent a tolerance when none is visible.
7. If the text is unclear, return the best reading but lower the confidence.
8. Try to associate a labeled callout with its dimension using the visual layout,
   leader line, nearby text, and drawing context.
9. A labeled characteristic can be on either a drawing or a process-control table.
10. Do not treat ordinary page numbers, table row numbers, revision numbers,
    item numbers, or unrelated notes as characteristic labels unless the visual
    context clearly indicates that they are measurement/control labels.

Return ONLY valid JSON in this structure:

{
  "pages": [
    {
      "page_number": 1,
      "page_type": "drawing | process_control | unknown",
      "confidence": 0.0,
      "characteristics": [
        {
          "label": "L1",
          "dimension": "25 mm",
          "tolerance": "±0.05 mm",
          "description": "",
          "view": "",
          "confidence": 0.0
        }
      ]
    }
  ]
}
"""


def call_openai(images, extra_text="", api_key=None, model=DEFAULT_MODEL):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    content = [{"type": "input_text", "text": build_prompt()}]
    if extra_text.strip():
        content.append({
            "type": "input_text",
            "text": "Additional DOCX extracted text:\n" + extra_text[:30000]
        })

    for page_no, img in images:
        content.append({"type": "input_text", "text": f"Page/Image {page_no}"})
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(img),
            "detail": "high"
        })

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        temperature=0
    )

    return extract_json(response.output_text)


def split_results(results):
    match = [r for r in results if r["result"] == "MATCH"]
    errors = [r for r in results if r["result"] != "MATCH" and r["result"] != "REVIEW"]
    return match, errors


def result_table(results):
    rows = []
    for r in results:
        d = r.get("drawing") or {}
        c = r.get("process_control") or {}
        rows.append({
            "Label": r["label"],
            "Drawing Dimension": d.get("dimension", ""),
            "Drawing Tolerance": d.get("tolerance", ""),
            "Process Control Dimension": c.get("dimension", ""),
            "Process Control Tolerance": c.get("tolerance", ""),
            "Result": r["result"],
            "Reason": r["reason"],
            "AI Confidence": d.get("confidence", c.get("confidence", ""))
        })
    return pd.DataFrame(rows)


def save_feedback(document_name, decision, comment):
    row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "document": document_name,
        "ai_decision": st.session_state.get("overall_decision", ""),
        "user_feedback": decision,
        "comment": comment
    }])

    if FEEDBACK_FILE.exists():
        row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(FEEDBACK_FILE, index=False)


# -----------------------------
# UI
# -----------------------------
st.title("🔎 M&Q AI Document Checker")
st.caption(
    "Compare labeled drawing characteristics against Process Control characteristics. "
    "Unlabeled drawing dimensions are intentionally ignored."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    model = st.text_input("AI model", value=DEFAULT_MODEL)

    st.divider()
    st.markdown("### Comparison rules")
    st.write("• Labels are case-sensitive.")
    st.write("• L1 ≠ l1.")
    st.write("• Unlabeled drawing dimensions are ignored.")
    st.write("• Dimension and tolerance are compared separately.")

uploaded = st.file_uploader(
    "Upload M&Q document",
    type=["pdf", "png", "jpg", "jpeg", "docx"],
    help="PDF is recommended for drawings because it preserves page layout."
)

if uploaded:
    try:
        images, docx_text = load_upload(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.success(f"Loaded {len(images)} page/image(s).")

    with st.expander("Document preview", expanded=False):
        cols = st.columns(min(3, len(images)))
        for idx, (page_no, img) in enumerate(images):
            with cols[idx % len(cols)]:
                st.image(img, caption=f"Page/Image {page_no}", use_container_width=True)

    if docx_text:
        with st.expander("Extracted DOCX text", expanded=False):
            st.text(docx_text[:10000])

    if st.button("🚀 Run M&Q AI Check", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter an OpenAI API key in the sidebar.")
            st.stop()

        with st.spinner("Reading labels, dimensions and tolerances..."):
            try:
                extracted = call_openai(
                    images=images,
                    extra_text=docx_text,
                    api_key=api_key,
                    model=model
                )
                st.session_state["extracted"] = extracted
                st.session_state["document_name"] = uploaded.name

                pages = extracted.get("pages", [])
                drawing_items = []
                control_items = []

                for page in pages:
                    page_type = page.get("page_type", "unknown")
                    items = page.get("characteristics", [])
                    if page_type == "drawing":
                        drawing_items.extend(items)
                    elif page_type == "process_control":
                        control_items.extend(items)

                # Fallback: if AI couldn't classify pages, infer from content.
                if not drawing_items or not control_items:
                    all_items = []
                    for page in pages:
                        all_items.extend(page.get("characteristics", []))

                    st.session_state["classification_warning"] = True

                results = compare_characteristics(drawing_items, control_items)
                st.session_state["results"] = results

                has_errors = any(r["result"] != "MATCH" for r in results)
                st.session_state["overall_decision"] = "FAIL" if has_errors else "PASS"

            except Exception as e:
                st.error(f"AI check failed: {e}")
                st.stop()

if "results" in st.session_state:
    results = st.session_state["results"]

    st.divider()
    st.subheader("Verification Result")

    matches = sum(r["result"] == "MATCH" for r in results)
    errors = sum(r["result"] != "MATCH" for r in results)

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Matches", matches)
    c2.metric("🔴 Issues", errors)
    c3.metric("Characteristics Checked", len(results))

    if errors == 0:
        st.success("PASS — all detected labeled characteristics match.")
    else:
        st.error("FAIL — one or more labeled characteristics require attention.")

    st.dataframe(result_table(results), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Error Details")

    error_results = [r for r in results if r["result"] != "MATCH"]

    if not error_results:
        st.success("No mismatches detected.")
    else:
        for r in error_results:
            d = r.get("drawing") or {}
            c = r.get("process_control") or {}

            with st.expander(f"🔴 {r['label']} — {r['result']}"):
                st.write(f"**Reason:** {r['reason']}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Drawing**")
                    st.write(f"Dimension: `{d.get('dimension', '—')}`")
                    st.write(f"Tolerance: `{d.get('tolerance', '—')}`")
                    st.write(f"Confidence: `{d.get('confidence', '—')}`")

                with col2:
                    st.markdown("**Process Control**")
                    st.write(f"Dimension: `{c.get('dimension', '—')}`")
                    st.write(f"Tolerance: `{c.get('tolerance', '—')}`")
                    st.write(f"Confidence: `{c.get('confidence', '—')}`")

    st.divider()
    st.subheader("🤖 AI Decision Feedback")
    st.write("Was the overall AI decision correct?")

    b1, b2 = st.columns(2)

    with b1:
        if st.button("👍 Good", use_container_width=True):
            st.session_state["feedback_selected"] = "Good"

    with b2:
        if st.button("👎 Bad", use_container_width=True):
            st.session_state["feedback_selected"] = "Bad"

    if st.session_state.get("feedback_selected"):
        st.info(f"Feedback selected: {st.session_state['feedback_selected']}")
        comment = st.text_area(
            "Optional comment",
            placeholder="What did the AI get right or wrong?"
        )

        if st.button("Save AI Feedback", type="primary"):
            try:
                save_feedback(
                    st.session_state.get("document_name", "unknown"),
                    st.session_state["feedback_selected"],
                    comment
                )
                st.success("AI feedback saved.")
            except Exception as e:
                st.error(f"Could not save feedback: {e}")

    # Simple performance view
    st.divider()
    st.subheader("📈 AI Performance")

    if FEEDBACK_FILE.exists():
        feedback = pd.read_csv(FEEDBACK_FILE)
        total = len(feedback)
        good = int((feedback["user_feedback"] == "Good").sum())
        bad = int((feedback["user_feedback"] == "Bad").sum())
        approval = (good / total * 100) if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Reviewed Decisions", total)
        m2.metric("Good", good)
        m3.metric("AI Approval Rate", f"{approval:.1f}%")

        st.dataframe(feedback.tail(20), use_container_width=True, hide_index=True)
    else:
        st.caption("No AI feedback has been recorded yet.")

st.divider()
st.caption(
    "MVP note: This tool is an engineering-document assistant, not a substitute "
    "for final manufacturing/quality approval."
)
