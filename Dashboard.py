"""
HR Contract Dashboard
----------------------
Reads the master Excel workbook produced by extract_to_excel.py and displays
it as an interactive dashboard: KPIs, review-flagged contracts, contracts
expiring soon, and the full searchable table.

The dashboard also includes an "Upload Contract PDFs" section at the top - a
front-end replacement for the extract_to_excel.py command line: upload new
contract PDFs, run them through the local LLM pipeline, review the extracted
record, and append it to the workbook without leaving the browser.

PROCESSING LIFECYCLE
    Each extraction run happens in its own OS subprocess (see
    pipeline_worker.py), not a thread. A Python thread can only be asked to
    stop cooperatively; it cannot be forced to abandon a blocking LLM call,
    OCR pass, or PDF parse mid-flight. A subprocess can - clicking Stop calls
    process.terminate() (escalating to kill() if needed), which halts
    whatever that process is doing immediately, at the OS level.

    The status panel is an st.fragment(run_every=...) block, so the
    once-a-second poll for progress only re-renders that panel, not the
    whole page - the rest of the dashboard (filters, charts, tables) stays
    completely still while extraction runs. A handful of one-time full-page
    reruns still happen at state *transitions* (start / stop / done / error)
    so the upload controls and preview panel outside the fragment update too
    - that's a deliberate, occasional refresh, not continuous blinking.

    Requires Streamlit >= 1.37 for st.fragment(run_every=...) and
    st.rerun(scope=...).

USAGE:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --file contracts_master.xlsx
"""

import argparse
import importlib.util
import multiprocessing
import os
import queue as pyqueue
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# Make the folder this script lives in importable, regardless of the working
# directory Streamlit was launched from - this is what makes
# `import pipeline_worker` below reliable.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import pipeline_worker
except ModuleNotFoundError:
    st.set_page_config(page_title="Dashboard", layout="wide", page_icon="📄")
    st.error(
        "Could not find **pipeline_worker.py**.\n\n"
        f"It needs to sit in the same folder as this dashboard script:\n\n"
        f"`{_THIS_DIR}`\n\n"
        "Save `pipeline_worker.py` there (next to Dashboard.py and "
        "extract_to_excel.py) and reload the page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard", layout="wide", page_icon="📄")

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="contracts_master.xlsx")
# Streamlit passes its own args too, so ignore anything we don't recognize
args, _ = parser.parse_known_args()

EXCEL_PATH = args.file

START_COL = "Term of Agreement - Start Date"
END_COL = "Term of Agreement - End Date"

DOC_TYPE_OPTIONS = ["MSA", "NDA", "Other"]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Contracts")
    df.columns = [c.strip() for c in df.columns]
    for date_col in [START_COL, END_COL]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # "Needs Review" is written to Excel as the STRING "Y"/"N" (see
    # append_row_to_excel in extract_to_excel.py, which converts the
    # Python bool to "Y"/"N" before saving) - not a real boolean. Normalize
    # it here to an actual bool column so every filter/KPI/chart below can
    # just compare against True/False consistently, regardless of whether
    # a given cell happens to hold "Y", "N", True, False, or is blank.
    if "Needs Review" in df.columns:
        df["Needs Review"] = (
            df["Needs Review"].astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])
        )
    return df


def guess_doc_type(filename: str) -> str:
    """Heuristically guess whether a PDF is an MSA, NDA, or other agreement."""
    name = filename.lower()
    if any(key in name for key in ("msa", "master service", "service agreement", "consulting service")):
        return "MSA"
    if any(key in name for key in ("nda", "non-disclosure", "confidentiality")):
        return "NDA"
    return "Other"


def _find_pipeline_script_path() -> Path:
    """Locate the extraction pipeline script next to Dashboard.py."""

    # Your actual file is: Extract_to_excel.PY
    candidates = [
        p for p in _THIS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".py"
        and "extract" in p.stem.lower()
        and "excel" in p.stem.lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find the extraction pipeline script in {_THIS_DIR}. "
            f"Expected a file such as 'Extract_to_excel.PY'."
        )

    return candidates[0]


@st.cache_resource
def load_pipeline_module():
    """Import the pipeline in THIS (main) process - used only for constants
    like EXCEL_COLUMNS/COLUMN_HEADERS and for append_row_to_excel. Actual
    document extraction always happens in the subprocess (pipeline_worker),
    never here.
    """
    pipeline_path = _find_pipeline_script_path()
    module_name = "contract_extraction_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# PROCESS LIFECYCLE: start / stop / cleanup
# ---------------------------------------------------------------------------
def _cleanup_after_run():
    """Removes temp input files, sweeps up any orphaned OCR page images the
    worker process may have left behind if it was killed mid-OCR, clears all
    run-scoped session state, and bumps the uploader's key so the file
    uploader widget resets to empty on the next full rerun."""
    for p in st.session_state.get("_tmp_paths", []):
        try:
            os.remove(p)
        except OSError:
            pass

    # extract_to_excel.py's OCR fallback writes "_temp_ocr_page_N.png" into
    # the current working directory and normally deletes it itself - but a
    # killed process never reaches that cleanup line, so sweep for leftovers.
    try:
        for p in Path.cwd().glob("_temp_ocr_page_*.png"):
            p.unlink(missing_ok=True)
    except OSError:
        pass

    for key in ("_process", "_progress_queue", "_result_queue", "_tmp_paths", "_log"):
        st.session_state.pop(key, None)

    st.session_state["_extracting"] = False
    st.session_state["_uploader_epoch"] = st.session_state.get("_uploader_epoch", 0) + 1


def _start_processing(uploaded_files, doc_types, model):
    if st.session_state.get("_extracting") or not uploaded_files:
        return

    st.session_state.pop("pending_record", None)
    st.session_state.pop("pending_review_reasons", None)
    st.session_state.pop("_stop_notice", None)
    st.session_state.pop("_error_notice", None)

    pipeline_path = str(_find_pipeline_script_path())

    tmp_paths, file_infos = [], []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_paths.append(tmp.name)
        file_infos.append((tmp.name, f.name))

    # "spawn" (not "fork") - safer to launch from inside an already-running,
    # multi-threaded Streamlit server, and it's what Windows uses anyway.
    ctx = multiprocessing.get_context("spawn")
    progress_queue = ctx.Queue()
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=pipeline_worker.run_worker,
        args=(pipeline_path, file_infos, doc_types, model, progress_queue, result_queue),
        daemon=True,
    )
    process.start()

    st.session_state["_process"] = process
    st.session_state["_progress_queue"] = progress_queue
    st.session_state["_result_queue"] = result_queue
    st.session_state["_tmp_paths"] = tmp_paths
    st.session_state["_log"] = []
    st.session_state["_extracting"] = True
    st.rerun()  # full-page: disables the uploader/editor outside the fragment


def _stop_processing():
    process = st.session_state.get("_process")
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)

    _cleanup_after_run()
    st.session_state["_stop_notice"] = "Processing stopped by user."
    st.rerun()  # full-page: clears the uploader and re-enables controls


@st.fragment(run_every=1)
def _processing_panel(uploaded_files, doc_types, model):
    """The only part of the page that re-renders on a timer. Everything
    outside this function is untouched by that timer, so the rest of the
    dashboard never blinks while extraction runs."""
    processing = st.session_state.get("_extracting", False)

    btn_col, stop_col = st.columns([4, 1])
    with btn_col:
        st.button(
            "🕒 Extracting..." if processing else "🚀 Process & Extract",
            type="primary",
            use_container_width=True,
            disabled=processing or not uploaded_files,
            key="start_extract_btn",
            on_click=_start_processing,
            args=(uploaded_files, doc_types, model),
        )
    with stop_col:
        st.button(
            "🛑 Stop",
            use_container_width=True,
            disabled=not processing,
            key="stop_extract_btn",
            on_click=_stop_processing,
        )

    if not processing:
        return

    progress_queue = st.session_state.get("_progress_queue")
    log = st.session_state.setdefault("_log", [])
    if progress_queue is not None:
        while True:
            try:
                log.append(progress_queue.get_nowait())
            except pyqueue.Empty:
                break

    status_box = st.status("Processing uploaded documents...", expanded=True)
    with status_box:
        for line in log:
            st.write(line)

    process = st.session_state.get("_process")
    result_queue = st.session_state.get("_result_queue")
    outcome = None
    if result_queue is not None:
        try:
            outcome = result_queue.get_nowait()
        except pyqueue.Empty:
            outcome = None

    if outcome is not None:
        kind = outcome[0]
        if kind == "done":
            _, final_record, review_reasons = outcome
            status_box.update(label="✅ Extraction done", state="complete", expanded=False)
            st.session_state["pending_record"] = final_record
            st.session_state["pending_review_reasons"] = review_reasons
        else:
            _, message = outcome
            status_box.update(label="❌ Processing failed", state="error", expanded=False)
            st.session_state["_error_notice"] = message
        if process is not None:
            process.join(timeout=2)
        _cleanup_after_run()
        st.rerun()  # full-page: reveals the preview panel, resets controls
    elif process is not None and not process.is_alive():
        # Process is gone with no result on the queue - e.g. it was killed
        # by something other than the Stop button. Treat as a stop, not an
        # error, so the user isn't shown a scary red message for it.
        status_box.update(label="🛑 Processing stopped", state="error", expanded=False)
        _cleanup_after_run()
        st.session_state.setdefault("_stop_notice", "Processing stopped.")
        st.rerun()


st.title("📄Dashboard")
st.caption(f"Source file: `{EXCEL_PATH}`")

st.subheader("📤 Upload Contract PDFs")
st.caption(
    "Upload the contract PDFs (MSA / NDA / others), review the fields the local LLM "
    "extracts, then append them to the workbook. No command line needed."
)

_processing = st.session_state.get("_extracting", False)

if st.session_state.get("_stop_notice"):
    st.info(st.session_state.pop("_stop_notice"))
if st.session_state.get("_error_notice"):
    st.error(f"Processing failed: {st.session_state.pop('_error_notice')}")

# Keying the uploader on a counter that bumps after every stop/completion is
# how it gets cleared back to empty - Streamlit file_uploader has no direct
# `.clear()` method, but swapping its key makes it a fresh widget.
uploader_key = f"contract_pdf_upload_{st.session_state.get('_uploader_epoch', 0)}"
uploaded = st.file_uploader(
    "Choose contract PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=uploader_key,
    disabled=_processing,
    help="Drag & drop all your contract documents here - MSA(s), NDA(s), or any other PDF.",
)

model_choice = st.selectbox(
    "Ollama model",
    ["phi4-mini", "llama3.2:1b"],
    index=0,
    disabled=_processing,
    help="phi4-mini is the recommended default; llama3.2:1b is faster but noticeably less accurate.",
)

if uploaded:
    edited = st.data_editor(
        pd.DataFrame(
            {
                "File": [f.name for f in uploaded],
                "Document Type": [guess_doc_type(f.name) for f in uploaded],
            }
        ),
        column_config={
            "File": st.column_config.TextColumn("File", disabled=True),
            "Document Type": st.column_config.SelectboxColumn(
                "Document Type",
                options=DOC_TYPE_OPTIONS,
                help="What kind of agreement is each file? Used for smart merging (e.g. MSA + NDA).",
            ),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=_processing,
        key=f"doc_type_editor_{st.session_state.get('_uploader_epoch', 0)}",
    )
    _processing_panel(uploaded, edited["Document Type"].tolist(), model_choice)
    st.divider()

# ---------------------------------------------------------------------------
# PREVIEW the extracted record and offer to append it to the workbook
# ---------------------------------------------------------------------------
pending_record = st.session_state.get("pending_record")
if pending_record and isinstance(pending_record, dict):
    pipeline = load_pipeline_module()  # cached after the first processing run
    st.subheader("📋 Extracted record (preview)")
    column_order = [c for c in pipeline.EXCEL_COLUMNS if c in pending_record]
    preview_df = pd.DataFrame(
        [{pipeline.COLUMN_HEADERS.get(c, c): pending_record.get(c) for c in column_order}]
    )
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Source file(s): `{pending_record.get('source_files')}`  ·  "
        f"Model: `{pending_record.get('model_used')}`  ·  Processed: `{pending_record.get('processed_at')}`"
    )

    review_reasons = st.session_state.get("pending_review_reasons") or []
    if review_reasons:
        st.warning("**Flagged for review:**\n\n" + "\n".join(f"- {reason}" for reason in review_reasons))
    else:
        st.success("No review flags - all extracted fields look good.")

    if st.button("✅ Append to Excel workbook", type="primary", use_container_width=True):
        try:
            pipeline.append_row_to_excel(EXCEL_PATH, pending_record)
        except Exception as e:
            st.error(f"Could not write to `{EXCEL_PATH}`: {e}")
        else:
            st.session_state.pop("pending_record", None)
            st.session_state.pop("pending_review_reasons", None)
            st.session_state["_uploader_epoch"] = st.session_state.get("_uploader_epoch", 0) + 1
            st.success(f"Appended to `{EXCEL_PATH}` - refreshing the dashboard...")
            st.cache_data.clear()
            st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# DATA LOADING - run after the upload UI so a brand-new workbook can be made here
# ---------------------------------------------------------------------------
try:
    df = load_data(EXCEL_PATH)
except FileNotFoundError:
    df = None

if df is None:
    st.error(
        f"Could not find '{EXCEL_PATH}'. Upload your first contract PDFs above and click "
        f"**Append to Excel workbook** to create it, or run extract_to_excel.py first, "
        f"e.g.:\n\npython extract_to_excel.py --msa msa.pdf --nda nda.pdf"
    )
    st.stop()

if df.empty:
    st.info("The workbook exists but has no contracts yet. Upload PDFs above and append the first one.")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

partner_filter = st.sidebar.multiselect(
    "Partner", sorted(df["Partner Name"].dropna().unique().tolist())
)
review_filter = st.sidebar.selectbox("Review status", ["All", "Needs review only", "Clean only"])
signed_filter = st.sidebar.selectbox("Signed status", ["All", "Signed only", "Unsigned only"])

filtered = df.copy()
if partner_filter:
    filtered = filtered[filtered["Partner Name"].isin(partner_filter)]
if review_filter == "Needs review only":
    filtered = filtered[filtered["Needs Review"] == True]  # noqa: E712
elif review_filter == "Clean only":
    filtered = filtered[filtered["Needs Review"] == False]  # noqa: E712
if signed_filter == "Signed only":
    filtered = filtered[filtered["Document Signed (Y/N)"] == "Y"]
elif signed_filter == "Unsigned only":
    filtered = filtered[filtered["Document Signed (Y/N)"] == "N"]

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
total_contracts = len(filtered)
needs_review_count = int((filtered["Needs Review"] == True).sum())  # noqa: E712
signed_count = int((filtered["Document Signed (Y/N)"] == "Y").sum())
today = pd.Timestamp(datetime.now().date())
expiring_soon = filtered[
    (filtered[END_COL].notna())
    & (filtered[END_COL] >= today)
    & (filtered[END_COL] <= today + pd.Timedelta(days=60))
]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Contracts", total_contracts)
k2.metric("Needs Review", needs_review_count,
          delta=f"{needs_review_count / total_contracts:.0%} of total" if total_contracts else None,
          delta_color="inverse")
k3.metric("Signed", f"{signed_count}/{total_contracts}")
k4.metric("Expiring in 60 Days", len(expiring_soon))

st.divider()

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Review Status")
    review_counts = filtered["Needs Review"].map({True: "Needs Review", False: "Clean"}).value_counts()
    if not review_counts.empty:
        fig = px.pie(
            names=review_counts.index, values=review_counts.values,
            color=review_counts.index,
            color_discrete_map={"Needs Review": "#F2A900", "Clean": "#2E7D32"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Penalty Clause Present?")
    # "Penalty Clause? (Y/N & Details)" values start with "Y" or "N"
    penalty_col = "Penalty Clause? (Y/N & Details)"
    if penalty_col in filtered.columns:
        penalty_yn = filtered[penalty_col].dropna().astype(str).str.strip().str.upper().str[0]
        penalty_counts = penalty_yn.map({"Y": "Has Penalty Clause", "N": "No Penalty Clause"}).value_counts()
        if not penalty_counts.empty:
            fig = px.pie(
                names=penalty_counts.index, values=penalty_counts.values,
                color_discrete_sequence=["#C0392B", "#2E7D32"],
                hole=0.45,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No penalty clause data yet.")

st.subheader("Contracts Expiring Soon (next 60 days)")
if not expiring_soon.empty:
    exp_view = expiring_soon[["Partner Name", END_COL]].sort_values(END_COL)
    st.dataframe(exp_view, use_container_width=True, hide_index=True)
else:
    st.success("Nothing expiring in the next 60 days.")

st.divider()

# ---------------------------------------------------------------------------
# REVIEW QUEUE
# ---------------------------------------------------------------------------
st.subheader("🚩 Contracts Needing Review")
review_queue = filtered[filtered["Needs Review"] == True]  # noqa: E712
if not review_queue.empty:
    st.dataframe(
        review_queue[["Partner Name", "Review Notes"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.success("No contracts currently flagged for review.")

st.divider()

# ---------------------------------------------------------------------------
# FULL TABLE
# ---------------------------------------------------------------------------
st.subheader("All Contracts")
search = st.text_input("Search (partner name, remarks, etc.)")
table_view = filtered
if search:
    mask = filtered.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
    table_view = filtered[mask]

st.dataframe(table_view, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download filtered data as CSV",
    data=table_view.to_csv(index=False).encode("utf-8"),
    file_name="contracts_filtered.csv",
    mime="text/csv",
)