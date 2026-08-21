"""
Subprocess worker for the contract extraction pipeline.

Runs as a separate OS process rather than a thread so that clicking Stop can
call process.terminate()/kill() and immediately halt whatever it's doing -
an in-flight LLM call, OCR pass, or PDF parse - which a Python thread simply
cannot be forced to do (there is no safe way to kill a thread mid-blocking-
call in Python).

Deliberately kept in its own file, with NO Streamlit import anywhere in this
module or anything it imports at call time. On Windows, multiprocessing's
default "spawn" start method boots the child by re-importing whichever
module the target function lives in. If the target lived inside the
Streamlit script itself, the child would re-execute every st.* call at
import time. Living here, the child only ever imports this file plus the
extraction pipeline - never dashboard.py.

There is no cooperative cancellation flag anywhere below. That's
intentional: cancellation happens from the OUTSIDE, by killing this whole
process. Whatever line is executing when that happens just stops.
"""

import importlib.util
import sys
import traceback
from datetime import datetime


def _load_pipeline(pipeline_path: str):
    from importlib.machinery import SourceFileLoader

    module_name = "contract_extraction_pipeline"

    loader = SourceFileLoader(module_name, pipeline_path)
    spec = importlib.util.spec_from_loader(module_name, loader)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline: {pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)

    return module

def run_worker(pipeline_path, file_infos, doc_types, model, progress_queue, result_queue):
    """
    file_infos: list of (temp_file_path, original_filename) tuples. Temp
    files are created and later cleaned up by the PARENT process, not here -
    that way cleanup still happens even if this process gets killed before
    it has a chance to run any of its own cleanup code.

    progress_queue: strings pushed as human-readable status lines.
    result_queue: gets exactly one final message, either
        ("done", final_record_dict, review_reasons_list)
        ("error", message_str)
    If this process is killed, result_queue simply never receives anything -
    the parent treats "process no longer alive + no result" as a stop.
    """
    try:
        pipeline = _load_pipeline(pipeline_path)
        dicts, labels, source_files = [], [], []

        for (tmp_path, orig_name), doc_type in zip(file_infos, doc_types):
            progress_queue.put(f"Processing '{orig_name}' ({doc_type}) - extracting text & running LLM...")
            dicts.append(pipeline.process_single_document(tmp_path, model, doc_type))
            labels.append(doc_type)
            source_files.append(orig_name)

        if len(dicts) > 1:
            progress_queue.put("Merging documents into a single combined record (MSA + NDA priority rules)...")
            merged, conflicts = pipeline.merge_records(dicts, labels)
            if conflicts:
                progress_queue.put(f"{len(conflicts)} field conflict(s) detected between sources")
        else:
            merged, conflicts = dicts[0], []

        progress_queue.put("Running field-level validation & review flagging...")
        record = pipeline.PartnerAgreementRecord(**merged)
        final_record, review_reasons = pipeline.validate_record(record)

        if conflicts:
            review_reasons.extend(conflicts)
            final_record["needs_review"] = True
            final_record["review_reasons"] = "; ".join(review_reasons)

        final_record["processed_at"] = datetime.now().strftime("%d-%b-%y %H:%M")
        final_record["model_used"] = model
        final_record["source_files"] = " + ".join(source_files)

        result_queue.put(("done", final_record, review_reasons))

    except SystemExit:
        result_queue.put((
            "error",
            "The extraction pipeline stopped itself, which usually means Ollama is not running "
            "or the document could not be processed. Start Ollama with `ollama serve` and try again.",
        ))
    except Exception:
        result_queue.put(("error", traceback.format_exc(limit=4)))