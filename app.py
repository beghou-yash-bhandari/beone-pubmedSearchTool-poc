# app.py

import os
import sys
import threading
import queue
import time
import streamlit as st
import pandas as pd

from pubmed_pipeline import SearchAndExtract, load_prompt_assets


st.set_page_config(page_title="PubMed Research Tool", layout="wide")
st.title("PubMed Research Tool")

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

def sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def clean_value(v):
        if isinstance(v, str):
            return ILLEGAL_CHARACTERS_RE.sub("", v)
        return v

    return df.applymap(clean_value)


# -------------------------
# Sidebar configuration
# -------------------------
# with st.sidebar:
#     st.header("Configuration")
#     excel_path = st.text_input(
#         "Prompt Excel path",
#         value="Prompt_Design_PubMed_Parameters_With_Prompts.xlsx",
#         help="Path to the Excel file containing Allowed, Data_num, Data, and LLM Extraction Prompt.",
#     )
#     

#     st.divider()
#     # st.caption("Secrets are read from environment variables (.env recommended).")
#     # st.caption("Required: OPENAI_API_KEY. Optional: ENTREZ_EMAIL, PINECONE_API_KEY, PINECONE_INDEX_US.")

model = "gpt-5.2"
excel_path = "Prompt_Design_PubMed_Parameters_With_Prompts.xlsx"

with st.sidebar:
    st.header("Configuration")
    # excel_path = st.text_input(
    #     "Prompt Excel path",
    #     value="",
    #     help="Path to the Excel file containing Allowed, Data_num, Data, and LLM Extraction Prompt.",
    # )

    # st.markdown("**Prompt Excel path**")
    # st.code("Prompt_Design_PubMed_Parameters_With_Prompts.xlsx")

    st.markdown("**LLM model**")
    st.code("gpt-5.2")

    st.divider()

# -------------------------
# Load rates (for dropdown)
# -------------------------
rate_chart_dict = {}
prompt_map = {}
rate_chart_dict_op = {}
rate_options = ["all"]
load_error = None

try:
    # Your load_prompt_assets must return (rate_chart_dict, prompt_map, rate_chart_dict_op)
    rate_chart_dict, prompt_map, rate_chart_dict_op = load_prompt_assets(excel_path)

    # You are using display names (values) as selectable options
    rate_options = ["all"] + list(rate_chart_dict.values())
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"Failed to load prompt assets from Excel: {load_error}")


# -------------------------
# Form inputs + submit
# -------------------------
with st.form("pubmed_form"):
    search_query = st.text_input(
        "What do you want to research today?",
        value="Progression rate for CML",
        help="This is passed into your EnhancedQuery() + PubMed retrieval flow.",
    )

    chosen_rate_name = st.multiselect(
        "Rate type",
        options=rate_options,
        disabled=bool(load_error),
        help="Choose one or more rates by name, or select 'all'.",
    )

    submitted = st.form_submit_button("Submit", disabled=bool(load_error))


# Resolve chosen_rate AFTER submit (do not do processing inside the form)
if submitted:
    # If user picked "all" or picked nothing => run all
    if (not chosen_rate_name) or ("all" in chosen_rate_name):
        chosen_rate = "all"
    else:
        # Map display-name(s) -> internal key(s) using your reverse map rate_chart_dict_op
        # Expectation: rate_chart_dict_op[display_name] = "1_1" style key
        chosen_rate = [rate_chart_dict_op[name] for name in chosen_rate_name]


# -------------------------
# Live print streaming helpers
# -------------------------
class StreamToQueue:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, message: str):
        if message:
            self.q.put(message)

    def flush(self):
        pass


def run_pipeline_in_thread(
    q: queue.Queue,
    result_container: dict,
    *,
    query: str,
    rate,
    model: str,
    excel_path: str,
):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = StreamToQueue(q)
    sys.stderr = StreamToQueue(q)

    try:
        df = SearchAndExtract(query=query, rate=rate, model=model)
        result_container["df"] = df
        result_container["error"] = None
    except Exception as e:
        result_container["df"] = None
        result_container["error"] = e
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        q.put("\n[Run complete]\n")


# -------------------------
# Run on submit
# -------------------------
if submitted:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in the environment. Add it to .env or your shell environment.")
        st.stop()

    # ---- Logs UI: use a single placeholder, not a widget
    st.subheader("Logs")
    log_placeholder = st.empty()

    q = queue.Queue()
    result_container = {}
    collected = []

    t = threading.Thread(
        target=run_pipeline_in_thread,
        args=(q, result_container),
        kwargs=dict(query=search_query, rate=chosen_rate, model=model, excel_path=excel_path),
        daemon=True,
    )
    t.start()

    with st.spinner("Running..."):
        while t.is_alive() or not q.empty():
            try:
                msg = q.get(timeout=0.1)
                collected.append(msg)

                # Update the SAME placeholder (no new widget created)
                log_placeholder.code("".join(collected), language="text")

            except queue.Empty:
                pass

            time.sleep(0.05)

    # Final update
    log_placeholder.code("".join(collected), language="text")

    st.subheader("Results")
    err = result_container.get("error")
    df = result_container.get("df")
 

    if err is not None:
        if "No PubMed IDs found for this query" in str(err):
            st.warning(
                "No PubMed IDs were returned for this query. "
                "Try broadening the search query, removing field tags like [tiab], or increasing page size."
            )
        else:
            st.error(f"Run failed: {err}")
    elif isinstance(df, pd.DataFrame) and not df.empty:

        # ------------------------------------------------------------------
        # 1. Columns that uniquely identify an article
        # ------------------------------------------------------------------
        index_cols = [
            'title', 'pmid', 'doi', 'journal', 'authors',
            'publication_date', 'keywords', 'url', 'affiliations'
        ]

        # ------------------------------------------------------------------
        # 2. Make list-valued columns hashable (required for pivot_table)
        #    This is CRITICAL for authors / keywords / affiliations
        # ------------------------------------------------------------------
        list_cols = ['authors', 'keywords', 'affiliations']

        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: "; ".join(map(str, x)) if isinstance(x, list) else x
                )

        # ------------------------------------------------------------------
        # 3. Pivot extracted values (parameters → columns)
        # ------------------------------------------------------------------
        values_df = df.pivot_table(
            index=index_cols,
            columns='parameter',
            values='extracted_value',
            aggfunc='first'
        )

        # ------------------------------------------------------------------
        # 4. Pivot evidence sentences (same shape)
        # ------------------------------------------------------------------
        evidence_df = df.pivot_table(
            index=index_cols,
            columns='parameter',
            values='exact_evidence_sentence',
            aggfunc='first'
        )

        # Add suffix so columns don’t collide
        evidence_df.columns = [f"{c}_exact_evidence_statement" for c in evidence_df.columns]

        # ------------------------------------------------------------------
        # 5. Combine + flatten
        # ------------------------------------------------------------------
        pivot_df = (
            pd.concat([values_df, evidence_df], axis=1)
            .reset_index()
        )

        # Optional: sort parameter columns for readability
        pivot_df = pivot_df.reindex(
            columns=index_cols + sorted(
                [c for c in pivot_df.columns if c not in index_cols]
            )
        )

        # ------------------------------------------------------------------
        # 6. Display in Streamlit
        # ------------------------------------------------------------------
        pivot_df = pivot_df.iloc[1:].reset_index(drop=True)
        st.dataframe(pivot_df, use_container_width=True)


        # st.dataframe(df.columns, use_container_width=True)
        # st.dataframe(pivot_df, use_container_width = True)
        
        df = sanitize_df_for_excel(df)
        
        df = df[['title', 'pmid','doi', 'journal',   'authors', 'publication_date', 'keywords', 'parameter', 'extracted_value', 'exact_evidence_sentence', 'url', 'affiliations']].transpose()
        df.to_excel(rf"output\test{search_query.replace(' ','_')}.xlsx", index=False)

    else:
        st.warning("No rows returned.")
