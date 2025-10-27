import os
import io
import asyncio
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Import the async prioritization flow from the existing module
from agents_oai_inline_eval import prioritize_features


st.set_page_config(
    page_title="Feature Request Prioritizer",
    page_icon="✨",
    layout="wide",
)


def write_uploaded_file_to_temp(uploaded_file) -> str:
    """
    Persist an uploaded file-like object to a real temporary file and return its path.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def ensure_customers_csv_path(user_uploaded_customers) -> str:
    """
    Resolve the customers ARR mapping CSV path, preferring user upload,
    else falling back to repo default `customers.csv` if present.
    """
    if user_uploaded_customers is not None:
        return write_uploaded_file_to_temp(user_uploaded_customers)

    repo_default = Path(__file__).parent / "customers.csv"
    if repo_default.exists():
        return str(repo_default)

    raise FileNotFoundError(
        "No customers.csv provided and default file not found in the repository."
    )


def validate_tickets_df(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that the uploaded tickets CSV contains required columns.
    """
    required_cols = {"ticket_id", "description", "customers_requesting", "customer_priority"}
    missing = [c for c in required_cols if c not in df.columns]
    return (len(missing) == 0, missing)


def run_prioritization(input_csv_path: str, customers_csv_path: str) -> pd.DataFrame:
    """
    Run the async prioritization flow and return the resulting scored DataFrame.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out_tmp:
        output_path = out_tmp.name

    # Execute the async workflow
    asyncio.run(
        prioritize_features(
            input_csv=input_csv_path,
            output_csv=output_path,
            customers_csv=customers_csv_path,
        )
    )

    return pd.read_csv(output_path)


def header():
    left, right = st.columns([0.7, 0.3])
    with left:
        st.title("✨ Feature Request Prioritizer")
        st.markdown(
            "Prioritize feature tickets using an agentic flow that scores reach, impact, alignment, and effort."
        )
    with right:
        st.metric("Status", "Ready")

    st.markdown(
        """
        <style>
        /* Light theme with strong contrast */
        .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%); }
        html, body, .stApp { color: #0f172a; }
        h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
        div[data-testid="stFileUploader"] > label { font-weight: 600; color: #0f172a; }
        .soft-card {
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .soft-divider { height: 1px; background: #e5e7eb; margin: 0.75rem 0 1rem 0; }
        /* Improve table readability */
        div[data-testid="stDataFrame"] { filter: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar():
    with st.sidebar:
        st.header("How it works")
        st.caption(
            "Upload your feature requests CSV. Optionally provide a customers ARR mapping CSV.\n"
            "The app runs the same async prioritization used in `agents_oai_inline_eval.py`."
        )
        st.markdown("---")
        st.write("Expected columns (tickets CSV):")
        st.code("ticket_id, description, customers_requesting, customer_priority")
        st.markdown("---")
        st.caption("Tip: `customers_requesting` can be a JSON array or comma-separated list.")


def main():
    header()
    sidebar()

    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("1) Upload your CSVs")

    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        tickets_file = st.file_uploader(
            "Feature requests CSV",
            type=["csv"],
            help="Must contain columns: ticket_id, description, customers_requesting, customer_priority",
        )

    with c2:
        customers_file = st.file_uploader(
            "Customers ARR mapping CSV (optional)",
            type=["csv"],
            help="If omitted, uses repository default customers.csv if available",
        )

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    st.subheader("2) Preview")

    valid_input = False
    if tickets_file is not None:
        try:
            df_preview = pd.read_csv(tickets_file)
            is_valid, missing = validate_tickets_df(df_preview)
            if is_valid:
                valid_input = True
                st.dataframe(df_preview.head(20), use_container_width=True)
            else:
                st.error(
                    f"Missing required columns: {', '.join(missing)}. Please upload a valid CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.info("Upload a feature requests CSV to preview and continue.")

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    st.subheader("3) Run prioritization")

    run_col, _ = st.columns([0.25, 0.75])
    with run_col:
        run = st.button("Run scoring 🚀", type="primary", disabled=not valid_input)

    st.markdown("</div>", unsafe_allow_html=True)  # close soft-card

    if run and tickets_file is not None and valid_input:
        try:
            with st.spinner("Running agentic prioritization... This can take a moment."):
                # Write uploaded tickets to temp path
                input_csv_path = write_uploaded_file_to_temp(tickets_file)
                customers_csv_path = ensure_customers_csv_path(customers_file)

                df_scored = run_prioritization(
                    input_csv_path=input_csv_path, customers_csv_path=customers_csv_path
                )

            st.success("Scoring complete!")

            # Show results preview
            st.subheader("Scored tickets preview")
            st.dataframe(df_scored.head(50), use_container_width=True)

            # Offer download
            csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download scored CSV",
                data=csv_bytes,
                file_name="scored_tickets.csv",
                mime="text/csv",
            )

            # Lightweight summary metrics
            try:
                avg_score = float(df_scored["final_score"].mean())
                st.metric("Average final score", f"{avg_score:.2f}")
            except Exception:
                pass

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An error occurred while scoring tickets: {e}")


if __name__ == "__main__":
    main()


