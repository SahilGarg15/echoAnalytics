# """Manage multi-step cleaning pipeline and undo."""
# import streamlit as st
# import pandas as pd

# def init_pipeline():
#     if "pipeline_steps" not in st.session_state:
#         st.session_state.pipeline_steps = []
#     if "original_df" not in st.session_state:
#         st.session_state.original_df = st.session_state.get("df").copy() if "df" in st.session_state else None

# def add_step(step_name: str, df_snapshot: pd.DataFrame):
#     st.session_state.pipeline_steps.append({"name": step_name, "df": df_snapshot.copy()})

# def undo_last_step():
#     if st.session_state.pipeline_steps:
#         st.session_state.pipeline_steps.pop()
#         if st.session_state.pipeline_steps:
#             st.session_state.df = st.session_state.pipeline_steps[-1]["df"].copy()
#         else:
#             st.session_state.df = st.session_state.original_df.copy()
#         st.success("Reverted last step.")
#     else:
#         st.warning("No steps to undo.")


# """Manage multi-step cleaning pipeline and undo/reset."""
# import streamlit as st
# import pandas as pd

# def init_pipeline():
#     if "pipeline_steps" not in st.session_state:
#         st.session_state.pipeline_steps = []
#     if "original_df" not in st.session_state:
#         st.session_state.original_df = (
#             st.session_state.get("df").copy()
#             if "df" in st.session_state else None
#         )

# def add_step(step_name: str, df_snapshot: pd.DataFrame, details: dict = None):
#     """Save snapshot with step description and optional details."""
#     st.session_state.pipeline_steps.append({
#         "name": step_name,
#         "details": details or {},
#         "df": df_snapshot.copy()
#     })
#     st.session_state.df = df_snapshot.copy()

# def undo_last_step():
#     if st.session_state.pipeline_steps:
#         st.session_state.pipeline_steps.pop()
#         if st.session_state.pipeline_steps:
#             st.session_state.df = st.session_state.pipeline_steps[-1]["df"].copy()
#         else:
#             st.session_state.df = st.session_state.original_df.copy()
#         st.success("Reverted last step.")
#     else:
#         st.warning("No steps to undo.")

# def reset_pipeline():
#     if "original_df" in st.session_state and st.session_state.original_df is not None:
#         st.session_state.df = st.session_state.original_df.copy()
#         st.session_state.pipeline_steps = []
#         st.success("Reset to original dataset.")
#     else:
#         st.warning("No original dataset available.")


"""
Simple pipeline manager with step history and reset.
Save as: echoAnalytics/modules/pipeline_manager.py
"""

import streamlit as st
import pandas as pd
from typing import Any

def init_pipeline():
    if "pipeline_steps" not in st.session_state:
        st.session_state.pipeline_steps = []
    if "original_df" not in st.session_state:
        st.session_state.original_df = st.session_state.get("df").copy() if "df" in st.session_state else None


def add_step(step_name: str, df_snapshot: pd.DataFrame, details: Any = None):
    """Append a pipeline step and keep updated current df in session state."""
    st.session_state.pipeline_steps.append({
        "name": step_name,
        "details": details or {},
        "df": df_snapshot.copy()
    })
    st.session_state.df = df_snapshot.copy()


def undo_last_step():
    if st.session_state.get("pipeline_steps"):
        st.session_state.pipeline_steps.pop()
        if st.session_state.pipeline_steps:
            st.session_state.df = st.session_state.pipeline_steps[-1]["df"].copy()
        else:
            st.session_state.df = st.session_state.original_df.copy()
        st.success("Reverted last step.")
    else:
        st.warning("No steps to undo.")


def reset_pipeline():
    if "original_df" in st.session_state and st.session_state.original_df is not None:
        st.session_state.pipeline_steps = []
        st.session_state.df = st.session_state.original_df.copy()
        st.success("Reset to original dataset.")
    else:
        st.warning("No original dataset available.")
    