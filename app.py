# # # app.py
# # """Streamlit UI for EchoAnalytics MVP with Memory Controls & Normalization."""
# # import streamlit as st
# # import pandas as pd

# # from modules.data_handler import load_dataset
# # from modules.query_parser import run_query, CONTROL_CLEAR, CONTROL_SHOW_HISTORY
# # from modules.voice_handler import handle_voice_query
# # from modules.context_manager import (
# #     load_context, save_context, clear_context,
# #     log_query, add_history, get_history
# # )

# # st.set_page_config(page_title="EchoAnalytics - Phase 6.5", layout="wide")
# # st.title("📊 EchoAnalytics — Voice/Text Data Assistant (Phase 6.5 Upgrades)")

# # # Persistent context loaded at startup
# # if "query_context" not in st.session_state:
# #     st.session_state.query_context = load_context()

# # # Sidebar controls
# # st.sidebar.header("⚙️ Settings")
# # use_memory = st.sidebar.toggle("Enable Memory", value=True, help="Turn off to ignore saved context")
# # if st.sidebar.button("🗑 Clear Memory (Sidebar)"):
# #     clear_context()
# #     st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
# #     st.sidebar.success("Memory cleared.")

# # # Dataset upload
# # uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xls", "xlsx"])
# # df = None

# # if uploaded_file:
# #     df = load_dataset(uploaded_file)
# #     st.subheader("📄 Dataset Preview")
# #     st.dataframe(df.head())

# #     st.markdown("---")
# #     st.markdown("### 🎙 Speak or Type a Question")

# #     col1, col2, col3 = st.columns([3, 1, 1])

# #     query = ""
# #     result = None
# #     used_context = False
# #     control = None

# #     with col1:
# #         query = st.text_input(
# #             "Ask about your data",
# #             value="",
# #             placeholder="E.g., 'Unique values in proto', 'Average sales', 'show history', 'clear memory'"
# #         )

# #     with col2:
# #         if st.button("🎤 Speak"):
# #             # Voice path returns (result, raw_query)
# #             v_result, raw_query = handle_voice_query(df)
# #             # We will re-run the same raw query via unified path below to keep behavior consistent
# #             query = raw_query or ""
# #             if raw_query:
# #                 st.info(f"You said: **{raw_query}**")

# #     with col3:
# #         if st.button("🗑 Clear Memory (Main)"):
# #             clear_context()
# #             st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
# #             st.success("Memory cleared!")

# #     run_clicked = st.button("Run Query")

# #     # Execute the query
# #     if run_clicked and query.strip():
# #         context_to_pass = st.session_state.query_context if use_memory else {}
# #         result, updated_context, control, used_context = run_query(df, query, context_to_pass)

# #         # Handle control commands
# #         if control == CONTROL_CLEAR:
# #             clear_context()
# #             st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
# #             st.success("Memory cleared!")
# #         elif control == CONTROL_SHOW_HISTORY:
# #             st.info("Showing memory history (most recent last):")
# #             history = get_history(st.session_state.query_context)
# #             if history:
# #                 st.table(pd.DataFrame(history))
# #             else:
# #                 st.write("No history yet.")
# #         else:
# #             # Normal results path
# #             if use_memory:
# #                 st.session_state.query_context = updated_context
# #                 save_context(st.session_state.query_context)

# #             # Display result
# #             if result is not None:
# #                 if isinstance(result, pd.DataFrame):
# #                     st.dataframe(result)
# #                 elif isinstance(result, (list, pd.Series, pd.Index)):
# #                     st.write(result)
# #                 else:
# #                     st.success(f"Result: {result}")

# #                 # Context usage feedback + logging + history
# #                 if use_memory and used_context:
# #                     st.caption("🧠 Used previous context (last column/operation).")
# #                 log_query(query, result, used_context)
# #                 # store a compact history record
# #                 st.session_state.query_context = add_history(
# #                     st.session_state.query_context, query, repr(result), used_context
# #                 )
# #                 save_context(st.session_state.query_context)

# #     elif run_clicked and not query.strip():
# #         st.warning("Please enter or speak a question.")

# # else:
# #     st.warning("Please upload a dataset first.")


# """Streamlit UI for EchoAnalytics MVP with Persistent Query Memory + Logging."""
# import streamlit as st
# import pandas as pd
# from modules.data_handler import load_dataset
# from modules.query_parser import run_query
# from modules.voice_handler import handle_voice_query, ERROR_NO_SPEECH, ERROR_SERVICE
# from modules.context_manager import load_context, save_context, clear_context, log_query

# # --------------------
# # Page Config
# # --------------------
# st.set_page_config(page_title="EchoAnalytics - Phase 6", layout="wide")
# st.title("📊 EchoAnalytics - Voice/Text Data Assistant (Phase 6)")

# # Load memory at startup
# if "query_context" not in st.session_state:
#     st.session_state.query_context = load_context()

# # Upload dataset
# uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xls", "xlsx"])
# df = None

# if uploaded_file:
#     df = load_dataset(uploaded_file)
#     st.subheader("📄 Dataset Preview")
#     st.dataframe(df.head())

#     st.markdown("---")
#     st.markdown("### 🎙 Speak or Type a Question")
#     col1, col2, col3 = st.columns([3, 1, 1])

#     query = ""
#     result = None

#     with col1:
#         query = st.text_input(
#             "Ask about your data",
#             value="",
#             placeholder="E.g., 'Unique values in proto', 'Average sales'"
#         )

#     with col2:
#         if st.button("🎤 Speak"):
#             v_result, v_context, raw_query = handle_voice_query(df)
#             query = raw_query or ""
#             if raw_query:
#                 if raw_query in {ERROR_NO_SPEECH, ERROR_SERVICE}:
#                     st.warning(f"Voice input error: {raw_query}")
#                 else:
#                     st.info(f"You said: **{raw_query}**")
#             # If we got a result from voice, persist context immediately
#             if v_result is not None:
#                 result = v_result
#                 if v_context is not None:
#                     st.session_state.query_context = v_context
#                     save_context(v_context)
#                     log_query(raw_query, v_result)

#     with col3:
#         if st.button("🗑 Clear Memory"):
#             clear_context()
#             st.session_state.query_context = {}
#             st.success("Memory cleared!")

#     run_clicked = st.button("Run Query")

#     if run_clicked or (query and result is None):
#         if not query.strip():
#             st.warning("Please enter or speak a question.")
#         else:
#             # Only run text query if it wasn't already executed via voice path
#             result, updated_context = run_query(df, query, st.session_state.query_context)
#             st.session_state.query_context = updated_context
#             save_context(updated_context)
#             log_query(query, result)

#     if result is not None:
#         if isinstance(result, pd.DataFrame):
#             st.dataframe(result)
#         elif isinstance(result, (list, pd.Series, pd.Index)):
#             st.write(result)
#         else:
#             st.success(f"Result: {result}")

# else:
#     st.warning("Please upload a dataset first.")


"""Streamlit UI for EchoAnalytics – Phase 6 (Context-Aware + Persistent Memory)."""
import streamlit as st
import pandas as pd

from modules.data_handler import load_dataset
from modules.query_parser import run_query, CONTROL_CLEAR, CONTROL_SHOW_HISTORY
from modules.voice_handler import capture_voice, ERROR_NO_SPEECH, ERROR_SERVICE
from modules.context_manager import (
    load_context, save_context, clear_context,
    log_query, add_history, get_history
)

# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="EchoAnalytics – Phase 6", layout="wide")
st.title("📊 EchoAnalytics — Voice/Text Data Assistant (Phase 6)")

# --------------------
# Load persistent context once
# --------------------
if "query_context" not in st.session_state:
    st.session_state.query_context = load_context()

# --------------------
# Sidebar controls
# --------------------
st.sidebar.header("⚙️ Settings")
use_memory = st.sidebar.toggle("Enable Memory", value=True, help="Turn off to ignore saved context")
if st.sidebar.button("🗑 Clear Memory (Sidebar)"):
    clear_context()
    st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
    st.sidebar.success("Memory cleared.")

# --------------------
# Dataset upload
# --------------------
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xls", "xlsx"])
df = None

if uploaded_file:
    df = load_dataset(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")
    st.markdown("### 🎙 Speak or Type a Question")

    col1, col2, col3 = st.columns([3, 1, 1])

    # Working variables per interaction
    query = ""
    auto_run_from_voice = False
    result = None
    used_context = False
    control = None

    with col1:
        # Keep last spoken query in the box for easy edit
        if "last_voice_query" not in st.session_state:
            st.session_state.last_voice_query = ""
        query = st.text_input(
            "Ask about your data",
            value=st.session_state.last_voice_query,
            placeholder="E.g., 'unique values in proto', 'average sales', 'show history', 'clear memory'"
        )

    with col2:
        if st.button("🎤 Speak"):
            raw = capture_voice()
            if raw in {ERROR_NO_SPEECH, ERROR_SERVICE, None, ""}:
                st.warning(f"Voice input error: {raw}")
            else:
                st.info(f"You said: **{raw}**")
                st.session_state.last_voice_query = raw
                query = raw
                auto_run_from_voice = True  # run via the same unified path below

    with col3:
        if st.button("🗑 Clear Memory (Main)"):
            clear_context()
            st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
            st.success("Memory cleared!")

    run_clicked = st.button("Run Query")

    # --------------------
    # Unified execution path (text or voice)
    # --------------------
    if (run_clicked or auto_run_from_voice) and query.strip():
        context_to_pass = st.session_state.query_context if use_memory else {}
        result, updated_context, control, used_context = run_query(df, query, context_to_pass)

        # Handle control commands
        if control == CONTROL_CLEAR:
            clear_context()
            st.session_state.query_context = {"last_operation": None, "last_column": None, "history": []}
            st.success("Memory cleared!")
        elif control == CONTROL_SHOW_HISTORY:
            st.info("Showing memory history (most recent last):")
            history = get_history(st.session_state.query_context)
            if history:
                st.table(pd.DataFrame(history))
            else:
                st.write("No history yet.")
        else:
            # Normal result path
            if use_memory:
                st.session_state.query_context = updated_context
                save_context(st.session_state.query_context)

            if result is not None:
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, (list, pd.Series, pd.Index)):
                    st.write(result)
                else:
                    st.success(f"Result: {result}")

                if use_memory and used_context:
                    st.caption("🧠 Used previous context (last column/operation).")

                # Logging & compact history
                log_query(query, result, used_context)
                st.session_state.query_context = add_history(
                    st.session_state.query_context, query, repr(result), used_context
                )
                save_context(st.session_state.query_context)

    elif (run_clicked or auto_run_from_voice) and not query.strip():
        st.warning("Please enter or speak a question.")

else:
    st.warning("Please upload a dataset first.")
