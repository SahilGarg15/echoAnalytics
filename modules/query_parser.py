# # # # # import pandas as pd
# # # # # import math

# # # # # def run_query(df: pd.DataFrame, query: str):
# # # # #     """Parse a simple natural language query and return the result."""
# # # # #     if not query or not isinstance(query, str):
# # # # #         return "No query provided"

# # # # #     q = query.lower().strip()
# # # # #     result = None

# # # # #     try:
# # # # #         # First N rows
# # # # #         if "first" in q and "rows" in q:
# # # # #             num = [int(s) for s in q.split() if s.isdigit()]
# # # # #             result = df.head(num[0] if num else 5) if not df.empty else "No data available"

# # # # #         # Last N rows
# # # # #         elif "last" in q and "rows" in q:
# # # # #             num = [int(s) for s in q.split() if s.isdigit()]
# # # # #             result = df.tail(num[0] if num else 5) if not df.empty else "No data available"

# # # # #         # List Columns
# # # # #         elif "columns" in q:
# # # # #             result = df.columns.tolist()

# # # # #         # Average / Mean
# # # # #         elif "average" in q or "mean" in q:
# # # # #             for col in df.select_dtypes(include='number').columns:
# # # # #                 if col.lower() in q:
# # # # #                     mean_val = df[col].mean()
# # # # #                     result = "No data available" if math.isnan(mean_val) else mean_val
# # # # #                     break

# # # # #         # Sum
# # # # #         elif "sum" in q or "total" in q:
# # # # #             for col in df.select_dtypes(include='number').columns:
# # # # #                 if col.lower() in q:
# # # # #                     result = df[col].sum() if not df[col].empty else "No data available"
# # # # #                     break

# # # # #         # Maximum
# # # # #         elif "max" in q or "maximum" in q:
# # # # #             for col in df.select_dtypes(include='number').columns:
# # # # #                 if col.lower() in q:
# # # # #                     result = df[col].max() if not df[col].empty else "No data available"
# # # # #                     break

# # # # #         # Minimum
# # # # #         elif "min" in q or "minimum" in q:
# # # # #             for col in df.select_dtypes(include='number').columns:
# # # # #                 if col.lower() in q:
# # # # #                     result = df[col].min() if not df[col].empty else "No data available"
# # # # #                     break

# # # # #         # Unique values
# # # # #         elif "unique" in q or "distinct" in q:
# # # # #             for col in df.columns:
# # # # #                 if col.lower() in q:
# # # # #                     result = df[col].unique().tolist()
# # # # #                     break

# # # # #         # Fallback
# # # # #         if result is None:
# # # # #             result = "Query not understood or data not found"

# # # # #     except Exception as e:
# # # # #         return f"Error: {e}"

# # # # #     return result

# # # # import pandas as pd

# # # # def run_query(df, query, context=None):
# # # #     query = query.strip().lower()
# # # #     result = None
# # # #     updated_context = context.copy() if context else {}

# # # #     col_name = extract_column(query, df, context)

# # # #     # --------------------
# # # #     # Unique Values
# # # #     # --------------------
# # # #     if "unique values" in query:
# # # #         if col_name:
# # # #             result = df[col_name].unique()
# # # #             updated_context.update({"last_operation": "unique_values", "last_column": col_name})

# # # #     # --------------------
# # # #     # Head / First Rows
# # # #     # --------------------
# # # #     elif "show first" in query or "head" in query:
# # # #         result = df.head()
# # # #         updated_context.update({"last_operation": "show_head", "last_column": None})

# # # #     # --------------------
# # # #     # Average
# # # #     # --------------------
# # # #     elif "average" in query or "mean" in query:
# # # #         if col_name and pd.api.types.is_numeric_dtype(df[col_name]):
# # # #             result = df[col_name].mean()
# # # #             updated_context.update({"last_operation": "average", "last_column": col_name})

# # # #     # --------------------
# # # #     # Follow-Up Queries
# # # #     # --------------------
# # # #     else:
# # # #         if context and "last_operation" in context:
# # # #             if context["last_operation"] == "unique_values" and col_name:
# # # #                 result = df[col_name].unique()
# # # #                 updated_context["last_column"] = col_name
# # # #             elif context["last_operation"] == "average" and col_name:
# # # #                 result = df[col_name].mean()
# # # #                 updated_context["last_column"] = col_name
# # # #             else:
# # # #                 result = "Query not understood or data not found."
# # # #         else:
# # # #             result = "Query not understood or data not found."

# # # #     return result, updated_context


# # # # def extract_column(query, df, context=None):
# # # #     """Find column name from query text or memory."""
# # # #     for col in df.columns:
# # # #         if col.lower() in query:
# # # #             return col
# # # #     if context and context.get("last_column"):
# # # #         return context["last_column"]
# # # #     return None


# # # # modules/query_parser.py
# # # import re
# # # import difflib
# # # import pandas as pd

# # # CONTROL_CLEAR = "__CONTROL_CLEAR_MEMORY__"
# # # CONTROL_SHOW_HISTORY = "__CONTROL_SHOW_HISTORY__"

# # # # Lightweight synonyms map (extend over time)
# # # SYNONYMS = {
# # #     "avg": "average",
# # #     "mean": "average",
# # #     "top": "largest",
# # #     "bottom": "smallest",
# # #     "unique": "unique values",
# # #     "uniq": "unique values",
# # #     "first": "head",
# # #     "last": "tail",
# # #     "distinct": "unique values",
# # #     "show": "display",
# # #     "display": "show",
# # #     "rows": "rows",
# # #     "columns": "columns",
# # #     "clear": "forget",
# # #     "forget": "clear",
# # #     "memory": "context",
# # #     "context": "memory",
# # #     "history": "log",
# # #     "log": "history"
# # # }

# # # def normalize_query(raw: str) -> str:
# # #     if not isinstance(raw, str):
# # #         return ""
# # #     q = raw.lower().strip()
# # #     q = re.sub(r"\s+", " ", q)  # collapse spaces
# # #     # apply simple synonyms
# # #     for k, v in SYNONYMS.items():
# # #         q = q.replace(f" {k} ", f" {v} ")
# # #         if q.startswith(k + " "):
# # #             q = q.replace(k + " ", v + " ", 1)
# # #         if q.endswith(" " + k):
# # #             q = q[:-len(k)] + v
# # #     return q

# # # def detect_control_command(q: str) -> str | None:
# # #     if q in {"clear memory", "reset memory", "forget memory"}:
# # #         return CONTROL_CLEAR
# # #     if q in {"show history", "display history", "history"}:
# # #         return CONTROL_SHOW_HISTORY
# # #     return None

# # # def extract_column(query: str, df: pd.DataFrame, context: dict | None = None) -> str | None:
# # #     """Find a column by direct match or fuzzy match; fallback to last_column in memory."""
# # #     # direct containment
# # #     for col in df.columns:
# # #         if col.lower() in query:
# # #             return col
# # #     # fuzzy (best partial ratio via get_close_matches on tokens)
# # #     tokens = query.split()
# # #     lowered_cols = [c.lower() for c in df.columns]
# # #     for tok in tokens:
# # #         match = difflib.get_close_matches(tok, lowered_cols, n=1, cutoff=0.9)
# # #         if match:
# # #             # find original case
# # #             idx = lowered_cols.index(match[0])
# # #             return df.columns[idx]
# # #     # memory fallback
# # #     if context and context.get("last_column"):
# # #         return context["last_column"]
# # #     return None

# # # def run_query(df: pd.DataFrame, raw_query: str, context: dict | None = None):
# # #     """
# # #     Returns: (result, updated_context, control_command or None, used_context_flag)
# # #     - result: computation result or message
# # #     - updated_context: possibly modified context dict
# # #     - control: control token when user asks for 'clear memory' or 'show history'
# # #     - used_context: bool indicating if 'last_column' or prior op were used
# # #     """
# # #     q = normalize_query(raw_query)
# # #     if not q:
# # #         return "No query provided", (context or {}), None, False

# # #     control = detect_control_command(q)
# # #     if control:
# # #         # We don't alter context here; app handles control commands.
# # #         return None, (context or {}), control, False

# # #     updated_context = (context or {}).copy()
# # #     used_context = False
# # #     result = None

# # #     col_name = extract_column(q, df, context)
# # #     if not col_name and context and context.get("last_column"):
# # #         used_context = True  # fell back to memory

# # #     # ---- Unique values ----
# # #     if "unique values" in q:
# # #         if col_name:
# # #             result = pd.unique(df[col_name])
# # #             updated_context.update({"last_operation": "unique_values", "last_column": col_name})
# # #             used_context = used_context or (col_name == (context or {}).get("last_column"))

# # #     # ---- Head / First Rows ----
# # #     elif "show head" in q or "head" in q:
# # #         # Optionally parse "show head N"
# # #         m = re.search(r"head (\d+)", q)
# # #         n = int(m.group(1)) if m else 5
# # #         result = df.head(n)
# # #         updated_context.update({"last_operation": "show_head", "last_column": None})
        
# # #     # ---- Tail / Last Rows ----
# # #     elif "show tail" in q or "tail" in q:
# # #         # Optionally parse "show tail N"
# # #         m = re.search(r"tail (\d+)", q)
# # #         n = int(m.group(1)) if m else 5
# # #         result = df.tail(n)
# # #         updated_context.update({"last_operation": "show_tail", "last_column": None})
        
# # #     # ---- Sum / Total ----
# # #     elif "sum" in q or "total" in q:
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df[col_name].sum()
# # #                 updated_context.update({"last_operation": "sum", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #             result = "Which column should I sum?"
            
# # #     # ---- Largest / Top N ----
# # #     elif "largest" in q or "top" in q:
# # #         # Optionally parse "largest N" or "top N"
# # #         m = re.search(r"(largest|top) (\d+)", q)
# # #         n = int(m.group(2)) if m else 5
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df.nlargest(n, col_name)
# # #                 updated_context.update({"last_operation": "largest", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #                 result = "Which column should I find the largest values in?"
# # #     # ---- Smallest / Bottom N ----
# # #     elif "smallest" in q or "bottom" in q:
# # #         # Optionally parse "smallest N" or "bottom N"
# # #         m = re.search(r"(smallest|bottom) (\d+)", q)
# # #         n = int(m.group(2)) if m else 5
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df.nsmallest(n, col_name)
# # #                 updated_context.update({"last_operation": "smallest", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #             result = "Which column should I find the smallest values in?"
            
# # #     # ---- Count / Size ----
# # #     elif "count" in q or "size" in q:
# # #         if col_name:
# # #             result = df[col_name].count()
# # #             updated_context.update({"last_operation": "count", "last_column": col_name})
# # #             used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #         else:
# # #             result = "Which column should I count?"
# # #     elif "size" in q:
# # #         if col_name:
# # #             result = df[col_name].size
# # #             updated_context.update({"last_operation": "size", "last_column": col_name})
# # #             used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #         else:
# # #             result = "Which column should I get the size of?"

# # #     # ---- Average / Mean ----
# # #     elif "average" in q:
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df[col_name].mean()
# # #                 updated_context.update({"last_operation": "average", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #             result = "Which column should I average?"

# # #     # ---- Max ----
# # #     elif "max" in q or "maximum" in q:
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df[col_name].max()
# # #                 updated_context.update({"last_operation": "max", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #             result = "Which column do you want the maximum from?"

# # #     # ---- Min ----
# # #     elif "min" in q or "minimum" in q:
# # #         if col_name:
# # #             if pd.api.types.is_numeric_dtype(df[col_name]):
# # #                 result = df[col_name].min()
# # #                 updated_context.update({"last_operation": "min", "last_column": col_name})
# # #                 used_context = used_context or (col_name == (context or {}).get("last_column"))
# # #             else:
# # #                 result = f"Column '{col_name}' is not numeric."
# # #         else:
# # #             result = "Which column do you want the minimum from?"
            
    
# # #     # ---- Fallback for other queries ----
# # #     elif "show columns" in q or "list columns" in q:
# # #         result = df.columns.tolist()
# # #         updated_context.update({"last_operation": "list_columns", "last_column": None})
# # #         used_context = True
# # #     elif "show rows" in q or "list rows" in q:
# # #         result = df.index.tolist()
# # #         updated_context.update({"last_operation": "list_rows", "last_column": None})
# # #         used_context = True
# # #     elif "show first" in q or "first rows" in q:
# # #         m = re.search(r"first (\d+)", q)
# # #         n = int(m.group(1)) if m else 5
# # #         result = df.head(n)
# # #         updated_context.update({"last_operation": "show_first", "last_column": None})
# # #         used_context = True
# # #     elif "show last" in q or "last rows" in q:
# # #         m = re.search(r"last (\d+)", q)
# # #         n = int(m.group(1)) if m else 5
# # #         result = df.tail(n)
# # #         updated_context.update({"last_operation": "show_last", "last_column": None})
# # #         used_context = True
# # #     elif "show history" in q or "display history" in q:
# # #         control = CONTROL_SHOW_HISTORY
# # #         result = None
# # #     elif "clear memory" in q or "forget memory" in q:
# # #         control = CONTROL_CLEAR
# # #         result = None
# # #     elif "clear context" in q or "forget context" in q:
# # #         control = CONTROL_CLEAR
# # #         result = None

# # #     # ---- Fallback follow-ups ----
# # #     else:
# # #         if context and context.get("last_operation"):
# # #             # try to reuse op with new column if provided
# # #             last_op = context["last_operation"]
# # #             if last_op == "unique_values" and col_name:
# # #                 result = pd.unique(df[col_name])
# # #                 updated_context.update({"last_operation": "unique_values", "last_column": col_name})
# # #                 used_context = True
# # #             elif last_op in {"average", "max", "min"} and col_name:
# # #                 if not pd.api.types.is_numeric_dtype(df[col_name]):
# # #                     result = f"Column '{col_name}' is not numeric."
# # #                 else:
# # #                     if last_op == "average":
# # #                         result = df[col_name].mean()
# # #                     elif last_op == "max":
# # #                         result = df[col_name].max()
# # #                     else:
# # #                         result = df[col_name].min()
# # #                     updated_context.update({"last_operation": last_op, "last_column": col_name})
# # #                     used_context = True
# # #             else:
# # #                 result = "Query not understood or data not found."
# # #         else:
# # #             result = "Query not understood or data not found."

# # #     return result, updated_context, None, used_context


# # import pandas as pd

# # def run_query(df, query, context=None):
# #     query = query.strip().lower()
# #     result = None
# #     updated_context = context.copy() if context else {}

# #     col_name = extract_column(query, df, context)

# #     # --------------------
# #     # Unique Values
# #     # --------------------
# #     if "unique values" in query:
# #         if col_name:
# #             result = df[col_name].unique()
# #             updated_context.update({"last_operation": "unique_values", "last_column": col_name})

# #     # --------------------
# #     # Head / First Rows
# #     # --------------------
# #     elif "show first" in query or "head" in query:
# #         result = df.head()
# #         updated_context.update({"last_operation": "show_head", "last_column": None})

# #     # --------------------
# #     # Average
# #     # --------------------
# #     elif "average" in query or "mean" in query:
# #         if col_name and pd.api.types.is_numeric_dtype(df[col_name]):
# #             result = df[col_name].mean()
# #             updated_context.update({"last_operation": "average", "last_column": col_name})

# #     # --------------------
# #     # Follow-Up Queries
# #     # --------------------
# #     else:
# #         if context and "last_operation" in context:
# #             if context["last_operation"] == "unique_values" and col_name:
# #                 result = df[col_name].unique()
# #                 updated_context["last_column"] = col_name
# #             elif context["last_operation"] == "average" and col_name:
# #                 result = df[col_name].mean()
# #                 updated_context["last_column"] = col_name
# #             else:
# #                 result = "Query not understood or data not found."
# #         else:
# #             result = "Query not understood or data not found."

# #     return result, updated_context


# # def extract_column(query, df, context=None):
# #     """Find column name from query text or memory."""
# #     for col in df.columns:
# #         if col.lower() in query:
# #             return col
# #     if context and context.get("last_column"):
# #         return context["last_column"]
# #     return None


# # modules/query_parser.py
# import re
# import difflib
# import pandas as pd
# from typing import Tuple, Any, Dict, Optional

# CONTROL_CLEAR = "__CONTROL_CLEAR_MEMORY__"
# CONTROL_SHOW_HISTORY = "__CONTROL_SHOW_HISTORY__"

# # Lightweight synonyms map (extend over time)
# SYNONYMS = {
#     "avg": "average",
#     "mean": "average",
#     "top": "largest",
#     "bottom": "smallest",
#     "unique": "unique values",
#     "uniq": "unique values",
#     "first": "head",
#     "last": "tail",
#     "distinct": "unique values",
#     "show": "display",
#     "display": "show",
#     "cols": "columns",
#     "rows": "rows",
#     "clear": "forget",
#     "forget": "clear",
#     "memory": "context",
#     "context": "memory",
#     "history": "log",
#     "log": "history",
# }

# def normalize_query(raw: str) -> str:
#     if not isinstance(raw, str):
#         return ""
#     q = raw.lower().strip()
#     q = re.sub(r"\s+", " ", q)
#     # simple replacements (word-boundary-ish handling)
#     for k, v in SYNONYMS.items():
#         q = q.replace(f" {k} ", f" {v} ")
#         if q.startswith(k + " "):
#             q = q.replace(k + " ", v + " ", 1)
#         if q.endswith(" " + k):
#             q = q[: -len(k)] + v
#     return q

# def detect_control_command(q: str) -> Optional[str]:
#     # Normalize exact phrases
#     if q in {"clear memory", "reset memory", "forget memory", "clear context", "forget context"}:
#         return CONTROL_CLEAR
#     if q in {"show history", "display history", "history"}:
#         return CONTROL_SHOW_HISTORY
#     return None

# def extract_column(query: str, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
#     """Find a column by direct match or fuzzy match; fallback to last_column in memory."""
#     # direct containment
#     for col in df.columns:
#         if col.lower() in query:
#             return col
#     # fuzzy token matching
#     tokens = query.split()
#     lowered_cols = [c.lower() for c in df.columns]
#     for tok in tokens:
#         match = difflib.get_close_matches(tok, lowered_cols, n=1, cutoff=0.9)
#         if match:
#             idx = lowered_cols.index(match[0])
#             return df.columns[idx]
#     # memory fallback
#     if context and context.get("last_column"):
#         return context["last_column"]
#     return None

# def run_query(
#     df: pd.DataFrame,
#     raw_query: str,
#     context: Optional[Dict[str, Any]] = None
# ) -> Tuple[Any, Dict[str, Any], Optional[str], bool]:
#     """
#     Returns: (result, updated_context, control_command or None, used_context_flag)
#     """
#     q = normalize_query(raw_query)
#     if not q:
#         return "No query provided", (context or {}), None, False

#     # Control commands
#     control = detect_control_command(q)
#     if control:
#         return None, (context or {}), control, False

#     updated_context = (context or {}).copy()
#     used_context = False
#     result: Any = None

#     col_name = extract_column(q, df, context)
#     if not col_name and context and context.get("last_column"):
#         used_context = True  # fell back to memory

#     # ---------- Operations ----------
#     # Unique values
#     if "unique values" in q:
#         if col_name:
#             result = pd.unique(df[col_name])
#             updated_context.update({"last_operation": "unique_values", "last_column": col_name})
#             used_context = used_context or (col_name == (context or {}).get("last_column"))

#     # Head / first N rows
#     elif "show head" in q or "head" in q or "show first" in q or "first rows" in q:
#         m = re.search(r"(head|first) (\d+)", q)
#         n = int(m.group(2)) if m else 5
#         result = df.head(n)
#         updated_context.update({"last_operation": "show_head", "last_column": None})

#     # Tail / last N rows
#     elif "show tail" in q or "tail" in q or "show last" in q or "last rows" in q:
#         m = re.search(r"(tail|last) (\d+)", q)
#         n = int(m.group(2)) if m else 5
#         result = df.tail(n)
#         updated_context.update({"last_operation": "show_tail", "last_column": None})

#     # List columns
#     elif "show columns" in q or "list columns" in q or q.strip() == "columns":
#         result = df.columns.tolist()
#         updated_context.update({"last_operation": "list_columns", "last_column": None})
#         used_context = False

#     # Count / Size
#     elif "count" in q or "size" in q:
#         if col_name:
#             result = df[col_name].count() if "count" in q else df[col_name].size
#             updated_context.update({"last_operation": "count", "last_column": col_name})
#             used_context = used_context or (col_name == (context or {}).get("last_column"))
#         else:
#             result = "Which column should I count/size?"

#     # Sum / Total
#     elif "sum" in q or "total" in q:
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df[col_name].sum()
#                 updated_context.update({"last_operation": "sum", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column should I sum?"

#     # Average / Mean
#     elif "average" in q:
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df[col_name].mean()
#                 updated_context.update({"last_operation": "average", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column should I average?"

#     # Max
#     elif "max" in q or "maximum" in q:
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df[col_name].max()
#                 updated_context.update({"last_operation": "max", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column do you want the maximum from?"

#     # Min
#     elif "min" in q or "minimum" in q:
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df[col_name].min()
#                 updated_context.update({"last_operation": "min", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column do you want the minimum from?"

#     # Largest / Top N
#     elif "largest" in q or "top" in q:
#         m = re.search(r"(largest|top) (\d+)", q)
#         n = int(m.group(2)) if m else 5
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df.nlargest(n, col_name)
#                 updated_context.update({"last_operation": "largest", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column should I find the largest values in?"

#     # Smallest / Bottom N
#     elif "smallest" in q or "bottom" in q:
#         m = re.search(r"(smallest|bottom) (\d+)", q)
#         n = int(m.group(2)) if m else 5
#         if col_name:
#             if pd.api.types.is_numeric_dtype(df[col_name]):
#                 result = df.nsmallest(n, col_name)
#                 updated_context.update({"last_operation": "smallest", "last_column": col_name})
#                 used_context = used_context or (col_name == (context or {}).get("last_column"))
#             else:
#                 result = f"Column '{col_name}' is not numeric."
#         else:
#             result = "Which column should I find the smallest values in?"

#     # ---------- Follow-up fallback ----------
#     else:
#         if context and context.get("last_operation"):
#             last_op = context["last_operation"]
#             if last_op == "unique_values" and col_name:
#                 result = pd.unique(df[col_name])
#                 updated_context.update({"last_operation": "unique_values", "last_column": col_name})
#                 used_context = True
#             elif last_op in {"average", "max", "min"} and col_name:
#                 if not pd.api.types.is_numeric_dtype(df[col_name]):
#                     result = f"Column '{col_name}' is not numeric."
#                 else:
#                     if last_op == "average":
#                         result = df[col_name].mean()
#                     elif last_op == "max":
#                         result = df[col_name].max()
#                     else:
#                         result = df[col_name].min()
#                     updated_context.update({"last_operation": last_op, "last_column": col_name})
#                     used_context = True
#             else:
#                 result = "Query not understood or data not found."
#         else:
#             result = "Query not understood or data not found."

#     return result, updated_context, None, used_context


# modules/query_parser.py
import re
import difflib
import pandas as pd

CONTROL_CLEAR = "__CONTROL_CLEAR_MEMORY__"
CONTROL_SHOW_HISTORY = "__CONTROL_SHOW_HISTORY__"

# Lightweight synonyms map
SYNONYMS = {
    "avg": "average",
    "mean": "average",
    "top": "largest",
    "bottom": "smallest",
    "unique": "unique values",
    "uniq": "unique values",
    "first": "head",
    "last": "tail",
    "distinct": "unique values",
    "show": "display",
    "display": "show",
    "clear": "forget",
    "forget": "clear",
    "history": "log",
    "log": "history",
}

def normalize_query(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    q = raw.lower().strip()
    q = re.sub(r"\s+", " ", q)  # collapse spaces
    # apply simple synonyms (boundary-aware-ish)
    for k, v in SYNONYMS.items():
        q = q.replace(f" {k} ", f" {v} ")
        if q.startswith(k + " "):
            q = q.replace(k + " ", v + " ", 1)
        if q.endswith(" " + k):
            q = q[:-len(k)] + v
    return q

def detect_control_command(q: str):
    # exact phrases are fine for now
    if q in {"clear memory", "reset memory", "forget memory", "clear context", "forget context"}:
        return CONTROL_CLEAR
    if q in {"show history", "display history", "history", "show log", "display log"}:
        return CONTROL_SHOW_HISTORY
    return None

def extract_column(query: str, df: pd.DataFrame, context: dict | None = None) -> str | None:
    """Find a column by direct substring or fuzzy match; fallback to last_column in memory."""
    # direct containment
    for col in df.columns:
        if col.lower() in query:
            return col
    # fuzzy by tokens
    tokens = query.split()
    lowered_cols = [c.lower() for c in df.columns]
    for tok in tokens:
        match = difflib.get_close_matches(tok, lowered_cols, n=1, cutoff=0.9)
        if match:
            idx = lowered_cols.index(match[0])
            return df.columns[idx]
    # memory fallback
    if context and context.get("last_column"):
        return context["last_column"]
    return None

def _set_ctx(ctx: dict, op: str | None, col: str | None):
    ctx["last_operation"] = op
    ctx["last_column"] = col

def run_query(df: pd.DataFrame, raw_query: str, context: dict | None = None):
    """
    Returns: (result, updated_context, control or None, used_context: bool)
    """
    q = normalize_query(raw_query)
    if not q:
        return "No query provided", (context or {"last_operation": None, "last_column": None, "history": []}), None, False

    control = detect_control_command(q)
    if control:
        # Let app.py handle control actions; do not mutate here
        return None, (context or {"last_operation": None, "last_column": None, "history": []}), control, False

    updated_context = (context or {"last_operation": None, "last_column": None, "history": []}).copy()
    used_context = False
    result = None

    col_name = extract_column(q, df, context)
    if not col_name and context and context.get("last_column"):
        used_context = True

    # --- common helpers for parsing N ---
    def _parse_n(pattern, default=5):
        m = re.search(pattern, q)
        return int(m.group(1)) if m else default

    # ---- Unique values ----
    if "unique values" in q or "unique" in q or "distinct" in q:
        if col_name:
            result = pd.unique(df[col_name])
            _set_ctx(updated_context, "unique_values", col_name)
            used_context = used_context or (col_name == (context or {}).get("last_column"))

    # ---- Head / First Rows ----
    elif "show head" in q or "head" in q or "show first" in q or "first rows" in q:
        n = _parse_n(r"(?:head|first)\s+(\d+)", default=5)
        result = df.head(n)
        _set_ctx(updated_context, "show_head", None)

    # ---- Tail / Last Rows ----
    elif "show tail" in q or "tail" in q or "show last" in q or "last rows" in q:
        n = _parse_n(r"(?:tail|last)\s+(\d+)", default=5)
        result = df.tail(n)
        _set_ctx(updated_context, "show_tail", None)

    # ---- Sum / Total ----
    elif "sum" in q or "total" in q:
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df[col_name].sum()
                _set_ctx(updated_context, "sum", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column should I sum?"

    # ---- Largest / Top N ----
    elif "largest" in q or "top" in q:
        n = _parse_n(r"(?:largest|top)\s+(\d+)", default=5)
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df.nlargest(n, col_name)
                _set_ctx(updated_context, "largest", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column should I find the largest values in?"

    # ---- Smallest / Bottom N ----
    elif "smallest" in q or "bottom" in q:
        n = _parse_n(r"(?:smallest|bottom)\s+(\d+)", default=5)
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df.nsmallest(n, col_name)
                _set_ctx(updated_context, "smallest", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column should I find the smallest values in?"

    # ---- Count / Size ----
    elif "count" in q or "size" in q:
        if col_name:
            result = df[col_name].count()
            _set_ctx(updated_context, "count", col_name)
            used_context = used_context or (col_name == (context or {}).get("last_column"))
        else:
            result = "Which column should I count?"

    # ---- Average / Mean ----
    elif "average" in q:
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df[col_name].mean()
                _set_ctx(updated_context, "average", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column should I average?"

    # ---- Max ----
    elif "max" in q or "maximum" in q:
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df[col_name].max()
                _set_ctx(updated_context, "max", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column do you want the maximum from?"

    # ---- Min ----
    elif "min" in q or "minimum" in q:
        if col_name:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                result = df[col_name].min()
                _set_ctx(updated_context, "min", col_name)
                used_context = used_context or (col_name == (context or {}).get("last_column"))
            else:
                result = f"Column '{col_name}' is not numeric."
        else:
            result = "Which column do you want the minimum from?"

    # ---- Meta / Listings ----
    elif "show columns" in q or "list columns" in q:
        result = df.columns.tolist()
        _set_ctx(updated_context, "list_columns", None)
        used_context = True
    elif "show rows" in q or "list rows" in q:
        result = df.index.tolist()
        _set_ctx(updated_context, "list_rows", None)
        used_context = True

    # ---- Follow-ups ----
    else:
        if context and context.get("last_operation"):
            last_op = context["last_operation"]

            if last_op == "unique_values" and col_name:
                result = pd.unique(df[col_name])
                _set_ctx(updated_context, "unique_values", col_name)
                used_context = True

            elif last_op in {"average", "max", "min", "sum"} and col_name:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    result = f"Column '{col_name}' is not numeric."
                else:
                    if last_op == "average":
                        result = df[col_name].mean()
                    elif last_op == "max":
                        result = df[col_name].max()
                    elif last_op == "min":
                        result = df[col_name].min()
                    else:  # sum
                        result = df[col_name].sum()
                    _set_ctx(updated_context, last_op, col_name)
                    used_context = True
            else:
                result = "Query not understood or data not found."
        else:
            result = "Query not understood or data not found."

    return result, updated_context, None, used_context
