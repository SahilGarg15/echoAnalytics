# # # modules/context_manager.py
# # import json
# # import os
# # from datetime import datetime

# # MEMORY_FILE = "query_memory.json"
# # LOG_FILE = "query_log.csv"

# # def load_context():
# #     """Load last query context and history from file."""
# #     if os.path.exists(MEMORY_FILE):
# #         try:
# #             with open(MEMORY_FILE, "r", encoding="utf-8") as f:
# #                 data = json.load(f)
# #                 # Ensure keys exist
# #                 data.setdefault("last_operation", None)
# #                 data.setdefault("last_column", None)
# #                 data.setdefault("history", [])
# #                 return data
# #         except json.JSONDecodeError:
# #             return {"last_operation": None, "last_column": None, "history": []}
# #     return {"last_operation": None, "last_column": None, "history": []}

# # def save_context(context: dict):
# #     """Save current query context to file."""
# #     with open(MEMORY_FILE, "w", encoding="utf-8") as f:
# #         json.dump(context, f, indent=4, ensure_ascii=False)

# # def clear_context():
# #     """Clear the memory file."""
# #     if os.path.exists(MEMORY_FILE):
# #         os.remove(MEMORY_FILE)

# # def log_query(query, result, used_context: bool):
# #     """Append each query and result to log file."""
# #     header = "timestamp,used_context,query,result\n"
# #     log_entry = f"{datetime.now().isoformat()},{used_context},{repr(query)},{repr(str(result))}\n"

# #     if not os.path.exists(LOG_FILE):
# #         with open(LOG_FILE, "w", encoding="utf-8") as f:
# #             f.write(header)

# #     with open(LOG_FILE, "a", encoding="utf-8") as f:
# #         f.write(log_entry)

# # def add_history(context: dict, query: str, result_repr: str, used_context: bool):
# #     """Store a compact history item inside the persisted context."""
# #     context.setdefault("history", [])
# #     context["history"].append({
# #         "ts": datetime.now().isoformat(timespec="seconds"),
# #         "query": query,
# #         "result": result_repr[:500],  # keep short
# #         "used_context": used_context
# #     })
# #     # Keep last 50 entries to avoid file bloat
# #     if len(context["history"]) > 50:
# #         context["history"] = context["history"][-50:]
# #     return context

# # def get_history(context: dict):
# #     """Return an array of history items."""
# #     return context.get("history", [])


# import json
# import os
# from datetime import datetime

# MEMORY_FILE = "query_memory.json"
# LOG_FILE = "query_log.csv"

# def load_context():
#     """Load last query context from file."""
#     if os.path.exists(MEMORY_FILE):
#         try:
#             with open(MEMORY_FILE, "r") as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             return {}
#     return {}

# def save_context(context):
#     """Save current query context to file."""
#     with open(MEMORY_FILE, "w") as f:
#         json.dump(context, f, indent=4)

# def clear_context():
#     """Clear the memory file."""
#     if os.path.exists(MEMORY_FILE):
#         os.remove(MEMORY_FILE)

# def log_query(query, result):
#     """Append each query and result to log file."""
#     header = "timestamp,query,result\n"
#     log_entry = f"{datetime.now().isoformat()},{repr(query)},{repr(str(result))}\n"

#     if not os.path.exists(LOG_FILE):
#         with open(LOG_FILE, "w", encoding="utf-8") as f:
#             f.write(header)

#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(log_entry)


# modules/context_manager.py
import json
import os
from datetime import datetime

MEMORY_FILE = "query_memory.json"
LOG_FILE = "query_log.csv"

def _default_ctx():
    return {"last_operation": None, "last_column": None, "history": []}

def load_context():
    """Load last query context and history from file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("last_operation", None)
            data.setdefault("last_column", None)
            data.setdefault("history", [])
            return data
        except json.JSONDecodeError:
            return _default_ctx()
    return _default_ctx()

def save_context(context: dict):
    """Save current query context to file."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=4, ensure_ascii=False)

def clear_context():
    """Clear the memory file."""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

def log_query(query, result, used_context: bool):
    """Append each query and result to log file."""
    header = "timestamp,used_context,query,result\n"
    log_entry = f"{datetime.now().isoformat()},{used_context},{repr(query)},{repr(str(result))}\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

def add_history(context: dict, query: str, result_repr: str, used_context: bool):
    """Store a compact history item inside the persisted context."""
    context.setdefault("history", [])
    context["history"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "result": result_repr[:500],
        "used_context": used_context,
    })
    # Keep last 50 entries
    if len(context["history"]) > 50:
        context["history"] = context["history"][-50:]
    return context

def get_history(context: dict):
    """Return an array of history items."""
    return context.get("history", [])
