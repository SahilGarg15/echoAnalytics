# # modules/data_handler.py
# import pandas as pd

# def load_dataset(uploaded_file):
#     """Loads CSV or Excel into a pandas DataFrame."""
#     if uploaded_file.name.endswith(".csv"):
#         return pd.read_csv(uploaded_file)
#     else:
#         return pd.read_excel(uploaded_file)


# modules/data_handler.py
import pandas as pd

def load_dataset(uploaded_file):
    """Loads CSV or Excel into a pandas DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    # default try csv
    return pd.read_csv(uploaded_file)
