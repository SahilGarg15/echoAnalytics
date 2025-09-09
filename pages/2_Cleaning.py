# import streamlit as st
# import pandas as pd
# from modules import preprocessing as prep

# st.title("🧹 Data Cleaning & Preprocessing")

# # Check if dataset exists
# if "df" not in st.session_state:
#     st.warning("⚠ Please upload a dataset from the Home tab first.")
#     st.stop()

# df = st.session_state["df"]

# st.subheader("Preview of Current Dataset")
# st.dataframe(df.head())

# # Missing values
# st.markdown("### Handle Missing Values")
# missing_strategy = st.selectbox(
#     "Choose strategy:",
#     ["none", "drop", "mean", "median", "mode", "constant"]
# )
# fill_value = None
# if missing_strategy == "constant":
#     fill_value = st.text_input("Enter constant fill value:")

# if missing_strategy != "none":
#     df = prep.handle_missing(df, strategy=missing_strategy, fill_value=fill_value)

# # Outliers
# st.markdown("### Handle Outliers")
# if st.checkbox("Remove outliers"):
#     method = st.radio("Method:", ["zscore", "iqr"])
#     threshold = st.number_input("Z-score threshold (for zscore)", 1.0, 10.0, 3.0)
#     df = prep.remove_outliers(df, method=method, threshold=threshold)

# # Convert dtypes
# st.markdown("### Convert Column Types")
# col_to_convert = st.multiselect("Select columns:", df.columns)
# dtype_map = {}
# for col in col_to_convert:
#     dtype = st.selectbox(f"Convert {col} to:", ["int", "float", "str", "datetime"])
#     dtype_map[col] = dtype

# if st.button("Convert Types"):
#     df = prep.convert_dtypes(df, dtype_map)

# # Rename columns
# st.markdown("### Rename Columns")
# rename_map = {}
# for col in df.columns:
#     new_name = st.text_input(f"Rename {col} to:", col)
#     if new_name != col:
#         rename_map[col] = new_name

# if rename_map:
#     df = prep.rename_columns(df, rename_map)

# # Normalize units
# st.markdown("### Normalize Units")
# col_norm = st.selectbox("Select numeric column to normalize:", df.select_dtypes(include=['number']).columns)
# factor = st.number_input("Multiply by factor:", value=1.0)
# if st.button("Apply Normalization"):
#     df = prep.normalize_units(df, col_norm, factor)

# # Save dataset
# st.markdown("### Save Cleaned Dataset")
# fmt = st.radio("Format:", ["csv", "excel", "json"])
# if st.button("Save"):
#     path = prep.save_cleaned(df, path=f"cleaned.{fmt}", format=fmt)
#     st.success(f"Dataset saved as {path}")
#     st.download_button("Download File", open(path, "rb"), file_name=f"cleaned.{fmt}")

# # Update session state
# st.session_state["df"] = df

# st.success("✅ Cleaning applied successfully!")

# """Streamlit UI for EchoAnalytics – Phase 7 (Cleaning Tab)."""
# import streamlit as st
# import pandas as pd
# from modules import preprocessing as pp

# if st.button("⬅️ Back to Home"):
#     st.switch_page("app.py")

# st.title("🧹 Data Cleaning")

# # Ensure dataset exists
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload one in the **Home tab** first.")
# else:
#     df = st.session_state["df"]

#     st.subheader("📄 Current Dataset Preview")
#     st.dataframe(df.head())

#     st.markdown("---")

#     # --------------------
#     # Cleaning Operations
#     # --------------------
#     st.header("⚙️ Cleaning Options")

#     # Missing values
#     st.subheader("1. Handle Missing Values")
#     missing_method = st.radio("Choose method:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
#     if missing_method != "None":
#         if st.button("Apply Missing Value Handling"):
#             df = pp.handle_missing_values(df, method=missing_method)
#             st.session_state["df"] = df
#             st.success(f"Missing values handled using: {missing_method}")
#             st.dataframe(df.head())

#     st.markdown("---")

#     # Outliers
#     st.subheader("2. Handle Outliers")
#     outlier_col = st.selectbox("Select column for outlier removal", ["None"] + list(df.select_dtypes(include="number").columns))
#     if outlier_col != "None":
#         if st.button("Remove Outliers"):
#             df = pp.remove_outliers(df, column=outlier_col)
#             st.session_state["df"] = df
#             st.success(f"Outliers removed from column: {outlier_col}")
#             st.dataframe(df.head())

#     st.markdown("---")

#     # Data type conversion
#     st.subheader("3. Convert Data Type")
#     dtype_col = st.selectbox("Select column to convert", ["None"] + list(df.columns))
#     new_type = st.selectbox("Convert to type", ["None", "int", "float", "str"])
#     if dtype_col != "None" and new_type != "None":
#         if st.button("Convert Type"):
#             df = pp.convert_dtype(df, dtype_col, new_type)
#             st.session_state["df"] = df
#             st.success(f"Column {dtype_col} converted to {new_type}")
#             st.dataframe(df.head())

#     st.markdown("---")

#     # Rename columns
#     st.subheader("4. Rename Column")
#     col_to_rename = st.selectbox("Select column", ["None"] + list(df.columns))
#     new_name = st.text_input("Enter new column name")
#     if col_to_rename != "None" and new_name.strip():
#         if st.button("Rename Column"):
#             df = pp.rename_column(df, col_to_rename, new_name.strip())
#             st.session_state["df"] = df
#             st.success(f"Column {col_to_rename} renamed to {new_name}")
#             st.dataframe(df.head())

#     st.markdown("---")

#     # Normalization
#     st.subheader("5. Normalize Column")
#     norm_col = st.selectbox("Select column for normalization", ["None"] + list(df.select_dtypes(include="number").columns))
#     if norm_col != "None":
#         if st.button("Normalize Column"):
#             df = pp.normalize_column(df, norm_col)
#             st.session_state["df"] = df
#             st.success(f"Column {norm_col} normalized (0–1 scale).")
#             st.dataframe(df.head())

#     st.markdown("---")

#     # Save/export
#     st.subheader("6. Export Cleaned Data")
#     export_format = st.selectbox("Choose format", ["CSV", "Excel"])
#     if st.button("Download Cleaned Data"):
#         if export_format == "CSV":
#             st.download_button("Download CSV", df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
#         else:
#             from io import BytesIO
#             buffer = BytesIO()
#             df.to_excel(buffer, index=False, engine="openpyxl")
#             st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel")

# """Streamlit UI for EchoAnalytics – Enhanced Cleaning & Preprocessing Tab."""
# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from modules import preprocessing as pp

# st.title("🧹 Data Cleaning & Preprocessing")

# # --------------------
# # Back to Home
# # --------------------
# if st.button("⬅️ Back to Home"):
#     st.switch_page("app.py")

# # --------------------
# # Dataset existence
# # --------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload one in the **Home tab** first.")
#     st.stop()

# df = st.session_state["df"]

# # --------------------
# # Step 0: Dataset Overview
# # --------------------
# st.subheader("📄 Dataset Overview")
# st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
# st.dataframe(df.head())

# if st.checkbox("Show Column Info / Basic Statistics"):
#     st.write(df.info())
#     st.write(df.describe(include="all"))
#     st.write("Missing values per column:")
#     st.write(df.isna().sum())

# # --------------------
# # Step 1: Cleaning / Preprocessing Options
# # --------------------
# st.subheader("⚙️ Cleaning & Preprocessing Options")

# # Missing Values
# st.markdown("### 1. Handle Missing Values")
# missing_method = st.selectbox("Choose method:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
# if missing_method != "None" and st.button("Apply Missing Values Handling"):
#     df = pp.handle_missing_values(df, method=missing_method)
#     st.session_state["df"] = df
#     st.success(f"Missing values handled using: {missing_method}")
#     st.dataframe(df.head())

# # Outliers
# st.markdown("### 2. Handle Outliers")
# outlier_col = st.selectbox("Select column for outlier removal", ["None"] + list(df.select_dtypes(include="number").columns))
# if outlier_col != "None" and st.button("Remove Outliers"):
#     df = pp.remove_outliers(df, column=outlier_col)
#     st.session_state["df"] = df
#     st.success(f"Outliers removed from column: {outlier_col}")
#     st.dataframe(df.head())

# # Data Type Conversion
# st.markdown("### 3. Convert Data Type")
# dtype_col = st.selectbox("Select column to convert", ["None"] + list(df.columns))
# new_type = st.selectbox("Convert to type", ["None", "int", "float", "str"])
# if dtype_col != "None" and new_type != "None" and st.button("Convert Type"):
#     df = pp.convert_dtype(df, dtype_col, new_type)
#     st.session_state["df"] = df
#     st.success(f"Column {dtype_col} converted to {new_type}")
#     st.dataframe(df.head())

# # Rename / Drop Columns
# st.markdown("### 4. Rename / Drop Columns")
# col_to_rename = st.selectbox("Rename column", ["None"] + list(df.columns))
# new_name = st.text_input("New column name")
# if col_to_rename != "None" and new_name.strip() and st.button("Rename Column"):
#     df = pp.rename_column(df, col_to_rename, new_name.strip())
#     st.session_state["df"] = df
#     st.success(f"Column {col_to_rename} renamed to {new_name}")
#     st.dataframe(df.head())

# col_to_drop = st.multiselect("Drop columns (optional)", options=df.columns)
# if col_to_drop and st.button("Drop Selected Columns"):
#     df = df.drop(columns=col_to_drop)
#     st.session_state["df"] = df
#     st.success(f"Dropped columns: {col_to_drop}")
#     st.dataframe(df.head())

# # Normalization / Scaling
# st.markdown("### 5. Normalize / Scale Columns")
# norm_col = st.selectbox("Select column to normalize", ["None"] + list(df.select_dtypes(include="number").columns))
# if norm_col != "None" and st.button("Normalize Column"):
#     df = pp.normalize_column(df, norm_col)
#     st.session_state["df"] = df
#     st.success(f"Column {norm_col} normalized (0–1 scale).")
#     st.dataframe(df.head())

# # --------------------
# # Step 2: Train/Test Split
# # --------------------
# st.subheader("📊 Train/Test Split (Optional)")
# if st.checkbox("Split dataset for ML"):
#     target_col = st.selectbox("Select target column", ["None"] + list(df.columns))
#     test_size = st.slider("Test size fraction", min_value=0.1, max_value=0.5, value=0.2)
#     random_state = st.number_input("Random seed (optional)", min_value=0, value=42, step=1)
#     if st.button("Perform Train/Test Split"):
#         if target_col != "None":
#             train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
#             st.session_state["train_df"] = train_df
#             st.session_state["test_df"] = test_df
#             st.success(f"Dataset split into Train ({len(train_df)}) / Test ({len(test_df)}) rows")
#             st.write("Train sample:")
#             st.dataframe(train_df.head())
#             st.write("Test sample:")
#             st.dataframe(test_df.head())
#         else:
#             st.warning("Please select a target column for splitting.")

# # --------------------
# # Step 3: Export / Save Cleaned Data
# # --------------------
# st.subheader("💾 Export Cleaned Dataset")
# export_format = st.selectbox("Choose format", ["CSV", "Excel"])
# if st.button("Download Cleaned Data"):
#     if export_format == "CSV":
#         st.download_button("Download CSV", df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
#     else:
#         from io import BytesIO
#         buffer = BytesIO()
#         df.to_excel(buffer, index=False, engine="openpyxl")
#         st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel")

# import streamlit as st
# import pandas as pd
# from modules import preprocessing as pp
# from modules import visualizations as vz
# from modules import pipeline_manager as pm
# from sklearn.model_selection import train_test_split
# from modules import visual_analysis as va

# st.title("🧹 Data Cleaning & Preprocessing")

# # -----------------------------
# # Ensure dataset exists
# # -----------------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload in Home tab.")
# else:
#     df = st.session_state["df"]
#     pm.init_pipeline()

#     # -----------------------------
#     # Dataset preview
#     # -----------------------------
#     with st.expander("📄 Current Dataset Preview", expanded=True):
#         st.dataframe(df.head())
        
    
#     # --------------------
# # Visual Analysis
# # --------------------
#     if df is not None:
#         va.analyze_dataset(df)


#     # # -----------------------------
#     # # Visual Analysis
#     # # -----------------------------
#     # st.header("📊 Visual Analysis")
#     # col_to_plot = st.selectbox("Select column for visualization", ["None"] + list(df.columns))
#     # if col_to_plot != "None":
#     #     plot_type = st.radio("Plot type", ["Histogram", "Boxplot", "Bar Plot"])
#     #     if st.button("Generate Plot"):
#     #         if plot_type=="Histogram": vz.plot_histogram(df, col_to_plot)
#     #         elif plot_type=="Boxplot": vz.plot_boxplot(df, col_to_plot)
#     #         else: vz.plot_bar(df, col_to_plot)

#     # if st.button("Correlation Heatmap"):
#     #     vz.plot_correlation_heatmap(df)

#     # st.markdown("---")
#     # st.header("⚙️ Cleaning Operations")

#     # -----------------------------
#     # Missing values
#     # -----------------------------
#     missing_method = st.selectbox("Handle Missing Values", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
#     if missing_method != "None" and st.button("Apply Missing Values"):
#         df = pp.handle_missing_values(df, missing_method)
#         pm.add_step(f"Missing: {missing_method}", df)
#         st.session_state.df = df
#         st.success(f"Applied missing value handling: {missing_method}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Outliers
#     # -----------------------------
#     numeric_cols = list(df.select_dtypes(include="number").columns)
#     outlier_col = st.selectbox("Remove Outliers Column", ["None"] + numeric_cols)
#     if outlier_col != "None" and st.button("Remove Outliers"):
#         df = pp.remove_outliers(df, outlier_col)
#         pm.add_step(f"Outliers: {outlier_col}", df)
#         st.session_state.df = df
#         st.success(f"Outliers removed from {outlier_col}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Data type conversion
#     # -----------------------------
#     dtype_col = st.selectbox("Convert Column Type", ["None"] + list(df.columns))
#     new_type = st.selectbox("Convert to type", ["None", "int", "float", "str"])
#     if dtype_col!="None" and new_type!="None" and st.button("Convert Type"):
#         df = pp.convert_dtype(df, dtype_col, new_type)
#         pm.add_step(f"Convert {dtype_col} to {new_type}", df)
#         st.session_state.df = df
#         st.success(f"{dtype_col} converted to {new_type}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Rename columns
#     # -----------------------------
#     col_to_rename = st.selectbox("Rename Column", ["None"] + list(df.columns))
#     new_name = st.text_input("New name")
#     if col_to_rename!="None" and new_name.strip() and st.button("Rename Column"):
#         df = pp.rename_column(df, col_to_rename, new_name.strip())
#         pm.add_step(f"Rename {col_to_rename} → {new_name.strip()}", df)
#         st.session_state.df = df
#         st.success(f"{col_to_rename} renamed to {new_name.strip()}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Normalize / scale
#     # -----------------------------
#     norm_col = st.selectbox("Normalize Column", ["None"] + numeric_cols)
#     scale_method = st.selectbox("Scaling Method", ["MinMax", "Standard", "Robust"])
#     if norm_col!="None" and st.button("Apply Scaling"):
#         df = pp.normalize_column(df, norm_col, scale_method)
#         pm.add_step(f"Scale {norm_col} ({scale_method})", df)
#         st.session_state.df = df
#         st.success(f"{norm_col} scaled using {scale_method}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Train/Test Split
#     # -----------------------------
#     st.markdown("---")
#     st.header("🔀 Train/Test Split")
#     split_col = st.selectbox("Column for Split (optional)", ["None"] + list(df.columns))
#     test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
#     if st.button("Preview Train/Test Split"):
#         if split_col=="None":
#             train, test = train_test_split(df, test_size=test_size, random_state=42)
#         else:
#             train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#         st.subheader("Train Sample")
#         st.dataframe(train.head())
#         st.subheader("Test Sample")
#         st.dataframe(test.head())

#     # -----------------------------
#     # Undo last step
#     # -----------------------------
#     if st.button("↩️ Undo Last Step"):
#         pm.undo_last_step()
#         st.dataframe(st.session_state.df.head())

#     # -----------------------------
#     # Export cleaned dataset
#     # -----------------------------
#     st.markdown("---")
#     st.header("💾 Export Cleaned Data")
#     export_format = st.selectbox("Format", ["CSV", "Excel"])
#     if st.button("Download Cleaned Data"):
#         df_final = st.session_state.df
#         if export_format=="CSV":
#             st.download_button("Download CSV", df_final.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
#         else:
#             from io import BytesIO
#             buffer = BytesIO()
#             df_final.to_excel(buffer, index=False, engine="openpyxl")
#             st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel")

# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Local modules
# from modules import preprocessing as pp
# from modules import pipeline_manager as pm
# from modules import visual_analysis as va

# # ==============================================
# # Title
# # ==============================================
# st.title("🧹 Data Cleaning & Preprocessing — Pro Workspace")

# # ==============================================
# # Guards & Setup
# # ==============================================
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload a dataset in the Home tab.")
#     st.stop()

# # working DataFrame reference
# st.session_state.setdefault("df", st.session_state["df"])  # ensure key exists

# # history & originals
# pm.init_pipeline()  # sets original_df and pipeline_steps if not present

# # queue container for pipeline steps (to-be-applied)
# st.session_state.setdefault("queued_steps", [])  # list[dict]

# # ==============================================
# # Sidebar: Pipeline Manager & History
# # ==============================================
# st.sidebar.header("🧭 Pipeline Manager")

# # --- queued steps list (to be applied) ---
# if st.session_state["queued_steps"]:
#     for i, step in enumerate(st.session_state["queued_steps"]):
#         with st.sidebar.container():
#             cols = st.sidebar.columns([1, 6, 1])
#             cols[1].markdown(f"**{i+1}. {step['label']}**")
#             if cols[2].button("✖", key=f"rmq_{i}", help="Remove this pending step"):
#                 st.session_state["queued_steps"].pop(i)
#                 st.rerun()
# else:
#     st.sidebar.caption("No pending steps. Configure steps in the main area and click ‘Add to Pipeline’.")

# # --- actions row ---
# apply_col, undo_col, revert_col = st.sidebar.columns([2, 1, 1])

# if apply_col.button("▶️ Apply Pipeline", use_container_width=True, help="Run all queued steps sequentially on the current dataset"):
#     df = st.session_state.df

#     def _apply_single_step(df_in: pd.DataFrame, step: dict) -> pd.DataFrame:
#         action = step.get("action")
#         try:
#             # Missing values
#             if action == "missing":
#                 method = step["method"]
#                 return pp.handle_missing_values(df_in, method)

#             # Outliers
#             if action == "outliers":
#                 column = step["column"]
#                 return pp.remove_outliers(df_in, column)

#             # Dtype conversion
#             if action == "convert":
#                 return pp.convert_dtype(df_in, step["column"], step["to_type"])

#             # Rename column
#             if action == "rename":
#                 return pp.rename_column(df_in, step["old"], step["new"]) 

#             # Scaling / normalization
#             if action == "scale":
#                 return pp.normalize_column(df_in, step["column"], step["method"]) 

#             # Encoding — One-Hot (uses pandas)
#             if action == "encode_onehot":
#                 cols = step["columns"]
#                 return pd.get_dummies(df_in, columns=cols, drop_first=False)

#             # Encoding — Label (uses sklearn LabelEncoder) single column
#             if action == "encode_label":
#                 col = step["column"]
#                 enc = LabelEncoder()
#                 df_out = df_in.copy()
#                 df_out[col] = enc.fit_transform(df_out[col].astype(str))
#                 return df_out

#             # No-op fallback
#             return df_in
#         except Exception as e:
#             st.warning(f"Step '{step.get('label','?')}' failed: {e}")
#             return df_in

#     # apply all queued steps in order, recording each snapshot for undo
#     for step in st.session_state["queued_steps"]:
#         new_df = _apply_single_step(df, step)
#         pm.add_step(step["label"], new_df)
#         df = new_df

#     st.session_state.df = df
#     st.session_state["queued_steps"] = []  # clear queue after apply
#     st.success("✅ Pipeline applied.")
#     st.rerun()

# # Undo last applied step (uses pipeline_manager history)
# if undo_col.button("↩️ Undo", use_container_width=True, help="Revert to the previous applied snapshot"):
#     pm.undo_last_step()
#     st.rerun()

# # Revert to original upload
# if revert_col.button("🧼 Revert", use_container_width=True, help="Restore the dataset to the original uploaded state"):
#     if st.session_state.get("original_df") is not None:
#         st.session_state.df = st.session_state.original_df.copy()
#         st.session_state.pipeline_steps = []
#         st.session_state.queued_steps = []
#         st.success("Dataset reverted to the original uploaded state.")
#         st.rerun()
#     else:
#         st.warning("Original dataset copy not found.")

# # ==============================================
# # Workspace: Data Overview & Visual Guidance
# # ==============================================
# with st.expander("📄 Current Dataset Preview", expanded=True):
#     st.dataframe(st.session_state.df.head())
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Rows", st.session_state.df.shape[0])
#     c2.metric("Columns", st.session_state.df.shape[1])
#     c3.metric("Missing Cells", int(st.session_state.df.isna().sum().sum()))

# # Visual guidance to decide cleaning
# with st.expander("📊 Visual Analysis (to guide cleaning decisions)", expanded=True):
#     va.analyze_dataset(st.session_state.df)

# st.markdown("---")

# # ==============================================
# # Cleaning & Preprocessing – Add to Pipeline
# # ==============================================
# st.header("⚙️ Cleaning Workspace (Queue Steps)")

# # ---------- 1. Missing Values ----------
# with st.expander("1️⃣ Handle Missing Values", expanded=False):
#     st.caption("Tip: Use mean/median for numeric columns and mode for categorical columns. 'Drop Rows' removes any row with missing values.")
#     missing_method = st.selectbox(
#         "Method",
#         ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
#         help="Choose how to handle NaNs across the dataset"
#     )
#     if st.button("➕ Add to Pipeline", key="add_missing"):
#         st.session_state.queued_steps.append({
#             "action": "missing",
#             "method": missing_method,
#             "label": f"Missing → {missing_method}"
#         })
#         st.success("Added missing value handling to pipeline.")

# # ---------- 2. Outlier Removal ----------
# with st.expander("2️⃣ Remove Outliers (IQR)", expanded=False):
#     df = st.session_state.df
#     numeric_cols = list(df.select_dtypes(include="number").columns)
#     outlier_col = st.selectbox("Column", ["None"] + numeric_cols, help="Uses IQR rule: keep values within [Q1-1.5*IQR, Q3+1.5*IQR]")
#     if outlier_col != "None" and st.button("➕ Add to Pipeline", key="add_outliers"):
#         st.session_state.queued_steps.append({
#             "action": "outliers",
#             "column": outlier_col,
#             "label": f"Outliers → {outlier_col}"
#         })
#         st.success(f"Added outlier removal for '{outlier_col}'.")

# # ---------- 3. Data Type Conversion ----------
# with st.expander("3️⃣ Convert Column Type", expanded=False):
#     dtype_col = st.selectbox("Column", ["None"] + list(st.session_state.df.columns))
#     new_type = st.selectbox("Convert to", ["None", "int", "float", "str"]) 
#     if dtype_col != "None" and new_type != "None" and st.button("➕ Add to Pipeline", key="add_convert"):
#         st.session_state.queued_steps.append({
#             "action": "convert",
#             "column": dtype_col,
#             "to_type": new_type,
#             "label": f"Convert → {dtype_col} → {new_type}"
#         })
#         st.success(f"Added type conversion for '{dtype_col}' → {new_type}.")

# # ---------- 4. Rename Columns ----------
# with st.expander("4️⃣ Rename Column", expanded=False):
#     col_to_rename = st.selectbox("Column", ["None"] + list(st.session_state.df.columns))
#     new_name = st.text_input("New name", help="Provide a unique, valid column name")
#     if col_to_rename != "None" and new_name.strip() and st.button("➕ Add to Pipeline", key="add_rename"):
#         st.session_state.queued_steps.append({
#             "action": "rename",
#             "old": col_to_rename,
#             "new": new_name.strip(),
#             "label": f"Rename → {col_to_rename} → {new_name.strip()}"
#         })
#         st.success(f"Added rename: {col_to_rename} → {new_name.strip()}.")

# # ---------- 5. Scaling / Normalization ----------
# with st.expander("5️⃣ Scale / Normalize", expanded=False):
#     numeric_cols = list(st.session_state.df.select_dtypes(include="number").columns)
#     norm_col = st.selectbox("Column", ["None"] + numeric_cols, help="Scale one numeric column at a time")
#     scale_method = st.selectbox("Method", ["MinMax", "Standard", "Robust"], help="Choose a scaling strategy")
#     if norm_col != "None" and st.button("➕ Add to Pipeline", key="add_scale"):
#         st.session_state.queued_steps.append({
#             "action": "scale",
#             "column": norm_col,
#             "method": scale_method,
#             "label": f"Scale → {norm_col} ({scale_method})"
#         })
#         st.success(f"Added scaling for '{norm_col}' using {scale_method}.")

# # ---------- 6. Encode Categorical (Advanced) ----------
# with st.expander("6️⃣ Encode Categorical (Advanced)", expanded=False):
#     cat_cols = list(st.session_state.df.select_dtypes(include=["object", "category"]).columns)
#     enc_type = st.radio("Encoding Type", ["One-Hot", "Label"], horizontal=True)
#     if enc_type == "One-Hot":
#         cols = st.multiselect("Columns to One-Hot Encode", cat_cols)
#         if cols and st.button("➕ Add to Pipeline", key="add_onehot"):
#             st.session_state.queued_steps.append({
#                 "action": "encode_onehot",
#                 "columns": cols,
#                 "label": f"One-Hot Encode → {', '.join(cols)}"
#             })
#             st.success("Added One-Hot encoding to pipeline.")
#     else:
#         col = st.selectbox("Column to Label Encode", ["None"] + cat_cols)
#         if col != "None" and st.button("➕ Add to Pipeline", key="add_label"):
#             st.session_state.queued_steps.append({
#                 "action": "encode_label",
#                 "column": col,
#                 "label": f"Label Encode → {col}"
#             })
#             st.success("Added Label encoding to pipeline.")

# # ---------- 7. Train/Test Split (Preview Only) ----------
# with st.expander("7️⃣ Train/Test Split (Preview)", expanded=False):
#     split_col = st.selectbox("Optional stratify column", ["None"] + list(st.session_state.df.columns))
#     test_size = st.slider("Test Size", 0.1, 0.5, 0.2, help="Proportion of data to hold out for testing")
#     if st.button("👀 Preview Split"):
#         df = st.session_state.df
#         if split_col == "None":
#             train, test = train_test_split(df, test_size=test_size, random_state=42)
#         else:
#             train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#         st.subheader("Train Sample")
#         st.dataframe(train.head())
#         st.subheader("Test Sample")
#         st.dataframe(test.head())

# # ==============================================
# # History & Current State
# # ==============================================
# st.markdown("---")

# with st.expander("📜 Applied Steps History (for Undo)", expanded=False):
#     if st.session_state.get("pipeline_steps"):
#         hist_df = pd.DataFrame([
#             {"#": i+1, "Step": item["name"]} for i, item in enumerate(st.session_state.pipeline_steps)
#         ])
#         st.dataframe(hist_df, use_container_width=True)
#     else:
#         st.caption("No applied steps yet. Queue steps and click ‘Apply Pipeline’.")

# # ==============================================
# # Export Cleaned Data
# # ==============================================
# st.markdown("---")

# st.header("💾 Export Cleaned Data")
# export_format = st.selectbox("Format", ["CSV", "Excel"], index=0)
# if st.button("Download Cleaned Data"):
#     df_final = st.session_state.df
#     if export_format == "CSV":
#         st.download_button("Download CSV", df_final.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
#     else:
#         from io import BytesIO
#         buffer = BytesIO()
#         df_final.to_excel(buffer, index=False, engine="openpyxl")
#         st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel")

# import streamlit as st
# import pandas as pd
# from modules import preprocessing as pp
# from modules import visualizations as vz
# from modules import pipeline_manager as pm
# from sklearn.model_selection import train_test_split
# from modules import visual_analysis as va

# st.title("🧹 Data Cleaning & Preprocessing")

# # -----------------------------
# # Ensure dataset exists
# # -----------------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload in Home tab.")
# else:
#     df = st.session_state["df"]
#     pm.init_pipeline()

#     # -----------------------------
#     # Dataset preview
#     # -----------------------------
#     with st.expander("📄 Current Dataset Preview", expanded=True):
#         st.dataframe(df.head())
    
#     # -----------------------------
#     # Visual Analysis
#     # -----------------------------
#     if df is not None:
#         va.analyze_dataset(df)

#     # -----------------------------
#     # Missing values
#     # -----------------------------
#     missing_method = st.selectbox(
#         "Handle Missing Values",
#         ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
#         key="missing_values_selectbox"
#     )
#     if missing_method != "None" and st.button("Apply Missing Values", key="apply_missing_values"):
#         df = pp.handle_missing_values(df, missing_method)
#         pm.add_step(f"Missing: {missing_method}", df)
#         st.session_state.df = df
#         st.success(f"Applied missing value handling: {missing_method}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Outliers
#     # -----------------------------
#     numeric_cols = list(df.select_dtypes(include="number").columns)
#     outlier_col = st.selectbox(
#         "Remove Outliers Column",
#         ["None"] + numeric_cols,
#         key="outlier_col_selectbox"
#     )
#     if outlier_col != "None" and st.button("Remove Outliers", key="remove_outliers_btn"):
#         df = pp.remove_outliers(df, outlier_col)
#         pm.add_step(f"Outliers: {outlier_col}", df)
#         st.session_state.df = df
#         st.success(f"Outliers removed from {outlier_col}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Data type conversion
#     # -----------------------------
#     dtype_col = st.selectbox(
#         "Convert Column Type",
#         ["None"] + list(df.columns),
#         key="dtype_col_selectbox"
#     )
#     new_type = st.selectbox(
#         "Convert to type",
#         ["None", "int", "float", "str"],
#         key="dtype_type_selectbox"
#     )
#     if dtype_col != "None" and new_type != "None" and st.button("Convert Type", key="convert_type_btn"):
#         df = pp.convert_dtype(df, dtype_col, new_type)
#         pm.add_step(f"Convert {dtype_col} to {new_type}", df)
#         st.session_state.df = df
#         st.success(f"{dtype_col} converted to {new_type}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Rename columns
#     # -----------------------------
#     col_to_rename = st.selectbox(
#         "Rename Column",
#         ["None"] + list(df.columns),
#         key="rename_col_selectbox"
#     )
#     new_name = st.text_input(
#         "New name",
#         key="rename_col_textinput"
#     )
#     if col_to_rename != "None" and new_name.strip() and st.button("Rename Column", key="rename_col_btn"):
#         df = pp.rename_column(df, col_to_rename, new_name.strip())
#         pm.add_step(f"Rename {col_to_rename} → {new_name.strip()}", df)
#         st.session_state.df = df
#         st.success(f"{col_to_rename} renamed to {new_name.strip()}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Normalize / scale
#     # -----------------------------
#     norm_col = st.selectbox(
#         "Normalize Column",
#         ["None"] + numeric_cols,
#         key="normalize_col_selectbox"
#     )
#     scale_method = st.selectbox(
#         "Scaling Method",
#         ["MinMax", "Standard", "Robust"],
#         key="scale_method_selectbox"
#     )
#     if norm_col != "None" and st.button("Apply Scaling", key="apply_scaling_btn"):
#         df = pp.normalize_column(df, norm_col, scale_method)
#         pm.add_step(f"Scale {norm_col} ({scale_method})", df)
#         st.session_state.df = df
#         st.success(f"{norm_col} scaled using {scale_method}")
#         st.dataframe(df.head())

#     # -----------------------------
#     # Train/Test Split
#     # -----------------------------
#     st.markdown("---")
#     st.header("🔀 Train/Test Split")
#     split_col = st.selectbox(
#         "Column for Split (optional)",
#         ["None"] + list(df.columns),
#         key="split_col_selectbox"
#     )
#     test_size = st.slider(
#         "Test Size",
#         0.1, 0.5, 0.2,
#         key="test_size_slider"
#     )
#     if st.button("Preview Train/Test Split", key="preview_split_btn"):
#         if split_col == "None":
#             train, test = train_test_split(df, test_size=test_size, random_state=42)
#         else:
#             train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#         st.subheader("Train Sample")
#         st.dataframe(train.head())
#         st.subheader("Test Sample")
#         st.dataframe(test.head())

#     # -----------------------------
#     # Undo last step
#     # -----------------------------
#     if st.button("↩️ Undo Last Step", key="undo_last_step_btn"):
#         pm.undo_last_step()
#         st.dataframe(st.session_state.df.head())

#     # -----------------------------
#     # Export cleaned dataset
#     # -----------------------------
#     st.markdown("---")
#     st.header("💾 Export Cleaned Data")
#     export_format = st.selectbox(
#         "Format",
#         ["CSV", "Excel"],
#         key="export_format_selectbox"
#     )
#     if st.button("Download Cleaned Data", key="download_cleaned_btn"):
#         df_final = st.session_state.df
#         if export_format == "CSV":
#             st.download_button(
#                 "Download CSV",
#                 df_final.to_csv(index=False),
#                 "cleaned_dataset.csv",
#                 "text/csv",
#                 key="download_csv_btn"
#             )
#         else:
#             from io import BytesIO
#             buffer = BytesIO()
#             df_final.to_excel(buffer, index=False, engine="openpyxl")
#             st.download_button(
#                 "Download Excel",
#                 buffer.getvalue(),
#                 "cleaned_dataset.xlsx",
#                 "application/vnd.ms-excel",
#                 key="download_excel_btn"
#             )


# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from modules import preprocessing as pp
# from modules import visualizations as vz
# from modules import pipeline_manager as pm
# from modules import visual_analysis as va

# st.title("🧹 Data Cleaning & Preprocessing")

# # -----------------------------
# # Ensure dataset exists
# # -----------------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload in Home tab.")
# else:
#     df = st.session_state["df"]
#     pm.init_pipeline()

#     # -----------------------------
#     # Dataset preview
#     # -----------------------------
#     with st.expander("📄 1. Dataset Preview", expanded=True):
#         st.dataframe(df.head())

#     # -----------------------------
#     # Visual Analysis
#     # -----------------------------
#     with st.expander("📊 2. Visual Analysis", expanded=False):
#         va.analyze_dataset(df)

#     # -----------------------------
#     # Missing Values
#     # -----------------------------
#     with st.expander("🩹 3. Handle Missing Values"):
#         st.info("Choose a method to handle missing values.")
#         missing_method = st.selectbox(
#             "Method",
#             ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
#             key="missing_values_selectbox"
#         )
#         if missing_method != "None" and st.button("Apply Missing Values", key="apply_missing_values"):
#             df = pp.handle_missing_values(df, missing_method)
#             pm.add_step(f"Missing: {missing_method}", df)
#             st.session_state.df = df
#             st.success(f"Applied missing value handling: {missing_method}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Outliers
#     # -----------------------------
#     with st.expander("📉 4. Outlier Removal"):
#         numeric_cols = list(df.select_dtypes(include="number").columns)
#         outlier_col = st.selectbox(
#             "Select column to clean",
#             ["None"] + numeric_cols,
#             key="outlier_col_selectbox"
#         )
#         if outlier_col != "None" and st.button("Remove Outliers", key="remove_outliers_btn"):
#             df = pp.remove_outliers(df, outlier_col)
#             pm.add_step(f"Outliers: {outlier_col}", df)
#             st.session_state.df = df
#             st.success(f"Outliers removed from {outlier_col}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Data Type Conversion
#     # -----------------------------
#     with st.expander("🔤 5. Data Type Conversion"):
#         dtype_col = st.selectbox(
#             "Select column",
#             ["None"] + list(df.columns),
#             key="dtype_col_selectbox"
#         )
#         new_type = st.selectbox(
#             "Convert to",
#             ["None", "int", "float", "str"],
#             key="dtype_type_selectbox"
#         )
#         if dtype_col != "None" and new_type != "None" and st.button("Convert Type", key="convert_type_btn"):
#             df = pp.convert_dtype(df, dtype_col, new_type)
#             pm.add_step(f"Convert {dtype_col} to {new_type}", df)
#             st.session_state.df = df
#             st.success(f"{dtype_col} converted to {new_type}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Rename / Drop Columns
#     # -----------------------------
#     with st.expander("✏️ 6. Rename / Drop Columns"):
#         # Rename
#         col_to_rename = st.selectbox(
#             "Rename Column",
#             ["None"] + list(df.columns),
#             key="rename_col_selectbox"
#         )
#         new_name = st.text_input("New name", key="rename_col_textinput")
#         if col_to_rename != "None" and new_name.strip() and st.button("Rename Column", key="rename_col_btn"):
#             df = pp.rename_column(df, col_to_rename, new_name.strip())
#             pm.add_step(f"Rename {col_to_rename} → {new_name.strip()}", df)
#             st.session_state.df = df
#             st.success(f"{col_to_rename} renamed to {new_name.strip()}")
#             st.dataframe(df.head())

#         # Drop
#         cols_to_drop = st.multiselect(
#             "Drop Columns",
#             df.columns,
#             key="drop_col_multiselect"
#         )
#         if cols_to_drop and st.button("Drop Selected Columns", key="drop_cols_btn"):
#             df = df.drop(columns=cols_to_drop)
#             pm.add_step(f"Drop columns: {cols_to_drop}", df)
#             st.session_state.df = df
#             st.success(f"Dropped columns: {cols_to_drop}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Encoding Categorical Variables
#     # -----------------------------
#     with st.expander("🔠 7. Encode Categorical Variables"):
#         cat_cols = list(df.select_dtypes(include="object").columns)
#         encode_col = st.selectbox(
#             "Select column",
#             ["None"] + cat_cols,
#             key="encode_col_selectbox"
#         )
#         encode_method = st.selectbox(
#             "Encoding Method",
#             ["One-Hot", "Label"],
#             key="encode_method_selectbox"
#         )
#         if encode_col != "None" and st.button("Apply Encoding", key="apply_encoding_btn"):
#             df = pp.encode_categorical(df, encode_col, encode_method)
#             pm.add_step(f"Encode {encode_col} ({encode_method})", df)
#             st.session_state.df = df
#             st.success(f"{encode_col} encoded with {encode_method}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Normalize / Scale
#     # -----------------------------
#     with st.expander("📏 8. Normalize / Scale Data"):
#         norm_col = st.selectbox(
#             "Normalize Column",
#             ["None"] + numeric_cols,
#             key="normalize_col_selectbox"
#         )
#         scale_method = st.selectbox(
#             "Scaling Method",
#             ["MinMax", "Standard", "Robust"],
#             key="scale_method_selectbox"
#         )
#         if norm_col != "None" and st.button("Apply Scaling", key="apply_scaling_btn"):
#             df = pp.normalize_column(df, norm_col, scale_method)
#             pm.add_step(f"Scale {norm_col} ({scale_method})", df)
#             st.session_state.df = df
#             st.success(f"{norm_col} scaled using {scale_method}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Train/Test Split
#     # -----------------------------
#     with st.expander("🔀 9. Train/Test Split"):
#         split_col = st.selectbox(
#             "Column for Stratify (optional)",
#             ["None"] + list(df.columns),
#             key="split_col_selectbox"
#         )
#         test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key="test_size_slider")
#         if st.button("Preview Train/Test Split", key="preview_split_btn"):
#             if split_col == "None":
#                 train, test = train_test_split(df, test_size=test_size, random_state=42)
#             else:
#                 train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#             st.subheader("Train Sample")
#             st.dataframe(train.head())
#             st.subheader("Test Sample")
#             st.dataframe(test.head())

#     # -----------------------------
#     # Undo / Revert
#     # -----------------------------
#     with st.expander("↩️ 10. Undo / Revert"):
#         if st.button("Undo Last Step", key="undo_last_step_btn"):
#             pm.undo_last_step()
#             st.dataframe(st.session_state.df.head())
#         if st.button("Reset to Original Dataset", key="reset_original_btn"):
#             pm.reset_pipeline()
#             st.dataframe(st.session_state.df.head())

#     # -----------------------------
#     # Export Cleaned Dataset
#     # -----------------------------
#     with st.expander("💾 11. Export Cleaned Dataset"):
#         export_format = st.selectbox("Format", ["CSV", "Excel"], key="export_format_selectbox")
#         if st.button("Download Cleaned Data", key="download_cleaned_btn"):
#             df_final = st.session_state.df
#             if export_format == "CSV":
#                 st.download_button(
#                     "Download CSV",
#                     df_final.to_csv(index=False),
#                     "cleaned_dataset.csv",
#                     "text/csv",
#                     key="download_csv_btn"
#                 )
#             else:
#                 from io import BytesIO
#                 buffer = BytesIO()
#                 df_final.to_excel(buffer, index=False, engine="openpyxl")
#                 st.download_button(
#                     "Download Excel",
#                     buffer.getvalue(),
#                     "cleaned_dataset.xlsx",
#                     "application/vnd.ms-excel",
#                     key="download_excel_btn"
#                 )


# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from modules import preprocessing as pp
# from modules import visualizations as vz
# from modules import pipeline_manager as pm
# from modules import visual_analysis as va

# st.set_page_config(layout="wide")
# st.title("🧹 Data Cleaning & Preprocessing")

# # -----------------------------
# # Ensure dataset exists
# # -----------------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload in Home tab.")
# else:
#     df = st.session_state["df"]
#     pm.init_pipeline()

#     # -----------------------------
#     # Sidebar Summary
#     # -----------------------------
#     with st.sidebar:
#         st.header("📋 Dataset Summary")
#         st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
#         st.write("Numeric Columns:", list(df.select_dtypes(include="number").columns))
#         st.write("Categorical Columns:", list(df.select_dtypes(include="object").columns))
#         if "pipeline_steps" in st.session_state and st.session_state.pipeline_steps:
#             st.subheader("🛠 Last Steps")
#             for step in st.session_state.pipeline_steps[-3:][::-1]:
#                 st.write(f"- {step['name']}")

#     # -----------------------------
#     # Tabs for major operations
#     # -----------------------------
#     tabs = st.tabs([
#         "Dataset & Visuals", 
#         "Missing & Outliers", 
#         "Data Types & Columns", 
#         "Encoding & Scaling", 
#         "Train/Test & Export", 
#         "Undo / Reset"
#     ])

#     # -----------------------------
#     # Tab 1: Dataset Preview & Visual Analysis
#     # -----------------------------
#     with tabs[0]:
#         st.subheader("📄 Dataset Preview")
#         st.dataframe(df.head())

#         st.subheader("📊 Visual Analysis")
#         va.analyze_dataset(df)

#     # -----------------------------
#     # Tab 2: Missing Values & Outliers
#     # -----------------------------
#     with tabs[1]:
#         st.subheader("🩹 Handle Missing Values")
#         missing_method = st.selectbox(
#             "Method",
#             ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
#             key="missing_values_selectbox"
#         )
#         if missing_method != "None" and st.button("Apply Missing Values", key="apply_missing_values"):
#             df = pp.handle_missing_values(df, missing_method)
#             pm.add_step(f"Missing: {missing_method}", df)
#             st.session_state.df = df
#             st.success(f"Applied missing value handling: {missing_method}")
#             st.dataframe(df.head())

#         st.subheader("📉 Outlier Removal")
#         numeric_cols = list(df.select_dtypes(include="number").columns)
#         outlier_col = st.selectbox("Select column to clean", ["None"] + numeric_cols, key="outlier_col_selectbox")
#         if outlier_col != "None" and st.button("Remove Outliers", key="remove_outliers_btn"):
#             df = pp.remove_outliers(df, outlier_col)
#             pm.add_step(f"Outliers: {outlier_col}", df)
#             st.session_state.df = df
#             st.success(f"Outliers removed from {outlier_col}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Tab 3: Data Types & Columns
#     # -----------------------------
#     with tabs[2]:
#         st.subheader("🔤 Data Type Conversion")
#         col1, col2, col3 = st.columns([2,2,1])
#         with col1:
#             dtype_col = st.selectbox("Select column", ["None"] + list(df.columns), key="dtype_col_selectbox")
#         with col2:
#             new_type = st.selectbox("Convert to", ["None", "int", "float", "str"], key="dtype_type_selectbox")
#         with col3:
#             if dtype_col != "None" and new_type != "None" and st.button("Convert", key="convert_type_btn"):
#                 df = pp.convert_dtype(df, dtype_col, new_type)
#                 pm.add_step(f"Convert {dtype_col} to {new_type}", df)
#                 st.session_state.df = df
#                 st.success(f"{dtype_col} converted to {new_type}")
#                 st.dataframe(df.head())

#         st.subheader("✏️ Rename / Drop Columns")
#         rename_col, new_name = st.columns([2,2])
#         with rename_col:
#             col_to_rename = st.selectbox("Rename Column", ["None"] + list(df.columns), key="rename_col_selectbox")
#         with new_name:
#             new_name_text = st.text_input("New name", key="rename_col_textinput")
#         if col_to_rename != "None" and new_name_text.strip() and st.button("Rename", key="rename_col_btn"):
#             df = pp.rename_column(df, col_to_rename, new_name_text.strip())
#             pm.add_step(f"Rename {col_to_rename} → {new_name_text.strip()}", df)
#             st.session_state.df = df
#             st.success(f"{col_to_rename} renamed to {new_name_text.strip()}")
#             st.dataframe(df.head())

#         drop_cols = st.multiselect("Drop Columns", df.columns, key="drop_col_multiselect")
#         if drop_cols and st.button("Drop Selected Columns", key="drop_cols_btn"):
#             df = df.drop(columns=drop_cols)
#             pm.add_step(f"Drop columns: {drop_cols}", df)
#             st.session_state.df = df
#             st.success(f"Dropped columns: {drop_cols}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Tab 4: Encoding & Scaling
#     # -----------------------------
#     with tabs[3]:
#         st.subheader("🔠 Encode Categorical Variables")
#         cat_cols = list(df.select_dtypes(include="object").columns)
#         col1, col2 = st.columns([2,2])
#         with col1:
#             encode_col = st.selectbox("Select column", ["None"] + cat_cols, key="encode_col_selectbox")
#         with col2:
#             encode_method = st.selectbox("Method", ["One-Hot", "Label"], key="encode_method_selectbox")
#         if encode_col != "None" and st.button("Apply Encoding", key="apply_encoding_btn"):
#             df = pp.encode_categorical(df, encode_col, encode_method)
#             pm.add_step(f"Encode {encode_col} ({encode_method})", df)
#             st.session_state.df = df
#             st.success(f"{encode_col} encoded with {encode_method}")
#             st.dataframe(df.head())

#         st.subheader("📏 Normalize / Scale Data")
#         norm_col = st.selectbox("Normalize Column", ["None"] + numeric_cols, key="normalize_col_selectbox")
#         scale_method = st.selectbox("Scaling Method", ["MinMax", "Standard", "Robust"], key="scale_method_selectbox")
#         if norm_col != "None" and st.button("Apply Scaling", key="apply_scaling_btn"):
#             df = pp.normalize_column(df, norm_col, scale_method)
#             pm.add_step(f"Scale {norm_col} ({scale_method})", df)
#             st.session_state.df = df
#             st.success(f"{norm_col} scaled using {scale_method}")
#             st.dataframe(df.head())

#     # -----------------------------
#     # Tab 5: Train/Test Split & Export
#     # -----------------------------
#     with tabs[4]:
#         st.subheader("🔀 Train/Test Split")
#         split_col = st.selectbox("Column for Stratify (optional)", ["None"] + list(df.columns), key="split_col_selectbox")
#         test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key="test_size_slider")
#         if st.button("Preview Train/Test Split", key="preview_split_btn"):
#             if split_col == "None":
#                 train, test = train_test_split(df, test_size=test_size, random_state=42)
#             else:
#                 train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#             st.subheader("Train Sample")
#             st.dataframe(train.head())
#             st.subheader("Test Sample")
#             st.dataframe(test.head())

#         st.subheader("💾 Export Cleaned Dataset")
#         export_format = st.selectbox("Format", ["CSV", "Excel"], key="export_format_selectbox")
#         if st.button("Download Cleaned Data", key="download_cleaned_btn"):
#             df_final = st.session_state.df
#             if export_format == "CSV":
#                 st.download_button("Download CSV", df_final.to_csv(index=False), "cleaned_dataset.csv", "text/csv", key="download_csv_btn")
#             else:
#                 from io import BytesIO
#                 buffer = BytesIO()
#                 df_final.to_excel(buffer, index=False, engine="openpyxl")
#                 st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel", key="download_excel_btn")

#     # -----------------------------
#     # Tab 6: Undo / Reset
#     # -----------------------------
#     with tabs[5]:
#         st.subheader("↩️ Undo / Reset")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Undo Last Step", key="undo_last_step_btn"):
#                 pm.undo_last_step()
#                 st.dataframe(st.session_state.df.head())
#         with col2:
#             if st.button("Reset to Original Dataset", key="reset_original_btn"):
#                 pm.reset_pipeline()
#                 st.dataframe(st.session_state.df.head())

# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from modules import preprocessing as pp
# from modules import pipeline_manager as pm
# from modules import visual_analysis as va
# from io import BytesIO

# st.set_page_config(layout="wide")
# st.title("🧹 Data Cleaning & Preprocessing")

# # ----------------------------- Helper Functions -----------------------------
# def update_df(new_df, step_name):
#     pm.add_step(step_name, new_df)
#     st.session_state.df = new_df
#     st.success(f"{step_name} applied")
#     st.dataframe(new_df.head())

# def download_df(df):
#     st.subheader("💾 Export Cleaned Dataset")
#     fmt = st.selectbox("Format", ["CSV","Excel"], key="export_format")
#     if fmt=="CSV":
#         st.download_button("Download CSV", df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
#     else:
#         buffer = BytesIO()
#         df.to_excel(buffer, index=False, engine="openpyxl")
#         st.download_button("Download Excel", buffer.getvalue(), "cleaned_dataset.xlsx", "application/vnd.ms-excel")

# def get_columns(df, dtype="number"):
#     if dtype=="number": return list(df.select_dtypes(include="number").columns)
#     if dtype=="object": return list(df.select_dtypes(include="object").columns)
#     return list(df.columns)

# # ----------------------------- Ensure dataset exists -----------------------------
# if "df" not in st.session_state or st.session_state.df is None:
#     st.warning("⚠️ No dataset loaded. Upload in Home tab.")
# else:
#     df = st.session_state.df
#     pm.init_pipeline()

#     # ----------------------------- Tabs -----------------------------
#     tabs = st.tabs([
#         "Dataset & Visuals", 
#         "Missing & Outliers", 
#         "Data Types & Columns", 
#         "Encoding & Scaling", 
#         "Train/Test & Export", 
#         "Undo / Reset"
#     ])

#     # Tab 1: Preview & Visuals
#     with tabs[0]:
#         st.subheader("📄 Dataset Preview")
#         st.dataframe(df.head())
#         st.subheader("📊 Visual Analysis")
#         va.analyze_dataset(df)

#     # Tab 2: Missing & Outliers
#     with tabs[1]:
#         st.subheader("🩹 Handle Missing Values")
#         missing_method = st.selectbox("Method", ["None","Drop Rows","Fill with Mean","Fill with Median","Fill with Mode"])
#         if missing_method!="None" and st.button("Apply Missing Values"):
#             update_df(pp.handle_missing_values(df, missing_method), f"Missing: {missing_method}")

#         st.subheader("📉 Remove Outliers")
#         num_cols = get_columns(df,"number")
#         col = st.selectbox("Select column", ["None"]+num_cols)
#         if col!="None" and st.button("Remove Outliers"):
#             update_df(pp.remove_outliers(df,col), f"Outliers removed: {col}")

#     # Tab 3: Data Types & Columns
#     with tabs[2]:
#         st.subheader("🔤 Convert Data Type")
#         col, new_type = st.columns(2)
#         with col: sel_col = st.selectbox("Column", ["None"]+get_columns(df,"all"))
#         with new_type: sel_type = st.selectbox("Convert to", ["None","int","float","str"])
#         if sel_col!="None" and sel_type!="None" and st.button("Convert Type"):
#             update_df(pp.convert_dtype(df, sel_col, sel_type), f"Convert {sel_col} → {sel_type}")

#         st.subheader("✏️ Rename / Drop Columns")
#         rcol, nname = st.columns(2)
#         with rcol: rename_col = st.selectbox("Rename Column", ["None"]+get_columns(df,"all"))
#         with nname: new_name = st.text_input("New name")
#         if rename_col!="None" and new_name.strip() and st.button("Rename Column"):
#             update_df(pp.rename_column(df, rename_col,new_name.strip()), f"Rename {rename_col} → {new_name.strip()}")

#         drop_cols = st.multiselect("Drop Columns", get_columns(df,"all"))
#         if drop_cols and st.button("Drop Selected Columns"):
#             update_df(df.drop(columns=drop_cols), f"Dropped columns: {drop_cols}")

#     # Tab 4: Encoding & Scaling
#     with tabs[3]:
#         st.subheader("🔠 Encode Categorical Columns")
#         cat_cols = get_columns(df,"object")
#         enc_col, enc_method = st.columns(2)
#         with enc_col: sel_col = st.selectbox("Column", ["None"]+cat_cols)
#         with enc_method: sel_method = st.selectbox("Method", ["One-Hot","Label"])
#         if sel_col!="None" and st.button("Apply Encoding"):
#             update_df(pp.encode_categorical(df, sel_col, sel_method), f"Encode {sel_col} → {sel_method}")

#         st.subheader("📏 Normalize / Scale")
#         norm_col = st.selectbox("Column", ["None"]+get_columns(df,"number"))
#         scale_method = st.selectbox("Method", ["MinMax","Standard","Robust"])
#         if norm_col!="None" and st.button("Apply Scaling"):
#             update_df(pp.normalize_column(df,norm_col,scale_method), f"Scale {norm_col} → {scale_method}")

#     # Tab 5: Train/Test & Export
#     with tabs[4]:
#         st.subheader("🔀 Train/Test Split")
#         split_col = st.selectbox("Stratify Column (optional)", ["None"]+get_columns(df,"all"))
#         test_size = st.slider("Test Size", 0.1,0.5,0.2)
#         if st.button("Preview Split"):
#             if split_col=="None":
#                 train,test = train_test_split(df,test_size=test_size,random_state=42)
#             else:
#                 train,test = train_test_split(df,test_size=test_size,random_state=42,stratify=df[split_col])
#             st.subheader("Train Sample"); st.dataframe(train.head())
#             st.subheader("Test Sample"); st.dataframe(test.head())

#         download_df(df)

#     # Tab 6: Undo / Reset
#     with tabs[5]:
#         col1,col2 = st.columns(2)
#         with col1: 
#             if st.button("Undo Last Step"): 
#                 pm.undo_last_step()
#                 st.dataframe(st.session_state.df.head())
#         with col2: 
#             if st.button("Reset to Original"): 
#                 pm.reset_pipeline()
#                 st.dataframe(st.session_state.df.head())

# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from modules import preprocessing as pp
# from modules import pipeline_manager as pm
# from modules import visual_analysis as va

# st.set_page_config(layout="wide")
# st.title("🧹 Data Cleaning & Preprocessing")

# # ------------------------------------------------------------------
# # Guard: dataset
# # ------------------------------------------------------------------
# if "df" not in st.session_state or st.session_state["df"] is None:
#     st.warning("⚠️ No dataset loaded. Please upload in Home.")
#     st.stop()

# df = st.session_state["df"]
# pm.init_pipeline()

# # ------------------------------------------------------------------
# # Sidebar context
# # ------------------------------------------------------------------
# with st.sidebar:
#     st.header("📋 Dataset Summary")
#     st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
#     st.caption("Numeric")
#     st.write(list(df.select_dtypes(include="number").columns))
#     st.caption("Categorical")
#     st.write(list(df.select_dtypes(include="object").columns))

#     if st.session_state.get("pipeline_steps"):
#         st.subheader("🛠 Recent Steps")
#         for step in st.session_state.pipeline_steps[-4:][::-1]:
#             st.write(f"- {step['name']}")

# # ------------------------------------------------------------------
# # Tabs
# # ------------------------------------------------------------------
# tabs = st.tabs([
#     "Dataset & Visuals",
#     "Missing & Outliers",
#     "Types & Columns",
#     "Encoding & Scaling",
#     "Train/Test & Export",
#     "Undo / Reset",
# ])

# # ------------------------------------------------------------------
# # Tab 1: dataset + visuals
# # ------------------------------------------------------------------
# with tabs[0]:
#     st.subheader("📄 Dataset Preview")
#     st.dataframe(df.head())
#     st.subheader("📊 Visual Analysis")
#     va.analyze_dataset(df)

# # ------------------------------------------------------------------
# # Tab 2: missing + outliers
# # ------------------------------------------------------------------
# with tabs[1]:
#     st.subheader("🩹 Handle Missing Values")
#     missing_method = st.selectbox(
#         "Method",
#         ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
#         key="c_missing_method",
#     )
#     if missing_method != "None" and st.button("Apply Missing Values", key="c_apply_missing"):
#         df = pp.handle_missing_values(df, missing_method)
#         pm.add_step(f"Missing: {missing_method}", df)
#         st.session_state.df = df
#         st.success(f"Applied: {missing_method}")
#         st.dataframe(df.head())

#     st.divider()
#     st.subheader("📉 Outlier Removal")
#     numeric_cols = list(df.select_dtypes(include="number").columns)
#     outlier_cols = st.multiselect(
#         "Columns to clean (IQR method, per column)", numeric_cols, key="c_outlier_cols"
#     )
#     if outlier_cols and st.button("Remove Outliers", key="c_remove_outliers"):
#         for col in outlier_cols:
#             df = pp.remove_outliers(df, col)
#         pm.add_step(f"Outliers: {outlier_cols}", df)
#         st.session_state.df = df
#         st.success(f"Outliers removed: {outlier_cols}")
#         st.dataframe(df.head())

# # ------------------------------------------------------------------
# # Tab 3: types & columns
# # ------------------------------------------------------------------
# with tabs[2]:
#     st.subheader("🔤 Data Type Conversion")
#     c1, c2, c3 = st.columns([2, 2, 1])
#     with c1:
#         dtype_cols = st.multiselect("Columns", list(df.columns), key="c_dtype_cols")
#     with c2:
#         new_type = st.selectbox("Convert to", ["int", "float", "str"], key="c_dtype_to")
#     with c3:
#         if dtype_cols and st.button("Convert", key="c_convert_types"):
#             for col in dtype_cols:
#                 df = pp.convert_dtype(df, col, new_type)
#             pm.add_step(f"Convert {dtype_cols} → {new_type}", df)
#             st.session_state.df = df
#             st.success(f"Converted {dtype_cols} to {new_type}")
#             st.dataframe(df.head())

#     st.divider()
#     st.subheader("✏️ Rename / Drop Columns")
#     rc1, rc2 = st.columns([2, 2])
#     with rc1:
#         rename_col = st.selectbox("Rename column", ["None"] + list(df.columns), key="c_rename_col")
#     with rc2:
#         new_name = st.text_input("New name", key="c_new_name")
#     if rename_col != "None" and new_name.strip() and st.button("Rename", key="c_btn_rename"):
#         df = pp.rename_column(df, rename_col, new_name.strip())
#         pm.add_step(f"Rename {rename_col} → {new_name.strip()}", df)
#         st.session_state.df = df
#         st.success(f"Renamed {rename_col} → {new_name.strip()}")
#         st.dataframe(df.head())

#     drop_cols = st.multiselect("Drop columns", list(df.columns), key="c_drop_cols")
#     if drop_cols and st.button("Drop Selected", key="c_btn_drop"):
#         df = df.drop(columns=drop_cols)
#         pm.add_step(f"Drop: {drop_cols}", df)
#         st.session_state.df = df
#         st.success(f"Dropped: {drop_cols}")
#         st.dataframe(df.head())

# # ------------------------------------------------------------------
# # Tab 4: encoding & scaling
# # ------------------------------------------------------------------
# with tabs[3]:
#     st.subheader("🔠 Encode Categorical Variables")
#     cat_cols = list(df.select_dtypes(include="object").columns)
#     ec1, ec2 = st.columns([2, 2])
#     with ec1:
#         encode_cols = st.multiselect("Columns to encode", cat_cols, key="c_encode_cols")
#     with ec2:
#         encode_method = st.selectbox("Method", ["One-Hot", "Label"], key="c_encode_method")

#     if encode_cols and st.button("Apply Encoding", key="c_btn_encode"):
#         # Multi-column support even if pp supports single internally
#         if encode_method == "One-Hot":
#             df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
#         else:  # Label
#             for col in encode_cols:
#                 df = pp.encode_categorical(df, col, "Label")
#         pm.add_step(f"Encode {encode_cols} ({encode_method})", df)
#         st.session_state.df = df
#         st.success(f"Encoded: {encode_cols} with {encode_method}")
#         st.dataframe(df.head())

#     st.divider()
#     st.subheader("📏 Normalize / Scale")
#     sc1, sc2 = st.columns([2, 2])
#     with sc1:
#         scale_cols = st.multiselect("Numeric columns to scale", numeric_cols, key="c_scale_cols")
#     with sc2:
#         scale_method = st.selectbox("Scaling method", ["MinMax", "Standard", "Robust"], key="c_scale_method")
#     if scale_cols and st.button("Apply Scaling", key="c_btn_scale"):
#         for col in scale_cols:
#             df = pp.normalize_column(df, col, scale_method)
#         pm.add_step(f"Scale {scale_cols} ({scale_method})", df)
#         st.session_state.df = df
#         st.success(f"Scaled: {scale_cols} using {scale_method}")
#         st.dataframe(df.head())

# # ------------------------------------------------------------------
# # Tab 5: train/test + export
# # ------------------------------------------------------------------
# with tabs[4]:
#     st.subheader("🔀 Train/Test Split (preview)")
#     split_col = st.selectbox("Stratify by (optional)", ["None"] + list(df.columns), key="c_split_col")
#     test_size = st.slider("Test size", 0.1, 0.5, 0.2, key="c_test_size")
#     if st.button("Preview Split", key="c_btn_split"):
#         if split_col == "None":
#             train, test = train_test_split(df, test_size=test_size, random_state=42)
#         else:
#             train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[split_col])
#         st.write("**Train sample**")
#         st.dataframe(train.head())
#         st.write("**Test sample**")
#         st.dataframe(test.head())

#     st.divider()
#     st.subheader("💾 Export Cleaned Dataset")
#     exp1, exp2 = st.columns([2, 1])
#     with exp1:
#         filename = st.text_input("File name (without extension)", value="cleaned_dataset", key="c_filename")
#     with exp2:
#         export_format = st.radio("Format", ["CSV", "Excel"], horizontal=True, key="c_export_fmt")

#     # Render the download button every run so it never "disappears"
#     df_final = st.session_state.df
#     if export_format == "CSV":
#         st.download_button(
#             label="⬇️ Download CSV",
#             data=df_final.to_csv(index=False),
#             file_name=f"{filename or 'cleaned_dataset'}.csv",
#             mime="text/csv",
#             key=f"c_dl_csv_{filename}",
#         )
#     else:
#         from io import BytesIO
#         buffer = BytesIO()
#         df_final.to_excel(buffer, index=False, engine="openpyxl")
#         st.download_button(
#             label="⬇️ Download Excel",
#             data=buffer.getvalue(),
#             file_name=f"{filename or 'cleaned_dataset'}.xlsx",
#             mime="application/vnd.ms-excel",
#             key=f"c_dl_xlsx_{filename}",
#         )

# # ------------------------------------------------------------------
# # Tab 6: undo / reset
# # ------------------------------------------------------------------
# with tabs[5]:
#     st.subheader("↩️ Undo / Reset")
#     u1, u2 = st.columns(2)
#     with u1:
#         if st.button("Undo Last Step", key="c_btn_undo"):
#             pm.undo_last_step()
#             st.dataframe(st.session_state.df.head())
#     with u2:
#         if st.button("Reset to Original Dataset", key="c_btn_reset"):
#             pm.reset_pipeline()
#             st.dataframe(st.session_state.df.head())

"""
Main cleaning page (Phase 7 complete) — tabs, multi-column ops, EDA report, sampling safeguards.
Save as: echoAnalytics/pages/2_Cleaning.py
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO

from modules import preprocessing as pp
from modules import pipeline_manager as pm
from modules import visual_analysis as va

st.set_page_config(layout="wide", page_title="EchoAnalytics — Cleaning")
st.title("🧹 EchoAnalytics — Data Cleaning & Preprocessing (Phase 7 Complete)")

# -----------------------------
# Guard dataset presence
# -----------------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset in Home before using the cleaning tools.")
    st.stop()

# ensure pipeline state
pm.init_pipeline()
df = st.session_state.df

# -----------------------------
# Sidebar summary & controls
# -----------------------------
with st.sidebar:
    st.header("📋 Dataset Summary")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    st.write("Numeric:", list(df.select_dtypes(include="number").columns))
    st.write("Categorical:", list(df.select_dtypes(include="object").columns))
    st.divider()
    st.write("Recent steps:")
    for s in st.session_state.get("pipeline_steps", [])[-5:][::-1]:
        st.write(f"- {s['name']}")

# -----------------------------
# Utility helpers
# -----------------------------
def _apply_and_record(new_df: pd.DataFrame, step_name: str):
    pm.add_step(step_name, new_df)
    st.session_state.df = new_df
    st.success(step_name)
    st.dataframe(new_df.head())

def _download_bytes(data_bytes: bytes, filename: str, mime: str, key: str):
    st.download_button(label=f"⬇️ {filename}", data=data_bytes, file_name=filename, mime=mime, key=key)

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Dataset & Visuals",
    "Missing & Outliers",
    "Types & Columns",
    "Encoding & Scaling",
    "Train/Test & Export",
    "Undo / Reset",
])

# Tab 0: preview + visuals
with tabs[0]:
    st.subheader("📄 Preview")
    st.dataframe(df.head())
    st.subheader("📊 Visual Analysis & EDA")
    va.analyze_dataset(st.session_state.df)

# Tab 1: missing & outliers (multi-column)
with tabs[1]:
    st.subheader("🩹 Missing Values (batch)")
    miss_action = st.selectbox("Method", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"], key="c_miss_method")
    if miss_action != "None" and st.button("Apply missing action", key="c_miss_apply"):
        mmap = {"Drop Rows":"drop","Fill with Mean":"mean","Fill with Median":"median","Fill with Mode":"mode"}
        new_df = pp.apply_steps(st.session_state.df, [{"action":"handle_missing","method":mmap.get(miss_action)}])
        _apply_and_record(new_df, f"Missing: {miss_action}")

    st.divider()
    st.subheader("📉 Outlier Removal (multi-column)")
    numeric_cols = list(st.session_state.df.select_dtypes(include="number").columns)
    outlier_cols = st.multiselect("Select numeric columns to apply IQR outlier removal", numeric_cols, key="c_outlier_cols")
    if outlier_cols and st.button("Remove outliers (IQR) from selected", key="c_outliers_apply"):
        steps = [{"action":"remove_outliers","columns":outlier_cols}]
        new_df = pp.apply_steps(st.session_state.df, steps)
        _apply_and_record(new_df, f"Outliers removed: {outlier_cols}")

# Tab 2: types & columns
with tabs[2]:
    st.subheader("🔤 Convert Types (multi-column)")
    cols = list(st.session_state.df.columns)
    sel_cols = st.multiselect("Columns to convert", cols, key="c_convert_cols")
    to_type = st.selectbox("Convert to", ["int","float","str"], key="c_convert_to")
    if sel_cols and st.button("Convert selected", key="c_convert_apply"):
        steps = [{"action":"convert_dtype","columns":sel_cols,"dtype":to_type}]
        new_df = pp.apply_steps(st.session_state.df, steps)
        _apply_and_record(new_df, f"Converted {sel_cols} → {to_type}")

    st.divider()
    st.subheader("✏️ Rename / Drop")
    rename_col = st.selectbox("Rename column (choose)", ["None"] + cols, key="c_rename_choice")
    new_name = st.text_input("New name", key="c_rename_newname")
    if rename_col != "None" and new_name and st.button("Rename selected", key="c_rename_apply"):
        new_df = pp.apply_steps(st.session_state.df, [{"action":"rename","old":rename_col,"new":new_name}])
        _apply_and_record(new_df, f"Rename {rename_col} → {new_name}")

    drop_cols = st.multiselect("Drop columns", cols, key="c_drop_multi")
    if drop_cols and st.button("Drop selected columns", key="c_drop_apply"):
        new_df = pp.apply_steps(st.session_state.df, [{"action":"drop_columns","columns":drop_cols}])
        _apply_and_record(new_df, f"Dropped: {drop_cols}")

# Tab 3: encoding & scaling (multi-column)
with tabs[3]:
    st.subheader("🔠 Encoding (multi-column)")
    cat_cols = list(st.session_state.df.select_dtypes(include="object").columns)
    encode_cols = st.multiselect("Categorical columns to encode", cat_cols, key="c_encode_multi")
    encode_method = st.selectbox("Method", ["One-Hot","Label"], key="c_encode_method")
    if encode_cols and st.button("Apply encoding to selected", key="c_encode_apply"):
        # For one-hot, use pandas get_dummies in a single step to preserve other columns
        if encode_method == "One-Hot":
            new_df = pd.get_dummies(st.session_state.df, columns=encode_cols, drop_first=True)
        else:
            # Label encode each selected column
            steps = [{"action":"encode","columns":encode_cols,"method":"label"}]
            new_df = pp.apply_steps(st.session_state.df, steps)
        _apply_and_record(new_df, f"Encoded {encode_cols} ({encode_method})")

    st.divider()
    st.subheader("📏 Scaling / Normalization (multi-column)")
    num_cols = list(st.session_state.df.select_dtypes(include="number").columns)
    scale_cols = st.multiselect("Numeric columns to scale", num_cols, key="c_scale_multi")
    scale_method = st.selectbox("Scaling method", ["MinMax","Standard","Robust"], key="c_scale_method")
    if scale_cols and st.button("Apply scaling to selected", key="c_scale_apply"):
        steps = [{"action":"scale","columns":scale_cols,"method":scale_method.lower()}]
        new_df = pp.apply_steps(st.session_state.df, steps)
        _apply_and_record(new_df, f"Scaled {scale_cols} ({scale_method})")

# Tab 4: train/test + export + EDA
with tabs[4]:
    st.subheader("🔀 Train/Test split (preview only)")
    cols = list(st.session_state.df.columns)
    split_col = st.selectbox("Stratify by (optional)", ["None"] + cols, key="c_split_col")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, key="c_test_size")
    if st.button("Preview split", key="c_split_preview"):
        if split_col == "None":
            tr, te = train_test_split(st.session_state.df, test_size=test_size, random_state=42)
        else:
            tr, te = train_test_split(st.session_state.df, test_size=test_size, random_state=42, stratify=st.session_state.df[split_col])
        st.subheader("Train sample")
        st.dataframe(tr.head())
        st.subheader("Test sample")
        st.dataframe(te.head())

    st.divider()
    st.subheader("💾 Export & EDA Report")
    filename = st.text_input("File name, without extension", value="cleaned_dataset", key="c_export_name")
    fmt = st.radio("Format", ["CSV","Excel","Parquet"], key="c_export_fmt", horizontal=True)

    # Always render download buttons (so they are available immediately)
    df_final = st.session_state.df
    if fmt == "CSV":
        csv_bytes = df_final.to_csv(index=False).encode("utf-8")
        _download_bytes(csv_bytes, f"{filename}.csv", "text/csv", key=f"dl_csv_{filename}")
    elif fmt == "Excel":
        buffer = BytesIO()
        df_final.to_excel(buffer, index=False, engine="openpyxl")
        _download_bytes(buffer.getvalue(), f"{filename}.xlsx", "application/vnd.ms-excel", key=f"dl_xlsx_{filename}")
    else:
        pq_bytes = df_final.to_parquet(index=False)
        _download_bytes(pq_bytes, f"{filename}.parquet", "application/octet-stream", key=f"dl_parquet_{filename}")

    st.divider()
    st.subheader("One-click EDA Report")
    if st.button("Generate & download EDA report (HTML)", key="c_generate_eda"):
        report_bytes = va.generate_eda_report(st.session_state.df, sample_n=2000)
        _download_bytes(report_bytes, "eda_report.html", "text/html", key="dl_eda_report")

# Tab 5: undo / reset
with tabs[5]:
    st.subheader("↩️ Undo / Reset")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo last step", key="c_undo"):
            pm.undo_last_step()
            st.dataframe(st.session_state.df.head())
    with c2:
        if st.button("Reset to original", key="c_reset"):
            pm.reset_pipeline()
            st.dataframe(st.session_state.df.head())

# Footer tip
st.markdown("---")
st.caption("Phase 7 complete — multi-column ops, improved visuals, EDA report, sampling for performance.")
