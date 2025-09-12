# """Visual Analysis utilities for EchoAnalytics."""
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# sns.set(style="whitegrid")

# def analyze_dataset(df: pd.DataFrame):
#     """Perform automatic visual analysis for any dataset."""
    
#     st.header("📊 Visual Analysis")
    
#     # Detect numeric and categorical columns
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
    
#     st.subheader("1️⃣ Summary Statistics")
#     with st.expander("View Summary Stats"):
#         if numeric_cols:
#             st.write("**Numeric Columns:**")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.write("**Categorical Columns:**")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)
    
#     st.subheader("2️⃣ Visual Plots")
#     with st.expander("View Plots"):
#         # Numeric Columns
#         for col in numeric_cols:
#             st.markdown(f"**Numeric Column:** {col}")
#             fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            
#             sns.histplot(df[col], kde=True, ax=ax[0], bins=30, color='skyblue')
#             ax[0].set_title(f"{col} Histogram")
            
#             sns.boxplot(x=df[col], ax=ax[1], color='lightgreen')
#             ax[1].set_title(f"{col} Boxplot")
            
#             st.pyplot(fig)
        
#         # Categorical Columns
#         for col in categorical_cols:
#             st.markdown(f"**Categorical Column:** {col}")
#             top_n = 20
#             value_counts = df[col].value_counts().nlargest(top_n)
#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#             ax.set_title(f"Top {top_n} values of {col}")
#             st.pyplot(fig)
        
#         # Correlation heatmap for numeric columns
#         if len(numeric_cols) > 1:
#             st.subheader("Correlation Heatmap (Numeric Columns)")
#             corr = df[numeric_cols].corr()
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#             st.pyplot(fig)
        
#         # Optional: Pairplot for numeric columns (limit to 5 columns for performance)
#         if 2 <= len(numeric_cols) <= 5:
#             st.subheader("Pairplot (Numeric Columns)")
#             fig = sns.pairplot(df[numeric_cols])
#             st.pyplot(fig)
        
#         # Optional interactive Plotly scatter matrix (numeric only)
#         if len(numeric_cols) > 1:
#             st.subheader("Interactive Scatter Matrix")
#             fig = px.scatter_matrix(df[numeric_cols])
#             fig.update_traces(diagonal_visible=False)
#             st.plotly_chart(fig, use_container_width=True)


# """Interactive Visual Analysis for EchoAnalytics."""
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# sns.set(style="whitegrid")

# def analyze_dataset(df: pd.DataFrame):
#     """Interactive visual analysis where user chooses columns and plot types."""
    
#     st.header("📊 Visual Analysis")
    
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
    
#     # --------------------
#     # Summary Statistics
#     # --------------------
#     st.subheader("1️⃣ Summary Statistics")
#     with st.expander("View Summary Stats"):
#         if numeric_cols:
#             st.write("**Numeric Columns:**")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.write("**Categorical Columns:**")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)
    
#     # --------------------
#     # Interactive Plotting
#     # --------------------
#     st.subheader("2️⃣ Plot Your Data")
    
#     col_type = st.radio("Choose column type to visualize", ["Numeric", "Categorical"])
    
#     if col_type == "Numeric" and numeric_cols:
#         chosen_col = st.selectbox("Select numeric column", numeric_cols)
#         plot_type = st.selectbox("Choose plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"])
        
#         fig, ax = plt.subplots(figsize=(6, 4))
#         if plot_type == "Histogram":
#             sns.histplot(df[chosen_col], kde=True, ax=ax, bins=30, color='skyblue')
#         elif plot_type == "Boxplot":
#             sns.boxplot(x=df[chosen_col], ax=ax, color='lightgreen')
#         elif plot_type == "Line Plot":
#             ax.plot(df[chosen_col])
#             ax.set_title(f"{chosen_col} Line Plot")
#         elif plot_type == "Scatter Plot":
#             # Scatter vs index
#             ax.scatter(df.index, df[chosen_col], alpha=0.6)
#             ax.set_xlabel("Index")
#             ax.set_ylabel(chosen_col)
#             ax.set_title(f"{chosen_col} Scatter Plot")
#         st.pyplot(fig)
        
#     elif col_type == "Categorical" and categorical_cols:
#         chosen_col = st.selectbox("Select categorical column", categorical_cols)
#         plot_type = st.selectbox("Choose plot type", ["Bar Plot", "Pie Chart"])
        
#         top_n = 20
#         value_counts = df[chosen_col].value_counts().nlargest(top_n)
        
#         fig, ax = plt.subplots(figsize=(6, 4))
#         if plot_type == "Bar Plot":
#             sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#             ax.set_title(f"Top {top_n} values of {chosen_col}")
#         elif plot_type == "Pie Chart":
#             ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
#             ax.set_title(f"Top {top_n} values of {chosen_col}")
#         st.pyplot(fig)
    
#     # --------------------
#     # Correlation Heatmap (Numeric)
#     # --------------------
#     if len(numeric_cols) > 1:
#         if st.checkbox("Show Correlation Heatmap"):
#             st.subheader("Correlation Heatmap (Numeric Columns)")
#             corr = df[numeric_cols].corr()
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#             st.pyplot(fig)
    
#     # --------------------
#     # Optional Pairplot
#     # --------------------
#     if 2 <= len(numeric_cols) <= 5:
#         if st.checkbox("Show Pairplot (Numeric Columns)"):
#             st.subheader("Pairplot")
#             fig = sns.pairplot(df[numeric_cols])
#             st.pyplot(fig)
    
#     # --------------------
#     # Optional Interactive Plotly Scatter Matrix
#     # --------------------
#     if len(numeric_cols) > 1:
#         if st.checkbox("Show Interactive Scatter Matrix (Plotly)"):
#             st.subheader("Interactive Scatter Matrix")
#             fig = px.scatter_matrix(df[numeric_cols])
#             fig.update_traces(diagonal_visible=False)
#             st.plotly_chart(fig, use_container_width=True)


# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import numpy as np

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.figure_factory as ff
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set(style="whitegrid")

# def analyze_dataset(df: pd.DataFrame):
#     """Interactive visual analysis where user chooses columns and plot types."""
    
#     st.header("📊 Visual Analysis")
    
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
    
#     # --------------------
#     # Summary Statistics
#     # --------------------
#     st.subheader("1️⃣ Summary Statistics")
#     with st.expander("View Summary Stats"):
#         if numeric_cols:
#             st.write("**Numeric Columns:**")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.write("**Categorical Columns:**")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)
    
#     # --------------------
#     # Interactive Plotting
#     # --------------------
#     st.subheader("2️⃣ Plot Your Data")
    
#     col_type = st.radio("Choose column type to visualize", ["Numeric", "Categorical"])
    
#     if col_type == "Numeric" and numeric_cols:
#         chosen_col = st.selectbox("Select numeric column", numeric_cols)
#         plot_type = st.selectbox("Choose plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"])
        
#         fig, ax = plt.subplots(figsize=(6, 4))
#         if plot_type == "Histogram":
#             sns.histplot(df[chosen_col], kde=True, ax=ax, bins=30, color='skyblue')
#         elif plot_type == "Boxplot":
#             sns.boxplot(x=df[chosen_col], ax=ax, color='lightgreen')
#         elif plot_type == "Line Plot":
#             ax.plot(df[chosen_col])
#             ax.set_title(f"{chosen_col} Line Plot")
#         elif plot_type == "Scatter Plot":
#             ax.scatter(df.index, df[chosen_col], alpha=0.6)
#             ax.set_xlabel("Index")
#             ax.set_ylabel(chosen_col)
#             ax.set_title(f"{chosen_col} Scatter Plot")
#         st.pyplot(fig)
        
#     elif col_type == "Categorical" and categorical_cols:
#         chosen_col = st.selectbox("Select categorical column", categorical_cols)
#         plot_type = st.selectbox("Choose plot type", ["Bar Plot", "Pie Chart"])
        
#         top_n = 20
#         value_counts = df[chosen_col].value_counts().nlargest(top_n)
        
#         fig, ax = plt.subplots(figsize=(6, 4))
#         if plot_type == "Bar Plot":
#             sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#             ax.set_title(f"Top {top_n} values of {chosen_col}")
#         elif plot_type == "Pie Chart":
#             ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
#             ax.set_title(f"Top {top_n} values of {chosen_col}")
#         st.pyplot(fig)
    
#     # --------------------
#     # Correlation Heatmap (Numeric)
#     # --------------------
#     if numeric_cols:
#         st.subheader("3️⃣ Correlation Heatmap")
#         selected_corr_cols = st.multiselect(
#             "Select numeric columns for correlation", numeric_cols, default=numeric_cols
#         )
#         if len(selected_corr_cols) > 1:
#             corr = df[selected_corr_cols].corr()
#             fig, ax = plt.subplots(figsize=(max(6, len(selected_corr_cols)), max(4, len(selected_corr_cols)//2)))
#             mask = None
#             if len(selected_corr_cols) > 10:
#                 mask = np.triu(np.ones_like(corr, dtype=bool))
#             sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, mask=mask)
#             st.pyplot(fig)
    
#     # --------------------
#     # Optional Interactive Scatter Matrix (Plotly)
#     # --------------------
#     if numeric_cols:
#         st.subheader("4️⃣ Interactive Scatter Matrix")
#         selected_scatter_cols = st.multiselect(
#             "Select numeric columns for scatter matrix (max 6 recommended)", numeric_cols
#         )
#         if len(selected_scatter_cols) > 1:
#             if len(selected_scatter_cols) > 6:
#                 st.warning("Too many columns selected. Limit to 6 for interactive scatter matrix.")
#             else:
#                 fig = px.scatter_matrix(df[selected_scatter_cols])
#                 fig.update_traces(diagonal_visible=False)
#                 st.plotly_chart(fig, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# sns.set(style="whitegrid")

# def analyze_dataset(df: pd.DataFrame):
#     """Interactive visual analysis with column selection and safe plots."""
    
#     st.header("📊 Visual Analysis")
    
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()

#     # --------------------
#     # Summary Statistics
#     # --------------------
#     with st.expander("1️⃣ Summary Statistics", expanded=True):
#         if numeric_cols:
#             st.subheader("Numeric Columns")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.subheader("Categorical Columns")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)

#     # --------------------
#     # Column-wise Plotting
#     # --------------------
#     with st.expander("2️⃣ Column-wise Plotting", expanded=False):
#         col_type = st.radio("Choose column type", ["Numeric", "Categorical"], key="va_col_type")
        
#         if col_type == "Numeric" and numeric_cols:
#             chosen_col = st.selectbox("Select numeric column", numeric_cols, key="va_numeric_col")
#             plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"], key="va_numeric_plot")
            
#             fig, ax = plt.subplots(figsize=(6, 4))
#             if plot_type == "Histogram":
#                 sns.histplot(df[chosen_col], kde=True, bins=30, color="skyblue", ax=ax)
#             elif plot_type == "Boxplot":
#                 sns.boxplot(x=df[chosen_col], color="lightgreen", ax=ax)
#             elif plot_type == "Line Plot":
#                 ax.plot(df[chosen_col])
#                 ax.set_title(f"{chosen_col} Line Plot")
#             elif plot_type == "Scatter Plot":
#                 ax.scatter(df.index, df[chosen_col], alpha=0.6)
#                 ax.set_xlabel("Index")
#                 ax.set_ylabel(chosen_col)
#                 ax.set_title(f"{chosen_col} Scatter Plot")
#             st.pyplot(fig)

#         elif col_type == "Categorical" and categorical_cols:
#             chosen_col = st.selectbox("Select categorical column", categorical_cols, key="va_cat_col")
#             plot_type = st.selectbox("Plot type", ["Bar Plot", "Pie Chart"], key="va_cat_plot")
            
#             top_n = 20
#             value_counts = df[chosen_col].value_counts().nlargest(top_n)
#             fig, ax = plt.subplots(figsize=(6, 4))
#             if plot_type == "Bar Plot":
#                 sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             else:
#                 ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%")
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             st.pyplot(fig)

#     # --------------------
#     # Correlation Heatmap
#     # --------------------
#     if numeric_cols:
#         with st.expander("3️⃣ Correlation Heatmap", expanded=False):
#             selected_corr_cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols, key="va_corr_cols")
#             if len(selected_corr_cols) > 1:
#                 corr = df[selected_corr_cols].corr()
#                 fig, ax = plt.subplots(figsize=(max(6, len(selected_corr_cols)), max(4, len(selected_corr_cols)//2)))
#                 mask = None
#                 if len(selected_corr_cols) > 10:
#                     mask = np.triu(np.ones_like(corr, dtype=bool))
#                 sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, mask=mask)
#                 st.pyplot(fig)

#     # --------------------
#     # Interactive Scatter Matrix
#     # --------------------
#     if numeric_cols:
#         with st.expander("4️⃣ Interactive Scatter Matrix (Plotly)", expanded=False):
#             selected_scatter_cols = st.multiselect(
#                 "Select numeric columns (max 6 recommended)", numeric_cols, key="va_scatter_cols"
#             )
#             if len(selected_scatter_cols) > 1:
#                 if len(selected_scatter_cols) > 6:
#                     st.warning("Too many columns selected. Limit to 6 for interactive scatter matrix.")
#                 else:
#                     fig = px.scatter_matrix(df[selected_scatter_cols])
#                     fig.update_traces(diagonal_visible=False)
#                     st.plotly_chart(fig, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# sns.set(style="whitegrid")


# def analyze_dataset(df: pd.DataFrame):
#     """Interactive visual analysis with column selection and dynamic plots."""
#     st.header("📊 Visual Analysis")

#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()

#     # --------------------
#     # Summary Statistics
#     # --------------------
#     with st.expander("1️⃣ Summary Statistics", expanded=True):
#         if numeric_cols:
#             st.subheader("Numeric Columns")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.subheader("Categorical Columns")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)

#     # --------------------
#     # Column-wise Plotting
#     # --------------------
#     with st.expander("2️⃣ Column-wise Plotting", expanded=False):
#         col_type = st.radio("Choose column type", ["Numeric", "Categorical"], key="va_col_type")
        
#         if col_type == "Numeric" and numeric_cols:
#             chosen_cols = st.multiselect("Select numeric columns (1-2 recommended)", numeric_cols, key="va_numeric_cols")
#             plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"], key="va_numeric_plot")
            
#             for col in chosen_cols:
#                 fig, ax = plt.subplots(figsize=(6, 4))
#                 if plot_type == "Histogram":
#                     sns.histplot(df[col], kde=True, bins=30, color="skyblue", ax=ax)
#                 elif plot_type == "Boxplot":
#                     sns.boxplot(x=df[col], color="lightgreen", ax=ax)
#                 elif plot_type == "Line Plot":
#                     ax.plot(df[col])
#                     ax.set_title(f"{col} Line Plot")
#                 elif plot_type == "Scatter Plot":
#                     ax.scatter(df.index, df[col], alpha=0.6)
#                     ax.set_xlabel("Index")
#                     ax.set_ylabel(col)
#                     ax.set_title(f"{col} Scatter Plot")
#                 st.pyplot(fig)

#         elif col_type == "Categorical" and categorical_cols:
#             chosen_col = st.selectbox("Select categorical column", categorical_cols, key="va_cat_col")
#             plot_type = st.selectbox("Plot type", ["Bar Plot", "Pie Chart"], key="va_cat_plot")
            
#             top_n = 20
#             value_counts = df[chosen_col].value_counts().nlargest(top_n)
#             fig, ax = plt.subplots(figsize=(6, 4))
#             if plot_type == "Bar Plot":
#                 sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             else:
#                 ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%")
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             st.pyplot(fig)

#     # --------------------
#     # Correlation Heatmap
#     # --------------------
#     if numeric_cols:
#         with st.expander("3️⃣ Correlation Heatmap", expanded=False):
#             selected_corr_cols = st.multiselect(
#                 "Select numeric columns", numeric_cols, default=numeric_cols, key="va_corr_cols"
#             )
#             if len(selected_corr_cols) > 1:
#                 corr = df[selected_corr_cols].corr()
#                 fig, ax = plt.subplots(figsize=(max(6, len(selected_corr_cols)), max(4, len(selected_corr_cols)//2)))
#                 mask = np.triu(np.ones_like(corr, dtype=bool)) if len(selected_corr_cols) > 10 else None
#                 sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, mask=mask)
#                 st.pyplot(fig)

#     # --------------------
#     # Interactive Scatter Matrix
#     # --------------------
#     if numeric_cols:
#         with st.expander("4️⃣ Interactive Scatter Matrix (Plotly)", expanded=False):
#             selected_scatter_cols = st.multiselect(
#                 "Select numeric columns (max 6 recommended)", numeric_cols, key="va_scatter_cols"
#             )
#             if len(selected_scatter_cols) > 1:
#                 if len(selected_scatter_cols) > 6:
#                     st.warning("Too many columns selected. Limit to 6 for interactive scatter matrix.")
#                 else:
#                     fig = px.scatter_matrix(df[selected_scatter_cols], dimensions=selected_scatter_cols)
#                     fig.update_traces(diagonal_visible=False)
#                     st.plotly_chart(fig, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# sns.set(style="whitegrid")

# def summarize_columns(df: pd.DataFrame):
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
#     return numeric_cols, categorical_cols

# def analyze_dataset(df: pd.DataFrame):
#     """Interactive visual analysis with column selection and safe plots."""
#     st.header("📊 Visual Analysis")

#     numeric_cols, categorical_cols = summarize_columns(df)

#     # -------------------- Summary --------------------
#     with st.expander("1️⃣ Summary Statistics", expanded=True):
#         if numeric_cols:
#             st.subheader("Numeric Columns")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.subheader("Categorical Columns")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)

#     # -------------------- Column-wise Plots --------------------
#     with st.expander("2️⃣ Column-wise Plotting", expanded=False):
#         col_type = st.radio("Choose column type", ["Numeric", "Categorical"], key="va_col_type")
        
#         if col_type == "Numeric" and numeric_cols:
#             chosen_col = st.selectbox("Select numeric column", numeric_cols, key="va_numeric_col")
#             plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"], key="va_numeric_plot")
#             fig, ax = plt.subplots(figsize=(6,4))
#             if plot_type == "Histogram":
#                 sns.histplot(df[chosen_col], kde=True, bins=30, color="skyblue", ax=ax)
#             elif plot_type == "Boxplot":
#                 sns.boxplot(x=df[chosen_col], color="lightgreen", ax=ax)
#             elif plot_type == "Line Plot":
#                 ax.plot(df[chosen_col])
#                 ax.set_title(f"{chosen_col} Line Plot")
#             elif plot_type == "Scatter Plot":
#                 ax.scatter(df.index, df[chosen_col], alpha=0.6)
#                 ax.set_xlabel("Index")
#                 ax.set_ylabel(chosen_col)
#             st.pyplot(fig)

#         elif col_type == "Categorical" and categorical_cols:
#             chosen_col = st.selectbox("Select categorical column", categorical_cols, key="va_cat_col")
#             plot_type = st.selectbox("Plot type", ["Bar Plot", "Pie Chart"], key="va_cat_plot")
#             top_n = 20
#             value_counts = df[chosen_col].value_counts().nlargest(top_n)
#             fig, ax = plt.subplots(figsize=(6,4))
#             if plot_type == "Bar Plot":
#                 sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             else:
#                 ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%")
#                 ax.set_title(f"Top {top_n} values of {chosen_col}")
#             st.pyplot(fig)

#     # -------------------- Correlation Heatmap --------------------
#     if numeric_cols:
#         with st.expander("3️⃣ Correlation Heatmap", expanded=False):
#             selected_corr_cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:min(10,len(numeric_cols))], key="va_corr_cols")
#             if len(selected_corr_cols) > 1:
#                 corr = df[selected_corr_cols].corr()
#                 fig, ax = plt.subplots(figsize=(max(6,len(selected_corr_cols)), max(4,len(selected_corr_cols)//2)))
#                 mask = np.triu(np.ones_like(corr, dtype=bool)) if len(selected_corr_cols)>10 else None
#                 sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, mask=mask)
#                 st.pyplot(fig)

#     # -------------------- Interactive Scatter Matrix --------------------
#     if numeric_cols:
#         with st.expander("4️⃣ Interactive Scatter Matrix (Plotly)", expanded=False):
#             selected_scatter_cols = st.multiselect("Select numeric columns (max 6 recommended)", numeric_cols, key="va_scatter_cols")
#             if len(selected_scatter_cols) > 1:
#                 if len(selected_scatter_cols) > 6:
#                     st.warning("Too many columns selected. Limit to 6 for scatter matrix.")
#                 else:
#                     fig = px.scatter_matrix(df[selected_scatter_cols])
#                     fig.update_traces(diagonal_visible=False)
#                     st.plotly_chart(fig, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# sns.set(style="whitegrid")


# def _readiness_table(df: pd.DataFrame) -> pd.DataFrame:
#     """Small ‘what needs cleaning’ table to guide users."""
#     out = pd.DataFrame(index=df.columns)
#     out["dtype"] = df.dtypes.astype(str)
#     out["missing_%"] = (df.isnull().mean() * 100).round(2)
#     out["n_unique"] = df.nunique(dropna=True)

#     num_cols = df.select_dtypes(include="number").columns
#     out.loc[num_cols, "min"] = df[num_cols].min(numeric_only=True)
#     out.loc[num_cols, "max"] = df[num_cols].max(numeric_only=True)
#     return out.reset_index(names="column")


# def analyze_dataset(df: pd.DataFrame) -> None:
#     """Interactive visual analysis with multi-column plotting and safe charts."""
#     st.header("📊 Visual Analysis")

#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()

#     # 1) Readiness snapshot
#     with st.expander("1️⃣ Readiness Snapshot (missing, unique, ranges)", expanded=True):
#         st.dataframe(_readiness_table(df))

#     # 2) Summary statistics
#     with st.expander("2️⃣ Summary Statistics", expanded=False):
#         if numeric_cols:
#             st.subheader("Numeric Summary")
#             st.dataframe(df[numeric_cols].describe().T)
#         if categorical_cols:
#             st.subheader("Categorical Summary")
#             cat_summary = pd.DataFrame({
#                 "Unique": df[categorical_cols].nunique(),
#                 "Top": df[categorical_cols].mode().iloc[0],
#                 "Freq": df[categorical_cols].apply(lambda x: x.value_counts().max())
#             })
#             st.dataframe(cat_summary)

#     # 3) Column-wise plotting
#     with st.expander("3️⃣ Column-wise Plotting", expanded=False):
#         col_type = st.radio("Choose column type", ["Numeric", "Categorical"], key="va_coltype")

#         if col_type == "Numeric" and numeric_cols:
#             chosen_cols = st.multiselect("Select numeric columns", numeric_cols, key="va_numcols")
#             plot_type = st.selectbox(
#                 "Plot type", ["Histogram", "Boxplot", "Line Plot", "Scatter Plot"], key="va_numplot"
#             )
#             for col in chosen_cols:
#                 fig, ax = plt.subplots(figsize=(6, 4))
#                 if plot_type == "Histogram":
#                     sns.histplot(df[col], kde=True, bins=30, ax=ax)
#                     ax.set_title(f"{col} • Histogram")
#                 elif plot_type == "Boxplot":
#                     sns.boxplot(x=df[col], ax=ax)
#                     ax.set_title(f"{col} • Boxplot")
#                 elif plot_type == "Line Plot":
#                     ax.plot(df[col])
#                     ax.set_title(f"{col} • Line")
#                 else:  # Scatter Plot
#                     ax.scatter(df.index, df[col], alpha=0.6)
#                     ax.set_xlabel("Index")
#                     ax.set_ylabel(col)
#                     ax.set_title(f"{col} • Scatter")
#                 st.pyplot(fig)

#         if col_type == "Categorical" and categorical_cols:
#             chosen_cols = st.multiselect("Select categorical columns", categorical_cols, key="va_catcols")
#             plot_type = st.selectbox("Plot type", ["Bar Plot", "Pie Chart"], key="va_catplot")
#             top_n = st.slider("Top categories to show", 5, 30, 20, key="va_cattop")
#             for col in chosen_cols:
#                 vc = df[col].value_counts().nlargest(top_n)
#                 fig, ax = plt.subplots(figsize=(6, 4))
#                 if plot_type == "Bar Plot":
#                     sns.barplot(x=vc.values, y=vc.index, ax=ax)
#                     ax.set_title(f"{col} • Top {top_n}")
#                 else:
#                     ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%")
#                     ax.set_title(f"{col} • Top {top_n}")
#                 st.pyplot(fig)

#     # 4) Correlation heatmap (with smart selection)
#     if numeric_cols:
#         with st.expander("4️⃣ Correlation Heatmap", expanded=False):
#             mode = st.radio(
#                 "Column selection mode",
#                 ["Manual pick", "Auto: Top-K by variance"],
#                 horizontal=True,
#                 key="va_corrmode",
#             )

#             selected_cols = []
#             if mode == "Manual pick":
#                 selected_cols = st.multiselect(
#                     "Select numeric columns", numeric_cols, default=numeric_cols, key="va_corrcols"
#                 )
#             else:
#                 k = st.slider("K (auto-select)", 3, min(30, len(numeric_cols)), min(10, len(numeric_cols)), key="va_k")
#                 # pick by variance
#                 variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
#                 selected_cols = variances.head(k).index.tolist()
#                 st.caption(f"Auto-selected: {', '.join(selected_cols)}")

#             if len(selected_cols) > 1:
#                 corr = df[selected_cols].corr()
#                 # Large matrices look cleaner with an upper-triangle mask
#                 use_mask = len(selected_cols) > 10
#                 mask = np.triu(np.ones_like(corr, dtype=bool)) if use_mask else None

#                 fig, ax = plt.subplots(
#                     figsize=(max(6, 0.7 * len(selected_cols)), max(4, 0.5 * len(selected_cols)))
#                 )
#                 sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, mask=mask)
#                 st.pyplot(fig)

#     # 5) Interactive scatter matrix (guarded)
#     if numeric_cols:
#         with st.expander("5️⃣ Interactive Scatter Matrix (Plotly)", expanded=False):
#             scatter_cols = st.multiselect(
#                 "Select numeric columns (≤ 6 recommended)", numeric_cols, key="va_scatter_cols"
#             )
#             if 2 <= len(scatter_cols) <= 6:
#                 fig = px.scatter_matrix(df[scatter_cols])
#                 fig.update_traces(diagonal_visible=False)
#                 st.plotly_chart(fig, use_container_width=True)
#             elif len(scatter_cols) > 6:
#                 st.warning("Too many columns selected. Please select 6 or fewer.")

# """
# Upgraded visual analysis helpers + EDA report generator.
# Save as: echoAnalytics/modules/visual_analysis.py
# """

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

import pypandoc
pypandoc.download_pandoc()


sns.set(style="whitegrid")


def _sample_df_for_plot(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Return a sampled dataframe for plotting to keep UI responsive."""
    if df.shape[0] <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s)


def _build_readiness_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.columns)
    out["dtype"] = df.dtypes.astype(str)
    out["missing_%"] = (df.isnull().mean() * 100).round(2)
    out["n_unique"] = df.nunique(dropna=True)
    num_cols = df.select_dtypes(include="number").columns
    out.loc[num_cols, "min"] = df[num_cols].min(numeric_only=True)
    out.loc[num_cols, "max"] = df[num_cols].max(numeric_only=True)
    return out.reset_index(names="column")


# -----------------------------
# EDA report generation (HTML)
# -----------------------------
def generate_eda_report(df: pd.DataFrame, sample_n: int = 1000) -> bytes:
    """
    Build a simple standalone HTML EDA report and return bytes.
    The report contains summary stats and a few interactive Plotly charts.
    """
    df_sample = df.sample(n=min(sample_n, len(df)), random_state=42) if len(df) > sample_n else df
    html_parts = []
    # Title
    html_parts.append("<html><head><meta charset='utf-8'><title>EDA Report</title></head><body>")
    html_parts.append("<h1>Auto EDA Report</h1>")

    # Basic table
    html_parts.append("<h2>Dataset snapshot</h2>")
    html_parts.append(df_sample.head(10).to_html(index=False))

    # Summary stats
    html_parts.append("<h2>Summary Statistics (numeric)</h2>")
    if not df.select_dtypes(include="number").empty:
        html_parts.append(df.select_dtypes(include="number").describe().to_html())

    # Top categorical counts
    cats = df.select_dtypes(include="object").columns.tolist()
    if cats:
        html_parts.append("<h2>Categorical top values</h2>")
        for c in cats[:10]:
            vc = df[c].value_counts().nlargest(10)
            html_parts.append(f"<h3>{c}</h3>")
            html_parts.append(vc.to_frame(name="count").to_html())

    # Add a few interactive plotly charts -> embed via to_html
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:8]
    if numeric_cols:
        try:
            fig = px.histogram(df_sample, x=numeric_cols[0], marginal="box", nbins=30, title=f"Distribution: {numeric_cols[0]}")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        except Exception:
            pass

        if len(numeric_cols) >= 2:
            try:
                fig = px.scatter(df_sample, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            except Exception:
                pass

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)
    return html.encode("utf-8")


# -----------------------------
# Main interactive UI
# -----------------------------
def analyze_dataset(df: pd.DataFrame) -> None:
    """Render interactive visual analysis. Uses Plotly for interactivity and matplotlib/seaborn for quick plots."""

    st.header("📊 Visual Analysis & EDA tools")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # readiness
    with st.expander("1️⃣ Readiness snapshot", expanded=True):
        st.dataframe(_build_readiness_table(df))

    # quick EDA report
    with st.expander("2️⃣ One-click EDA report (downloadable HTML)", expanded=False):
        st.write("Generates an HTML report containing summary tables and a few interactive charts.")
        col1, col2 = st.columns([3, 1])
        with col1:
            sample_n = st.number_input("Sample rows for plots", min_value=200, max_value=10000, value=1000, step=100, key="va_report_sample")
        with col2:
            if st.button("Generate & Download EDA Report", key="va_gen_report"):
                report_bytes = generate_eda_report(df, sample_n=sample_n)
                st.download_button("Download EDA report (HTML)", report_bytes, file_name="eda_report.html", mime="text/html")

    # column-wise plotting grid
    with st.expander("3️⃣ Column plots (grid)", expanded=False):
        # choose columns
        ctype = st.radio("Column type", ["Numeric", "Categorical", "Datetime"], key="va_grid_type")
        sample_df = _sample_df_for_plot(df, max_rows=5000)

        if ctype == "Numeric":
            pick = st.multiselect("Select numeric columns (compare multiple)", numeric_cols, default=numeric_cols[:2], key="va_grid_numcols")
            plot_kind = st.selectbox("Plot", ["Histogram (KDE)", "Boxplot", "Line (index)", "Density (KDE)"], key="va_grid_num_plot")
            if pick:
                # render two-column grid
                cols = st.columns(2)
                for i, col in enumerate(pick):
                    with cols[i % 2]:
                        st.markdown(f"**{col}**")
                        if plot_kind in {"Histogram (KDE)", "Density (KDE)"}:
                            # use plotly histogram with KDE-like marginal
                            try:
                                fig = px.histogram(sample_df, x=col, nbins=40, marginal="rug" if plot_kind=="Histogram (KDE)" else None)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                fig, ax = plt.subplots(figsize=(6,3))
                                sns.histplot(sample_df[col].dropna(), kde=True, ax=ax)
                                st.pyplot(fig)
                        elif plot_kind == "Boxplot":
                            fig, ax = plt.subplots(figsize=(6,3))
                            sns.boxplot(x=sample_df[col].dropna(), ax=ax)
                            st.pyplot(fig)
                        else:
                            fig = px.line(sample_df.reset_index(), x=sample_df.reset_index().index, y=col, title=f"{col} over index")
                            st.plotly_chart(fig, use_container_width=True)

        elif ctype == "Categorical":
            pick = st.multiselect("Select categorical columns", categorical_cols, key="va_grid_catcols")
            topn = st.slider("Top categories to show", 3, 50, 10, key="va_grid_cat_topn")
            if pick:
                cols = st.columns(2)
                for i, col in enumerate(pick):
                    with cols[i % 2]:
                        vc = df[col].value_counts().nlargest(topn)
                        fig = px.bar(x=vc.values, y=vc.index, orientation="h", labels={"x":"count","y":col})
                        st.plotly_chart(fig, use_container_width=True)

        else:  # Datetime
            if not datetime_cols:
                st.info("No datetime columns detected. Convert a column to datetime to enable time-series plots.")
            else:
                pick = st.selectbox("Select datetime column", datetime_cols, key="va_grid_dtcol")
                numeric_for_ts = st.multiselect("Numeric columns to plot vs time", numeric_cols, default=numeric_cols[:1], key="va_grid_ts_numcols")
                if pick and numeric_for_ts:
                    sample_ts = df[[pick] + numeric_for_ts].dropna().sort_values(pick)
                    # sample to keep responsive
                    sample_ts = _sample_df_for_plot(sample_ts, max_rows=5000)
                    for col in numeric_for_ts:
                        fig = px.line(sample_ts, x=pick, y=col, title=f"{col} over {pick}")
                        st.plotly_chart(fig, use_container_width=True)

    # correlation (advanced)
    if numeric_cols:
        with st.expander("4️⃣ Correlations & advanced visuals", expanded=False):
            corr_mode = st.radio("Select mode", ["Manual select", "Auto top-K by variance"], key="va_corr_mode")
            selected = []
            if corr_mode == "Manual select":
                selected = st.multiselect("Columns", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))], key="va_corr_manual")
            else:
                k = st.slider("Top-K by variance", 3, min(30, len(numeric_cols)), value=min(8, len(numeric_cols)), key="va_corr_k")
                vars_sorted = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
                selected = vars_sorted.head(k).index.tolist()
                st.caption(f"Auto-selected: {', '.join(selected)}")

            if len(selected) > 1:
                # heatmap (masked if wide)
                corr = df[selected].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool)) if len(selected) > 12 else None
                fig, ax = plt.subplots(figsize=(max(6, 0.6*len(selected)), max(4, 0.45*len(selected))))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, mask=mask)
                st.pyplot(fig)

                # guarded scatter matrix (plotly)
                if len(selected) <= 6:
                    if st.checkbox("Show interactive scatter matrix (Plotly)", value=False, key="va_show_scatter"):
                        fig2 = px.scatter_matrix(_sample_df_for_plot(df[selected], max_rows=2000), dimensions=selected)
                        fig2.update_traces(diagonal_visible=False)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Select 6 or fewer columns for interactive scatter matrix to avoid heavy rendering.")

    st.caption("Tip: when dataset is large, plots are sampled for responsiveness.")

# Add at the top
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import pypandoc


# -----------------------------
# EDA report generation (PDF & Markdown)
# -----------------------------
def generate_eda_report_pdf(df: pd.DataFrame, filename: str = "eda_report.pdf") -> bytes:
    """Generate a simple EDA report as PDF and return as bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Auto EDA Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Basic info
    elements.append(Paragraph(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Summary statistics
    if not df.select_dtypes(include="number").empty:
        desc = df.describe().reset_index()
        data = [desc.columns.tolist()] + desc.values.tolist()
        table = Table(data)
        elements.append(Paragraph("Summary Statistics (numeric):", styles["Heading2"]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Categorical counts
    cats = df.select_dtypes(include="object").columns.tolist()
    if cats:
        elements.append(Paragraph("Categorical Top Values:", styles["Heading2"]))
        for c in cats[:5]:
            vc = df[c].value_counts().nlargest(10)
            data = [[c, "Count"]] + [[idx, val] for idx, val in vc.items()]
            table = Table(data)
            elements.append(Paragraph(f"Column: {c}", styles["Heading3"]))
            elements.append(table)
            elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_eda_report_md(df: pd.DataFrame) -> bytes:
    """Generate EDA report as Markdown and return bytes."""
    md_parts = []
    md_parts.append("# Auto EDA Report\n")

    # Basic info
    md_parts.append(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}\n")

    # Summary stats
    if not df.select_dtypes(include="number").empty:
        md_parts.append("## Summary Statistics (numeric)\n")
        md_parts.append(df.describe().to_markdown())

    # Categorical counts
    cats = df.select_dtypes(include="object").columns.tolist()
    if cats:
        md_parts.append("\n## Categorical Top Values\n")
        for c in cats[:5]:
            vc = df[c].value_counts().nlargest(10)
            md_parts.append(f"### {c}\n")
            md_parts.append(vc.to_frame().to_markdown())

    md_text = "\n\n".join(md_parts)
    # Convert to standalone markdown (pandoc ensures proper encoding)
    output = pypandoc.convert_text(md_text, "md", format="md", extra_args=["--standalone"])
    return output.encode("utf-8")