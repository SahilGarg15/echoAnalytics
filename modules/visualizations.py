"""Visualization helpers for data cleaning."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_histogram(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    st.pyplot(fig)

def plot_boxplot(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column].dropna(), ax=ax)
    ax.set_title(f"Boxplot of {column}")
    st.pyplot(fig)

def plot_bar(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots()
    counts = df[column].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f"Bar Plot of {column}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
