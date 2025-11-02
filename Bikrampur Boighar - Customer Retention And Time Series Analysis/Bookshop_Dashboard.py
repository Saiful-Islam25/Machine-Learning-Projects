import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Bikrampur Boighar Interactive Dashboard (Excel Supported)")

# File uploader (CSV or Excel)
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Detect file type and read accordingly
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # 🧹 DATA CLEANING
    # ==========================
    if 'Repeated Customer?' in df.columns:
        df['Repeated_Customer_flag'] = (
            df['Repeated Customer?']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                'yes': 1, 'y': 1, 'true': 1, '1': 1, 'হ্যাঁ': 1,
                'no': 0, 'n': 0, 'false': 0, '0': 0, 'না': 0
            })
            .fillna(0)
            .astype(int)
        )
    else:
        st.warning("⚠️ 'Repeated Customer' column not found!")

    # ==========================
    # 🔍 SUMMARY METRICS
    # ==========================
    total_sales = df['Sell'].sum() if 'Sell' in df.columns else 0
    total_customers = len(df)
    repeated_customers = df['Repeated_Customer_flag'].sum()
    new_customers = total_customers - repeated_customers

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Sales", f"{total_sales}")
    col2.metric("👥 Total Customers", f"{total_customers}")
    col3.metric("🔁 Repeated Customers", f"{repeated_customers}")

    # ==========================
    # 📊 CHARTS
    # ==========================
    st.subheader("📚 Category-wise Sales")
    if 'Category Of Books' in df.columns and 'Sell' in df.columns:
        cat_sales = df.groupby('Category Of Books')['Sell'].sum().reset_index()
        fig1 = px.bar(cat_sales, x='Category Of Books', y='Sell', color='Category Of Books',
                      title="Sales by Book Category", text='Sell')
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

    if 'Gender' in df.columns and 'Sell' in df.columns:
        st.subheader("🧍 Gender-wise Sales Distribution")
        gender_sales = df.groupby('Gender')['Sell'].sum().reset_index()
        fig2 = px.pie(gender_sales, values='Sell', names='Gender', title="Gender-wise Sales Share")
        st.plotly_chart(fig2, use_container_width=True)

    if 'Year' in df.columns:
        st.subheader("📆 Year-wise Repeated Customer Count")
        yearly_repeat = df.groupby('Year')['Repeated_Customer_flag'].sum().reset_index()
        fig3 = px.bar(yearly_repeat, x='Year', y='Repeated_Customer_flag',
                      text='Repeated_Customer_flag', color='Year',
                      title='Year-wise Repeated Customer Count')
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)

    # ==========================
    # 🎯 FILTER OPTION
    # ==========================
    st.subheader("🎯 Filter by Category")
    if 'Category Of Books' in df.columns:
        categories = st.multiselect("Select categories:", df['Category Of Books'].unique())
        if categories:
            filtered_df = df[df['Category Of Books'].isin(categories)]
            st.dataframe(filtered_df)
            st.success(f"Filtered {len(filtered_df)} records")

else:
    st.info("📂 Please upload your Excel (.xlsx) or CSV file to start.")
