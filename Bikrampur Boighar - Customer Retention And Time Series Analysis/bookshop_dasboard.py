# bookshop_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import os

st.set_page_config(layout="wide", page_title="Bikrampur Boighar Dashboard (Pro)")

# ----------------- CONFIG -----------------
DATA_PATH = r"C:/Users/Saiful/Bookshop_Project/"  # <- আপনার ফোল্ডার
FILES = {
    "Sell-2023": "Sell-2023.xlsx",
    "Sell-2024": "Sell-2024.xlsx",
    "Sell-2025": "Sell-2025.xlsx",
    "All Books": "All Book List.xlsx"
}

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data(file_key):
    file_path = os.path.join(DATA_PATH, FILES[file_key])
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Could not read {file_key}: {e}")
        return None
    # clean columns
    df.columns = df.columns.str.strip()
    # convert numeric columns
    for col in ["Sell","Cost","Unit","Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Profit
    if "Sell" in df.columns and "Cost" in df.columns:
        df["Profit"] = df["Sell"] - df["Cost"]
    else:
        df["Profit"] = np.nan
    # Repeated Customer
    if "Repeated Customer?" in df.columns:
        df['Repeated_Customer_flag'] = (
            df['Repeated Customer?'].astype(str).str.strip().str.lower().replace({
                'yes':1,'y':1,'true':1,'1':1,'হ্যাঁ':1,
                'no':0,'n':0,'false':0,'0':0,'না':0
            })
        )
        df['Repeated_Customer_flag'] = pd.to_numeric(df['Repeated_Customer_flag'], errors='coerce').fillna(0).astype(int)
    else:
        df['Repeated_Customer_flag'] = np.nan
    # Date
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df["Month"] = df["Order Date"].dt.month
        df["Year"] = df["Order Date"].dt.year.fillna(2025).astype(int)
    else:
        df["Month"] = np.nan
        df["Year"] = 2025
    # Age
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df

# ----------------- SIDEBAR -----------------
st.sidebar.header("📂 Select Dataset")
dataset_choice = st.sidebar.selectbox("Choose file to analyze", list(FILES.keys()), index=0)
df = load_data(dataset_choice)
if df is None:
    st.stop()

# ----------------- FILTERS -----------------
st.sidebar.header("🎛 Filters")
# Year
available_years = sorted(df["Year"].dropna().unique().astype(int).tolist())
year_sel = st.sidebar.multiselect("Year", available_years, default=available_years)
# Category
cat_sel = st.sidebar.multiselect("Category", sorted(df["Category Of Books"].dropna().unique().tolist()) if "Category Of Books" in df.columns else [], default=None)
# Gender
gender_sel = st.sidebar.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else [], default=None)
# Delivery
delivery_sel = st.sidebar.multiselect("Delivery Method", sorted(df["Delivery Method"].dropna().unique().tolist()) if "Delivery Method" in df.columns else [], default=None)
# Top N books
top_n = st.sidebar.slider("Top N books (for table)", 5, 50, 10)

# Apply filters
df_filtered = df.copy()
if year_sel:
    df_filtered = df_filtered[df_filtered["Year"].isin(year_sel)]
if cat_sel:
    df_filtered = df_filtered[df_filtered["Category Of Books"].isin(cat_sel)]
if gender_sel:
    df_filtered = df_filtered[df_filtered["Gender"].isin(gender_sel)]
if delivery_sel:
    df_filtered = df_filtered[df_filtered["Delivery Method"].isin(delivery_sel)]

plot_df = df_filtered.copy()
plot_df["Sell"] = pd.to_numeric(plot_df["Sell"], errors="coerce")
plot_df["Profit"] = pd.to_numeric(plot_df["Profit"], errors="coerce")
plot_df["Unit"] = pd.to_numeric(plot_df.get("Unit",1), errors="coerce").fillna(1)
plot_df["Age"] = pd.to_numeric(plot_df["Age"], errors="coerce")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Overview","🚚 Delivery & Profit","👥 Customers & Segmentation","🧠 ML Predictions"])

# ----------------- TAB 1: Sales Overview -----------------
with tab1:
    st.subheader("Sales Overview")
    col1,col2,col3,col4 = st.columns(4)
    total_sell = int(plot_df["Sell"].sum() if not plot_df["Sell"].isna().all() else 0)
    total_orders = int(len(plot_df))
    total_books = int(plot_df["Name of Books"].nunique()) if "Name of Books" in plot_df.columns else 0
    pct_rep = plot_df["Repeated_Customer_flag"].mean()*100 if "Repeated_Customer_flag" in plot_df.columns else 0
    col1.metric("💰 Total Sell", f"{total_sell:,}")
    col2.metric("🧾 Total Orders", f"{total_orders:,}")
    col3.metric("📚 Unique Books", f"{total_books:,}")
    col4.metric("🔁 Repeated (%)", f"{pct_rep:.1f}%")

    st.markdown("### Monthly Sales Trend")
    monthly = plot_df.groupby("Month")["Sell"].sum().reindex(range(1,13), fill_value=0)
    fig = px.line(x=monthly.index, y=monthly.values, labels={"x":"Month","y":"Sell"}, title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Selling Books")
    if "Name of Books" in plot_df.columns:
        top_books = plot_df.groupby("Name of Books")["Unit"].sum().sort_values(ascending=False).head(top_n)
        st.dataframe(top_books.reset_index().rename(columns={"Unit":"Units Sold"}).head(top_n))

# ----------------- TAB 2: Delivery & Profit -----------------
with tab2:
    st.subheader("Delivery & Profit")
    if "Delivery Method" in plot_df.columns:
        delivery_counts = plot_df["Delivery Method"].value_counts()
        fig = px.pie(values=delivery_counts.values, names=delivery_counts.index, title="Delivery Method Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    if "Category Of Books" in plot_df.columns:
        cat_profit = plot_df.groupby("Category Of Books")["Profit"].sum().sort_values(ascending=False)
        fig = px.bar(x=cat_profit.index, y=cat_profit.values, labels={"x":"Category","y":"Profit"}, title="Profit by Category")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 3: Customer Insights -----------------
with tab3:
    st.subheader("Customer Insights & Segmentation")
    if "Gender" in plot_df.columns:
        gender_ret = plot_df.groupby("Gender")["Repeated_Customer_flag"].mean()*100
        fig = px.bar(x=gender_ret.index, y=gender_ret.values, labels={"x":"Gender","y":"Retention (%)"}, title="Gender-wise Retention (%)")
        st.plotly_chart(fig, use_container_width=True)

    if "Age" in plot_df.columns:
        fig = px.histogram(plot_df, x="Age", nbins=12, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # KMeans clustering
    seg_features = [c for c in ["Age","Unit","Sell","Profit"] if c in plot_df.columns]
    if len(seg_features)>=2:
        seg_df = plot_df[seg_features].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(seg_df)
        k = st.slider("Number of clusters", 2,6,3)
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(Xs)
        seg_df = seg_df.assign(Cluster=labels)
        st.write("Cluster centers:")
        centers = scaler.inverse_transform(km.cluster_centers_)
        st.dataframe(pd.DataFrame(centers, columns=seg_features))

# ----------------- TAB 4: ML Predictions -----------------
with tab4:
    st.subheader("ML Predictions")
    features_for_model = [c for c in ["Age","Gender","Unit","Profit","Sell"] if c in plot_df.columns]

    if len(features_for_model)<2:
        st.info("Not enough features to train models.")
    else:
        ml_df = plot_df[features_for_model + ["Repeated_Customer_flag"]].dropna()
        if len(ml_df)<30:
            st.warning("Too few rows for reliable ML predictions.")
        else:
            # Regression: Sell
            if "Sell" in features_for_model:
                Xr = ml_df[[c for c in features_for_model if c!="Sell"]]
                yr = ml_df["Sell"]
                Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
                reg = LinearRegression().fit(Xr_train, yr_train)
                r2 = reg.score(Xr_test, yr_test)
                st.write(f"Sell Regression R²: {r2:.3f}")
            else:
                reg = None

            # Classification: Retention
            if "Repeated_Customer_flag" in ml_df.columns:
                Xc = ml_df[[c for c in features_for_model if c!="Repeated_Customer_flag" and c!="Sell"]]
                yc = ml_df["Repeated_Customer_flag"]
                if len(yc.unique())>1:
                    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
                    clf = LogisticRegression(max_iter=1000).fit(Xc_train, yc_train)
                    acc = clf.score(Xc_test, yc_test)
                    st.write(f"Retention Classifier Accuracy: {acc:.3f}")
                else:
                    clf = None
            else:
                clf = None

            # Predict form
            st.markdown("### Make a Prediction")
            with st.form("predict_form"):
                inputs = {}
                columns_for_input = Xr.columns if reg is not None else (Xc.columns if clf is not None else [])
                for f in columns_for_input:
                    if f=="Gender":
                        opts = sorted(plot_df["Gender"].dropna().unique().tolist())
                        inputs[f] = st.selectbox(f, opts, index=0)
                    else:
                        val = float(plot_df[f].median()) if f in plot_df and not plot_df[f].dropna().empty else 0.0
                        inputs[f] = st.number_input(f, value=val)
                submitted = st.form_submit_button("Predict")
                if submitted:
                    X_new = pd.DataFrame([inputs])
                    if reg is not None:
                        sell_pred = reg.predict(X_new)[0]
                        st.success(f"Predicted Sell ≈ {sell_pred:,.0f}")
                    if clf is not None:
                        retain_prob = clf.predict_proba(X_new)[0][1] if hasattr(clf,"predict_proba") else clf.predict(X_new)[0]
                        if hasattr(clf,"predict_proba"):
                            st.success(f"Repeat Purchase Probability ≈ {retain_prob:.2%}")
                        else:
                            st.success(f"Repeat Purchase: {'Yes' if retain_prob==1 else 'No'}")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("Made with ❤️ by Saiful Islam — Bikrampur Boighar")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "filtered_data.csv", "text/csv")
