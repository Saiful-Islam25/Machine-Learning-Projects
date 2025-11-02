# bookshop_dashboard_final.py
"""
Bikrampur Boighar — Full Dashboard (with Deep Time Series Analysis)
Place this file and set DATA_PATH to the folder where your Excel files are located.
This script will try to locate files by fuzzy matching names you provided:
  Sell- 2023.xlxs , Sell - 2024.xlxs, Sell - 2025.xlxs, All Book List.xlxs
Note: common correct extension is .xlsx — the code attempts to handle small typos.
"""

import os
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Bikrampur Boighar — Pro Dashboard (Final)")

# ----------------- USER CONFIG -----------------
# <-- UPDATE THIS to the folder where your Excel files are stored -->
DATA_PATH = r"C:\Users\MY COMPUTER\Downloads\Bookshop"  # <- change this to your folder

# What filenames you *think* you have (from your message). We will fuzzy-search these.
EXPECTED_NAMES = [
    "Sell- 2023.xlsx",
    "Sell - 2024.xlsx",
    "Sell - 2025.xlsx",
    "All Book List.xlsx"
]

# accepted extensions to attempt
EXTS = [".xlsx", ".xls", ".csv"]

# ----------------- helper: find file by fuzzy name -----------------
def find_file_by_keywords(folder, keywords):
    """
    Search folder for a filename containing all keywords (case-insensitive),
    try different extensions, return first match Path or None.
    """
    folder = Path(folder)
    files = list(folder.iterdir()) if folder.exists() else []
    keywords = [re.sub(r'[^a-z0-9]', '', k.lower()) for k in keywords.split()]
    for f in files:
        name = f.name.lower()
        name_simple = re.sub(r'[^a-z0-9]', '', name)
        # ensure all keywords tokens present in filename (in order not required)
        if all(k in name_simple for k in keywords):
            return f
    # fallback: try contains whole phrase (with possible extension differences)
    for ext in EXTS:
        candidate = folder / (keywords[0] + ext)  # not very reliable, but try
        if candidate.exists():
            return candidate
    return None

def locate_file(folder, expected_phrase):
    """Return Path if found via fuzzy search, else None."""
    folder = Path(folder)
    if not folder.exists():
        return None
    # Search by tokens in expected_phrase against filenames
    tokens = re.sub(r'[^a-z0-9 ]', ' ', expected_phrase.lower()).split()
    tokens = [t for t in tokens if t]
    # Primary scan: filenames containing all tokens
    for f in folder.iterdir():
        if not f.is_file():
            continue
        fname = f.name.lower()
        fname_simple = re.sub(r'[^a-z0-9]', '', fname)
        if all(token in fname_simple for token in tokens):
            return f
    # Secondary: try matching year if phrase has year
    years = re.findall(r'20\d{2}', expected_phrase)
    if years:
        for y in years:
            for f in folder.iterdir():
                if y in f.name:
                    return f
    # tertiary: try any file that contains any token
    for f in folder.iterdir():
        fname = f.name.lower()
        if any(token in fname for token in tokens):
            return f
    return None

# ----------------- load Excel with fallback -----------------
def read_any_table(path):
    path = Path(path)
    if not path.exists():
        return None
    # if CSV
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # try excel
    try:
        return pd.read_excel(path)
    except Exception:
        try:
            # openpyxl sometimes fails on corrupted extension names; try engine None
            return pd.read_excel(path, engine='openpyxl')
        except Exception:
            # last attempt: read as csv
            try:
                return pd.read_csv(path)
            except Exception:
                return None

# ----------------- locate all required files -----------------
st.sidebar.header("📂 Data files & settings")
st.sidebar.write("Looking for your files in:")
st.sidebar.write(f"**{DATA_PATH}**")

found = {}
p = Path(DATA_PATH)
if not p.exists():
    st.error(f"DATA_PATH does not exist: {DATA_PATH}. Please update DATA_PATH variable at top of script.")
    st.stop()

for expect in EXPECTED_NAMES:
    fpath = locate_file(p, expect)
    found[expect] = fpath

# Show results in sidebar
st.sidebar.subheader("Detected files")
for name, path in found.items():
    if path is None:
        st.sidebar.write(f":x: {name} — **NOT FOUND**")
    else:
        st.sidebar.write(f":white_check_mark: {name} — `{path.name}`")

# If essential sales files missing, warn and stop
#if found["sell2023".replace("sell","sell")] is None and all(found[k] is None for k in found if "sell" in k):
    # fallback: require at least one sell file
    #pass  # we'll still attempt but warn later

# ----------------- LOAD & MERGE DATA -----------------
@st.cache_data
def load_and_merge_data(data_path, expected_names):
    # try to assemble list of sales files for 2023-2025
    folder = Path(data_path)
    sales_dfs = []
    years_order = []
    for expect in ["sell 2023", "sell 2024", "sell 2025"]:
        file_path = locate_file(folder, expect)
        if file_path is None:
            # try alternate patterns e.g. with hyphen or spaces
            # search for year token alone
            year_token = re.findall(r'20\d{2}', expect)
            if year_token:
                candidates = [f for f in folder.iterdir() if year_token[0] in f.name]
                file_path = candidates[0] if candidates else None
        if file_path is None:
            # skip missing year file
            continue
        df_temp = read_any_table(file_path)
        if df_temp is None:
            continue
        # add Year column if not present and infer from filename
        yr = None
        try:
            # try to find 4-digit year in filename
            y = re.search(r'20\d{2}', file_path.name)
            yr = int(y.group(0)) if y else None
        except Exception:
            yr = None
        if 'Year' not in df_temp.columns and yr is not None:
            df_temp['Year'] = yr
        sales_dfs.append(df_temp)
        years_order.append(yr)
    # merge if any found
    if sales_dfs:
        df_sales = pd.concat(sales_dfs, ignore_index=True)
    else:
        df_sales = pd.DataFrame()  # empty

    # load books metadata if exists
    book_file = locate_file(folder, "all book list")
    if book_file is not None:
        df_books = read_any_table(book_file)
    else:
        df_books = pd.DataFrame()

    # normalize 'Name of Books' for merging
    if not df_sales.empty and 'Name of Books' in df_sales.columns:
        df_sales['Name of Books'] = df_sales['Name of Books'].astype(str).str.strip().str.lower()
    if not df_books.empty and 'Name of Books' in df_books.columns:
        df_books['Name of Books'] = df_books['Name of Books'].astype(str).str.strip().str.lower()

    if not df_books.empty and 'Name of Books' in df_books.columns and 'Name of Books' in df_sales.columns:
        df = df_sales.merge(df_books.drop_duplicates(subset=['Name of Books']), on='Name of Books', how='left')
    else:
        df = df_sales.copy()

    # standard clean up: strip column names
    df.columns = df.columns.str.strip()

    # Convert Order Date
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

    # Numeric conversions & clean Sell values that might be in "100.088k" formats
    def convert_kmb(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            if s == '':
                return np.nan
            # remove commas and currency signs
            s = s.replace(',', '').replace('৳', '').replace('tk', '').strip()
            m = re.match(r'^([0-9,.]+)\s*([kKmMbB])$', s)
            if m:
                num = float(m.group(1))
                suf = m.group(2).lower()
                if suf == 'k':
                    return num * 1_000
                if suf == 'm':
                    return num * 1_000_000
                if suf == 'b':
                    return num * 1_000_000_000
            # try to directly parse
            return float(s)
        except:
            # attempt to extract digits
            digits = re.findall(r'[\d.]+', str(x))
            return float(digits[0]) if digits else np.nan

    for col in ['Sell', 'Cost', 'Unit', 'Price']:
        if col in df.columns:
            df[col] = df[col].apply(convert_kmb)

    # Profit
    if 'Sell' in df.columns and 'Cost' in df.columns:
        df['Profit'] = df['Sell'] - df['Cost']
    else:
        df['Profit'] = np.nan

    # Repeated Customer flag: normalize
    if 'Repeated Customer?' in df.columns:
        df['Repeated_Customer_flag'] = (
            df['Repeated Customer?'].astype(str)
            .str.strip()
            .str.lower()
            .replace({
                'yes':1, 'y':1, 'true':1, '1':1, 'হ্যাঁ':1, 'yes ':1,
                'no':0, 'n':0, 'false':0, '0':0, 'না':0
            })
        )
        df['Repeated_Customer_flag'] = pd.to_numeric(df['Repeated_Customer_flag'], errors='coerce').fillna(0).astype(int)
    else:
        # try alternate column name variants
        alt_cols = [c for c in df.columns if 'repeated' in c.lower() and 'customer' in c.lower()]
        if alt_cols:
            col0 = alt_cols[0]
            df['Repeated_Customer_flag'] = (
                df[col0].astype(str)
                .str.strip()
                .str.lower()
                .replace({
                    'yes':1, 'y':1, 'true':1, '1':1, 'হ্যাঁ':1,
                    'no':0, 'n':0, 'false':0, '0':0, 'না':0
                })
            )
            df['Repeated_Customer_flag'] = pd.to_numeric(df['Repeated_Customer_flag'], errors='coerce').fillna(0).astype(int)
        else:
            df['Repeated_Customer_flag'] = np.nan

    # Age numeric
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Month and Year fields
    if 'Order Date' in df.columns:
        df['Year'] = df['Order Date'].dt.year.fillna(df.get('Year', np.nan)).astype('Int64')
        df['Month'] = df['Order Date'].dt.month
    else:
        # if no order date, keep Year if present, else NaN
        df['Year'] = df.get('Year', pd.Series([np.nan]*len(df)))
        df['Month'] = np.nan

    return df

# Load data
df = load_and_merge_data(DATA_PATH, EXPECTED_NAMES)

if df is None or df.empty:
    st.error("No sales data found in DATA_PATH. Please check DATA_PATH and filenames.")
    st.stop()

# show small preview
st.title("📚 Bikrampur Boighar — Customer Retention & Time Series Dashboard (Final)")

with st.expander("Preview of loaded data (first 5 rows)"):
    st.dataframe(df.head())

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filters & Settings")
years_avail = sorted([int(x) for x in df['Year'].dropna().unique().astype(int).tolist()]) if 'Year' in df.columns else []
year_sel = st.sidebar.multiselect("Year", years_avail, default=years_avail if years_avail else None)
cat_sel = st.sidebar.multiselect("Category", sorted(df['Category Of Books'].dropna().unique().tolist()) if 'Category Of Books' in df.columns else [], default=None)
gender_sel = st.sidebar.multiselect("Gender", sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else [], default=None)
delivery_sel = st.sidebar.multiselect("Delivery Method", sorted(df['Delivery Method'].dropna().unique().tolist()) if 'Delivery Method' in df.columns else [], default=None)
top_n = st.sidebar.slider("Top N books", 5, 50, 10)

# Apply filters
df_filtered = df.copy()
if year_sel:
    df_filtered = df_filtered[df_filtered['Year'].isin(year_sel)]
if cat_sel:
    df_filtered = df_filtered[df_filtered['Category Of Books'].isin(cat_sel)]
if gender_sel:
    df_filtered = df_filtered[df_filtered['Gender'].isin(gender_sel)]
if delivery_sel:
    df_filtered = df_filtered[df_filtered['Delivery Method'].isin(delivery_sel)]

# numeric safe columns
df_filtered['Sell'] = pd.to_numeric(df_filtered['Sell'], errors='coerce').fillna(0)
df_filtered['Profit'] = pd.to_numeric(df_filtered['Profit'], errors='coerce').fillna(0)
df_filtered['Unit'] = pd.to_numeric(df_filtered.get('Unit', 1), errors='coerce').fillna(1)
df_filtered['Age'] = pd.to_numeric(df_filtered.get('Age', np.nan), errors='coerce')

# ----------------- TABS -----------------
tabs = st.tabs(["📈 Overview", "🚚 Delivery & Profit", "👥 Customers", "🧠 ML", "⏱ Time Series", "📥 Download"])
tab_overview, tab_delivery, tab_customers, tab_ml, tab_ts, tab_dl = tabs

# ----------------- TAB: Overview -----------------
with tab_overview:
    st.header("Sales Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_sell = int(df_filtered['Sell'].sum())
    total_profit = int(df_filtered['Profit'].sum())
    total_orders = int(len(df_filtered))
    unique_books = int(df_filtered['Name of Books'].nunique()) if 'Name of Books' in df_filtered.columns else 0
    rep_pct = df_filtered['Repeated_Customer_flag'].mean()*100 if 'Repeated_Customer_flag' in df_filtered.columns else np.nan

    col1.metric("💰 Total Sell", f"{total_sell:,}")
    col2.metric("🧾 Total Orders", f"{total_orders:,}")
    col3.metric("📚 Unique Books", f"{unique_books:,}")
    col4.metric("🔁 Repeated (%)", f"{rep_pct:.1f}%" if not np.isnan(rep_pct) else "N/A")

    st.subheader("Monthly Sales Trend")
    if 'Order Date' in df_filtered.columns:
        monthly = df_filtered.groupby(pd.Grouper(key='Order Date', freq='M'))['Sell'].sum().reset_index()
        monthly['Rolling_3M'] = monthly['Sell'].rolling(3).mean()
        fig = px.line(monthly, x='Order Date', y='Sell', markers=True, title="Monthly Sales")
        fig.add_scatter(x=monthly['Order Date'], y=monthly['Rolling_3M'], mode='lines', name='3M Rolling')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Order Date column to show monthly trend.")

    st.subheader("Top Selling Books")
    if 'Name of Books' in df_filtered.columns:
        top_books = df_filtered.groupby('Name of Books')['Unit'].sum().sort_values(ascending=False).head(top_n)
        st.dataframe(top_books.reset_index().rename(columns={'Unit':'Units Sold'}))

# ----------------- TAB: Delivery & Profit -----------------
with tab_delivery:
    st.header("Delivery & Profit")
    if 'Delivery Method' in df_filtered.columns:
        dcount = df_filtered['Delivery Method'].value_counts()
        fig = px.pie(values=dcount.values, names=dcount.index, title="Delivery Method Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Delivery Method column not found.")

    if 'Category Of Books' in df_filtered.columns:
        cat_profit = df_filtered.groupby('Category Of Books')['Profit'].sum().sort_values(ascending=False)
        fig2 = px.bar(x=cat_profit.index, y=cat_profit.values, labels={'x':'Category','y':'Profit'}, title="Profit by Category")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Category Of Books column not found for profit analysis.")

# ----------------- TAB: Customers -----------------
with tab_customers:
    st.header("Customer Insights & Segmentation")

    if 'Gender' in df_filtered.columns and 'Repeated_Customer_flag' in df_filtered.columns:
        gender_ret = df_filtered.groupby('Gender')['Repeated_Customer_flag'].mean()*100
        fig = px.bar(x=gender_ret.index, y=gender_ret.values, labels={'x':'Gender','y':'Retention (%)'}, title="Gender-wise Retention (%)")
        st.plotly_chart(fig, use_container_width=True)

    if 'Age' in df_filtered.columns:
        fig2 = px.histogram(df_filtered, x='Age', nbins=12, title="Age Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # RFM-like summary by customer phone if available
    if 'Customer Phone' in df_filtered.columns and 'Order Date' in df_filtered.columns:
        st.subheader("RFM Summary (by Customer Phone)")
        rfm = df_filtered.groupby('Customer Phone').agg(
            frequency=('Order Date','count'),
            monetary=('Sell','sum'),
            last_order=('Order Date','max')
        ).reset_index()
        rfm['recency_days'] = (pd.Timestamp.now() - rfm['last_order']).dt.days
        st.dataframe(rfm.sort_values('monetary', ascending=False).head(20))
    else:
        st.info("Customer-level RFM requires 'Customer Phone' and 'Order Date' columns.")

    # KMeans segmentation
    seg_feats = [c for c in ['Age','Unit','Sell','Profit'] if c in df_filtered.columns]
    if len(seg_feats) >= 2:
        seg_df = df_filtered[seg_feats].dropna()
        if len(seg_df) >= 10:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(seg_df)
            k = st.slider("Choose K (clusters)", 2, 6, 3)
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(Xs)
            seg_df = seg_df.assign(Cluster=labels)
            centers = scaler.inverse_transform(km.cluster_centers_)
            st.write("Cluster centers (approx):")
            st.dataframe(pd.DataFrame(centers, columns=seg_feats))
            if 'Age' in seg_feats and 'Sell' in seg_feats:
                fig = px.scatter(seg_df, x='Age', y='Sell', color='Cluster', title="Clusters (Age vs Sell)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough rows for clustering (need >=10 after dropna).")
    else:
        st.info("Not enough numeric features for clustering (need at least 2).")

# ----------------- TAB: ML -----------------
with tab_ml:
    st.header("Quick ML: Sell Regression & Retention Classification")
    features = [c for c in ['Age','Gender','Unit','Profit','Sell'] if c in df_filtered.columns]
    if len(features) < 2:
        st.info("Not enough features available for ML.")
    else:
        ml_df_cols = [c for c in features if c!='Sell'] + (['Sell'] if 'Sell' in features else [])
        # prepare dataset with target for classification if exists
        ml_df = df_filtered[[c for c in features if c in df_filtered.columns] + (['Repeated_Customer_flag'] if 'Repeated_Customer_flag' in df_filtered.columns else [])].dropna()
        if len(ml_df) < 40:
            st.warning("Too few rows to train reliable ML models. Try widening filters.")
        else:
            # Regression (Sell)
            if 'Sell' in features:
                Xr = ml_df[[c for c in features if c!='Sell']]
                yr = ml_df['Sell']
                Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
                reg = LinearRegression().fit(Xr_train, yr_train)
                st.write(f"Linear Regression R²: {reg.score(Xr_test, yr_test):.3f}")
            else:
                reg = None

            # Classification (Retention)
            if 'Repeated_Customer_flag' in ml_df.columns:
                Xc = ml_df[[c for c in features if c!='Sell' and c!='Repeated_Customer_flag' and c in ml_df.columns]]
                yc = ml_df['Repeated_Customer_flag']
                if len(yc.unique()) > 1:
                    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
                    clf = LogisticRegression(max_iter=1000).fit(Xc_train, yc_train)
                    st.write(f"Logistic Regression Accuracy: {clf.score(Xc_test, yc_test):.3f}")
                else:
                    clf = None
                    st.info("Not enough class variation for retention classification.")
            else:
                clf = None

            # Prediction form
            st.subheader("Make a prediction")
            with st.form("predict_form"):
                inputs = {}
                model_inputs = Xr.columns if reg is not None else (Xc.columns if clf is not None else [])
                for f in model_inputs:
                    if f=='Gender' and 'Gender' in df_filtered.columns:
                        opts = sorted(df_filtered['Gender'].dropna().unique().tolist())
                        inputs[f] = st.selectbox(f, opts, index=0)
                    else:
                        default = float(df_filtered[f].median()) if f in df_filtered.columns and not df_filtered[f].dropna().empty else 0.0
                        inputs[f] = st.number_input(f, value=default)
                sub = st.form_submit_button("Predict")
                if sub:
                    X_new = pd.DataFrame([inputs])
                    if reg is not None:
                        sell_p = reg.predict(X_new)[0]
                        st.success(f"Predicted Sell ≈ {sell_p:,.0f}")
                    if clf is not None:
                        prob = clf.predict_proba(X_new)[0][1] if hasattr(clf, "predict_proba") else clf.predict(X_new)[0]
                        if hasattr(clf, "predict_proba"):
                            st.success(f"Repeat Purchase Probability ≈ {prob:.2%}")
                        else:
                            st.success(f"Repeat Purchase: {'Yes' if prob==1 else 'No'}")

# ----------------- TAB: Time Series -----------------
with tab_ts:
    st.header("Deep Time Series Analysis")

    if 'Order Date' not in df_filtered.columns:
        st.info("Order Date column required for time series analysis.")
    else:
        ts_df = df_filtered.set_index('Order Date').sort_index()
        # monthly and weekly aggregation
        monthly = ts_df['Sell'].resample('M').sum().reset_index()
        weekly = ts_df['Sell'].resample('W').sum().reset_index()

        st.subheader("Monthly Sales (with Rolling Averages)")
        monthly['Rolling_3M'] = monthly['Sell'].rolling(3).mean()
        monthly['Rolling_6M'] = monthly['Sell'].rolling(6).mean()
        fig = px.line(monthly, x='Order Date', y='Sell', markers=True, title='Monthly Sales')
        fig.add_scatter(x=monthly['Order Date'], y=monthly['Rolling_3M'], mode='lines', name='3M Rolling')
        fig.add_scatter(x=monthly['Order Date'], y=monthly['Rolling_6M'], mode='lines', name='6M Rolling')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weekly Sales Trend")
        figw = px.line(weekly, x='Order Date', y='Sell', markers=True, title='Weekly Sales')
        st.plotly_chart(figw, use_container_width=True)

        # MoM growth
        monthly['Prev'] = monthly['Sell'].shift(1)
        monthly['MoM_%'] = (monthly['Sell'] - monthly['Prev']) / monthly['Prev'] * 100
        st.subheader("Month-over-Month Growth (%)")
        figm = px.bar(monthly, x='Order Date', y='MoM_%', title='MoM Growth %')
        st.plotly_chart(figm, use_container_width=True)

        # YoY
        yearly = ts_df['Sell'].resample('Y').sum().reset_index()
        yearly['Prev'] = yearly['Sell'].shift(1)
        yearly['YoY_%'] = (yearly['Sell'] - yearly['Prev']) / yearly['Prev'] * 100
        st.subheader("Yearly Sales & YoY Growth")
        figy = px.bar(yearly, x='Order Date', y='Sell', text=yearly['YoY_%'].round(2).astype(str) + '%', title='Yearly Sales')
        st.plotly_chart(figy, use_container_width=True)

        # seasonal decomposition if enough data
        if len(monthly) >= 24:
            from statsmodels.tsa.seasonal import seasonal_decompose
            ts_series = monthly.set_index('Order Date')['Sell']
            decomp = seasonal_decompose(ts_series, model='additive', period=12, extrapolate_trend='freq')
            st.subheader("Seasonal Decomposition")
            st.write("Trend:")
            st.line_chart(decomp.trend)
            st.write("Seasonal:")
            st.line_chart(decomp.seasonal)
            st.write("Residual:")
            st.line_chart(decomp.resid)
        else:
            st.info("At least 24 months of data required for seasonal decomposition (have {}).".format(len(monthly)))

        # heatmap: year vs month
        st.subheader("Month vs Year Heatmap (Total Sell)")
        df_temp = df_filtered.copy()
        df_temp['Year'] = df_temp['Order Date'].dt.year
        df_temp['Month_Num'] = df_temp['Order Date'].dt.month
        heat = df_temp.groupby(['Year','Month_Num'])['Sell'].sum().unstack(fill_value=0)
        # display as styled dataframe (works in Streamlit)
        st.dataframe(heat.style.background_gradient(cmap='YlGnBu'))

# ----------------- TAB: Download -----------------
with tab_dl:
    st.header("Download")
    st.write("Download the filtered dataset (CSV):")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "filtered_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Saiful Islam — Bikrampur Boighar")
