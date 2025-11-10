# dash_updated.py
# Bikrampur Boighar — PRO MAX Dashboard (Updated: added Sell/Profit/Customer Retention prediction features)
# Developer: Saiful Islam (modified by assistant)

import os
import re
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report

# Optional packages
try:
    from xgboost import XGBRegressor, XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    has_statsmodels = True
except Exception:
    has_statsmodels = False

try:
    # prophet can be heavy; optional
    from prophet import Prophet
    has_prophet = True
except Exception:
    has_prophet = False


# ------------------------
# USER CONFIG - update this path
# ------------------------
DATA_PATH = r"C:\Users\MY COMPUTER\Downloads\Boighar"  # <- change to your folder
EXPECTED_KEYWORDS = ["Sell- 2023.xlsx", "Sell - 2024.xlsx", "Sell - 2025.xlsx", "All Book List.xlsx"]
ACCEPT_EXTS = [".xlsx", ".xls", ".csv"]


# ------------------------
# Helpers: file locate & read
# ------------------------
def locate_file(folder: str, phrase: str):
    folder = Path(folder)
    if not folder.exists():
        return None
    tokens = re.sub(r'[^a-z0-9 ]', ' ', phrase.lower()).split()
    files = [f for f in folder.iterdir() if f.is_file()]
    for f in files:
        name = re.sub(r'[^a-z0-9]', '', f.name.lower())
        if all(t in name for t in tokens):
            return f
    years = re.findall(r'20\d{2}', phrase)
    if years:
        for y in years:
            for f in files:
                if y in f.name:
                    return f
    for f in files:
        if any(t in f.name.lower() for t in tokens):
            return f
    # try with ext variations
    base = ''.join(tokens)
    for ext in ACCEPT_EXTS:
        candidate = folder / (base + ext)
        if candidate.exists():
            return candidate
    return None

def read_table(path: Path):
    if path is None:
        return None
    try:
        if path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

# ------------------------
# Helpers: convert special numeric strings like '100.088k' etc
# ------------------------
def convert_kmb(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == '':
            return np.nan
        s = s.replace(',', '').replace('৳','').replace('tk','').strip()
        m = re.match(r'^([0-9]*\.?[0-9]+)\s*([kKmMbB])$', s)
        if m:
            num = float(m.group(1))
            suf = m.group(2).lower()
            if suf == 'k':
                return num * 1_000
            if suf == 'm':
                return num * 1_000_000
            if suf == 'b':
                return num * 1_000_000_000
        digits = re.findall(r'[-+]?[0-9]*\.?[0-9]+', s)
        if digits:
            return float(digits[0])
        return np.nan
    except Exception:
        return np.nan

# ------------------------
# Load & prepare dataset
# ------------------------
@st.cache_data
def load_and_prepare(data_path):
    folder = Path(data_path)
    found = {}
    for kw in EXPECTED_KEYWORDS:
        f = locate_file(folder, kw)
        found[kw] = f
    # read sales files
    sales = []
    for year_kw in ["Sell- 2023.xlsx", "Sell - 2024.xlsx", "Sell - 2025.xlsx"]:
        p = found.get(year_kw)
        if p:
            df_temp = read_table(p)
            if df_temp is not None:
                # set Year if missing
                yr = re.search(r'20\d{2}', p.name)
                if 'Year' not in df_temp.columns and yr:
                    try:
                        df_temp['Year'] = int(yr.group(0))
                    except:
                        pass
                sales.append(df_temp)
    df_sales = pd.concat(sales, ignore_index=True) if sales else pd.DataFrame()
    # books metadata
    book_p = found.get('all book list')
    df_books = read_table(book_p) if book_p else pd.DataFrame()
    # normalize book names if exist
    if not df_sales.empty and 'Name of Books' in df_sales.columns:
        df_sales['Name of Books'] = df_sales['Name of Books'].astype(str).str.strip().str.lower()
    if not df_books.empty and 'Name of Books' in df_books.columns:
        df_books['Name of Books'] = df_books['Name of Books'].astype(str).str.strip().str.lower()
    if not df_books.empty and 'Name of Books' in df_books.columns and 'Name of Books' in df_sales.columns:
        df = df_sales.merge(df_books.drop_duplicates(subset=['Name of Books']), on='Name of Books', how='left')
    else:
        df = df_sales.copy()
    # clean col names
    df.columns = df.columns.astype(str).str.strip()
    # Order Date
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    # numeric conversion
    for col in ['Sell','Cost','Unit','Price','Discount %','Sell Price','Buying Price']:
        if col in df.columns:
            df[col] = df[col].apply(convert_kmb)
    # Profit
    if 'Sell' in df.columns and 'Cost' in df.columns:
        df['Profit'] = df['Sell'] - df['Cost']
    elif 'Sell Price' in df.columns and 'Buying Price' in df.columns:
        df['Profit'] = (df['Sell Price'] - df['Buying Price']).fillna(0)
    else:
        df['Profit'] = np.nan
    # Repeated Customer normalize (detect col)
    rep_col = None
    for c in df.columns:
        if 'repeated' in c.lower() and 'customer' in c.lower():
            rep_col = c
            break
    if rep_col:
        df['Repeated_Customer_flag'] = df[rep_col].astype(str).str.strip().str.lower().replace({
            'yes':1,'y':1,'true':1,'1':1,'হ্যাঁ':1,'yes ':1,
            'no':0,'n':0,'false':0,'0':0,'না':0
        })
        df['Repeated_Customer_flag'] = pd.to_numeric(df['Repeated_Customer_flag'], errors='coerce').fillna(0).astype(int)
    else:
        df['Repeated_Customer_flag'] = np.nan
    # Age numeric
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Year/Month
    if 'Order Date' in df.columns:
        df['Year'] = df['Order Date'].dt.year.fillna(df.get('Year', np.nan)).astype('Int64')
        df['Month'] = df['Order Date'].dt.month
    else:
        df['Month'] = np.nan
    # fill object NaNs with 'Unknown' to avoid .str accessor errors later
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna('Unknown')
    return df, found

# ------------------------
# New: Modeling helpers
# ------------------------

def prepare_Xy(df, target_col, feature_cols=None, drop_cols=None):
    d = df.copy()
    if drop_cols is None:
        drop_cols = []
    if feature_cols is None:
        # choose numeric columns excluding target
        feature_cols = [c for c in d.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]
    X = d[feature_cols].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median())
    y = d[target_col]
    return X, y


def train_regressors(X_train, y_train):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42)
    }
    if has_xgb:
        models['XGBoost'] = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def train_classifiers(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    if has_xgb:
        try:
            models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
        except Exception:
            pass
    for m in models.values():
        m.fit(X_train, y_train)
    return models

# ------------------------
# Initialize / load
# ------------------------
df, found_files = load_and_prepare(DATA_PATH)

st.set_page_config(layout="wide", page_title="Bikrampur Boighar — Customer Retention And Time Series Analysis")
st.title("📚 Bikrampur Boighar — Customer Retention And Time Series Analysis")

st.sidebar.header("📂 Data files & status")
if found_files:
    for k,v in found_files.items():
        if v:
            st.sidebar.write(f"- {k}: `{v.name}`")
        else:
            st.sidebar.write(f"- {k}: **NOT FOUND**")
else:
    st.sidebar.write("No files detected in DATA_PATH.")

if df.empty:
    st.error("No sales data loaded. Check DATA_PATH and file names.")
    st.stop()

# Sidebar filters (same as before)
st.sidebar.header("🔎 Filters")
years_list = sorted([int(x) for x in df['Year'].dropna().unique().tolist()]) if 'Year' in df.columns else []
year_sel = st.sidebar.multiselect("Year", years_list, default=years_list if years_list else None)
cats = sorted(df['Category Of Books'].dropna().unique().tolist()) if 'Category Of Books' in df.columns else []
cat_sel = st.sidebar.multiselect("Category", cats, default=cats)
genders = sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else []
gender_sel = st.sidebar.multiselect("Gender", genders, default=genders)
delivery_opts = sorted(df['Delivery Method'].dropna().unique().tolist()) if 'Delivery Method' in df.columns else []
delivery_sel = st.sidebar.multiselect("Delivery Method", delivery_opts, default=delivery_opts)
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

# Ensure numeric safe

df_filtered['Sell'] = pd.to_numeric(df_filtered.get('Sell', df_filtered.get('Sell Price', 0)), errors='coerce').fillna(0)
df_filtered['Profit'] = pd.to_numeric(df_filtered.get('Profit', 0), errors='coerce').fillna(0)
df_filtered['Unit'] = pd.to_numeric(df_filtered.get('Unit', 1), errors='coerce').fillna(1)
df_filtered['Age'] = pd.to_numeric(df_filtered.get('Age', np.nan), errors='coerce')

# Tabs

tabs = st.tabs(["Overview","Sales","Delivery & Profit","Customers","ML & Models","Time Series","Forecast","Download"])
tab_overview, tab_sales, tab_delivery, tab_customers, tab_ml, tab_ts, tab_forecast, tab_dl = tabs

# ------------------ OVERVIEW ------------------
with tab_overview:
    st.header("Overview & KPIs")
    c1, c2, c3, c4 = st.columns(4)

    df_unique = df_filtered.drop_duplicates(subset=['Order Date','Customer Phone','Name of Books']) if {'Order Date','Customer Phone','Name of Books'}.issubset(df_filtered.columns) else df_filtered.copy()
    df_unique['Sell Price'] = pd.to_numeric(df_unique.get('Sell Price', df_unique.get('Sell',0)), errors='coerce').fillna(0)
    df_unique['Buying Price'] = pd.to_numeric(df_unique.get('Buying Price', df_unique.get('Cost',0)), errors='coerce').fillna(0)
    df_unique['Unit'] = pd.to_numeric(df_unique.get('Unit', 1), errors='coerce').fillna(1)
    df_unique['Profit'] = (df_unique['Sell Price'] - df_unique['Buying Price']) * df_unique['Unit']
    df_unique['Profit'] = df_unique['Profit'].apply(lambda x: max(x,0))

    total_sell = (df_unique['Sell'] * df_unique['Unit']).sum()
    total_profit = df_unique['Profit'].sum()
    total_orders = int(len(df_unique))
    unique_books = int(df_unique['Name of Books'].nunique()) if 'Name of Books' in df_unique.columns else 0
    rep_pct = df_unique['Repeated_Customer_flag'].mean() * 100 if 'Repeated_Customer_flag' in df_unique.columns else np.nan

    c1.metric("💰 Total Sell", f"{int(total_sell):,}")
    c2.metric("💵 Total Profit", f"{int(total_profit):,}")
    c3.metric("🧾 Total Orders", f"{total_orders:,}")
    c4.metric("🔁 Repeated (%)", f"{rep_pct:.1f}%" if not np.isnan(rep_pct) else "N/A")

    st.markdown("### Monthly Sales Trend")
    if 'Order Date' in df_unique.columns:
        monthly = df_unique.groupby(pd.Grouper(key='Order Date', freq='M')).apply(lambda x: (x['Sell'] * x['Unit']).sum()).reset_index(name='Sell')
        monthly = monthly.sort_values('Order Date')
        monthly['Rolling_3M'] = monthly['Sell'].rolling(3).mean()
        monthly['Rolling_6M'] = monthly['Sell'].rolling(6).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Sell'], mode='lines+markers', name='Monthly Sell'))
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Rolling_3M'], mode='lines', name='3M Rolling'))
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Rolling_6M'], mode='lines', name='6M Rolling'))
        fig.update_layout(title="Monthly Sales with Rolling Averages", xaxis_title='Month', yaxis_title='Sell')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Order Date not present — cannot compute monthly trend.")

    st.markdown("### Top Selling Books")
    if 'Name of Books' in df_unique.columns:
        top_books = df_unique.groupby('Name of Books').apply(lambda x: x['Unit'].sum()).sort_values(ascending=False).head(top_n)
        st.dataframe(top_books.reset_index().rename(columns={0:'Units Sold'}))
    else:
        st.info("No 'Name of Books' column to show top books.")

# ------------------ SALES ------------------
with tab_sales:
    st.header("Sales Detailed Analysis")
    if 'Year' in df_filtered.columns:
        yearly = df_filtered.groupby('Year')['Sell'].sum().reset_index()
        yearly['YoY_%'] = yearly['Sell'].pct_change() * 100
        figy = px.bar(yearly, x='Year', y='Sell', text=yearly['YoY_%'].round(2).astype(str)+'%')
        st.plotly_chart(figy, use_container_width=True)
        st.dataframe(yearly)
    if 'Category Of Books' in df_filtered.columns:
        cat_sales = df_filtered.groupby('Category Of Books')['Sell'].sum().sort_values(ascending=False)
        figc = px.bar(x=cat_sales.index, y=cat_sales.values, labels={'x':'Category','y':'Sell'}, title='Category-wise Sell')
        st.plotly_chart(figc, use_container_width=True)
    if 'Order Date' in df_filtered.columns:
        df_temp = df_filtered.copy()
        df_temp['Month_Num'] = df_temp['Order Date'].dt.month
        season = df_temp.groupby(['Year','Month_Num'])['Sell'].sum().reset_index()
        fig_s = px.line(season, x='Month_Num', y='Sell', color='Year', markers=True, labels={'Month_Num':'Month'})
        st.plotly_chart(fig_s, use_container_width=True)

# ------------------ DELIVERY & PROFIT ------------------
with tab_delivery:
    st.header("Delivery Methods & Profit")
    if 'Delivery Method' in df_filtered.columns:
        dcounts = df_filtered['Delivery Method'].value_counts()
        figp = px.pie(values=dcounts.values, names=dcounts.index, title='Delivery Method Distribution', hole=0.4)
        st.plotly_chart(figp, use_container_width=True)
    if 'Category Of Books' in df_filtered.columns:
        cat_profit = df_filtered.groupby('Category Of Books')['Profit'].sum().sort_values(ascending=False)
        figcp = px.bar(x=cat_profit.index, y=cat_profit.values, labels={'x':'Category','y':'Profit'}, title='Profit by Category')
        st.plotly_chart(figcp, use_container_width=True)

# ------------------ CUSTOMERS ------------------
with tab_customers:
    st.header("Customer Insights & Segmentation")
    if 'Gender' in df_filtered.columns and 'Repeated_Customer_flag' in df_filtered.columns:
        gr = df_filtered.groupby('Gender')['Repeated_Customer_flag'].mean().reset_index()
        gr['Repeated_%'] = gr['Repeated_Customer_flag'] * 100
        figg = px.bar(gr, x='Gender', y='Repeated_%', title='Repeated Customer % by Gender')
        st.plotly_chart(figg, use_container_width=True)
    if 'Age' in df_filtered.columns:
        fig_age = px.histogram(df_filtered, x='Age', nbins=20, title='Age Distribution')
        st.plotly_chart(fig_age, use_container_width=True)

    review_col = 'Review(Out of 5)'
    if review_col in df_filtered.columns:
        df_rev = df_filtered.copy()
        df_rev[review_col] = pd.to_numeric(df_rev[review_col], errors='coerce')
        df_rev = df_rev.dropna(subset=[review_col])
        rating_count = df_rev.groupby(review_col).size().reset_index(name='Count')
        rating_count['Percent'] = (rating_count['Count'] / rating_count['Count'].sum() * 100).round(2)
        st.markdown("### Review Distribution by Rating (Pie Chart)")
        fig = px.pie(rating_count, values='Percent', names=review_col, title='Percentage of Reviews by Rating', hover_data=['Count'], labels={review_col:'Rating', 'Percent':'% of Total Reviews'})
        fig.update_traces(textinfo='label+percent', pull=[0.05]*len(rating_count))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"'{review_col}' column dataset-এ নেই।")

    st.subheader("RFM Quick (by Customer Phone)")
    if 'Customer Phone' in df_filtered.columns and 'Order Date' in df_filtered.columns:
        rfm = df_filtered.groupby('Customer Phone').agg(frequency=('Order Date','count'), monetary=('Sell','sum'), last_order=('Order Date','max')).reset_index()
        rfm['recency_days'] = (pd.Timestamp.now() - rfm['last_order']).dt.days
        st.dataframe(rfm.sort_values('monetary', ascending=False).head(20))
    else:
        st.info("Provide Customer Phone and Order Date for RFM.")

    seg_features = [c for c in ['Age','Unit','Sell','Profit'] if c in df_filtered.columns]
    if len(seg_features) >= 2:
        seg_df = df_filtered[seg_features].dropna()
        if len(seg_df) >= 20:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(seg_df)
            k = st.slider("Choose K clusters", 2, 8, 3)
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(Xs)
            seg_df2 = seg_df.copy()
            seg_df2['Cluster'] = labels
            centers = scaler.inverse_transform(km.cluster_centers_)
            st.write("Cluster centers (approx):")
            st.dataframe(pd.DataFrame(centers, columns=seg_features))
            if 'Age' in seg_features and 'Sell' in seg_features:
                figc = px.scatter(seg_df2, x='Age', y='Sell', color='Cluster', title='Clusters (Age vs Sell)')
                st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Not enough rows for clustering (need >=20).")
    else:
        st.info("Not enough numeric features for clustering.")

# ------------------ ML & MODELS (Expanded for predictions) ------------------
with tab_ml:
    st.header("Machine Learning — Regression & Classification (Prediction Modules)")
    st.markdown("Use the panels below to train/predict Sell, Profit, and Customer Retention.")

    # Create three columns for three prediction modules
    p1, p2, p3 = st.columns(3)

    # --- Sell prediction ---
    with p1:
        st.subheader("Sell Prediction (Regressor)")
        sell_features = st.multiselect("Features for Sell model", df_filtered.select_dtypes(include=[np.number]).columns.tolist(), default=['Unit','Price'] if 'Unit' in df_filtered.columns else None)
        sell_run = st.button("Train Sell Model")
        if sell_run:
            if not sell_features or 'Sell' not in df_filtered.columns:
                st.error("Need numeric features and Sell column to train.")
            else:
                X, y = prepare_Xy(df_filtered, 'Sell', feature_cols=sell_features)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                regs = train_regressors(X_train, y_train)
                res = []
                for name, m in regs.items():
                    preds = m.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    rmse = math.sqrt(mean_squared_error(y_test, preds))
                    res.append((name, r2, rmse))
                res_df = pd.DataFrame(res, columns=['Model','R2','RMSE']).sort_values('R2', ascending=False)
                st.dataframe(res_df)
                st.success('Sell models trained — choose a model to run predictions.')
                sel_model = st.selectbox('Choose model for Sell prediction', res_df['Model'].tolist())
                st.markdown("### 🔢 Enter your own input to predict Sell")
                custom_input = {}
                for col in sell_features:
                    custom_input[col] = st.number_input(f"Enter {col}", value=float(X[sell_features][col].median()))

                if st.button("Predict Sell from your input"):
                    model = regs[sel_model]
                    input_df = pd.DataFrame([custom_input])
                    pred = model.predict(input_df)[0]
                    st.success(f"💰 Predicted Sell: {pred:.2f}")


    # --- Profit prediction ---
    with p2:
        st.subheader("Profit Prediction (Regressor)")
        profit_features = st.multiselect("Features for Profit model", df_filtered.select_dtypes(include=[np.number]).columns.tolist(), default=['Sell','Unit'] if 'Sell' in df_filtered.columns else None)
        profit_run = st.button("Train Profit Model")
        if profit_run:
            if not profit_features or 'Profit' not in df_filtered.columns:
                st.error("Need numeric features and Profit column to train.")
            else:
                Xp, yp = prepare_Xy(df_filtered, 'Profit', feature_cols=profit_features)
                X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(Xp, yp, test_size=0.2, random_state=42)
                regs_p = train_regressors(X_train_p, y_train_p)
                res_p = []
                for name, m in regs_p.items():
                    preds = m.predict(X_test_p)
                    r2 = r2_score(y_test_p, preds)
                    rmse = math.sqrt(mean_squared_error(y_test_p, preds))
                    res_p.append((name, r2, rmse))
                resp_df = pd.DataFrame(res_p, columns=['Model','R2','RMSE']).sort_values('R2', ascending=False)
                st.dataframe(resp_df)
                st.success('Profit models trained.')
                sel_model_p = st.selectbox('Choose model for Profit prediction', resp_df['Model'].tolist())
                st.markdown("### 💹 Enter your own input to predict Profit")
                custom_profit_input = {}
                for col in profit_features:
                    custom_profit_input[col] = st.number_input(f"Enter {col}", value=float(Xp[profit_features][col].median()))

                if st.button("Predict Profit from your input"):
                    model = regs_p[sel_model_p]
                    input_df = pd.DataFrame([custom_profit_input])
                    pred = model.predict(input_df)[0]
                    st.success(f"💰 Predicted Profit: {pred:.2f}")

    # --- Customer retention prediction ---
    with p3:
        st.subheader("Customer Retention (Classification)")
        if 'Repeated_Customer_flag' not in df_filtered.columns or df_filtered['Repeated_Customer_flag'].isna().all():
            st.info('No Repeated_Customer_flag column — cannot train retention model.')
        else:
            cat_features = st.multiselect('Features for retention model', df_filtered.select_dtypes(include=[np.number]).columns.tolist(), default=['Frequency','monetary'] if 'Frequency' in df_filtered.columns else ['Sell','Unit'])
            retention_run = st.button('Train Retention Model')
            if retention_run:
                Xr, yr = prepare_Xy(df_filtered, 'Repeated_Customer_flag', feature_cols=cat_features)
                # ensure binary
                yr = yr.fillna(0).astype(int)
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr if yr.nunique()>1 else None)
                cls = train_classifiers(X_train_r, y_train_r)
                res_r = []
                for name, m in cls.items():
                    preds = m.predict(X_test_r)
                    acc = accuracy_score(y_test_r, preds)
                    res_r.append((name, acc))
                resr_df = pd.DataFrame(res_r, columns=['Model','Accuracy']).sort_values('Accuracy', ascending=False)
                st.dataframe(resr_df)
                st.success('Retention models trained.')
                sel_model_r = st.selectbox('Choose model for retention prediction', resr_df['Model'].tolist())
                st.markdown("### 🔁 Enter your own input to predict Customer Retention")
                custom_retention_input = {}
                for col in cat_features:
                    custom_retention_input[col] = st.number_input(f"Enter {col}", value=float(Xr[cat_features][col].median()))

                if st.button("Predict Retention from your input"):
                    model = cls[sel_model_r]
                    input_df = pd.DataFrame([custom_retention_input])
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

                    if pred == 1:
                        st.success("✅ Predicted: Customer is **LIKELY TO RETURN** (Repeated Customer)")
                    else:
                        st.warning("🚫 Predicted: Customer is **NOT LIKELY TO RETURN**")

                    if proba is not None:
                        st.write(f"Probability of Return: {proba[1]*100:.1f}% | Not Return: {proba[0]*100:.1f}%")


# ------------------ TIME SERIES ------------------
with tab_ts:
    st.header("Deep Time Series Analysis")
    if 'Order Date' not in df_filtered.columns:
        st.info("Order Date required for time series.")
    else:
        ts_df = df_filtered.set_index('Order Date').sort_index()
        monthly = ts_df['Sell'].resample('M').sum().reset_index()
        monthly['Rolling_3M'] = monthly['Sell'].rolling(3).mean()
        monthly['Rolling_6M'] = monthly['Sell'].rolling(6).mean()
        figts = go.Figure()
        figts.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Sell'], mode='lines+markers', name='Monthly Sell'))
        figts.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Rolling_3M'], mode='lines', name='3M Rolling'))
        figts.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Rolling_6M'], mode='lines', name='6M Rolling'))
        figts.update_layout(title='Monthly Sales & Rolling Averages', xaxis_title='Date', yaxis_title='Sell')
        st.plotly_chart(figts, use_container_width=True)
        st.subheader("Seasonality & Decomposition")
        if has_statsmodels and len(monthly) >= 24:
            try:
                decomp = seasonal_decompose(monthly.set_index('Order Date')['Sell'], model='additive', period=12)
                st.write("Trend:")
                st.line_chart(pd.Series(decomp.trend).dropna())
                st.write("Seasonal:")
                st.line_chart(pd.Series(decomp.seasonal).dropna())
                st.write("Residual:")
                st.line_chart(pd.Series(decomp.resid).dropna())
            except Exception as e:
                st.warning(f"Seasonal decomposition failed: {e}")
        else:
            st.info("At least 24 months required for decomposition or statsmodels missing.")
        st.subheader("Month vs Year Heatmap")
        tmp = df_filtered.copy()
        tmp['Year'] = tmp['Order Date'].dt.year
        tmp['Month_Num'] = tmp['Order Date'].dt.month
        heat = tmp.groupby(['Year','Month_Num'])['Sell'].sum().unstack(fill_value=0)
        st.dataframe(heat.style.background_gradient(cmap='YlGnBu'))

# ------------------ FORECAST ------------------
with tab_forecast:
    st.header("Forecasting — Prophet / Holt-Winters")
    if 'Order Date' not in df_filtered.columns:
        st.info("Order Date required.")
    else:
        ts = df_filtered.set_index('Order Date')['Sell'].resample('M').sum()
        st.write("Historical months:", len(ts))
        if len(ts) < 12:
            st.info("Need at least 12 months of data for basic forecasting.")
        else:
            months_ahead = st.number_input("Forecast months ahead", min_value=1, max_value=36, value=6)
            method = st.selectbox("Method", ["Prophet (if installed)", "Holt-Winters (additive)"])
            if method.startswith("Prophet") and has_prophet:
                df_prop = ts.reset_index().rename(columns={'Order Date':'ds','Sell':'y'})
                m = Prophet()
                m.fit(df_prop)
                future = m.make_future_dataframe(periods=months_ahead, freq='M')
                forecast = m.predict(future)
                figf = px.line(forecast, x='ds', y='yhat', title='Prophet Forecast')
                st.plotly_chart(figf, use_container_width=True)
                st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(months_ahead))
            else:
                if not has_statsmodels:
                    st.info("statsmodels not available; Holt-Winters not supported.")
                else:
                    try:
                        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12)
                        fit = model.fit(optimized=True)
                        pred = fit.forecast(months_ahead)
                        combined_x = list(ts.index) + list(pred.index)
                        combined_y = list(ts.values) + list(pred.values)
                        figf = px.line(x=combined_x, y=combined_y, title='Historical + Holt-Winters Forecast')
                        st.plotly_chart(figf, use_container_width=True)
                        fc_df = pd.DataFrame({'ds': pred.index, 'yhat': pred.values})
                        st.dataframe(fc_df)
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

# ------------------ DOWNLOAD ------------------
with tab_dl:
    st.header("Download Filtered Data")
    st.write("Download the currently filtered dataset as CSV.")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "filtered_data.csv", "text/csv")

st.markdown("---")
st.markdown("Made by Saiful Islam — Bikrampur Boighar (PRO MAX) — Updated")
