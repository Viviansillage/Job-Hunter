import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import joblib
import time

st.set_page_config(layout="wide")

# =============== Unified data loader (supports .parquet / .csv.gz / .zip / .csv) ===============
@st.cache_data
def load_data():
    usecols = ["firm","date_review","current","overall_rating",
               "work_life_balance","career_opp","comp_benefits","recommend"]

    base = Path(__file__).parent / "data"
    candidates = [
        base / "glassdoor_reviews.parquet",
        base / "glassdoor_reviews.csv.gz",
        base / "glassdoor_reviews.csv.zip",
        base / "glassdoor_reviews.zip",
        base / "glassdoor_reviews.csv",
    ]

    df = None
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p, columns=usecols)
            else:
                df = pd.read_csv(
                    p,
                    usecols=usecols,
                    parse_dates=["date_review"],
                    compression="infer"
                )
            break

    if df is None:
        up = st.file_uploader("Upload glassdoor_reviews (.csv / .csv.gz / .csv.zip / .parquet)",
                              type=["csv","gz","zip","parquet"])
        if up is None:
            st.info("Please upload the dataset to proceed.")
            st.stop()
        if up.name.endswith(".parquet"):
            df = pd.read_parquet(up, columns=usecols)
        else:
            df = pd.read_csv(up, usecols=usecols, parse_dates=["date_review"], compression="infer")

    # Normalize/alias just in case column variations appear
    df.columns = (df.columns.str.strip()
                           .str.lower()
                           .str.replace(r"[^a-z0-9]+", "_", regex=True))
    alias = {
        "firm": ["firm","company","company_name"],
        "date_review": ["date_review","review_date","date"],
        "current": ["current","is_current","currently_employed"],
        "overall_rating": ["overall_rating","overall","rating"],
        "work_life_balance": ["work_life_balance","wrok_life_balance","wlb","worklife_balance"],
        "career_opp": ["career_opp","career_opportunities","career"],
        "comp_benefits": ["comp_benefits","compensation_and_benefits","benefits","compensation"],
        "recommend": ["recommend","recommendation","recommend_company","recommend_firm",
                      "recommend_to_friend","recommendation_category"],
    }
    rename_map = {}
    for tgt, cands in alias.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = tgt
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure all 8 columns exist (avoid KeyError downstream)
    for c in usecols:
        if c not in df.columns:
            df[c] = pd.NA

    # Types
    df["date_review"] = pd.to_datetime(df["date_review"], errors="coerce")
    for c in ["overall_rating","work_life_balance","career_opp","comp_benefits"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[usecols]

# =============== Helpers used across pages ===============
DISPLAY_MAP = {
    "firm": "Firm",
    "date_review": "Review Date",
    "current": "Worker Status",
    "overall_rating": "Overall Rating",
    "work_life_balance": "Work/Life Balance",
    "career_opp": "Career Opportunities",
    "comp_benefits": "Comp & Benefits",
    "recommend": "Recommend"
}

def map_current_former(x: str):
    if not isinstance(x, str): return None
    t = x.strip().lower()
    if "key not found" in t or t == "": return None
    if t in {"true","1","yes","y","t"}: return "Current"
    if t in {"false","0","no","n","f"}: return "Former"
    if t.startswith("current"): return "Current"
    if t.startswith("former"): return "Former"
    return None

def map_role_type(x: str):
    if not isinstance(x, str): return None
    t = x.strip().lower()
    if "key not found" in t or t == "": return None
    if "intern" in t: return "Intern"
    if ("contractor" in t) or ("contract" in t) or ("freelancer" in t) \
       or ("temporary" in t) or ("per diem" in t) or ("contingent" in t) or ("temp" in t):
        return "Contractor"
    if "employee" in t: return "Employee"
    return "Employee"

def map_recommend(v: str):
    t = str(v).strip().lower()
    if t == "v": return "Positive"
    if t == "x": return "Negative"
    if t == "o": return "No opinion"
    if t == "r": return "Mild"
    return None

# =============== Navigation state ===============
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
def change_page(page):
    st.session_state.current_page = page

st.sidebar.title("Find YOUR Job!")
pages = ["Home", "Data Viewer", "Explore Jobs", "Find a Job", "About Me"]
for page in pages:
    if st.sidebar.button(page):
        change_page(page)

# ================================== PAGES ==================================
if st.session_state.current_page == 'Home':
    st.markdown("<h1 style='font-size: 65px;'>Join us and find the job just for YOU! </h1>", unsafe_allow_html=True)

    st.markdown("## Why we are here")
    st.markdown("""
Our goal is to **reduce anxiety caused by opaque and scattered job information** and give you
a clearer, more interactive way to explore the job market and plan your career.

- **Complex job landscape**: Many role names and evaluation dimensions (pay, career growth,
culture, work–life balance) make apples-to-apples comparisons hard.
- **Fragmented resources**: Official blurbs, social posts, and personal anecdotes are
scattered and vary in **trustworthiness and comparability**.
- **High decision costs**: Without a unified, interactive view, candidates spend a lot of time
collecting and reconciling data.

Here, you can use **clean visuals + personalized filters** to quickly understand company/industry
sentiment and trends — practical input for **smarter career decisions**.
    """)

    st.divider()

    st.markdown("## What data we used")
    st.markdown("""
We use the **Glassdoor Job Reviews** dataset from Kaggle, covering **838,566** employee reviews
from **2008-08-21** to **2021-03-31**.

**Source URLs**  
- Main dataset page: <https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews>  
- Data files: <https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews/data>
    """)

    st.markdown("""
**About Dataset**  
This large dataset contains job descriptions and rankings among various criteria such as work-life balance, income, culture, etc. The data covers the various industries in the UK. Great dataset for multidimensional sentiment analysis.

This data set complements the Glassdoor dataset located [here](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews-2).

**Features**  
The columns correspond to the date of the review, the status of the reviewers, and the reviews. Reviews are divided in sub-categories Career Opportunities, Comp & Benefits, and Work/Life Balance. In addition, employees can add recommendations on the firm.

**Other information**  
Ranking for the recommendation of the firm is allocated categories v, x, and o, with the following meanings: v - Positive, x - Negative, o - No opinion
    """)

    st.divider()

    st.markdown("## What is Glassdoor")
    st.markdown("""
**Glassdoor** is a career community and recruiting platform centered on **anonymous employee reviews**.
You can browse overall company ratings, dimension-level sentiment (work–life balance, compensation & benefits,
culture & values, etc.), salary data, and interview experiences. These real-world employee perspectives
help candidates get a fuller picture, while encouraging employers to improve culture and employer branding.  
Official site: <https://www.glassdoor.com>
    """)

    st.divider()

    st.markdown("## What can you get from us")
    st.markdown("""
This app lets you tailor your view of the job market with **personalized filters** across firm, date range, employment status, overall rating, and dimension ratings (work–life balance, career opportunities, compensation & benefits). Your selections are transformed into **clear visual insights**—from distribution snapshots to top-firm comparisons and dimension-level averages—so you can quickly read current sentiment and differences across firms and industries, then use those signals to **plan your next steps** with greater confidence. Our aim is to make data more visual, search more efficient, and decisions more evidence-based.
    """)

    st.divider()

elif st.session_state.current_page == 'Data Viewer':
    st.markdown("<h1 style='font-size: 50px;'>Welcome to the Data Viewer</h1>", unsafe_allow_html=True)
    st.markdown('You can get a good grasp of the data on this page and view it quickly. Only the relevant columns are displayed but there are others.')

    data = load_data()
    filtered_data = data.rename(columns=DISPLAY_MAP)
    st.dataframe(filtered_data, height=1000, use_container_width=True)

elif st.session_state.current_page == 'Explore Jobs':
    data = load_data()

    firm_col, date_col = "firm", "date_review"
    current_col = "current"
    wlb_col, career_col, comp_col = "work_life_balance", "career_opp", "comp_benefits"
    rec_col = "recommend"

    st.markdown("# Explore Jobs")
    st.markdown("Use the collapsible sections below to explore each column with focused filters and charts.")

    # 1) Firm
    with st.expander("Firm", expanded=True):
        n_companies = data[firm_col].dropna().nunique()
        st.write(f"**Total unique firms:** {n_companies}")

        top20 = data[firm_col].dropna().astype(str).value_counts().head(20).sort_values(ascending=True)
        st.write("**Top 20 firms by number of reviews**")
        fig, ax = plt.subplots()
        top20.plot(kind="barh", ax=ax)
        ax.set_xlabel("Count"); ax.set_ylabel("Firm")
        for i, v in enumerate(top20.values):
            ax.text(v, i, f" {v}", va='center')
        st.pyplot(fig)

    # 2) Date Review — Month/Quarter select slider
    with st.expander("Date Review", expanded=True):
        d = data.dropna(subset=[date_col]).copy()
        if d.empty:
            st.warning("No valid dates to filter.")
        else:
            hard_min = pd.Timestamp("2008-08-01")
            hard_max = pd.Timestamp("2021-03-31")
            real_min = d[date_col].min()
            real_max = d[date_col].max()
            span_min = max(hard_min, real_min)
            span_max = min(hard_max, real_max)

            gran = st.radio("Granularity", ["Month", "Quarter"], horizontal=True)
            if gran == "Month":
                start_p = span_min.to_period("M")
                end_p   = span_max.to_period("M")
                periods = pd.period_range(start_p, end_p, freq="M")
                options = [p.strftime("%Y-%m") for p in periods]
                start_label, end_label = st.select_slider(
                    "Select month range", options=options, value=(options[0], options[-1])
                )
                sel_start = pd.Period(start_label, freq="M").start_time
                sel_end   = pd.Period(end_label,   freq="M").end_time
            else:
                start_p = span_min.to_period("Q")
                end_p   = span_max.to_period("Q")
                periods = pd.period_range(start_p, end_p, freq="Q")
                def qlabel(p): return f"{p.start_time:%Y}-Q{p.quarter}"
                options = [qlabel(p) for p in periods]
                start_label, end_label = st.select_slider(
                    "Select quarter range", options=options, value=(options[0], options[-1])
                )
                def parse_q(label):
                    y, q = label.split("-Q"); return pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
                sel_start = parse_q(start_label).start_time
                sel_end   = parse_q(end_label).end_time

            filtered_data = d[(d[date_col] >= sel_start) & (d[date_col] <= sel_end)]
            st.write(f"Showing reviews from **{sel_start:%Y-%m}** to **{sel_end:%Y-%m}** — rows: **{len(filtered_data)}**")
            show_cols = [firm_col, date_col, current_col, "overall_rating", wlb_col, career_col, comp_col, rec_col]
            show_cols = [c for c in show_cols if c in filtered_data.columns]
            st.dataframe(filtered_data[show_cols].head(1000), use_container_width=True)

    # 3) Worker Status
    with st.expander("Worker Status", expanded=True):
        s_raw = data[current_col].astype(str).fillna("").str.strip()
        status_cf = s_raw.map(map_current_former)
        role_type = s_raw.map(map_role_type)

        st.markdown("**Current vs. Former**")
        cf_counts = status_cf.dropna().value_counts().reindex(["Current","Former"], fill_value=0)
        total_cf = int(cf_counts.sum())
        fig1, ax1 = plt.subplots()
        cf_counts.plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Status"); ax1.set_ylabel("Count")
        for i, v in enumerate(cf_counts.values):
            pct = (v / total_cf * 100) if total_cf else 0
            ax1.text(i, v, f"{v} ({pct:.1f}%)", ha='center', va='bottom')
        st.pyplot(fig1)

        st.markdown("**Contractor vs. Intern vs. Employee**")
        role_counts = role_type.dropna().value_counts().reindex(["Employee","Contractor","Intern"], fill_value=0)
        total_role = int(role_counts.sum())
        fig2, ax2 = plt.subplots()
        role_counts.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Role Type"); ax2.set_ylabel("Count")
        for i, v in enumerate(role_counts.values):
            pct = (v / total_role * 100) if total_role else 0
            ax2.text(i, v, f"{v} ({pct:.1f}%)", ha='center', va='bottom')
        st.pyplot(fig2)

    # helper for 1–5 categorical hist + mean
    def rating_block(title, col_name):
        with st.expander(title, expanded=True):
            s = pd.to_numeric(data[col_name], errors="coerce").dropna()
            if s.empty:
                st.info("No valid numeric ratings."); return
            counts = s.astype(int).value_counts().sort_index()
            cdf = counts.reindex([1,2,3,4,5], fill_value=0)
            mean_v = s.mean()
            fig, ax = plt.subplots()
            cdf.plot(kind="bar", ax=ax)
            ax.set_xlabel("Rating"); ax.set_ylabel("Frequency")
            for i, v in enumerate(cdf.values):
                ax.text(i, v, f"{v}", ha='center', va='bottom')
            st.pyplot(fig)
            st.write(f"**Mean {title}: {mean_v:.2f}** (n={int(s.shape[0])})")

    rating_block("Overall Rating", "overall_rating")
    rating_block("Work Life Balance", wlb_col)
    rating_block("Career Opportunity", career_col)
    rating_block("Company Benefits", comp_col)

    with st.expander("Recommend or not", expanded=True):
        s_map = data[rec_col].map(map_recommend)
        cnt = s_map.value_counts()
        total = cnt.sum()
        fig, ax = plt.subplots()
        cnt.plot(kind="bar", ax=ax)
        ax.set_xlabel("Recommendation"); ax.set_ylabel("Count")
        for i, v in enumerate(cnt.values):
            pct = v / total * 100 if total else 0
            ax.text(i, v, f"{v} ({pct:.1f}%)", ha='center', va='bottom')
        st.pyplot(fig)

elif st.session_state.current_page == 'Find a Job':
    data = load_data()

    firm_col, date_col = "firm", "date_review"
    current_col = "current"
    overall_col = "overall_rating"
    wlb_col, career_col, comp_col = "work_life_balance", "career_opp", "comp_benefits"
    rec_col = "recommend"

    st.markdown("# Find a Job")
    st.write("Filter by firm, worker status, rating ranges, and recommendation to see matching rows below.")

    # Firm: Top 20 + Other
    firm_counts = data[firm_col].dropna().astype(str).value_counts()
    top20_firms = firm_counts.head(20).index.tolist()
    firm_options = top20_firms + ["Other"]
    selected_firms = st.multiselect("Firm (Top 20 + Other)", options=firm_options, default=top20_firms)

    allowed_firms = set([f for f in selected_firms if f != "Other"])
    if "Other" in selected_firms:
        others = set(data[firm_col].dropna().astype(str).unique()) - set(top20_firms)
        allowed_firms |= others

    # Worker Status: Employee vs Non-employee
    def worker_bucket(x: str):
        t = str(x).lower()
        if "employee" in t: return "Employee"
        if ("intern" in t) or ("contract" in t) or ("contractor" in t) or ("freelancer" in t) \
           or ("temporary" in t) or ("per diem" in t) or ("contingent" in t) or ("temp" in t):
            return "Non-employee"
        return None

    worker_choice = st.selectbox("Worker Status", ["Employee", "Non-employee"])

    # Rating sliders (1–5)
    def rating_range(label, col):
        return st.slider(f"{label} (1–5)", min_value=1, max_value=5, value=(1,5))
    overall_rng = rating_range("Overall Rating", overall_col)
    wlb_rng     = rating_range("Work Life Balance", wlb_col)
    career_rng  = rating_range("Career Opportunity", career_col)
    comp_rng    = rating_range("Company Benefits", comp_col)

    # Recommend or not
    rec_options = ["Positive", "Negative", "No opinion"]
    selected_recs = st.multiselect("Recommend or not", options=rec_options, default=rec_options)

    # Apply filters
    q = data.copy()
    if len(allowed_firms) > 0:
        q = q[q[firm_col].astype(str).isin(allowed_firms)]

    q["_worker_status_"] = q[current_col].map(worker_bucket)
    q = q[q["_worker_status_"] == worker_choice]

    def between(series, lo, hi):
        s = pd.to_numeric(series, errors="coerce")
        return (s >= lo) & (s <= hi)

    q = q[between(q[overall_col], *overall_rng)]
    q = q[between(q[wlb_col], *wlb_rng)]
    q = q[between(q[career_col], *career_rng)]
    q = q[between(q[comp_col], *comp_rng)]

    q["_rec_mapped_"] = q[rec_col].map(map_recommend)
    q = q[q["_rec_mapped_"].isin(selected_recs)]

    st.write(f"**Matched rows:** {len(q)}")

    show_cols = [firm_col, date_col, current_col, overall_col, wlb_col, career_col, comp_col, rec_col]
    pretty = q[show_cols].rename(columns={
        firm_col: "Firm",
        date_col: "Review Date",
        current_col: "Worker Status (raw)",
        overall_col: "Overall Rating",
        wlb_col: "Work/Life Balance",
        career_col: "Career Opportunities",
        comp_col: "Comp & Benefits",
        rec_col: "Recommend (raw)"
    })
    st.dataframe(pretty.head(1000), use_container_width=True)

    st.download_button(
        "Download filtered rows (CSV)",
        data=pretty.to_csv(index=False).encode("utf-8"),
        file_name="find_a_job_filtered.csv",
        mime="text/csv"
    )

elif st.session_state.current_page == "About Me":
    st.markdown("<h1 style='font-size: 32px;'>Built by Yiheng Wang, Johns Hopkins University</h1>", unsafe_allow_html=True)
