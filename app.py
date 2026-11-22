"""
Customer Segmentation & Purchase Classification dashboard
Data: UCI Online Retail II (cleaned file exported from the project notebook)
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# Settings (taken from the project notebook)
# ----------------------------------------------------------------------------
DATA_FILE = "online_retail_cleaned.csv"
RANDOM_STATE = 42
N_CLUSTERS = 4
KPCA_GAMMA = 0.1
CUTOFF = pd.Timestamp("2010-11-10")          # history ends here
PREDICTION_END = pd.Timestamp("2010-12-10")  # next-30-day window ends here

SEGMENT_ORDER = ["High-Value Loyal", "Moderate-Value",
                 "Recent Occasional", "Inactive / At-Risk"]
SEGMENT_COLORS = {"High-Value Loyal": "#1b9e77", "Moderate-Value": "#7570b3",
                  "Recent Occasional": "#e6ab02", "Inactive / At-Risk": "#d95f02"}
SEGMENT_MEANING = {
    "High-Value Loyal": "Buy recently, buy often, spend the most. Protect and reward them.",
    "Moderate-Value": "Decent spend but not very recent. Nudge them to order again.",
    "Recent Occasional": "Bought recently but only a few times and spend little. Grow them.",
    "Inactive / At-Risk": "Long time since last order, usually one order, low spend. Win back or let go.",
}


# ----------------------------------------------------------------------------
# Data loading and feature building
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["InvoiceDate"],
        dtype={"CustomerID": str, "InvoiceNo": str, "StockCode": str},
    )
    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data
def build_rfm():
    """One row per customer: Recency, Frequency, Monetary."""
    df = load_data()
    reference = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    g = df.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": (reference - g["InvoiceDate"].max().dt.normalize()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["TotalAmount"].sum(),
    })
    return rfm


def name_clusters(rfm, labels):
    """Give each K-Means cluster a business name from its median RFM profile,
    so names stay correct even if the cluster numbers change."""
    med = rfm.assign(c=labels).groupby("c")[["Recency", "Frequency", "Monetary"]].median()
    names = {}
    loyal = med["Monetary"].idxmax()
    names[loyal] = "High-Value Loyal"
    rest = med.drop(index=loyal)
    risk = rest["Recency"].idxmax()
    names[risk] = "Inactive / At-Risk"
    rest = rest.drop(index=risk)
    moderate = rest["Monetary"].idxmax()
    names[moderate] = "Moderate-Value"
    for c in rest.index:
        if c not in names:
            names[c] = "Recent Occasional"
    return names


@st.cache_resource
def fit_segmentation():
    """log -> standardise -> KPCA (2D) -> K-Means (K=4)."""
    rfm = build_rfm()
    logged = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler().fit(logged)
    X = scaler.transform(logged)
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=KPCA_GAMMA)
    Z = kpca.fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(Z)
    names = name_clusters(rfm, km.labels_)
    seg = rfm.copy()
    seg["Segment"] = pd.Series(km.labels_, index=rfm.index).map(names)
    seg["KPC1"], seg["KPC2"] = Z[:, 0], Z[:, 1]
    return {"scaler": scaler, "kpca": kpca, "km": km, "names": names, "seg": seg}


def predict_segment(model, r, f, m):
    x = model["scaler"].transform(pd.DataFrame(
        np.log1p([[r, f, m]]), columns=["Recency", "Frequency", "Monetary"]))
    cluster = model["km"].predict(model["kpca"].transform(x))[0]
    return model["names"][cluster]


# ----------------------------------------------------------------------------
# Classification experiments (five models each)
# ----------------------------------------------------------------------------
def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    }


def run_experiment(X, y, average):
    """80/20 stratified split, train all models, return metrics + confusion matrices."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    rows, cms, fitted = [], {}, {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if name == "SVM":
            score = model.decision_function(Xte)
        else:
            score = model.predict_proba(Xte)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, average=average, zero_division=0),
            "Recall": recall_score(yte, pred, average=average, zero_division=0),
            "F1 Score": f1_score(yte, pred, average=average, zero_division=0),
            "ROC-AUC": roc_auc_score(yte, score),
        })
        cms[name] = confusion_matrix(yte, pred, labels=[0, 1])
        fitted[name] = model
    results = (pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
               .reset_index(drop=True))
    return {"results": results, "cms": cms, "models": fitted,
            "n_train": len(Xtr), "n_test": len(Xte), "y": y}


@st.cache_resource
def run_all_experiments():
    rfm = build_rfm()
    df = load_data()

    # Experiment 1: Frequent vs Infrequent (label from Frequency, so features exclude it)
    f_thr = rfm["Frequency"].median()
    y1 = (rfm["Frequency"] > f_thr).astype(int)
    exp1 = run_experiment(rfm[["Recency", "Monetary"]], y1, "macro")
    exp1["threshold"] = f_thr

    # Experiment 2: High-value vs Low-value (label from Monetary, so features exclude it)
    m_thr = rfm["Monetary"].median()
    y2 = (rfm["Monetary"] > m_thr).astype(int)
    exp2 = run_experiment(rfm[["Recency", "Frequency"]], y2, "macro")
    exp2["threshold"] = m_thr

    # Experiment 3: will the customer buy in the next 30 days?
    hist = df[df["InvoiceDate"] < CUTOFF]
    g = hist.groupby("CustomerID")
    feat = pd.DataFrame({
        "Recency": (CUTOFF - g["InvoiceDate"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["TotalAmount"].sum(),
    })
    feat["Average_Order_Value"] = feat["Monetary"] / feat["Frequency"]
    future = df[(df["InvoiceDate"] >= CUTOFF) & (df["InvoiceDate"] < PREDICTION_END)]
    y3 = feat.index.isin(future["CustomerID"].unique()).astype(int)
    exp3 = run_experiment(feat, pd.Series(y3, index=feat.index), "binary")
    exp3["features"] = list(feat.columns)
    return {"freq": exp1, "value": exp2, "future": exp3}


def confusion_figure(cm, labels, title):
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Customers"), title=title)
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ----------------------------------------------------------------------------
# Guard: data file must exist
# ----------------------------------------------------------------------------
try:
    df_all = load_data()
except FileNotFoundError:
    st.error(f"Could not find '{DATA_FILE}'. Put it in the same folder as app.py and rerun.")
    st.stop()

PAGES = ["How to read this", "Overview", "Customer segments", "Classification",
         "Future purchase", "Predict a customer"]
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.caption("UCI Online Retail II, Dec 2009 to 9 Dec 2010")

# ----------------------------------------------------------------------------
# Page: How to read this
# ----------------------------------------------------------------------------
if page == "How to read this":
    st.title("Customer Segmentation and Purchase Classification")
    st.write("This dashboard shows what we learned from a UK online retailer's sales "
             "data: who the customers are, how they group together, and which ones "
             "are likely to buy again.")
    st.subheader("Plain-English glossary")
    st.markdown("""
- **Recency**: days since the customer's last order (smaller is better).
- **Frequency**: how many separate orders the customer placed.
- **Monetary**: total money the customer spent (in pounds).
- **RFM**: these three numbers together. Every customer gets one row of them.
- **Clustering (K-Means)**: the computer sorts customers into groups of similar customers. We asked for 4 groups.
- **PCA / Kernel PCA**: a way to squash the three RFM numbers into two so the groups can be drawn on a flat chart. Kernel PCA can also follow curved patterns.
- **Classification**: the computer learns from labelled examples, then labels new customers (for example High-value or Low-value).
- **Accuracy**: the share of customers labelled correctly.
- **ROC-AUC**: how well a model ranks customers, from 0.5 (guessing) to 1.0 (perfect).
- **Precision / Recall / F1**: precision is how many flagged customers were right; recall is how many of the real ones were found; F1 balances the two.
""")
    st.subheader("Pages")
    st.markdown("""
1. **Overview**: sales, revenue by month, top products and countries.
2. **Customer segments**: the four customer groups and what defines them.
3. **Classification**: Frequent vs Infrequent and High-value vs Low-value, five models compared.
4. **Future purchase**: will a customer buy in the next 30 days?
5. **Predict a customer**: look up a customer or type in numbers to get a prediction.
""")

# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------
elif page == "Overview":
    st.title("Business Overview")

    min_d, max_d = df_all["InvoiceDate"].min().date(), df_all["InvoiceDate"].max().date()
    c_a, c_b = st.columns(2)
    date_range = c_a.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    countries = c_b.multiselect("Countries (leave empty for all)",
                                sorted(df_all["Country"].unique()))
    df = df_all
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["InvoiceDate"].dt.date >= start) & (df["InvoiceDate"].dt.date <= end)]
    if countries:
        df = df[df["Country"].isin(countries)]
    if df.empty:
        st.warning("No data for this filter.")
        st.stop()

    invoice_value = df.groupby("InvoiceNo")["TotalAmount"].sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Revenue", f"£{df['TotalAmount'].sum():,.0f}")
    k2.metric("Customers", f"{df['CustomerID'].nunique():,}")
    k3.metric("Orders", f"{df['InvoiceNo'].nunique():,}")
    k4.metric("Avg order value", f"£{invoice_value.mean():,.2f}")
    k5.metric("Products", f"{df['StockCode'].nunique():,}")

    monthly = df.set_index("InvoiceDate")["TotalAmount"].resample("MS").sum().reset_index()
    fig = px.line(monthly, x="InvoiceDate", y="TotalAmount", markers=True,
                  title="Monthly revenue", labels={"InvoiceDate": "Month", "TotalAmount": "Revenue (£)"})
    st.plotly_chart(fig)
    st.caption("The last month is incomplete: the data stops on 9 December 2010, "
               "so its drop is not a real fall in sales.")

    left, right = st.columns(2)
    products = (df[~df["Description"].str.contains("POSTAGE", case=False, na=False)]
                .groupby("Description")["TotalAmount"].sum().nlargest(10)
                .sort_values().reset_index())
    left.plotly_chart(px.bar(products, x="TotalAmount", y="Description", orientation="h",
                             title="Top 10 products by revenue (postage excluded)",
                             labels={"TotalAmount": "Revenue (£)", "Description": ""}))
    ctry = (df.groupby("Country")["TotalAmount"].sum().nlargest(10)
            .sort_values().reset_index())
    right.plotly_chart(px.bar(ctry, x="TotalAmount", y="Country", orientation="h",
                              title="Top 10 countries by revenue",
                              labels={"TotalAmount": "Revenue (£)"}))

    orders_per_cust = df.groupby("CustomerID")["InvoiceNo"].nunique()
    kind = pd.Series(np.where(orders_per_cust == 1, "One-time", "Repeat")).value_counts().reset_index()
    kind.columns = ["Type", "Customers"]
    st.plotly_chart(px.pie(kind, names="Type", values="Customers", hole=0.4,
                           title="One-time vs repeat customers"))

# ----------------------------------------------------------------------------
# Page: Customer segments
# ----------------------------------------------------------------------------
elif page == "Customer segments":
    st.title("Customer Segments")
    st.caption("Method: RFM, then log transform, scaling, Kernel PCA (2D) and K-Means with 4 clusters.")
    with st.spinner("Fitting the segmentation model (first load only)..."):
        model = fit_segmentation()
    seg = model["seg"]

    counts = seg["Segment"].value_counts().reindex(SEGMENT_ORDER).fillna(0).astype(int)
    cols = st.columns(4)
    for col, name in zip(cols, SEGMENT_ORDER):
        col.metric(name, f"{counts[name]:,}", f"{counts[name] / len(seg):.0%} of customers",
                   delta_color="off")

    st.plotly_chart(px.scatter(
        seg.reset_index(), x="KPC1", y="KPC2", color="Segment",
        color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"], opacity=0.6,
        title="Customers on the two Kernel PCA axes"))

    st.subheader("What defines each segment (median values)")
    profile = seg.groupby("Segment")[["Recency", "Frequency", "Monetary"]].median().round(1)
    profile["Customers"] = counts
    profile["Share of revenue"] = (seg.groupby("Segment")["Monetary"].sum()
                                   / seg["Monetary"].sum()).map("{:.0%}".format)
    profile["Meaning"] = pd.Series(SEGMENT_MEANING)
    st.dataframe(profile.reindex(SEGMENT_ORDER))

    st.plotly_chart(px.scatter(
        seg.reset_index(), x="Recency", y="Monetary", color="Segment", log_y=True,
        color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
        hover_data=["CustomerID", "Frequency"], opacity=0.6,
        title="Recency vs spend (spend on a log scale)"))

    st.download_button("Download customers with segments (CSV)",
                       seg[["Recency", "Frequency", "Monetary", "Segment"]].to_csv().encode("utf-8"),
                       "customer_segments.csv", "text/csv")

# ----------------------------------------------------------------------------
# Page: Classification
# ----------------------------------------------------------------------------
elif page == "Classification":
    st.title("Classification: Frequent / High-value customers")
    with st.spinner("Training five models per experiment (first load only)..."):
        exps = run_all_experiments()

    choice = st.radio("Experiment", ["Frequent vs Infrequent", "High-value vs Low-value"],
                      horizontal=True)
    if choice.startswith("Frequent"):
        e, labels = exps["freq"], ["Infrequent", "Frequent"]
        st.info(f"Label: **Frequent** if the customer has more than {e['threshold']:.0f} orders. "
                "Inputs: Recency and Monetary (Frequency is left out because it defines the label).")
    else:
        e, labels = exps["value"], ["Low-value", "High-value"]
        st.info(f"Label: **High-value** if total spend is above £{e['threshold']:,.2f} (the median). "
                "Inputs: Recency and Frequency (Monetary is left out because it defines the label).")

    st.write(f"Training customers: {e['n_train']:,} | Testing customers: {e['n_test']:,} "
             "(80/20 split, class balance kept).")
    st.subheader("Model comparison")
    st.dataframe(e["results"].round(4))
    st.plotly_chart(px.bar(e["results"], x="Model", y=["Accuracy", "F1 Score", "ROC-AUC"],
                           barmode="group", title="Accuracy, F1 and ROC-AUC by model",
                           labels={"value": "Score", "variable": "Metric"}, range_y=[0.5, 1]))
    model_name = st.selectbox("Confusion matrix for", e["results"]["Model"].tolist())
    st.plotly_chart(confusion_figure(e["cms"][model_name], labels, f"{model_name}: actual vs predicted"))
    st.caption("Frequency and spend are strongly related, so these tasks are easier than they look. "
               "All models score about 82 to 85 percent.")

# ----------------------------------------------------------------------------
# Page: Future purchase
# ----------------------------------------------------------------------------
elif page == "Future purchase":
    st.title("Will the customer buy in the next 30 days?")
    with st.spinner("Training five models (first load only)..."):
        e = run_all_experiments()["future"]
    st.info(f"History used: everything before {CUTOFF:%d %b %Y}. "
            f"Target: did the customer buy between {CUTOFF:%d %b %Y} and {PREDICTION_END:%d %b %Y}? "
            "Inputs: Recency, Frequency, Monetary, Average order value.")
    buyers = int(e["y"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in the data", f"{len(e['y']):,}")
    c2.metric("Bought in next 30 days", f"{buyers:,}")
    c3.metric("Share who bought", f"{buyers / len(e['y']):.0%}")

    st.subheader("Model comparison")
    st.dataframe(e["results"].round(4))
    st.warning("This is the hardest task here. Most models miss more than half of the real "
               "buyers (low recall), because only about one customer in three buys in the window. "
               "Treat the predictions as a guide, not a guarantee.")
    model_name = st.selectbox("Confusion matrix for", e["results"]["Model"].tolist())
    st.plotly_chart(confusion_figure(e["cms"][model_name], ["No purchase", "Purchase"],
                                     f"{model_name}: actual vs predicted"))
    imp = pd.DataFrame({"Feature": e["features"],
                        "Importance": e["models"]["Random Forest"].feature_importances_}
                       ).sort_values("Importance")
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                           title="Random Forest: which inputs matter most"))

# ----------------------------------------------------------------------------
# Page: Predict a customer
# ----------------------------------------------------------------------------
else:
    st.title("Predict a Customer")
    with st.spinner("Preparing models (first load only)..."):
        seg_model = fit_segmentation()
        exps = run_all_experiments()

    tab_lookup, tab_new = st.tabs(["Look up an existing customer", "Enter new values"])
    r = f = m = None

    with tab_lookup:
        rfm = seg_model["seg"]
        cid = st.selectbox("Customer ID", rfm.index.tolist())
        row = rfm.loc[cid]
        st.write(f"Recency {int(row['Recency'])} days | Frequency {int(row['Frequency'])} orders | "
                 f"Monetary £{row['Monetary']:,.2f}")
        st.success(f"Segment: **{row['Segment']}**. {SEGMENT_MEANING[row['Segment']]}")

    with tab_new:
        c1, c2, c3 = st.columns(3)
        r = c1.number_input("Recency (days since last order)", 1, 400, 30)
        f = c2.number_input("Frequency (number of orders)", 1, 300, 3)
        m = c3.number_input("Monetary (total spend, £)", 1.0, 500000.0, 800.0, step=50.0)
        if st.button("Predict"):
            segment = predict_segment(seg_model, r, f, m)
            st.success(f"Segment: **{segment}**. {SEGMENT_MEANING[segment]}")

            svm1 = exps["freq"]["models"]["SVM"]
            p1 = svm1.predict_proba(pd.DataFrame([[r, m]], columns=["Recency", "Monetary"]))[0, 1]
            svm2 = exps["value"]["models"]["SVM"]
            p2 = svm2.predict_proba(pd.DataFrame([[r, f]], columns=["Recency", "Frequency"]))[0, 1]
            rf = exps["future"]["models"]["Random Forest"]
            p3 = rf.predict_proba(pd.DataFrame(
                [[r, f, m, m / f]], columns=exps["future"]["features"]))[0, 1]

            a, b, c = st.columns(3)
            a.metric("Frequent buyer?", "Yes" if p1 >= 0.5 else "No", f"{p1:.0%} chance", delta_color="off")
            b.metric("High-value?", "Yes" if p2 >= 0.5 else "No", f"{p2:.0%} chance", delta_color="off")
            c.metric("Buys in next 30 days?", "Likely" if p3 >= 0.5 else "Unlikely",
                     f"{p3:.0%} chance", delta_color="off")
            st.caption("The future-purchase model is the weakest of the three (see the Future purchase page).")
