# streamlit_mva_checker.py
# Prototype med ML + regelmotor + fiktiv eskalering + rapport + grafer + tidsserie + PDF + Tabs + syntetisk datasett + varselsymboler + fargekoding + varselgrafer i PDF + toggle for grafer

import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import zipfile
import matplotlib.pyplot as plt

# streamlit_mva_checker.py (med valgfri logo i PDF for manglende kolonner)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

EXPECTED_COLS = [
    "invoice_id","date","amount_net","vat_rate","vat_claimed","invoice_type",
    "industry_code","counterparty_country","reverse_charge","export","intra_eu","mva_class"
]

MVA_CLASSES = {
    "full": 25,
    "matvarer": 15,
    "transport": 12,
    "eksport": 0,
    "intra_eu": 0
}

# -----------------------------
# PDF-generator for manglende kolonner
# -----------------------------

def generate_missing_cols_pdf(missing_cols):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Valgfri logo hvis filen finnes
    logo_path = "jaujau.png"
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=120, height=60))
        story.append(Spacer(1, 12))

    # Header
    story.append(Paragraph("Skatteetaten – Fiktiv prototype", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Manglende kolonner i opplastet fil", styles['Heading2']))
    story.append(Spacer(1, 12))

    text = "Følgende kolonner manglet og ble satt til standardverdier:"
    story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    for col in missing_cols:
        if col == "mva_class":
            story.append(Paragraph(f"- {col}: satt til 'full' (standardklasse)", styles['Normal']))
        else:
            story.append(Paragraph(f"- {col}: satt til 0 (standardverdi)", styles['Normal']))

    story.append(Spacer(1, 24))
    story.append(Paragraph("⚠️ Merk: Dette er kun en faglig prototype, ikke en faktisk skattekontroll.", styles['Italic']))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

# -----------------------------
# Hjelpefunksjoner
# -----------------------------

def ensure_expected_cols(df):
    missing = []
    for c in EXPECTED_COLS:
        if c not in df.columns:
            missing.append(c)
            if c == "mva_class":
                df[c] = "full"   # default-klasse hvis mangler
            else:
                df[c] = 0
    if missing:
        st.warning(f"Følgende kolonner manglet i filen og ble satt til standardverdier: {', '.join(missing)}")
        pdf_bytes = generate_missing_cols_pdf(missing)
        st.download_button("Last ned manglende kolonner-rapport (PDF)", data=pdf_bytes,
                           file_name="manglende_kolonner.pdf", mime="application/pdf")
    return df

# resten av koden beholdes som før

def feature_engineering(df):
    df = df.copy()
    df["vat_expected"] = (df["amount_net"].astype(float) * (df["vat_rate"].astype(float)/100)).round(2)
    df["vat_diff"] = (df["vat_claimed"].astype(float) - df["vat_expected"]).round(2)
    df["rule_high_diff"] = np.where(df["vat_diff"].abs() > 0.05*df["vat_expected"].abs(),1,0)

    # Forventet sats fra klasse
    df["expected_rate"] = df["mva_class"].map(MVA_CLASSES).fillna(df["vat_rate"])

    # Sjekk om faktisk sats stemmer med forventet
    df["rule_wrong_class"] = np.where(df["vat_rate"] != df["expected_rate"], 1, 0)

    return df

def build_model(model_type="gb"):
    num = ["amount_net", "vat_rate"]
    cat = ["invoice_type", "industry_code", "counterparty_country", "reverse_charge", "export", "intra_eu"]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat",
         Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         cat)
    ])
    clf = LogisticRegression(max_iter=200) if model_type == "logreg" else GradientBoostingClassifier()
    return Pipeline([("pre", pre), ("clf", clf)])


def train_model(df, model_type="gb"):
    df = feature_engineering(ensure_expected_cols(df))
    if "label_mva_correct" not in df.columns:
        return None, None
    X = df[
        ["amount_net", "vat_rate", "invoice_type", "industry_code", "counterparty_country", "reverse_charge", "export",
         "intra_eu"]]
    y = df["label_mva_correct"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_model(model_type)
    pipe.fit(Xtr, ytr)
    yprob = pipe.predict_proba(Xte)[:, 1]
    ypred = (yprob >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(yte, yprob),
        "cm": confusion_matrix(yte, ypred).tolist(),
        "report": classification_report(yte, ypred, output_dict=True)
    }
    return pipe, metrics


def hybrid_score(df, pipe, ml_weight=0.6):
    df = feature_engineering(ensure_expected_cols(df))
    X = df[
        ["amount_net", "vat_rate", "invoice_type", "industry_code", "counterparty_country", "reverse_charge", "export",
         "intra_eu"]]
    try:
        p_correct = pipe.predict_proba(X)[:, 1]
    except:
        p_correct = np.full(len(df), 0.5)
    df["p_correct_ml"] = p_correct
    df["risk"] = 1 - p_correct
    return df


# -----------------------------
# Varselsymboler og fargekoding
# -----------------------------

def add_warnings(df, risk_threshold=0.4):
    df = df.copy()
    df["Varsel"] = "✅"
    df.loc[df["risk"] >= risk_threshold, "Varsel"] = "⚠️"
    df.loc[df["risk"] >= 0.8, "Varsel"] = "🚨"
    df.loc[df["rule_wrong_class"] == 1, "Varsel"] = "🚨"  # Feil MVA-klasse overstyrer
    return df


def style_warnings(df):
    def highlight(val):
        if val == "✅":
            return "background-color: #c6f6d5"
        elif val == "⚠️":
            return "background-color: #fefcbf"
        elif val == "🚨":
            return "background-color: #feb2b2"
        return ""

    return df.style.applymap(highlight, subset=["Varsel"])


# -----------------------------
# Grafer
# -----------------------------

def plot_risk_distribution(df, threshold):
    fig, ax = plt.subplots()
    ax.hist(df["risk"], bins=20)
    ax.axvline(threshold, linestyle="--", label="Terskel")
    ax.set_title("Fordeling av risiko-score")
    ax.set_xlabel("Risiko")
    ax.set_ylabel("Antall bilag")
    ax.legend()
    return fig


def plot_time_series(df):
    df_ts = df.copy()
    df_ts["date"] = pd.to_datetime(df_ts["date"], errors="coerce")
    ts = df_ts.groupby("date")["risk"].mean().dropna()
    fig, ax = plt.subplots()
    ts.plot(ax=ax)
    ax.set_title("Gjennomsnittlig risiko over tid")
    ax.set_xlabel("Dato")
    ax.set_ylabel("Gj.snittlig risiko")
    return fig


def plot_warning_counts(df):
    counts = df["Varsel"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Antall bilag per varseltype")
    ax.set_xlabel("Varseltype")
    ax.set_ylabel("Antall")
    return fig


def plot_warning_pie(df):
    counts = df["Varsel"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))  # juster ned til 4x4 tommer
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax.set_title("Prosentvis fordeling av varseltyper")
    return fig

# -----------------------------
# PDF med grafer (toggle for valg)
# -----------------------------

def _fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def generate_graph_pdf(df, threshold, include_bar=True, include_pie=True, include_time=True, include_hist=True):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Risikografi-rapport", styles['Title']), Spacer(1, 12)]

    if include_hist:
        fig1 = plot_risk_distribution(df, threshold)
        img_buf1 = _fig_to_img(fig1)
        story.append(Image(img_buf1, width=400, height=250))
        story.append(Spacer(1, 12))

    if include_time:
        fig2 = plot_time_series(df)
        img_buf2 = _fig_to_img(fig2)
        story.append(Image(img_buf2, width=400, height=250))
        story.append(Spacer(1, 12))

    if "Varsel" not in df.columns:
        df = add_warnings(df, threshold)

    if include_bar:
        fig3 = plot_warning_counts(df)
        img_buf3 = _fig_to_img(fig3)
        story.append(Image(img_buf3, width=400, height=250))
        story.append(Spacer(1, 12))

    if include_pie:
        fig4 = plot_warning_pie(df)
        img_buf4 = _fig_to_img(fig4)
        story.append(Image(img_buf4, width=400, height=250))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# -----------------------------
# PDF-generator for brev
# -----------------------------

def generate_pdf(df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Fiktiv melding fra Skatteetaten", styles['Title']))
    story.append(Spacer(1, 12))
    for _, row in df.iterrows():
        text = f"Bilag {row['invoice_id']} datert {row['date']} er flagget med risiko {row['risk']:.2f}. Vennligst kontroller oppgitt MVA."
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 12))
    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# -----------------------------
# Samlet rapport ZIP
# -----------------------------

def generate_report_zip(flagged, auto_notify, result, threshold, include_bar, include_pie, include_time, include_hist):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        csv_buf = io.StringIO()
        flagged.to_csv(csv_buf, index=False)
        zf.writestr("flagged.csv", csv_buf.getvalue())
        if not auto_notify.empty:
            pdf_bytes = generate_pdf(auto_notify)
            zf.writestr("auto_notify.pdf", pdf_bytes)
        pdf_graphs = generate_graph_pdf(result, threshold, include_bar, include_pie, include_time, include_hist)
        zf.writestr("risk_graphs.pdf", pdf_graphs)
    zip_buf.seek(0)
    return zip_buf.getvalue()


# -----------------------------
# Syntetisk testdatasett
# -----------------------------

def get_synthetic_dataset():
    data = [
        # Korrekt: full sats (25 %)
        {"invoice_id": "INV001", "date": "2025-01-10", "amount_net": 1000, "vat_rate": 25, "vat_claimed": 250,
         "mva_class": "full", "invoice_type": "salg", "industry_code": "47",
         "counterparty_country": "NO", "reverse_charge": 0, "export": 0, "intra_eu": 0},

        # Feil: matvarer skulle hatt 15 %, men her står 25 %
        {"invoice_id": "INV002", "date": "2025-01-15", "amount_net": 2000, "vat_rate": 25, "vat_claimed": 600,
         "mva_class": "matvarer", "invoice_type": "salg", "industry_code": "47",
         "counterparty_country": "NO", "reverse_charge": 0, "export": 0, "intra_eu": 0},

        # Korrekt: matvarer 15 %
        {"invoice_id": "INV003", "date": "2025-02-01", "amount_net": 500, "vat_rate": 15, "vat_claimed": 75,
         "mva_class": "matvarer", "invoice_type": "salg", "industry_code": "56",
         "counterparty_country": "NO", "reverse_charge": 0, "export": 0, "intra_eu": 0},

        # Korrekt: eksport 0 %
        {"invoice_id": "INV004", "date": "2025-02-05", "amount_net": 3000, "vat_rate": 0, "vat_claimed": 0,
         "mva_class": "eksport", "invoice_type": "eksport", "industry_code": "47",
         "counterparty_country": "US", "reverse_charge": 0, "export": 1, "intra_eu": 0},

        # Feil: transport skulle hatt 12 %, men ført som 25 %
        {"invoice_id": "INV005", "date": "2025-02-10", "amount_net": 1200, "vat_rate": 25, "vat_claimed": 200,
         "mva_class": "transport", "invoice_type": "salg", "industry_code": "49",
         "counterparty_country": "NO", "reverse_charge": 0, "export": 0, "intra_eu": 0},

        # Korrekt: intra-EU 0 %
        {"invoice_id": "INV006", "date": "2025-02-15", "amount_net": 800, "vat_rate": 0, "vat_claimed": 0,
         "mva_class": "intra_eu", "invoice_type": "intra_eu", "industry_code": "46",
         "counterparty_country": "SE", "reverse_charge": 1, "export": 0, "intra_eu": 1},

        # Feil: full sats, men feil beløp (260 i stedet for 250)
        {"invoice_id": "INV007", "date": "2025-03-01", "amount_net": 1000, "vat_rate": 25, "vat_claimed": 260,
         "mva_class": "full", "invoice_type": "salg", "industry_code": "47",
         "counterparty_country": "NO", "reverse_charge": 0, "export": 0, "intra_eu": 0},
    ]
    return pd.DataFrame(data)

# -----------------------------
# Tabs med emoji
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Hjem",
    "🤖 Tren modellen",
    "📂 Vurder filer",
    "ℹ️ Om",
    "🧪 Test med syntetiske data"
])

with tab1:
    st.title("Velkommen til MVA-kontroll prototypen")
    st.write(
        "I menyen her kan du gjerne bruke fanene øverst for å trene modellen, vurdere filer, teste med syntetiske data eller lese mer om løsningen under Om-fanen.:)")

with tab2:
    model_type = st.selectbox("Modelltype", ["gb", "logreg"],
                              format_func=lambda x: "Gradient Boosting" if x == "gb" else "Logistisk regresjon")
    if "trained_pipe" not in st.session_state:
        st.session_state["trained_pipe"] = None
    uploaded = st.file_uploader("Last opp treningsdata (CSV)", type="csv", key="train")
    if uploaded:
        df = pd.read_csv(uploaded)
        pipe, metrics = train_model(df, model_type)
        st.session_state["trained_pipe"] = pipe
        st.success("Modell trent!")
        if metrics:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
            st.json(metrics["report"])

with tab3:
    risk_threshold = st.slider("Risiko-grense for flagging",0.0,1.0,0.4,0.05)
    include_hist = st.checkbox("Inkluder histogram i PDF", value=True)
    include_time = st.checkbox("Inkluder tidsserie i PDF", value=True)
    include_bar = st.checkbox("Inkluder stolpediagram i PDF", value=True)
    include_pie = st.checkbox("Inkluder kakediagram i PDF", value=True)

    if "trained_pipe" not in st.session_state:
        st.session_state["trained_pipe"] = None
    uploaded = st.file_uploader("Last opp skattemeldingsdata (CSV)", type="csv", key="predict")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Ingen fil lastet opp. Du kan teste i 🧪 Test-fanen med syntetiske data.")
        df = None
    if df is not None:
        pipe = st.session_state.get("trained_pipe") or build_model()
        result = hybrid_score(df, pipe)
        result = add_warnings(result, risk_threshold)
        result_sorted = result.sort_values("risk",ascending=False)

        st.subheader("Resultater")
        styled_result = style_warnings(result_sorted[["invoice_id","date","amount_net","vat_rate","vat_claimed","risk","Varsel"]])
        st.dataframe(style_warnings(
            result_sorted[
                ["invoice_id", "date", "amount_net", "vat_rate", "expected_rate", "vat_claimed", "risk", "Varsel"]]
        ), use_container_width=True)

        flagged = result_sorted[result_sorted["risk"]>=risk_threshold]
        to_caseworker = flagged[flagged["risk"]<0.8]
        auto_notify = flagged[flagged["risk"]>=0.8]

        st.subheader("Fiktiv videre behandling")
        st.write("Bilag sendt til fiktiv **saksbehandler**:")
        st.dataframe(style_warnings(to_caseworker[["invoice_id","date","amount_net","vat_rate","vat_claimed","risk","Varsel"]]), use_container_width=True)

        st.write("Bilag sendt til fiktiv **automatisk henvendelse** til avgiftspliktig:")
        st.dataframe(style_warnings(auto_notify[["invoice_id","date","amount_net","vat_rate","vat_claimed","risk","Varsel"]]), use_container_width=True)

        if not flagged.empty:
            zip_bytes = generate_report_zip(flagged, auto_notify, result, risk_threshold,
                                            include_bar, include_pie, include_time, include_hist)
            st.download_button("Last ned samlet rapport (CSV+PDF med grafer i ZIP)",
                               data=zip_bytes, file_name="rapport.zip", mime="application/zip")

            st.subheader("Oppsummering med grafer")
            st.pyplot(plot_risk_distribution(result, risk_threshold))
            st.pyplot(plot_time_series(result))
            st.pyplot(plot_warning_counts(result))
            st.pyplot(plot_warning_pie(result))

            st.metric("Totalt antall bilag", len(result))
            st.metric("Flaggede bilag", len(flagged))
            st.metric("Til saksbehandler", len(to_caseworker))
            st.metric("Automatisk henvendelse", len(auto_notify))

with tab4:
    st.title("Om denne appen")
    st.markdown("""
    Denne prototypen demonstrerer hvordan man kan bruke **maskinlæring** kombinert med en enkel **regelmotor**
    for å flagge potensielle feil i MVA-rapportering.

    ### Metoder som er brukt her
    - **Logistisk regresjon**: Dette her er en klassisk statistisk metode som estimerer sannsynligheten for at MVA er korrekt.
    - **Gradient Boosting**: Dette er en en kraftigere "ensemble-metode" som bygger mange små beslutningstrær i sekvens.
      Hvert nytt tre prøver å rette opp feilene fra de foregående, slik at modellen gradvis blir mer presis.

    ### Hva man får her:
    - Risiko-score for hvert bilag
    - Automatisert splitting mellom bilag til saksbehandler og automatisk varsling
    - Rapporter i CSV + PDF (brev og grafer)
    - Grafer for å se risikofordeling og utvikling over tid
    - Varselsymboler (✅, ⚠️, 🚨) med fargekoding (grønn/gul/rød) for enkel visuell tolkning
    - Fordeling av varsler både som stolpediagram og kakediagram

    ⚠️ Merk: Dette er kun en **faglig prototype** og ikke en faktisk skattekontroll. Denne appen er kun laget i forbindelse med et regnskapsfag på NMBU av Andreas Bolton Seielstad.
    """)
# ...

with tab5:
    st.title("Test med syntetiske data")
    st.write("Her kan du teste løsningen uten å laste opp egne filer.")
    df = get_synthetic_dataset()
    st.dataframe(df)

    include_hist = st.checkbox("Inkluder histogram i PDF (syntetisk)", value=True, key="synt_hist")
    include_time = st.checkbox("Inkluder tidsserie i PDF (syntetisk)", value=True, key="synt_time")
    include_bar = st.checkbox("Inkluder stolpediagram i PDF (syntetisk)", value=True, key="synt_bar")
    include_pie = st.checkbox("Inkluder kakediagram i PDF (syntetisk)", value=True, key="synt_pie")

    if st.button("Kjør vurdering på syntetiske data"):
        pipe = st.session_state.get("trained_pipe") or build_model()
        result = hybrid_score(df, pipe)
        result = add_warnings(result, 0.4)
        styled_result = style_warnings(result[["invoice_id","date","amount_net","vat_rate","vat_claimed","risk","Varsel"]])
        st.dataframe(styled_result, use_container_width=True)

        st.pyplot(plot_risk_distribution(result, 0.4))
        st.pyplot(plot_time_series(result))
        st.pyplot(plot_warning_counts(result))
        st.pyplot(plot_warning_pie(result))

        # Last ned PDF med syntetiske data
        pdf_bytes = generate_graph_pdf(result, 0.4,
                                       include_bar=include_bar,
                                       include_pie=include_pie,
                                       include_time=include_time,
                                       include_hist=include_hist)
        st.download_button("Last ned PDF-rapport (syntetiske data)",
                           data=pdf_bytes, file_name="synt_report.pdf", mime="application/pdf")



