
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Maternal–Fetal Epigenomic AI Prototype",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main {background:#f7fafc;}
.block-container {max-width:1200px; padding-top:2rem;}
.hero {padding:1.4rem 1.6rem; border-radius:18px; background:linear-gradient(135deg,#12324a,#0d6e7a); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.25rem; margin:0 0 .35rem 0;}
.hero p {font-size:1.02rem; margin:0; opacity:.94;}
.card {padding:1.05rem 1.15rem; border:1px solid #d9e4ea; border-radius:14px; background:white; height:100%;}
.section {font-size:1.35rem; font-weight:800; color:#12324a; margin:1.1rem 0 .6rem;}
.small {font-size:.88rem; color:#526774;}
.disclaimer {padding:1rem 1.1rem; border-left:5px solid #0d6e7a; background:#edf7f8; border-radius:8px; color:#234;}
.result {padding:1.2rem; border-radius:15px; background:white; border:1px solid #d9e4ea; text-align:center;}
.result h3 {margin:.1rem 0 .4rem; color:#12324a;}
.beta {font-size:2rem; font-weight:800; color:#0d6e7a;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Avishkar_Epigenetics_Pregnancy_Dataset.csv")

@st.cache_resource
def train_models():
    df = load_data()

    targets = {
        "AHRR": "AHRR_Gene_Methylation_BetaValue",
        "NR3C1": "NR3C1_Gene_Methylation_BetaValue",
        "PGC1α": "PGC1a_Gene_Methylation_BetaValue",
    }

    feature_sets = {
        "AHRR": ["Maternal_Age", "Smoking_Status"],
        "NR3C1": ["Maternal_Age", "Maternal_Stress_Index_1to10"],
        "PGC1α": [
            "Maternal_Age",
            "Diet_Quality_Score_1to100",
            "Physical_Activity_Hours_Per_Week",
            "Fasting_Glucose_mgdL",
        ],
    }

    models, metrics, importances, distributions = {}, {}, {}, {}

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    categorical_by_model = {
        "AHRR": ["Smoking_Status"],
        "NR3C1": [],
        "PGC1α": [],
    }

    for name, target in targets.items():
        features = feature_sets[name]
        X = df[features]
        y = df[target]

        cats = categorical_by_model[name]
        nums = [c for c in features if c not in cats]

        transformers = []

        if cats:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    cats,
                )
            )

        if nums:
            transformers.append(
                (
                    "num",
                    StandardScaler(),
                    nums,
                )
            )

        prep = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        pipe = Pipeline(
            [
                ("prep", prep),
                ("model", Ridge(alpha=1.0)),
            ]
        )

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            return_train_score=False,
        )

        metrics[name] = {
            "MAE": float(-scores["test_mae"].mean()),
            "RMSE": float(-scores["test_rmse"].mean()),
            "R2": float(scores["test_r2"].mean()),
        }

        pipe.fit(X, y)
        models[name] = pipe

        importances[name] = permutation_importance(
            pipe,
            X,
            y,
            n_repeats=30,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        distributions[name] = y.to_numpy()

    return models, metrics, importances, distributions
        pipe.fit(X, y)
        models[name] = pipe
        p = permutation_importance(
            pipe, X, y, n_repeats=30, random_state=42,
            scoring="neg_mean_absolute_error"
        )
        importances[name] = sorted(
            zip(features, p.importances_mean),
            key=lambda x: abs(x[1]), reverse=True
        )
        distributions[name] = {
            "mean": float(y.mean()),
            "sd": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }
    return models, metrics, importances, distributions

def diet_score(fruit, veg, whole, pulses, protein, processed, sugary):
    # Prototype mapping only; not a validated clinical diet score.
    positive = {
        "Fruit": fruit, "Vegetables": veg, "Whole grains/millets/oats": whole,
        "Pulses/beans/lentils": pulses, "Protein-rich foods": protein
    }
    negative = {"Packaged/ultra-processed foods": processed, "Sugary drinks/sweets": sugary}
    pos_map = {"Rarely":0, "1–2 days/week":1, "3–4 days/week":2, "5–6 days/week":3, "Daily":4}
    neg_map = {"Rarely":4, "1–2 days/week":3, "3–4 days/week":2, "5–6 days/week":1, "Daily":0}
    raw = sum(pos_map[v] for v in positive.values()) + sum(neg_map[v] for v in negative.values())
    return round(35 + (raw / 28) * 56.4, 1)

def stress_score(qs):
    # Maps four frequency questions to the dataset's 1–10 prototype scale.
    mp = {"Never":0, "Sometimes":1, "Often":2, "Very often":3}
    raw = sum(mp[q] for q in qs)
    return round(1 + (raw / 12) * 9, 1)

def percentile(value, arr):
    return float(np.mean(np.asarray(arr) <= value) * 100)

models, metrics, importances, distributions = train_models()
df = load_data()

st.markdown("""
<div class="hero">
<h1>🧬 Maternal–Fetal Epigenomic AI Prototype</h1>
<p>Research proof-of-concept for estimating epigenetic patterns associated with maternal environmental and metabolic factors.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["About", "Maternal Questionnaire", "AI Results", "Model & Methods"])

with tab1:
    st.markdown('<div class="section">What is this project?</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><b>Epigenomics</b><br><br>DNA methylation is an epigenetic modification that can influence gene regulation without changing the underlying DNA sequence.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>Maternal environment</b><br><br>Nutrition, activity, tobacco exposure, stress and metabolic factors are investigated as variables that may be associated with epigenetic patterns.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><b>Three markers</b><br><br><b>AHRR</b>: tobacco-exposure-associated marker<br><b>NR3C1</b>: glucocorticoid receptor / stress biology<br><b>PGC1α</b>: energy metabolism</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">Biological framework</div>', unsafe_allow_html=True)
    st.info("Maternal environment  →  maternal–fetal interface / placenta  →  DNA methylation  →  gene regulation  →  fetal development")
    st.markdown('<div class="section">Important interpretation rule</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">The prototype estimates patterns learned from a synthetic dataset. It does not establish that a maternal factor causes methylation changes, and it does not directly measure methylation in a new user.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section">Basic information</div>', unsafe_allow_html=True)
    age = st.number_input("Maternal age (years)", min_value=18, max_value=60, value=28)
    st.caption("The current training dataset contains ages 18–36; values outside that range are accepted by the interface but should be treated as extrapolation.")

    st.markdown('<div class="section">Tobacco exposure</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking status", ["Never", "Quit Pre-pregnancy", "Active (1-5/day)", "Active (>5/day)"])

    st.markdown('<div class="section">Diet</div>', unsafe_allow_html=True)
    freq = ["Rarely", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"]
    cols = st.columns(2)
    with cols[0]:
        fruit = st.selectbox("Fruit", freq, index=3)
        veg = st.selectbox("Vegetables", freq, index=3)
        whole = st.selectbox("Whole grains / millets / oats", freq, index=2)
        pulses = st.selectbox("Pulses / beans / lentils / chickpeas", freq, index=3)
    with cols[1]:
        protein = st.selectbox("Protein-rich foods", freq, index=3)
        processed = st.selectbox("Packaged / ultra-processed foods", freq, index=2)
        sugary = st.selectbox("Sugary drinks / sweets", freq, index=2)
    diet = diet_score(fruit, veg, whole, pulses, protein, processed, sugary)
    st.caption(f"Prototype diet-quality score generated from these responses: **{diet}/100**. This is not a clinically validated dietary score.")

    st.markdown('<div class="section">Physical activity</div>', unsafe_allow_html=True)
    days = st.slider("Days of activity per week", 0, 7, 3)
    mins = st.slider("Average minutes per session", 0, 180, 45)
    activity = round(days * mins / 60, 2)
    st.caption(f"Calculated activity exposure: **{activity} hours/week**")

    st.markdown('<div class="section">Stress-related responses</div>', unsafe_allow_html=True)
    stress_options = ["Never", "Sometimes", "Often", "Very often"]
    s1 = st.selectbox("How often have you felt overwhelmed by things you had to do?", stress_options, index=1)
    s2 = st.selectbox("How often have you felt unable to control important things?", stress_options, index=1)
    s3 = st.selectbox("How often have you felt nervous or stressed?", stress_options, index=1)
    s4 = st.selectbox("How often have you found it difficult to relax?", stress_options, index=1)
    stress = stress_score([s1,s2,s3,s4])
    st.caption(f"Prototype stress value mapped to the training dataset scale: **{stress}/10**. This is not a validated clinical stress score.")

    st.markdown('<div class="section">Metabolic information</div>', unsafe_allow_html=True)
    glucose = st.number_input("Fasting glucose (mg/dL)", min_value=50, max_value=250, value=90)
    st.caption("A fasting-glucose value is required for the current PGC1α model because it was one of the selected training predictors.")

    if st.button("Analyze Profile", type="primary", use_container_width=True):
        st.session_state["profile"] = {
            "Maternal_Age": age, "Smoking_Status": smoking,
            "Diet_Quality_Score_1to100": diet,
            "Physical_Activity_Hours_Per_Week": activity,
            "Maternal_Stress_Index_1to10": stress,
            "Fasting_Glucose_mgdL": glucose
        }
        st.success("Profile submitted. Open the **AI Results** tab.")

with tab3:
    if "profile" not in st.session_state:
        st.info("Complete the questionnaire and click **Analyze Profile** first.")
    else:
        p = st.session_state["profile"]
        inputs = pd.DataFrame([p])
        st.markdown('<div class="section">Model-estimated epigenetic patterns</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        preds = {}
        for i,name in enumerate(["AHRR","NR3C1","PGC1α"]):
            pred = float(models[name].predict(inputs[[c for c in inputs.columns if c in models[name].feature_names_in_]])[0])
            preds[name] = pred
            d = distributions[name]
            pct = percentile(pred, df[{"AHRR":"AHRR_Gene_Methylation_BetaValue","NR3C1":"NR3C1_Gene_Methylation_BetaValue","PGC1α":"PGC1a_Gene_Methylation_BetaValue"}[name]])
            with cols[i]:
                st.markdown(f'<div class="result"><h3>{name}</h3><div class="beta">{pred:.3f}</div><div>model-estimated beta value</div><br><div class="small">Training-distribution percentile: {pct:.0f}th</div></div>', unsafe_allow_html=True)
                st.progress(min(max(pred,0),1))
        st.caption("Beta values are shown on a 0–1 scale. The percentile is relative to this synthetic training dataset and is not a clinical reference range.")

        st.markdown('<div class="section">Factors contributing to these model estimates</div>', unsafe_allow_html=True)
        for name in ["AHRR","NR3C1","PGC1α"]:
            st.write(f"**{name}**")
            imp = pd.DataFrame(importances[name], columns=["Variable","Permutation importance"])
            imp["Absolute importance"] = imp["Permutation importance"].abs()
            imp = imp.sort_values("Absolute importance", ascending=False).drop(columns=["Absolute importance"])
            st.bar_chart(imp.set_index("Variable")["Permutation importance"])
            st.caption("Permutation importance describes how much predictive performance changes when a variable is shuffled. It is not evidence of biological causation.")

        st.markdown('<div class="section">Human-readable interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="card">
The model has estimated patterns for AHRR, NR3C1 and PGC1α from the questionnaire-derived variables. 
The contributing factors shown above are variables that influenced model performance in this dataset.
They should be interpreted as <b>model associations</b>, not causes of methylation change.
<br><br>
For example, the AHRR model includes smoking status because AHRR-associated methylation has been studied in relation to tobacco exposure. 
The NR3C1 model includes the stress-related prototype variable because NR3C1 encodes the glucocorticoid receptor and is biologically relevant to stress/HPA-axis research.
The PGC1α model uses diet, activity and fasting glucose because these variables represent the metabolic/lifestyle environment represented in the current dataset.
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section">General health-information note</div>', unsafe_allow_html=True)
        st.info("This prototype does not diagnose disease or predict a baby's health. If a user has concerns about smoking, glucose results, nutrition, stress or other pregnancy-related issues, those should be discussed with a qualified healthcare professional.")

        st.markdown('<div class="disclaimer"><b>Clinical disclaimer:</b> This is a research prototype and not a medical diagnostic tool. The model estimates patterns learned from the available dataset and does not directly measure DNA methylation in an individual. The current prototype requires validation using large, independent real-world pregnancy cohorts before any clinical application could be considered.</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section">Training design</div>', unsafe_allow_html=True)
    st.write("The three targets are continuous methylation beta values, so the prototype uses regression.")
    rows = []
    for name in ["AHRR","NR3C1","PGC1α"]:
        rows.append({
            "Target": name,
            "Predictors": ", ".join(models[name].feature_names_in_),
            "CV MAE": f"{metrics[name]['MAE']:.3f}",
            "CV RMSE": f"{metrics[name]['RMSE']:.3f}",
            "CV R²": f"{metrics[name]['R2']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Metrics are means from repeated 5-fold cross-validation (5 folds × 10 repeats). Because the dataset is synthetic and contains only 100 observations, these metrics demonstrate a computational proof-of-concept rather than real-world predictive performance.")

    st.markdown('<div class="section">Why these predictors?</div>', unsafe_allow_html=True)
    st.write("""
- **AHRR:** maternal age + smoking status. Smoking is the principal exposure represented in the dataset that has a direct biological rationale for AHRR-associated methylation.
- **NR3C1:** maternal age + the stress-related prototype variable. This keeps the model small and interpretable.
- **PGC1α:** maternal age + diet quality + physical activity + fasting glucose. These represent the metabolic/lifestyle environment captured by the dataset.
- **Patient_ID is excluded** because it is an identifier with no biological meaning.
- **Birth weight is excluded** because it is a downstream outcome and using it to estimate methylation would create an inappropriate leakage pathway for the intended user-facing application.
- GDM status and the three OGTT measures are not used in the current final models because they overlap strongly with glucose/metabolic information and would add complexity relative to the sample size. They remain useful for descriptive analysis.
""")

    st.markdown('<div class="section">Dataset limitations</div>', unsafe_allow_html=True)
    st.write("""
The uploaded dataset has 100 observations, no missing values, four smoking categories and two GDM categories. The three methylation columns are continuous beta-value targets. The dataset is synthetic, so its correlations and model performance cannot be treated as evidence from real pregnancies. The current variables also do not capture the full epigenome, CpG-site specificity, tissue/cell composition, batch effects or the broader confounding structure of real epigenetic cohorts.
""")

    st.markdown('<div class="section">Project separation</div>', unsafe_allow_html=True)
    st.write("""
**Published evidence:** provides biological rationale for studying maternal exposures and offspring methylation.

**Synthetic-data analysis:** demonstrates relationships and model behaviour in this constructed dataset.

**AI application:** demonstrates how questionnaire responses could be converted into model inputs and used to estimate methylation-associated patterns.
""")
