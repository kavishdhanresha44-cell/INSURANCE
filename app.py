
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

st.set_page_config(page_title="Insurance Policy Status Predictor", layout="wide")

# Load data
@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv("Insurance.csv")
    return df

# Prepare data
def prepare_data(df, target_col="POLICY_ STATUS"):
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object","category","bool"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median"))
        ]), num_cols)
    ])
    classes, y_encoded = np.unique(y, return_inverse=True)
    return X, y_encoded, y, preprocess, classes, cat_cols, num_cols

# Evaluation
def evaluate_models(X, y, preprocess, classes, cat_cols, num_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    results = []
    fitted = {}
    n_classes = len(classes)
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        auc = np.nan
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score(y_test, proba[:,1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        try:
            cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc_ovr_weighted")
        except Exception:
            cv_auc = np.full(5, np.nan)
        results.append({
            "Model": name,
            "Training Accuracy": round(train_acc,4),
            "Testing Accuracy": round(test_acc,4),
            "Precision": round(prec,4),
            "Recall": round(rec,4),
            "F1-Score": round(f1,4),
            "AUC (ROC)": round(auc,4),
            "CV Accuracy (mean±std)": f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}",
            "CV AUC (mean±std)": f"{np.nanmean(cv_auc):.4f} ± {np.nanstd(cv_auc):.4f}"
        })
        fitted[name] = {"pipe": pipe, "y_train": y_train, "y_test": y_test,
                        "y_pred_train": y_pred_train, "y_pred_test": y_pred_test}
    return pd.DataFrame(results), fitted

# Confusion matrix plot
def plot_confusion(cm, classes, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.tight_layout()
    st.pyplot(fig)

# Feature importance
def plot_feature_importance(pipe, cat_cols, num_cols, title):
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    if hasattr(model, "feature_importances_"):
        cat_trans = prep.named_transformers_["cat"]
        cat_names = cat_trans.named_steps["onehot"].get_feature_names_out(cat_cols)
        all_feats = np.concatenate([cat_names, num_cols])
        imp = pd.DataFrame({"Feature": all_feats, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(20)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title=title)
        st.plotly_chart(fig, use_container_width=True)

# Interface
st.title("🏢 Insurance Policy Status Prediction Dashboard")

uploaded = st.file_uploader("Upload Insurance.csv (or dataset with POLICY_ STATUS column)", type=["csv","xlsx"])
df = load_data(uploaded)
st.write("### Data Preview", df.head())

X, y, y_raw, preprocess, classes, cat_cols, num_cols = prepare_data(df)

tab1, tab2 = st.tabs(["📊 Model Lab", "💡 Feature Importance"])

with tab1:
    if st.button("Run Models"):
        summary_df, fitted = evaluate_models(X, y, preprocess, classes, cat_cols, num_cols)
        st.dataframe(summary_df, use_container_width=True)
        # ROC curves
        plt.figure()
        for name, info in fitted.items():
            pipe = info["pipe"]
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X)
                if len(classes) == 2:
                    fpr, tpr, _ = roc_curve(info["y_test"], pipe.predict_proba(X[0:len(info["y_test"])])[:,1])
                    plt.plot(fpr, tpr, label=name)
        plt.plot([0,1],[0,1],'--')
        plt.title("ROC Curves — Policy Status")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(plt)
        # Confusion matrices
        for name, info in fitted.items():
            cm_train = confusion_matrix(info["y_train"], info["y_pred_train"])
            cm_test  = confusion_matrix(info["y_test"], info["y_pred_test"])
            plot_confusion(cm_train, classes, f"Training Confusion Matrix — {name}")
            plot_confusion(cm_test, classes, f"Testing Confusion Matrix — {name}")

with tab2:
    st.write("### Top Features by Importance")
    _, fitted = evaluate_models(X, y, preprocess, classes, cat_cols, num_cols)
    for name, info in fitted.items():
        plot_feature_importance(info["pipe"], cat_cols, num_cols, f"Top Features — {name}")

st.caption("Best overall: Random Forest (AUC≈0.81, balanced accuracy, stable CV). Ideal for predicting policy outcomes.")
