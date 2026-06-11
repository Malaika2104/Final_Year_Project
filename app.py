"""
FYP2025-HealthTech — Head & Neck Cancer Outcome Dashboard
No WSI modality. Upload patient CSV → see predictions + analytics.
"""

import io
import base64

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ── colour tokens ────────────────────────────────────────────────────────────
C = dict(
    bg="#0D1117", panel="#161B22", border="#30363D",
    teal="#00B4D8", teal_dim="#0077A8",
    green="#3FB950", red="#F85149", amber="#E3B341",
    text="#E6EDF3", muted="#8B949E", white="#FFFFFF",
)

CARD = {
    "background": C["panel"], "border": f"1px solid {C['border']}",
    "borderRadius": "10px", "padding": "20px", "marginBottom": "18px",
}
BADGE = {
    "display": "inline-block", "padding": "3px 10px", "borderRadius": "20px",
    "fontSize": "12px", "fontWeight": "600", "marginLeft": "8px",
}

# ── clinical feature columns we expect (subset; extras are silently ignored) ──
CLINICAL_COLS = [
    "age", "gender", "t_stage", "n_stage", "m_stage", "overall_stage",
    "hpv_status", "smoking_status", "alcohol_use", "primary_site",
]
BLOOD_COLS = [
    "wbc", "rbc", "hemoglobin", "hematocrit", "platelets",
    "neutrophils", "lymphocytes", "monocytes", "eosinophils",
    "albumin", "creatinine", "alt", "ast", "bilirubin",
]
PATH_COLS = [
    "histology", "grade", "perineural_invasion", "lymphovascular_invasion",
    "margin_status", "extranodal_extension",
]
TARGET_SURVIVAL   = "survival_status"   # 1=alive, 0=deceased
TARGET_RECURRENCE = "recurrence_status" # 1=recurred, 0=no recurrence

# ── app ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="HNC Outcome Dashboard",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"background": C["bg"], "minHeight": "100vh", "color": C["text"],
           "fontFamily": "'Inter', 'Segoe UI', sans-serif", "padding": "0"},
    children=[

        # ── top bar ──────────────────────────────────────────────────────────
        html.Div(
            style={"background": C["panel"], "borderBottom": f"1px solid {C['border']}",
                   "padding": "14px 32px", "display": "flex",
                   "alignItems": "center", "justifyContent": "space-between"},
            children=[
                html.Div([
                    html.Span("HNC·AI", style={"color": C["teal"], "fontWeight": 800,
                                               "fontSize": "22px", "letterSpacing": "1px"}),
                    html.Span(" — Head & Neck Cancer Outcome Dashboard",
                              style={"color": C["muted"], "fontSize": "14px", "marginLeft": "12px"}),
                ]),
                html.Span("FYP2025-HealthTech · UET Peshawar",
                          style={"color": C["muted"], "fontSize": "12px"}),
            ]
        ),

        # ── main body ─────────────────────────────────────────────────────────
        html.Div(style={"padding": "28px 32px"}, children=[

            # upload card
            html.Div(style=CARD, children=[
                html.H5("Upload Patient Data", style={"color": C["teal"], "marginBottom": "14px",
                                                       "fontWeight": 700}),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.Span("📂  Drag & drop or "),
                        html.A("select CSV / Excel", style={"color": C["teal"], "cursor": "pointer"}),
                        html.Span("  file", style={"color": C["muted"]}),
                    ]),
                    style={
                        "width": "100%", "padding": "28px", "textAlign": "center",
                        "border": f"2px dashed {C['teal_dim']}", "borderRadius": "8px",
                        "background": "#0D1117", "cursor": "pointer", "color": C["muted"],
                    },
                    multiple=False,
                ),
                html.Div(id="upload-status", style={"marginTop": "10px", "fontSize": "13px"}),
            ]),

            # KPI row
            html.Div(id="kpi-row"),

            # tabs
            dbc.Tabs(
                id="main-tabs", active_tab="tab-overview",
                style={"borderBottom": f"1px solid {C['border']}", "marginBottom": "20px"},
                children=[
                    dbc.Tab(label="📊  Overview",     tab_id="tab-overview"),
                    dbc.Tab(label="🔬  Patient Table", tab_id="tab-table"),
                    dbc.Tab(label="📈  Predictions",   tab_id="tab-pred"),
                    dbc.Tab(label="🧬  Feature Insights", tab_id="tab-feat"),
                ],
            ),
            html.Div(id="tab-content"),

            # hidden store
            dcc.Store(id="stored-data"),
        ]),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_upload(contents, filename):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return None, str(e)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df, None


def encode_df(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def run_models(df, target_col):
    if target_col not in df.columns:
        return None, None, None, None
    df_enc = encode_df(df)
    y = df_enc[target_col]
    X = df_enc.drop(columns=[c for c in [TARGET_SURVIVAL, TARGET_RECURRENCE] if c in df_enc.columns])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = None
    cm = confusion_matrix(y_te, y_pred)
    importances = pipe.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(12)
    return acc, auc, cm, feat_imp


def kpi_card(label, value, color):
    return html.Div(style={**CARD, "textAlign": "center", "flex": "1", "margin": "0 8px"}, children=[
        html.Div(value, style={"fontSize": "32px", "fontWeight": 800, "color": color}),
        html.Div(label, style={"color": C["muted"], "fontSize": "13px", "marginTop": "4px"}),
    ])


def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        font_color=C["text"], margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(gridcolor=C["border"]), yaxis=dict(gridcolor=C["border"]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("stored-data", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def store_data(contents, filename):
    if contents is None:
        return None, ""
    df, err = parse_upload(contents, filename)
    if err:
        return None, html.Span(f"❌ Error: {err}", style={"color": C["red"]})
    msg = html.Span(
        f"✅  Loaded {len(df)} rows × {len(df.columns)} columns from {filename}",
        style={"color": C["green"]},
    )
    return df.to_json(date_format="iso", orient="split"), msg


@app.callback(
    Output("kpi-row", "children"),
    Input("stored-data", "data"),
)
def update_kpis(data):
    if not data:
        return ""
    df = pd.read_json(io.StringIO(data), orient="split")
    n = len(df)
    surv = f"{(df[TARGET_SURVIVAL].mean()*100):.0f}%" if TARGET_SURVIVAL in df.columns else "—"
    recu = f"{(df[TARGET_RECURRENCE].mean()*100):.0f}%" if TARGET_RECURRENCE in df.columns else "—"
    age_m = f"{df['age'].median():.0f}" if "age" in df.columns else "—"
    return html.Div(
        style={"display": "flex", "marginBottom": "4px"},
        children=[
            kpi_card("Total Patients", n,   C["teal"]),
            kpi_card("Survival Rate", surv, C["green"]),
            kpi_card("Recurrence Rate", recu, C["red"]),
            kpi_card("Median Age", age_m, C["amber"]),
        ]
    )


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("stored-data", "data"),
)
def render_tab(tab, data):
    if not data:
        return html.Div("⬆  Upload a patient file to begin.",
                        style={"color": C["muted"], "textAlign": "center", "marginTop": "60px"})
    df = pd.read_json(io.StringIO(data), orient="split")

    # ── Overview ─────────────────────────────────────────────────────────────
    if tab == "tab-overview":
        figs = []

        # survival bar
        if TARGET_SURVIVAL in df.columns:
            counts = df[TARGET_SURVIVAL].value_counts().rename({0: "Deceased", 1: "Alive"})
            fig1 = go.Figure(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[C["red"], C["green"]],
            ))
            fig1.update_layout(title="Survival Status", xaxis_title="", yaxis_title="Patients")
            figs.append(dcc.Graph(figure=dark_fig(fig1), style={"flex": "1"}))

        # age dist
        if "age" in df.columns:
            fig2 = go.Figure(go.Histogram(
                x=df["age"], nbinsx=20, marker_color=C["teal"],
            ))
            fig2.update_layout(title="Age Distribution", xaxis_title="Age", yaxis_title="Count")
            figs.append(dcc.Graph(figure=dark_fig(fig2), style={"flex": "1"}))

        # stage distribution
        stage_col = next((c for c in ["overall_stage", "t_stage", "stage"] if c in df.columns), None)
        if stage_col:
            vc = df[stage_col].value_counts().sort_index()
            fig3 = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=C["amber"]))
            fig3.update_layout(title=f"{stage_col.replace('_', ' ').title()} Distribution")
            figs.append(dcc.Graph(figure=dark_fig(fig3), style={"flex": "1"}))

        # recurrence donut
        if TARGET_RECURRENCE in df.columns:
            rc = df[TARGET_RECURRENCE].value_counts().rename({0: "No Recurrence", 1: "Recurred"})
            fig4 = go.Figure(go.Pie(
                labels=rc.index, values=rc.values,
                hole=0.5, marker_colors=[C["green"], C["red"]],
            ))
            fig4.update_layout(title="Recurrence")
            figs.append(dcc.Graph(figure=dark_fig(fig4), style={"flex": "1"}))

        rows = []
        for i in range(0, len(figs), 2):
            rows.append(html.Div(figs[i:i+2], style={"display": "flex", "gap": "16px"}))
        return html.Div(rows) if rows else html.Div("No plottable columns found.",
                                                     style={"color": C["muted"]})

    # ── Patient Table ─────────────────────────────────────────────────────────
    if tab == "tab-table":
        return html.Div(style=CARD, children=[
            dash_table.DataTable(
                data=df.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=15,
                style_table={"overflowX": "auto"},
                style_header={"background": C["teal_dim"], "color": C["white"],
                               "fontWeight": 700, "border": "none"},
                style_cell={"background": C["panel"], "color": C["text"],
                             "border": f"1px solid {C['border']}", "fontSize": "12px",
                             "padding": "8px", "textAlign": "left"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": C["bg"]},
                ],
                filter_action="native",
                sort_action="native",
            )
        ])

    # ── Predictions ───────────────────────────────────────────────────────────
    if tab == "tab-pred":
        cards = []
        for target, label, color in [
            (TARGET_SURVIVAL, "Survival", C["green"]),
            (TARGET_RECURRENCE, "Recurrence", C["red"]),
        ]:
            acc, auc, cm, _ = run_models(df, target)
            if acc is None:
                cards.append(html.Div(
                    f"Column '{target}' not found in data.",
                    style={"color": C["muted"], "marginBottom": "12px"},
                ))
                continue

            # confusion matrix heatmap
            cm_fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                colorscale=[[0, C["bg"]], [1, color]],
                text=cm, texttemplate="%{text}",
            ))
            cm_fig.update_layout(title=f"{label} — Confusion Matrix")

            cards.append(html.Div(style={**CARD, "borderLeft": f"3px solid {color}"}, children=[
                html.H5(f"{label} Prediction", style={"color": color, "fontWeight": 700}),
                html.Div(style={"display": "flex", "gap": "30px", "marginBottom": "14px"}, children=[
                    html.Div([
                        html.Div(f"{acc*100:.1f}%", style={"fontSize": "28px", "fontWeight": 800, "color": color}),
                        html.Div("Accuracy (GB)", style={"color": C["muted"], "fontSize": "12px"}),
                    ]),
                    html.Div([
                        html.Div(f"{auc:.3f}" if auc else "—",
                                 style={"fontSize": "28px", "fontWeight": 800, "color": color}),
                        html.Div("ROC-AUC", style={"color": C["muted"], "fontSize": "12px"}),
                    ]),
                ]),
                dcc.Graph(figure=dark_fig(cm_fig)),
            ]))

        return html.Div(cards)

    # ── Feature Insights ──────────────────────────────────────────────────────
    if tab == "tab-feat":
        plots = []
        for target, label, color in [
            (TARGET_SURVIVAL, "Survival", C["green"]),
            (TARGET_RECURRENCE, "Recurrence", C["red"]),
        ]:
            _, _, _, feat_imp = run_models(df, target)
            if feat_imp is None:
                continue
            fig = go.Figure(go.Bar(
                x=feat_imp.values[::-1], y=feat_imp.index[::-1],
                orientation="h", marker_color=color,
            ))
            fig.update_layout(title=f"Top Features — {label}", xaxis_title="Importance",
                               height=420)
            plots.append(dcc.Graph(figure=dark_fig(fig)))

        # correlation heatmap (numeric cols)
        num_df = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
        if len(num_df.columns) >= 4:
            corr = num_df.corr().round(2)
            heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdBu", zmid=0, text=corr.values,
                texttemplate="%{text:.1f}",
            ))
            heat.update_layout(title="Feature Correlation Matrix", height=500)
            plots.append(dcc.Graph(figure=dark_fig(heat)))

        return html.Div(plots) if plots else html.Div("No target columns found.",
                                                        style={"color": C["muted"]})

    return html.Div()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
