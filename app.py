import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import base64
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from uncertainty_utils import EPS
from uncertainty_transformer import UncertaintyTransformer  # required for unpickling

# Suppress scikit-learn version warnings for unpickling
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- CONSTANTS ---
# Renk Paleti
COLOR_ORANGE = "#F39C12"  # Grup 2 (AKS)
COLOR_BLUE = "#3498DB"  # Grup 1 (Miyokardit) & Ana Tema
COLOR_BACKGROUND = "#FFFFFF" 
COLOR_TEXT = "#333333" 
COLOR_NEW_PATIENT = "#E74C3C" 

# Artifact Dosya Yolları (absolute, Streamlit Cloud uyumlu)
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "best_model_finetuned.pkl")
METADATA_PATH = os.path.join(_BASE, "model_metadata.pkl")
TEST_METRICS_PATH = os.path.join(_BASE, "final_test_metrics.json")
EMBEDDING_PATH = os.path.join(_BASE, "app_artifacts", "embedding_data.npz")
LANDSCAPE_PATH = os.path.join(_BASE, "app_artifacts", "diagnostic_landscape.png")
TSNE_MODEL_PATH = os.path.join(_BASE, "app_artifacts", "tsne_model.pkl")
TSNE_SCALER_PATH = os.path.join(_BASE, "app_artifacts", "tsne_scaler.pkl")

# --- MODEL SABİTLERİ ---
G1, G2 = 1, 2 # Sadece iki grubumuz var
# EPS imported from uncertainty_utils (1e-12) for consistency 
IMPUTATION_THRESHOLD = 0 # Hiç eksik alan kabul edilmez - tüm feature'lar zorunlu

# --- DİL (ÇEVİRİ) SÖZLÜĞÜ (GÜNCELLENDİ) ---
LANG_STRINGS = {
    "app_title": {
        "ENG": "Clinical Uncertainty Positioning",
        "TR": "Klinik Belirsizlik Konumlandırma"
    },
    "main_title": {
        "ENG": "Clinical Uncertainty Positioning Tool (ACS vs Myocarditis)",
        "TR": "Klinik Belirsizlik Konumlandırma Aracı (AKS vs Miyokardit)"
    },
    "header_input": {
        "ENG": "New Patient Input",
        "TR": "Yeni Hasta Girişi"
    },
    "info_note": {
        "ENG": "NOTE: All fields are required. Please fill in all patient data.",
        "TR": "NOT: Tüm alanlar zorunludur. Lütfen tüm hasta verilerini doldurun."
    },
    "key_features": {"ENG": "Key Features", "TR": "Anahtar Özellikler"},
    "symptoms": {"ENG": "Symptoms & Presentation", "TR": "Semptomlar ve Başvuru"},
    "history": {"ENG": "Patient History & Comorbidities", "TR": "Hasta Geçmişi ve Komorbiditeler"},
    "labs": {"ENG": "Lab Results", "TR": "Laboratuvar Sonuçları"},
    "other_features": {"ENG": "Other Features", "TR": "Diğer Özellikler"},
    "calculate_button": {"ENG": "Calculate Patient Position", "TR": "Hasta Pozisyonunu Hesapla"},
    "header_output": {"ENG": "Patient Position Analysis", "TR": "Hasta Pozisyon Analizi"},
    "error_missing_max": {
        "ENG": f"Error: {{num_missing}} fields are missing. Max allowed is {IMPUTATION_THRESHOLD}.",
        "TR": f"Hata: {{num_missing}} alan eksik. İzin verilen en fazla eksik alan sayısı {IMPUTATION_THRESHOLD}."
    },
    "error_missing_fields": {"ENG": "Missing Fields:", "TR": "Eksik Alanlar:"},
    "missing_and_more": {
        "ENG": "...and {count} more field(s) are missing.",
        "TR": "...ve {count} alan daha eksik."
    },
    "warning_all_required": {
        "ENG": "Please complete all missing fields before running the analysis. This tool requires complete patient data to produce a reliable clinical result.",
        "TR": "Analizi çalıştırmadan önce lütfen eksik alanların tamamını doldurun. Bu araç, güvenilir bir klinik sonuç üretebilmek için hastaya ait tüm verilerin eksiksiz girilmesini gerektirmektedir."
    },
    "warn_imputed": {
        "ENG": f"**Warning:** {{num_missing}} feature(s) were missing and have been imputed using the population average (mean/mode). Results may be less accurate.",
        "TR": f"**Uyarı:** {{num_missing}} özellik eksikti ve popülasyon ortalaması (mean/mode) kullanılarak otomatik dolduruldu. Sonuçlar daha az güvenilir olabilir."
    },
    "view_imputed": {"ENG": "View Imputed Features", "TR": "Doldurulan Özellikleri Gör"},
    "imputed_as": {"ENG": "imputed as", "TR": "olarak dolduruldu"},
    "critical_error_impute": {
        "ENG": "Critical Error: No imputation value found for {feature}. Calculation cannot proceed.",
        "TR": "Kritik Hata: {feature} için doldurma değeri bulunamadı. Hesaplama devam edemez."
    },
    "success_all_data": {
        "ENG": "All data fields were provided. No imputation was needed.",
        "TR": "Tüm veri alanları sağlandı. Otomatik doldurma gerekmedi."
    },
    "plot_title_tsne": {
        "ENG": "Diagnostic Landscape (ACS vs Myocarditis)",
        "TR": "Tanısal Manzara (AKS vs Miyokardit)"
    },
    "plot_title_bar": {"ENG": "Feature-based Uncertainty", "TR": "Özellik Bazlı Belirsizlik"},
    "plot_top20": {
        "ENG": "Top 20 features contributing to uncertainty:",
        "TR": "Belirsizliğe katkıda bulunan ilk 20 özellik:"
    },
    "plot_title_global_unc": {"ENG": "Features Contributing to Certainty", "TR": "Belirliğe Katkıda Bulunan Özellikler"},
    "plot_top20_global": {
        "ENG": "Top 20 features with the lowest mean absolute uncertainty score across all training patients:",
        "TR": "Tüm eğitim hastalarında en düşük ortalama mutlak belirsizlik skoruna sahip 20 özellik:"
    },
    "plot_title_patient_unc": {"ENG": "Features Contributing to Certainty (This Patient)", "TR": "Belirliğe Katkıda Bulunan Özellikler (Bu Hasta)"},
    "plot_top20_patient": {
        "ENG": "Top 20 features with the lowest uncertainty score for this patient:",
        "TR": "Bu hastada en düşük belirsizlik skoruna sahip 20 özellik:"
    },
    "legend_g1": {"ENG": "Grup 1 (Myocarditis)", "TR": "Grup 1 (Miyokardit)"},
    "legend_g2": {"ENG": "Grup 2 (ACS)", "TR": "Grup 2 (AKS)"},
    "legend_new": {"ENG": "New Patient", "TR": "Yeni Hasta"},
    "legend_title": {"ENG": "Diagnosis", "TR": "Tanı"},
    "bar_chart_title": {"ENG": "Patient's Uncertainty Vector", "TR": "Hastanın Belirsizlik Vektörü"},
    "bar_xaxis": {"ENG": "Uncertainty Score", "TR": "Belirsizlik Skoru"},
    "bar_yaxis": {"ENG": "Feature", "TR": "Özellik"},
    "about_banner": {
        "ENG": "This tool positions a new patient on the **ACS vs. Myocarditis** diagnostic landscape based on their clinical features. Enter patient data on the left and click **Calculate**.",
        "TR": "Bu araç, klinik özellikler temelinde yeni bir hastayı **AKS ve Miyokardit** tanısal haritası üzerinde konumlandırır. Sol taraftaki forma hasta verilerini girin ve **Hesapla**'ya tıklayın."
    },
    "welcome_header": {"ENG": "About this Tool", "TR": "Araç Hakkında"},
    "welcome_info": {
        "ENG": "Please enter patient data on the left and click 'Calculate' to begin analysis.",
        "TR": "Lütfen soldaki alana hasta verilerini girin ve analize başlamak için 'Hesapla' butonuna tıklayın."
    },
    "welcome_text": {
        "ENG": """
            This tool positions a new patient within the diagnostic landscape
            of **Grup 2 (ACS) vs. Grup 1 (Myocarditis)**
            based on their clinical features.
            
            1.  **Enter** the patient's data in the form on the left.
            2.  **Click** 'Calculate' to see their position.
            3.  **Analyze** their position on the t-SNE map.
        """,
        "TR": """
            Bu araç, yeni bir hastayı klinik özelliklerine göre
            **Grup 2 (AKS) ve Grup 1 (Miyokardit)**
            tanısal manzarası üzerinde konumlandırır.
            
            1.  Sol taraftaki forma hasta verilerini **girin**.
            2.  Hastanın pozisyonunu görmek için 'Hesapla' butonuna **tıklayın**.
            3.  t-SNE haritası üzerindeki konumunu **analiz edin**.
        """
    },
    "load_error": {
        "ENG": "Application failed to load. Please check the terminal for errors.",
        "TR": "Uygulama yüklenemedi. Lütfen terminali kontrol edin."
    },
    "placeholder_number": {
        "ENG": "Enter value...",
        "TR": "Değer girin..."
    },
    "placeholder_select": {
        "ENG": "Select an option...",
        "TR": "Bir seçenek seçin..."
    },
    "demo_selector_label": {
        "ENG": "Quick Demo Patient",
        "TR": "Hızlı Demo Hasta"
    },
    "landscape_view_label": {"ENG": "Show", "TR": "Göster"},
    "view_all": {"ENG": "All Patients", "TR": "Tüm Hastalar"},
    "view_myo": {"ENG": "Myocarditis Only", "TR": "Sadece Miyokardit"},
    "view_acs": {"ENG": "ACS Only", "TR": "Sadece AKS"},
    "prediction_header": {
        "ENG": "Model Prediction",
        "TR": "Modelin Tahmini"
    },
    "prediction_text_myo": {
        "ENG": "{prob:.0f}% probability of Myocarditis",
        "TR": "%{prob:.0f} olasılıkla Miyokardit"
    },
    "prediction_text_acs": {
        "ENG": "{prob:.0f}% probability of ACS",
        "TR": "%{prob:.0f} olasılıkla AKS"
    },
    "risk_factors_note": {
        "ENG": (
            "**Major Risk Factors**  \n"
            "🔵 **ACS:** Diabetes mellitus (DM), hypertension (HT), smoking, hyperlipidemia (HL)  \n"
            "🔴 **Myocarditis:** Recent upper respiratory tract infection (URI)"
        ),
        "TR": (
            "**Major Risk Faktörleri**  \n"
            "🔵 **AKS:** Diyabet (DM), hipertansiyon (HT), sigara, hiperlipidemi (HL)  \n"
            "🔴 **Miyokardit:** Yakın dönemde geçirilmiş üst solunum yolu enfeksiyonu (ÜSYE)"
        )
    }
}

# Çeviri (Translation) için yardımcı fonksiyon
def T(key):
    """Geçerli dile göre çevrilmiş metni alır."""
    lang = st.session_state.language
    return LANG_STRINGS.get(key, {}).get(lang, f"MISSING_KEY: {key}")

# --- ARTIFACT YÜKLEME ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Tüm artifact'ları diskten yükler ve cache'ler."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        _raw = np.load(EMBEDDING_PATH)
        embedding_data = {k: np.array(_raw[k]) for k in _raw.files}
        _raw.close()

        test_metrics = {}
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH) as f:
                test_metrics = json.load(f)

        tsne_model = None
        if os.path.exists(TSNE_MODEL_PATH):
            with open(TSNE_MODEL_PATH, "rb") as f:
                tsne_model = pickle.load(f)

        # tsne_scaler: the StandardScaler fitted on the sorted uncertainty matrix
        # in NEW_uncertainty.ipynb — must be used for kNN positioning because
        # embedding_data['X_std'] was produced by this exact scaler.
        with open(TSNE_SCALER_PATH, "rb") as f:
            tsne_scaler = pickle.load(f)

        feature_list = list(metadata["features"])
        return model, metadata, embedding_data, test_metrics, feature_list, tsne_model, tsne_scaler
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.error("Please ensure best_model_finetuned.pkl, model_metadata.pkl, and app_artifacts/embedding_data.npz exist.")
        return None

@st.cache_data
def load_demo_patients():
    """demo_patients.txt dosyasından demo hasta verilerini yükler."""
    path = os.path.join(_BASE, "demo_patients.txt")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("patients", raw) if isinstance(raw, dict) else raw
    except Exception:
        return []

# --- TAHMİNLEME FONKSİYONLARI ---
def find_tsne_position(x_new_std, X_std_train, X_emb_train, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean'); nn.fit(X_std_train)
    distances, indices = nn.kneighbors(x_new_std)
    neighbor_coords_2d = X_emb_train[indices.flatten()]
    new_position = np.mean(neighbor_coords_2d, axis=0)
    return new_position[0], new_position[1]


@st.cache_data(show_spinner="Computing diagnostic landscape...")
def _compute_landscape_layers():
    """
    Computes KDE-based RGBA colour layers for the diagnostic landscape.
    Identical algorithm to NEW_uncertainty.ipynb Cells 23-27.
    Cached permanently — only runs once per Streamlit session.
    """
    _raw = np.load(EMBEDDING_PATH)
    X_emb = np.array(_raw['X_emb'])
    y     = np.array(_raw['y'])
    _raw.close()

    pad = 2.0
    xmin = float(X_emb[:, 0].min()) - pad
    xmax = float(X_emb[:, 0].max()) + pad
    ymin = float(X_emb[:, 1].min()) - pad
    ymax = float(X_emb[:, 1].max()) + pad

    resolution = 600
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    class1 = X_emb[y == 1]
    class2 = X_emb[y == 2]

    kde1 = gaussian_kde(class1.T, bw_method="scott")
    kde2 = gaussian_kde(class2.T, bw_method="scott")
    z1 = kde1(grid).reshape(xx.shape)
    z2 = kde2(grid).reshape(xx.shape)

    q = 0.6
    level1 = np.quantile(z1, q)
    level2 = np.quantile(z2, q)

    def normalise(z, clip=0.98):
        zmax = np.quantile(z, clip)
        return np.clip(z / zmax, 0, 1)

    alpha1 = normalise(z1) ** 0.5
    alpha2 = normalise(z2) ** 0.5
    alpha1[z1 < level1] = 0.0
    alpha2[z2 < level2] = 0.0

    overlap_mask  = (alpha1 > 0) & (alpha2 > 0)
    alpha_overlap = np.maximum(alpha1, alpha2)
    alpha_overlap[~overlap_mask] = 0.0
    alpha1[overlap_mask] = 0.0
    alpha2[overlap_mask] = 0.0

    eps = 1e-12
    total = z1 + z2 + eps
    t = (z1 - z2) / total
    shift_strength = 0.3

    R = np.full_like(t, 0.5); G = np.full_like(t, 0.5); B = np.full_like(t, 0.5)
    pos = t > 0
    R[pos] += shift_strength * t[pos]; G[pos] -= shift_strength * t[pos]; B[pos] -= shift_strength * t[pos]
    neg = t < 0
    B[neg] += shift_strength * (-t[neg]); R[neg] -= shift_strength * (-t[neg]); G[neg] -= shift_strength * (-t[neg])

    shape = (*alpha1.shape, 4)
    red_img  = np.zeros(shape); red_img[..., 0]  = 1.0; red_img[..., 3]  = alpha1
    blue_img = np.zeros(shape); blue_img[..., 2] = 1.0; blue_img[..., 3] = alpha2

    shape2 = (*alpha_overlap.shape, 4)
    over_img = np.zeros(shape2)
    over_img[..., 0] = R; over_img[..., 1] = G; over_img[..., 2] = B
    over_img[..., 3] = alpha_overlap

    return red_img, blue_img, over_img, xmin, xmax, ymin, ymax


@st.cache_data
def render_landscape_with_patient(star_x, star_y, legend_g1, legend_g2, legend_new):
    """
    Renders the KDE diagnostic landscape + new-patient star in one matplotlib figure.
    Identical coordinate system to the notebook (origin='lower', same extents).
    Returns PNG bytes for st.image().
    """
    red_img, blue_img, over_img, xmin, xmax, ymin, ymax = _compute_landscape_layers()

    fig, ax = plt.subplots(figsize=(7, 6.5))

    for layer in [over_img, red_img, blue_img]:
        ax.imshow(layer, extent=(xmin, xmax, ymin, ymax),
                  origin="lower", interpolation="bilinear")

    ax.scatter([star_x], [star_y], marker='*', s=500, c='limegreen',
               edgecolors='darkgreen', linewidths=1.5, zorder=5)

    legend_handles = [
        Patch(color='red',       label=legend_g1),
        Patch(color='blue',      label=legend_g2),
        Patch(color='lightgray', label='Uncertain'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='limegreen', markeredgecolor='darkgreen',
               markersize=14, label=legend_new),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#cccccc')

    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data
def render_landscape_filtered(star_x, star_y, filter_group, legend_label, legend_new):
    """
    Renders the KDE landscape for a single group (filter_group=1 or 2).
    Uses the same coordinate extents as the full landscape so the patient
    star stays in the correct position.
    star_x / star_y can be None when no patient has been calculated yet.
    """
    _raw = np.load(EMBEDDING_PATH)
    X_emb = np.array(_raw['X_emb'])
    y     = np.array(_raw['y'])
    _raw.close()

    # Keep full extents so patient position is consistent across views
    pad = 2.0
    xmin = float(X_emb[:, 0].min()) - pad
    xmax = float(X_emb[:, 0].max()) + pad
    ymin = float(X_emb[:, 1].min()) - pad
    ymax = float(X_emb[:, 1].max()) + pad

    X_g = X_emb[y == filter_group]

    resolution = 600
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    kde = gaussian_kde(X_g.T, bw_method="scott")
    z   = kde(grid).reshape(xx.shape)

    level = np.quantile(z, 0.6)

    def normalise(z, clip=0.98):
        zmax = np.quantile(z, clip)
        return np.clip(z / zmax, 0, 1)

    alpha = normalise(z) ** 0.5
    alpha[z < level] = 0.0

    shape = (*alpha.shape, 4)
    img = np.zeros(shape)
    if filter_group == 1:          # Miyokardit → kırmızı
        img[..., 0] = 1.0
        patch_color = 'red'
    else:                          # AKS → mavi
        img[..., 2] = 1.0
        patch_color = 'blue'
    img[..., 3] = alpha

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # White background spanning full extent — prevents bbox_inches='tight'
    # from cropping transparent (zero-density) regions where the patient
    # star might be located.
    ax.imshow(np.ones((10, 10, 3), dtype=np.float32),
              extent=(xmin, xmax, ymin, ymax),
              origin="lower", aspect='auto', zorder=0)

    ax.imshow(img, extent=(xmin, xmax, ymin, ymax),
              origin="lower", interpolation="bilinear", zorder=1)

    if star_x is not None and star_y is not None:
        ax.scatter([star_x], [star_y], marker='*', s=500, c='limegreen',
                   edgecolors='darkgreen', linewidths=1.5, zorder=5)

    legend_handles = [Patch(color=patch_color, label=legend_label)]
    if star_x is not None:
        legend_handles.append(
            Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='limegreen', markeredgecolor='darkgreen',
                   markersize=14, label=legend_new)
        )
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#cccccc')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --- PLOT/GRAFİK FONKSİYONLARI (ÇEVİRİLİ) ---

@st.cache_data
def _load_landscape_as_b64(path):
    """Notebook'tan kaydedilen PNG'yi base64'e çevirir (cache'li)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def plot_diagnostic_landscape(X_emb_train, y_train, lang, new_patient_coords=None):
    """
    Plotly interaktif t-SNE grafiği.
    Arka plan: notebook'tan kaydedilen diagnostic_landscape.png (birebir aynı görsel).
    Yeni hasta varsa yeşil yıldız olarak plotly scatter trace ile eklenir.
    """
    pad = 2.0
    xmin = float(X_emb_train[:, 0].min()) - pad
    xmax = float(X_emb_train[:, 0].max()) + pad
    ymin = float(X_emb_train[:, 1].min()) - pad
    ymax = float(X_emb_train[:, 1].max()) + pad

    b64 = _load_landscape_as_b64(LANDSCAPE_PATH)

    fig = go.Figure()

    # Eksen aralığını sabitleyen görünmez iz
    fig.add_trace(go.Scatter(
        x=[xmin, xmax], y=[ymin, ymax],
        mode="markers", marker=dict(opacity=0, size=0.1),
        showlegend=False, hoverinfo="skip"
    ))

    # Notebook PNG'si arka plan olarak
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{b64}",
        xref="x", yref="y",
        x=xmin, y=ymax,
        sizex=xmax - xmin, sizey=ymax - ymin,
        sizing="stretch",
        layer="below",
        xanchor="left", yanchor="top",
    ))

    # Legend proxy izleri
    for color, label in [
        ("red",  T("legend_g1")),
        ("blue", T("legend_g2")),
        ("gray", "Uncertain"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=12, color=color),
            name=label, showlegend=True
        ))

    # Yeni hasta yıldızı
    if new_patient_coords is not None:
        fig.add_trace(go.Scatter(
            x=[new_patient_coords[0]], y=[new_patient_coords[1]],
            mode="markers",
            marker=dict(symbol="star", size=20, color="limegreen",
                        line=dict(color="darkgreen", width=2)),
            name=T("legend_new"), showlegend=True
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[xmin, xmax]),
        yaxis=dict(visible=False, range=[ymin, ymax], scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        height=450,
    )
    return fig

def plot_uncertainty_vector(x_new_vec_df, lang):
    """Çubuk grafiği çizer."""
    y_vals  = x_new_vec_df['Feature'].tolist()
    scores  = x_new_vec_df['Uncertainty Score'].tolist()
    max_val = max(scores) if scores else 1.0
    min_val = min(scores) if scores else 0.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=y_vals,
        orientation='h',
        marker=dict(color=COLOR_BLUE),
        showlegend=False,
    ))

    # Colored star annotations at the right end of each major-risk bar
    shown_acs, shown_myo = [], []
    for f, unc in zip(y_vals, scores):
        if f in ACS_RISK_FEATURES:
            color = COLOR_BLUE
            shown_acs.append(f)
        elif f in MYO_RISK_FEATURES:
            color = COLOR_NEW_PATIENT
            shown_myo.append(f)
        else:
            continue
        fig.add_annotation(
            x=unc, y=f,
            xref='x', yref='y',
            text=' ★',
            showarrow=False,
            font=dict(color=color, size=18),
            xanchor='left',
        )

    # Legend entries (only when that group appears in top-20)
    if shown_acs:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='star', size=14, color=COLOR_BLUE),
            name="★ AKS Major Risk Faktörü" if lang == 'TR' else "★ ACS Major Risk Factor",
            showlegend=True,
        ))
    if shown_myo:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='star', size=14, color=COLOR_NEW_PATIENT),
            name="★ Miyokardit Major Risk Faktörü" if lang == 'TR' else "★ Myocarditis Major Risk Factor",
            showlegend=True,
        ))

    x_margin = abs(max_val) * 0.15 + 0.01
    fig.update_layout(
        title=T("bar_chart_title"),
        xaxis_title=T("bar_xaxis"),
        yaxis_title=T("bar_yaxis"),
        plot_bgcolor=COLOR_BACKGROUND,
        paper_bgcolor=COLOR_BACKGROUND,
        font_color=COLOR_TEXT,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[min(0, min_val), max_val + x_margin]),
        height=max(400, len(x_new_vec_df) * 20),
    )
    return fig

# --- KATEGORİK & GRUP HARİTALARI ---
yes_no_map = {"No": 0, "Yes": 1}
categorical_map = {
    "SEX": {"Female": 0, "Male": 1},
    "Socioeconomic Status": {"Poor": 0, "Good": 1},
    "Chest Pain Character": {"None": 0, "Stabbing / Localized": 1, "Pressure / Anginal": 2},
    "Infection type": {"URTI": 1, "Diarrhea": 2, "Vaccine": 3, "Other": 4},
    "DM": yes_no_map, "HT": yes_no_map, "HL": yes_no_map, "FH": yes_no_map,
    "SIGARA": yes_no_map, "KBY": yes_no_map, "PRIOR_KAH": yes_no_map,
    "HIPOTIROIDI": yes_no_map, "Chest Pain": yes_no_map, "Radiation": yes_no_map,
    "Arm Pain": yes_no_map, "Back Pain": yes_no_map, "Epigastric Pain": yes_no_map,
    "Relation with exercise": yes_no_map, "Relation with Position": yes_no_map,
    "Dyspnea": yes_no_map, "Fatigue": yes_no_map, "Nausea": yes_no_map,
    "Çarpıntı": yes_no_map, "Recent Infection(4 hafta)": yes_no_map,
    "INHOSPITAL_EX": yes_no_map, "Segmentary Wall Motion Abnormality": yes_no_map,
    "Pericardial Effusion": yes_no_map, "ECG_ST depression": yes_no_map,
    "ECG_T neg": yes_no_map, "ECG_Q waves": yes_no_map, "MRI_T2": yes_no_map,
    "MRI_LGE": yes_no_map, "KANSER_KEMOTERAPI": yes_no_map,
    "TASI_BRADIKARDI": yes_no_map, "MADDE_ILAC_KULLANIMI": yes_no_map,
    "Alcohol": yes_no_map, "KOAH": yes_no_map, "PAH": yes_no_map,
    "HIPERTIROIDI": yes_no_map, "REYNAULD": yes_no_map,
}
ACS_RISK_FEATURES = {"DM", "HT", "SIGARA", "HL"}
MYO_RISK_FEATURES = {"Recent Infection(4 hafta)"}
MAJOR_RISK_FEATURES = ACS_RISK_FEATURES | MYO_RISK_FEATURES

KEY_FEATURES = [
    "AGE", "SEX",
    "Segmentary Wall Motion Abnormality", "MRI_LGE"
]
SYMPTOM_FEATURES = [
    "Chest Pain", "Chest Pain Character", "Chest Pain Duration(saat)", "Radiation", "Arm Pain",
    "Back Pain", "Epigastric Pain", "Relation with exercise",
    "Relation with Position", "Dyspnea", "Fatigue", "Nausea", "Çarpıntı",
    "Any Previous Pain Attacks"
]
HISTORY_FEATURES = [
    "DM", "HT", "HL", "FH", "SIGARA", "KBY", "PRIOR_KAH", "HIPOTIROIDI",
    "Socioeconomic Status", "Recent Infection(4 hafta)", "Infection type",
    "KANSER_KEMOTERAPI", "TASI_BRADIKARDI", "MADDE_ILAC_KULLANIMI",
    "Alcohol", "KOAH", "PAH", "HIPERTIROIDI", "REYNAULD"
]
LAB_ECG_FEATURES = [
    "Troponin_Sonucu", "TROPONIN_CUTOFF_DEGERI", "TROP_KATSAYISI", "CK-MB",
    "GLUKOZ", "WBCpik", "NEUpik", "LYMPpik", "EOSpik", "MONOpik", "HB",
    "HTC", "PLT", "KREATIN", "AST", "ALT", "ALBUMIN", "TOTAL_KOLESTEROL",
    "TG", "LDL", "HDL", "hs-CRP", "hs-CRP_CUTOFF", "hs-CRP_FOLD",
    "Sedimantation", "BNP", "HBAB1C", "D-DIMER", "EF", "Pericardial Effusion",
    "ECG_ST depression", "ECG_Location of ST depression ", "Level of ST-Dep_mm",
    "ECG_T neg", "ECG_Location of T negativity", "Level of T invertion_mm",
    "ECG_Q waves", "MRI_T2", "INHOSPITAL_EX", "EX_TARIHI", "BMI"
]

# Klinik olarak tam sayı rapor edilen numerik özellikler
_INTEGER_FEATURES = {
    "AGE", "Any Previous Pain Attacks",
    "CK-MB", "GLUKOZ", "PLT",
    "AST", "ALT", "TOTAL_KOLESTEROL", "TG", "LDL", "HDL",
}

# --- ANINDA DOĞRULAMA YAPAN WIDGET FONKSİYONU ---
def render_feature_widget(feature, data_dict, prefill=None, key_suffix=""):
    # key_suffix, demo hasta değiştiğinde widget'ların sıfırlanmasını sağlar
    wkey = f"w__{feature}__{key_suffix}"
    if feature in categorical_map:
        options_map = categorical_map[feature]
        options_list = list(options_map.keys())
        default_index = None
        if prefill is not None:
            for i, val in enumerate(options_map.values()):
                if val == prefill:
                    default_index = i
                    break
        selected_option = st.selectbox(
            label=feature,
            options=options_list,
            index=default_index,
            placeholder=T("placeholder_select"),
            key=wkey,
        )
        data_dict[feature] = options_map[selected_option] if selected_option is not None else None
    elif feature in _INTEGER_FEATURES:
        min_val = 0
        max_val = 120 if feature == "AGE" else None
        data_dict[feature] = st.number_input(
            feature,
            value=int(prefill) if prefill is not None else None,
            placeholder=T("placeholder_number"),
            format="%d",
            step=1,
            min_value=min_val,
            max_value=max_val,
            key=wkey,
        )
    else:
        data_dict[feature] = st.number_input(
            feature,
            value=float(prefill) if prefill is not None else None,
            placeholder=T("placeholder_number"),
            format="%.2f",
            step=0.01,
            min_value=0.0,
            key=wkey,
        )

# --- ANA UYGULAMA MANTIĞI ---
st.set_page_config(
    page_title=LANG_STRINGS["app_title"]["ENG"],
    layout="wide"
)

# --- Dil Durumunu Başlat ---
if 'language' not in st.session_state:
    st.session_state.language = 'TR' # DEĞİŞİKLİK: Varsayılan dil Türkçe

lang = st.session_state.language

# Artifact'ları yükle
artifacts = load_artifacts()

if artifacts is not None:
    model, metadata, embedding_data, test_metrics, feature_list, tsne_model, tsne_scaler = artifacts
    
    # --- BASİT BAŞLIK VE DİL SEÇİMİ (APP BAR İPTAL EDİLDİ) ---
    st.title(T("main_title"))
    st.warning(T("risk_factors_note"))

    demo_patients = load_demo_patients()
    _NO_SEL = "───"
    _demo_options = [_NO_SEL] + [p["name"] for p in demo_patients]

    _hdr_lang, _hdr_demo = st.columns([1, 2])
    with _hdr_lang:
        st.radio(
            "Language / Dil",
            options=['TR', 'ENG'],
            index=0 if lang == 'TR' else 1,
            key="language",
            horizontal=True,
        )
    with _hdr_demo:
        if demo_patients:
            _selected_demo = st.selectbox(
                T("demo_selector_label"),
                options=_demo_options,
                index=0,
                key="demo_patient_key",
            )
            _demo_idx = _demo_options.index(_selected_demo) - 1
            prefill_data = demo_patients[_demo_idx]["data"] if _demo_idx >= 0 else {}
        else:
            prefill_data = {}

    st.info(T("about_banner"))
    st.divider()

    # --- İki sütunlu düzen: solda tanısal manzara, sağda belirsizlik grafikleri; form altta ---
    top_col1, top_col2 = st.columns([1, 1])

    st.divider()
    st.subheader(T("header_input"))
    st.info(T("info_note"))

    with st.form("patient_form"):
        patient_data = {}
        processed_features = set()

        with st.expander(T("key_features"), expanded=False):
            for feature in KEY_FEATURES:
                if feature in feature_list:
                    render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                    processed_features.add(feature)
        with st.expander(T("symptoms"), expanded=False):
            for feature in SYMPTOM_FEATURES:
                if feature in feature_list:
                    render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                    processed_features.add(feature)
        with st.expander(T("history"), expanded=False):
            for feature in HISTORY_FEATURES:
                if feature in feature_list:
                    render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                    processed_features.add(feature)
        with st.expander(T("labs"), expanded=False):
            for feature in LAB_ECG_FEATURES:
                if feature in feature_list:
                    render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                    processed_features.add(feature)

        other_features = [f for f in feature_list if f not in processed_features]
        if other_features:
            with st.expander(T("other_features"), expanded=False):
                for feature in other_features:
                    render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))

        submit_button = st.form_submit_button(
            T("calculate_button"),
            type="primary"
        )

    if submit_button:
        missing_features = [feature for feature, value in patient_data.items() if value is None]
        num_missing = len(missing_features)

        if num_missing > 0:
            with top_col1:
                st.error(T("error_missing_max").format(num_missing=num_missing))
                st.write(f"**{T('error_missing_fields')}**")
                for f in missing_features[:10]: st.write(f"- {f}")
                if len(missing_features) > 10:
                    st.write(T("missing_and_more").format(count=len(missing_features) - 10))
                st.warning(T("warning_all_required"))

        else:
            X_new_df = pd.DataFrame([patient_data])[feature_list]
            x_new_unc = model.named_steps['uncertainty'].transform(X_new_df)
            x_new_vec_raw = x_new_unc[0]

            _unc_feats  = list(model.named_steps['uncertainty'].feature_names_in_)
            _tsne_feats = list(tsne_scaler.feature_names_in_)
            x_new_unc_df     = pd.DataFrame(x_new_unc, columns=_unc_feats)
            x_new_unc_sorted = x_new_unc_df[_tsne_feats].values
            x_new_std        = tsne_scaler.transform(x_new_unc_sorted)
            new_coords_xy = find_tsne_position(x_new_std, embedding_data['X_std'], embedding_data['X_emb'], k=5)
            x_new_vec_df = pd.DataFrame({"Feature": feature_list, "Uncertainty Score": x_new_vec_raw}).sort_values(by="Uncertainty Score", ascending=False).head(20)

            with top_col1:
                st.subheader(T("header_output"))
                st.success(T("success_all_data"))
                st.subheader(T("plot_title_tsne"))
                _view_options = [T("view_all"), T("view_myo"), T("view_acs")]
                _view_sel = st.radio(
                    T("landscape_view_label"),
                    _view_options,
                    horizontal=True,
                    key="landscape_view"
                )
                if _view_sel == T("view_myo"):
                    landscape_img = render_landscape_filtered(
                        new_coords_xy[0], new_coords_xy[1], 1,
                        T("legend_g1"), T("legend_new")
                    )
                elif _view_sel == T("view_acs"):
                    landscape_img = render_landscape_filtered(
                        new_coords_xy[0], new_coords_xy[1], 2,
                        T("legend_g2"), T("legend_new")
                    )
                else:
                    landscape_img = render_landscape_with_patient(
                        new_coords_xy[0], new_coords_xy[1],
                        T("legend_g1"), T("legend_g2"), T("legend_new")
                    )
                st.image(landscape_img, use_container_width=True)

            with top_col2:
                try:
                    proba = model.predict_proba(X_new_df)[0]
                    classes = list(model.classes_)
                    pred_idx = int(np.argmax(proba))
                    pred_class = classes[pred_idx]
                    pred_prob = proba[pred_idx] * 100
                    if pred_class == G1:
                        pred_text = T("prediction_text_myo").format(prob=pred_prob)
                    else:
                        pred_text = T("prediction_text_acs").format(prob=pred_prob)
                    st.metric(T("prediction_header"), pred_text)
                except Exception:
                    pass
                st.subheader(T("plot_title_bar"))
                st.write(T("plot_top20"))
                fig_bar = plot_uncertainty_vector(x_new_vec_df, lang)
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader(T("plot_title_patient_unc"))
                st.write(T("plot_top20_patient"))
                _patient_cert_df = pd.DataFrame({
                    "Feature": _tsne_feats,
                    "Uncertainty Score": x_new_unc_sorted[0]
                })
                _patient_cert_df = _patient_cert_df[_patient_cert_df["Uncertainty Score"] > 0] \
                    .sort_values("Uncertainty Score", ascending=True).head(20)
                _patient_cert_df["Label"] = _patient_cert_df["Feature"].apply(
                    lambda f: f"★ {f}" if f in MAJOR_RISK_FEATURES else f
                )
                fig_patient_cert = go.Figure()
                fig_patient_cert.add_trace(go.Bar(
                    x=_patient_cert_df["Uncertainty Score"],
                    y=_patient_cert_df["Label"],
                    orientation='h',
                    marker=dict(color=COLOR_BLUE)
                ))
                fig_patient_cert.update_layout(
                    xaxis_title=T("bar_xaxis"),
                    yaxis_title=T("bar_yaxis"),
                    plot_bgcolor=COLOR_BACKGROUND,
                    paper_bgcolor=COLOR_BACKGROUND,
                    font_color=COLOR_TEXT,
                    yaxis=dict(autorange="reversed"),
                    height=max(400, 20 * 20)
                )
                st.plotly_chart(fig_patient_cert, use_container_width=True)

    else:
        with top_col1:
            st.subheader(T("plot_title_tsne"))
            _view_options_w = [T("view_all"), T("view_myo"), T("view_acs")]
            _view_sel_w = st.radio(
                T("landscape_view_label"),
                _view_options_w,
                horizontal=True,
                key="landscape_view"
            )
            if _view_sel_w == T("view_myo"):
                st.image(render_landscape_filtered(None, None, 1, T("legend_g1"), T("legend_new")), use_container_width=True)
            elif _view_sel_w == T("view_acs"):
                st.image(render_landscape_filtered(None, None, 2, T("legend_g2"), T("legend_new")), use_container_width=True)
            elif os.path.exists(LANDSCAPE_PATH):
                with open(LANDSCAPE_PATH, "rb") as _f:
                    st.image(_f.read(), use_container_width=True)
            else:
                st.warning("Landscape PNG bulunamadı. NEW_uncertainty.ipynb'deki Cell 27'yi çalıştırın.")
                st.plotly_chart(
                    plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], lang),
                    use_container_width=True
                )

        with top_col2:
            st.subheader(T("plot_title_global_unc"))
            st.write(T("plot_top20_global"))
            _tsne_feats = list(tsne_scaler.feature_names_in_)
            _mean_raw = tsne_scaler.mean_
            _global_df = pd.DataFrame({"Feature": _tsne_feats, "Uncertainty Score": _mean_raw})
            _global_df = _global_df[_global_df["Uncertainty Score"] > 0] \
                .sort_values("Uncertainty Score", ascending=True).head(20)
            _global_df['Label'] = _global_df['Feature'].apply(
                lambda f: f"★ {f}" if f in MAJOR_RISK_FEATURES else f
            )
            fig_global = go.Figure()
            fig_global.add_trace(go.Bar(
                x=_global_df['Uncertainty Score'],
                y=_global_df['Label'],
                orientation='h',
                marker=dict(color=COLOR_BLUE)
            ))
            fig_global.update_layout(
                xaxis_title=T("bar_xaxis"),
                yaxis_title=T("bar_yaxis"),
                plot_bgcolor=COLOR_BACKGROUND,
                paper_bgcolor=COLOR_BACKGROUND,
                font_color=COLOR_TEXT,
                yaxis=dict(autorange="reversed"),
                height=max(400, 20 * 20)
            )
            st.plotly_chart(fig_global, use_container_width=True)

        with st.expander(T("welcome_header"), expanded=False):
            st.markdown(
                f'<div style="background-color:#FFF9C4;border-left:4px solid #F9A825;padding:12px 16px;border-radius:4px;">'
                f'{T("welcome_info")}</div>',
                unsafe_allow_html=True
            )
            st.markdown(T("welcome_text"), unsafe_allow_html=True)
else:
    # Artifact'lar yüklenemezse
    st.error(T("load_error"))