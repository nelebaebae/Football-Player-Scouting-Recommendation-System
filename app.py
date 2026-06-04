
import pickle
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Player Recommendation System | Heatmap Visualization",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ARTIFACT_PATH = Path("artifacts/player_recommender_artifacts.pkl")
DEFAULT_DATASET = "5liga"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f3f6fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        padding-bottom: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .topbar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        background:white;
        border:1px solid #d8dee9;
        box-shadow:0 2px 8px rgba(15,23,42,0.06);
        padding:18px 22px;
        margin-bottom:18px;
    }

    .topbar-title {
        font-size:1.35rem;
        font-weight:900;
        color:#172033;
        letter-spacing:-0.2px;
    }

    .topbar-nav {
        color:#24466f;
        font-size:1rem;
        font-weight:800;
    }

    .section-title {
        font-size:1.45rem;
        font-weight:900;
        color:#172033;
        margin-bottom:0.55rem;
    }

    .panel {
        background:white;
        border:1px solid #d8dee9;
        border-radius:18px;
        box-shadow:0 2px 10px rgba(15,23,42,0.06);
        padding:18px;
        margin-bottom:16px;
    }

    .player-card {
        background:#edf7f3;
        border-radius:14px;
        padding:14px 18px;
        display:flex;
        align-items:center;
        justify-content:center;
        gap:12px;
        min-height:82px;
    }

    .avatar {
        width:48px;
        height:48px;
        min-width:48px;
        border-radius:999px;
        background:linear-gradient(145deg,#ffffff,#dfe8f3);
        border:2px solid white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:900;
        color:#24466f;
        box-shadow:0 1px 8px rgba(15,23,42,0.12);
    }

    .card-label {
        color:#172033;
        font-weight:900;
        font-size:1rem;
        margin-bottom:1px;
    }

    .card-name {
        color:#24466f;
        font-weight:900;
        font-size:1.08rem;
    }

    .score-big {
        text-align:center;
        margin-top:8px;
        font-size:1.05rem;
        color:#172033;
        font-weight:600;
    }

    .score-big span {
        color:#2f6f46;
        font-size:1.25rem;
        font-weight:900;
    }

    .rec-header {
        display:grid;
        grid-template-columns: 42px 1fr 96px;
        gap:10px;
        color:#202a3a;
        font-weight:900;
        font-size:0.94rem;
        border-bottom:1px solid #e5e9f0;
        padding:8px 0 10px 0;
        margin-top:4px;
    }

    .rec-row {
        display:grid;
        grid-template-columns: 42px 1fr 96px;
        gap:10px;
        align-items:center;
        border-bottom:1px solid #edf1f6;
        padding:10px 0;
    }

    .rank {
        font-weight:900;
        color:#172033;
        text-align:center;
    }

    .player-line {
        display:flex;
        align-items:center;
        gap:10px;
        min-width:0;
    }

    .mini-avatar {
        width:39px;
        height:39px;
        min-width:39px;
        border-radius:999px;
        background:linear-gradient(145deg,#ffffff,#e0e8f3);
        border:1.5px solid white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:0.8rem;
        font-weight:900;
        color:#24466f;
        box-shadow:0 1px 5px rgba(15,23,42,0.12);
    }

    .player-name {
        font-weight:900;
        color:#172033;
        font-size:1rem;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }

    .player-sub {
        color:#64748b;
        font-size:0.78rem;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }

    .score-text {
        color:#2f6f46;
        font-weight:900;
        font-size:1.02rem;
        text-align:right;
    }

    .bar-bg {
        height:7px;
        background:#dfeae6;
        border-radius:999px;
        margin-top:5px;
        overflow:hidden;
    }

    .bar-fill {
        height:7px;
        background:linear-gradient(90deg,#78b8a5,#4d8f76);
        border-radius:999px;
    }

    .caption-box {
        background:#f8fafc;
        border:1px dashed #cbd5e1;
        color:#475569;
        border-radius:12px;
        padding:12px 14px;
        font-size:0.92rem;
        line-height:1.45;
        margin-top:12px;
    }

    .low-high {
        display:flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        color:#202a3a;
        margin-top:-10px;
        margin-bottom:8px;
        font-weight:700;
    }

    .gradient-bar {
        width:220px;
        height:17px;
        border:1px solid #b8c0cc;
        background:linear-gradient(90deg,#f7f0dc,#91bfd3,#f5d567,#d45643);
    }

    .mini-card {
        background:white;
        border:1px solid #e1e7ef;
        border-radius:14px;
        padding:10px;
        box-shadow:0 1px 7px rgba(15,23,42,0.05);
        margin-bottom:12px;
    }

    .mini-title {
        text-align:center;
        color:#172033;
        font-weight:900;
        font-size:0.92rem;
        margin-bottom:4px;
    }

    .mini-score {
        text-align:center;
        color:#2f6f46;
        font-weight:900;
        font-size:0.88rem;
        margin-bottom:2px;
    }

    div[data-testid="stTextInput"] input {
        border-radius:11px;
        min-height:44px;
        font-size:1rem;
    }

    div[data-testid="stSelectbox"] div {
        border-radius:11px;
    }

    div[data-testid="stButton"] button {
        border-radius:11px;
        background:#5c7ead;
        color:white;
        border:none;
        min-height:43px;
        font-weight:900;
    }

    div[data-testid="stButton"] button:hover {
        background:#2f4f7f;
        color:white;
        border:none;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background:#dfe7f4;
        padding:6px;
        border-radius:999px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 8px 16px;
        background: #b9c9e4;
        color:#1f2937 !important;
        font-weight: 800;
        border:1px solid #9fb3d4;
    }

    .stTabs [aria-selected="true"] {
        background: #496da4 !important;
        color: #ffffff !important;
        border-color:#365887;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_display_name(x):
    if pd.isna(x):
        return x

    s = str(x)

    try:
        if "\\u" in s or "\\x" in s:
            s = s.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass

    for _ in range(3):
        try:
            if "Ã" in s or "Â" in s:
                s = s.encode("latin1").decode("utf-8")
            else:
                break
        except Exception:
            break

    s = s.replace("\u00ad", "").replace("\\u00ad", "")
    return unicodedata.normalize("NFC", s).strip()


def name_key(x):
    if pd.isna(x):
        return ""

    s = clean_display_name(x)
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ł", "l")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def initials(name):
    if not name:
        return "⚽"
    parts = [p for p in str(name).split() if p]
    if len(parts) == 1:
        return parts[0][0].upper()
    return "".join(p[0].upper() for p in parts[:2])


@st.cache_data(show_spinner=False)
def load_artifacts(path_str):
    path = Path(path_str)
    if not path.exists():
        return None

    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    for dataset_name, art in artifacts.items():
        if "player_emb_all" in art and art["player_emb_all"] is not None:
            df = art["player_emb_all"].copy()
            df["player_name"] = df["player_name"].apply(clean_display_name)
            df["_name_key"] = df["player_name"].apply(name_key)
            art["player_emb_all"] = df

        if art.get("sample_level_df", None) is not None:
            sdf = art["sample_level_df"].copy()
            if "player_name" in sdf.columns:
                sdf["player_name"] = sdf["player_name"].apply(clean_display_name)
                sdf["_name_key"] = sdf["player_name"].apply(name_key)
            art["sample_level_df"] = sdf

    return artifacts


def score_pct(score, base=1.0):
    try:
        v = float(score) / float(base)
    except Exception:
        v = 0
    v = max(0, min(1, v))
    return int(round(v * 100))


def get_filtered_names(player_df, query, max_options=60):
    df = player_df.copy()
    q = name_key(query)

    if not q:
        return df["player_name"].dropna().drop_duplicates().sort_values().head(max_options).tolist()

    matches = df[df["_name_key"].str.contains(q, na=False)].copy()

    if len(matches) == 0:
        return []

    # Prioritaskan yang mengandung token paling dekat, lalu yang total touch besar kalau ada.
    if "total_touches" in matches.columns:
        matches = matches.sort_values("total_touches", ascending=False)

    return matches["player_name"].dropna().drop_duplicates().head(max_options).tolist()


def get_all_player_names(player_df):
    names = player_df["player_name"].dropna().drop_duplicates().tolist()
    return sorted(names, key=lambda x: str(x).lower())


def recommend_by_player_name(player_name, player_df, emb_cols, top_k=10):
    df = player_df.copy().reset_index(drop=True)
    q = name_key(player_name)

    exact = df[df["_name_key"] == q]
    if len(exact) == 0:
        partial = df[df["_name_key"].str.contains(q, na=False)]
        if len(partial) == 0:
            return None, None
        exact = partial.head(1)

    target_idx = exact.index[0]
    emb = df[emb_cols].to_numpy(dtype=float)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    target_vec = emb[target_idx]
    df["similarity_score"] = emb @ target_vec

    target = df.loc[[target_idx]].copy()
    recs = (
        df[df.index != target_idx]
        .sort_values("similarity_score", ascending=False)
        .head(top_k)
        .copy()
    )

    return target, recs


def get_player_grid(player_name, sample_df, X):
    if sample_df is None:
        return None

    q = name_key(player_name)
    if "_name_key" in sample_df.columns:
        idx = sample_df.index[sample_df["_name_key"] == q].to_numpy()
        if len(idx) == 0:
            idx = sample_df.index[sample_df["_name_key"].str.contains(q, na=False)].to_numpy()
    else:
        tmp_names = sample_df["player_name"].apply(name_key)
        idx = sample_df.index[tmp_names == q].to_numpy()
        if len(idx) == 0:
            idx = sample_df.index[tmp_names.str.contains(q, na=False)].to_numpy()

    if len(idx) == 0:
        return None

    if X is not None:
        X_np = np.asarray(X)
        grids = X_np[idx]
        if grids.ndim == 4:
            if grids.shape[1] == 1:
                grids = grids[:, 0, :, :]
            else:
                grids = grids.mean(axis=1)
        grid = grids.mean(axis=0)
    else:
        grid_cols = [c for c in sample_df.columns if str(c).startswith("grid_") or str(c).startswith("cell_")]
        if len(grid_cols) == 0:
            return None
        flat = sample_df.loc[idx, grid_cols].to_numpy(dtype=float).mean(axis=0)
        side = int(np.sqrt(len(grid_cols)))
        if side * side != len(grid_cols):
            return None
        grid = flat.reshape(side, side)

    total = np.nansum(grid)
    if total > 0:
        grid = grid / total

    return grid


def smooth_grid(grid, sigma=0.55):
    if grid is None:
        return None

    arr = np.asarray(grid, dtype=float)

    # Prefer scipy if installed, fallback ke simple convolution.
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        # simple 3x3 smoothing fallback
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype=float)
        kernel = kernel / kernel.sum()

        padded = np.pad(arr, 1, mode="edge")
        out = np.zeros_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        return out


def transform_heatmap_orientation(grid):
    arr = np.asarray(grid, dtype=float)
    # Keep native orientation: own goal on the left, opponent goal on the right.
    return arr


def draw_pitch(ax, color="#18202c", alpha=0.65, lw=1.0):
    length = 105.0
    width = 68.0
    half = width / 2.0
    left_box = (width - 40.3) / 2.0
    right_box = width - left_box
    left_6 = (width - 18.3) / 2.0
    right_6 = width - left_6

    # outline
    ax.plot([0, length], [0, 0], color=color, alpha=alpha, lw=lw)
    ax.plot([0, length], [width, width], color=color, alpha=alpha, lw=lw)
    ax.plot([0, 0], [0, width], color=color, alpha=alpha, lw=lw)
    ax.plot([length, length], [0, width], color=color, alpha=alpha, lw=lw)
    ax.plot([length / 2.0, length / 2.0], [0, width], color=color, alpha=alpha, lw=lw)

    circle = plt.Circle((length / 2.0, half), 9.15, fill=False, color=color, alpha=alpha, lw=lw)
    ax.add_patch(circle)
    ax.scatter([length / 2.0], [half], s=9, color=color, alpha=alpha)

    # boxes
    for x0, x1 in [(0, 16.5), (length, length - 16.5)]:
        ax.plot([x0, x1], [left_box, left_box], color=color, alpha=alpha, lw=lw)
        ax.plot([x1, x1], [left_box, right_box], color=color, alpha=alpha, lw=lw)
        ax.plot([x1, x0], [right_box, right_box], color=color, alpha=alpha, lw=lw)

    for x0, x1 in [(0, 5.5), (length, length - 5.5)]:
        ax.plot([x0, x1], [left_6, left_6], color=color, alpha=alpha, lw=lw)
        ax.plot([x1, x1], [left_6, right_6], color=color, alpha=alpha, lw=lw)
        ax.plot([x1, x0], [right_6, right_6], color=color, alpha=alpha, lw=lw)

    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_xticks([])
    ax.set_yticks([])


def make_heatmap_fig(grid, title="", vmax=None, smooth=True, mini=False, smooth_sigma=0.55, interpolation_mode="bilinear"):
    fig, ax = plt.subplots(figsize=(5.8, 3.85) if not mini else (3.2, 2.2), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fbff")

    if grid is None:
        ax.text(0.5, 0.5, "Heatmap tidak tersedia", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return fig

    arr = smooth_grid(grid, sigma=smooth_sigma) if smooth else np.asarray(grid, dtype=float)
    arr = transform_heatmap_orientation(arr)

    if vmax is None:
        vmax = np.percentile(arr[arr > 0], 97) if np.any(arr > 0) else None

    ax.imshow(
        arr,
        origin="lower",
        extent=[0, 105, 0, 68],
        cmap="RdYlBu_r",
        interpolation=interpolation_mode,
        aspect="auto",
        vmin=0,
        vmax=vmax,
        alpha=0.88,
    )
    draw_pitch(ax, alpha=0.68 if not mini else 0.5, lw=1.0 if not mini else 0.7)
    ax.set_title(title, fontsize=9 if mini else 11, fontweight="bold", pad=5)
    plt.tight_layout(pad=0.5)
    return fig


def make_opta_style_heatmap_fig(
    grid,
    player_name,
    subtitle="",
    touches=None,
    smooth=True,
    smooth_sigma=0.85,
):
    fig = plt.figure(figsize=(8.8, 5.1), dpi=130)
    fig.patch.set_facecolor("#ececec")
    ax = fig.add_axes([0.07, 0.20, 0.86, 0.67])
    ax.set_facecolor("#ececec")

    if grid is None:
        ax.text(0.5, 0.5, "Heatmap tidak tersedia", ha="center", va="center", fontsize=11, color="#4b5563")
        ax.axis("off")
        return fig

    arr = smooth_grid(grid, sigma=smooth_sigma) if smooth else np.asarray(grid, dtype=float)
    arr = transform_heatmap_orientation(arr)
    vmax = np.percentile(arr[arr > 0], 96) if np.any(arr > 0) else 1.0

    opta_cmap = LinearSegmentedColormap.from_list(
        "opta_like",
        ["#aebee8", "#7e95da", "#57de7a", "#fff56a", "#ff964f", "#ea4d5b"],
        N=256,
    )

    ax.imshow(
        arr,
        origin="lower",
        extent=[0, 105, 0, 68],
        cmap=opta_cmap,
        interpolation="bicubic",
        vmin=0,
        vmax=vmax,
        alpha=0.9,
        aspect="auto",
    )
    draw_pitch(ax, color="#8f95a3", alpha=0.75, lw=1.25)

    fig.text(0.07, 0.93, f"{player_name} Open-Play Touches", fontsize=16, weight="bold", color="#1f2937")
    if subtitle:
        fig.text(0.07, 0.895, subtitle, fontsize=10, color="#4b5563")

    # Bottom information row (touches + attacking direction + legend)
    if touches is not None:
        fig.text(0.10, 0.095, str(int(touches)), fontsize=14, weight="bold", color="#ffffff",
                 bbox=dict(boxstyle="circle,pad=0.35", facecolor="#df6d7c", edgecolor="none"))
        fig.text(0.157, 0.095, "open-play\ntouches", fontsize=8.5, color="#374151", va="center")

    fig.text(0.37, 0.085, "▶  ▶  ▶  ▶", fontsize=16, color="#7c828f")
    fig.text(0.375, 0.055, "Attacking Direction", fontsize=8.5, color="#4b5563")

    cax = fig.add_axes([0.68, 0.075, 0.18, 0.03])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(grad, aspect="auto", cmap=opta_cmap)
    cax.set_axis_off()
    fig.text(0.63, 0.087, "Fewer\ntouches", fontsize=8.5, color="#4b5563", ha="center")
    fig.text(0.90, 0.087, "More\ntouches", fontsize=8.5, color="#4b5563", ha="center")
    return fig


def render_rec_row(rank, row, base_score):
    name = clean_display_name(row.get("player_name", "-"))
    pct = score_pct(row.get("similarity_score", 0), base=base_score)

    role = row.get("role_name", "")
    sp = row.get("specific_position", "")
    team = row.get("team_name", "")
    sub = " · ".join([str(x) for x in [role, sp, team] if pd.notna(x) and str(x) not in ["", "nan"]])

    st.markdown(
        f"""
        <div class="rec-row">
          <div class="rank">{rank}</div>
          <div class="player-line">
            <div class="mini-avatar">{initials(name)}</div>
            <div style="min-width:0;">
              <div class="player-name">{name}</div>
              <div class="player-sub">{sub}</div>
            </div>
          </div>
          <div>
            <div class="score-text">{pct}%</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_player_card(label, name):
    return f"""
    <div class="player-card">
        <div class="avatar">{initials(name)}</div>
        <div>
            <div class="card-label">{label}</div>
            <div class="card-name">{name}</div>
        </div>
    </div>
    """


# ============================================================
# LOAD ARTIFACTS
# ============================================================

artifacts = load_artifacts(str(ARTIFACT_PATH))

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">Player Recommendation System | Heatmap Visualization</div>
        <div class="topbar-nav">Home&nbsp;&nbsp;|&nbsp;&nbsp;About</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if artifacts is None:
    st.error(
        f"Artifact tidak ditemukan: `{ARTIFACT_PATH}`. "
        "Jalankan export artifact dari notebook terlebih dahulu."
    )
    st.stop()

dataset_names = list(artifacts.keys())
default_idx = dataset_names.index(DEFAULT_DATASET) if DEFAULT_DATASET in dataset_names else 0

with st.sidebar:
    st.header("Settings")
    dataset_name = st.selectbox("Dataset", dataset_names, index=default_idx)
    top_k = st.slider("Jumlah rekomendasi", 5, 15, 10)
    smooth = st.toggle("Smooth heatmap", value=True)
    smooth_sigma = st.slider("Smooth level", 0.0, 1.5, 0.55, 0.05)
    st.caption("Dataset 5 liga disarankan sebagai hasil utama skripsi.")

art = artifacts[dataset_name]
player_df = art["player_emb_all"].copy()
emb_cols = art["emb_cols"]
sample_df = art.get("sample_level_df", None)
X = art.get("X", None)

# ============================================================
# SEARCH AREA
# ============================================================

left, right = st.columns([0.33, 0.67], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Player Recommendations</div>', unsafe_allow_html=True)
        all_players = get_all_player_names(player_df)
        default_anchor = st.session_state.get("selected_player", all_players[0] if all_players else "")
        default_index = all_players.index(default_anchor) if default_anchor in all_players else 0

        selected = st.selectbox(
            "Cari dan pilih anchor player",
            options=all_players,
            index=default_index,
            help="Ketik langsung di kotak ini untuk filter nama pemain. Tidak perlu kotak search terpisah.",
        )
        st.session_state["selected_player"] = selected

        target_df, recs_df = recommend_by_player_name(selected, player_df, emb_cols, top_k=top_k)

        if target_df is None or recs_df is None or len(recs_df) == 0:
            st.warning("Rekomendasi tidak tersedia untuk pemain ini.")
            st.stop()

        target = target_df.iloc[0].copy()
        target["similarity_score"] = 1.0
        target["player_name"] = clean_display_name(target["player_name"])

        recs_show = pd.concat([pd.DataFrame([target]), recs_df], ignore_index=True).head(top_k)
        base_score = float(recs_show.iloc[0]["similarity_score"])

        st.markdown(
            """
            <div class="rec-header">
                <div>Rank</div><div>Player</div><div style="text-align:right;">Similarity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for rank, (_, row) in enumerate(recs_show.iterrows(), start=1):
            render_rec_row(rank, row, base_score)

        st.markdown(
            """
            <div class="caption-box">
                Rekomendasi dihitung dari cosine similarity pada embedding touch map x,y.
                Similarity ditampilkan relatif terhadap anchor player.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HEATMAP MAIN AREA
# ============================================================

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">Heatmap Visualization</div>', unsafe_allow_html=True)

        target_name = clean_display_name(target_df.iloc[0]["player_name"])
        top_candidate = recs_df.iloc[0]
        candidate_name = clean_display_name(top_candidate["player_name"])
        candidate_score = float(top_candidate["similarity_score"])

        card_col1, card_col2 = st.columns(2, gap="large")
        with card_col1:
            st.markdown(render_player_card("Anchor Player", target_name), unsafe_allow_html=True)
            st.markdown('<div class="score-big">Similarity Score: <span>100%</span></div>', unsafe_allow_html=True)
        with card_col2:
            st.markdown(render_player_card("Top Candidate", candidate_name), unsafe_allow_html=True)
            st.markdown(
                f'<div class="score-big">Similarity Score: <span>{score_pct(candidate_score)}%</span></div>',
                unsafe_allow_html=True,
            )

        top_match = target_df.iloc[0]
        target_grid = get_player_grid(target_name, sample_df, X)
        cand_grid = get_player_grid(candidate_name, sample_df, X)

        grids_for_scale = []
        for g in [target_grid, cand_grid]:
            if g is not None:
                sg = smooth_grid(g, sigma=smooth_sigma) if smooth else g
                if np.any(sg > 0):
                    grids_for_scale.append(sg[sg > 0].ravel())
        vmax = np.percentile(np.concatenate(grids_for_scale), 97) if grids_for_scale else None

        target_touches = top_match.get("total_touches", np.nan)
        candidate_touches = top_candidate.get("total_touches", np.nan)

        st.markdown("#### Side-by-Side Compare")
        hm1, hm2 = st.columns(2, gap="large")
        with hm1:
            subtitle = f"Dataset: {dataset_name} | Role: {top_match.get('role_name', '-')}"
            fig = make_opta_style_heatmap_fig(
                target_grid,
                player_name=target_name,
                subtitle=subtitle,
                touches=target_touches if pd.notna(target_touches) else None,
                smooth=smooth,
                smooth_sigma=max(smooth_sigma, 0.75),
            )
            st.pyplot(fig, use_container_width=True)
        with hm2:
            subtitle = f"Similar to: {target_name} | Similarity: {score_pct(candidate_score)}%"
            fig = make_opta_style_heatmap_fig(
                cand_grid,
                player_name=candidate_name,
                subtitle=subtitle,
                touches=candidate_touches if pd.notna(candidate_touches) else None,
                smooth=smooth,
                smooth_sigma=max(smooth_sigma, 0.75),
            )
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            """
            <div class="caption-box">
                Area merah/kuning menunjukkan lokasi sentuhan yang lebih sering.
                Heatmap digunakan untuk membantu menjelaskan mengapa pemain direkomendasikan berdasarkan pola sebaran x,y.
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_more, tab_table = st.tabs(["Top 4 Heatmaps", "Recommendation Table"])

        with tab_more:
            st.write("4 kandidat teratas selain anchor player.")

            candidates = recs_df.head(4).copy()
            cols = st.columns(2, gap="medium")

            for i, (_, row) in enumerate(candidates.iterrows()):
                pname = clean_display_name(row["player_name"])
                score = float(row["similarity_score"])
                g = get_player_grid(pname, sample_df, X)

                with cols[i % 2]:
                    st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="mini-title">{i+1}. {pname}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="mini-score">Similarity: {score_pct(score)}%</div>', unsafe_allow_html=True)
                    fig = make_opta_style_heatmap_fig(
                        g,
                        player_name=pname,
                        subtitle=f"Similarity: {score_pct(score)}%",
                        touches=row.get("total_touches", None),
                        smooth=smooth,
                        smooth_sigma=max(smooth_sigma, 0.75),
                    )
                    st.pyplot(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        with tab_table:
            show_cols = [
                "player_id", "player_name", "role_name", "specific_position",
                "position_status", "team_name", "n_matches", "total_touches",
                "similarity_score"
            ]
            show_cols = [c for c in show_cols if c in recs_df.columns]
            table = recs_df[show_cols].copy()
            if "similarity_score" in table.columns:
                table["similarity_percent"] = table["similarity_score"].apply(lambda x: score_pct(x))
            st.dataframe(table, use_container_width=True)
