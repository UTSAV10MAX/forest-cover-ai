import io
import json
import math
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="ForestWatch AI", page_icon="🌲", layout="wide")

MODEL_PATH = "forest_cnn_model.pth"
METRICS_PATH = "metrics.json"
SAMPLES_DIR = "samples"

GROUPS = {
    "Forest": ["Forest"],
    "Vegetation & Agriculture": ["AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop"],
    "Water": ["River", "SeaLake"],
    "Built-up": ["Highway", "Industrial", "Residential"],
}
GROUP_NAMES = list(GROUPS)
GROUP_COLORS = {
    "Forest": (34, 160, 70),
    "Vegetation & Agriculture": (240, 200, 40),
    "Water": (40, 130, 230),
    "Built-up": (225, 60, 60),
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
FOREST = GROUP_NAMES.index("Forest")


def rgb(group):
    return "rgb(%d,%d,%d)" % GROUP_COLORS[group]


def stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_image(img, **kwargs):
    return stretch(st.image, img, **kwargs)


def show_chart(fig):
    return stretch(st.plotly_chart, fig)


def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def legend_html():
    items = "".join(
        f"<span style='display:inline-block;margin-right:14px'>"
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
        f"background:{rgb(g)};margin-right:5px;vertical-align:middle'></span>{g}</span>"
        for g in GROUP_NAMES
    )
    return f"<div style='font-size:0.85rem'>{items}</div>"


st.markdown(
    """
    <style>
    .hero {padding:1.4rem 1.6rem;border-radius:16px;margin-bottom:1rem;color:#fff;
           background:linear-gradient(135deg,#0f5132 0%,#198754 55%,#20c997 100%);}
    .hero h1 {margin:0;font-size:2rem;color:#fff;padding:0}
    .hero p {margin:.35rem 0 0;opacity:.93;font-size:1.02rem}
    .card {padding:1rem 1.2rem;border-radius:14px;border:1px solid rgba(128,128,128,.25);
           background:rgba(128,128,128,.08);margin-bottom:.6rem}
    .card .big {font-size:1.5rem;font-weight:700}
    </style>
    <div class="hero">
      <h1>🌲 ForestWatch AI</h1>
      <p>Satellite-based land-cover classification, Grad-CAM explainability, and deforestation driver attribution
      powered by a fine-tuned ResNet18 trained on EuroSAT (Sentinel-2).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["classes"]
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(class_names))
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()
    with torch.no_grad():
        net(torch.zeros(1, 3, 64, 64, device=device))
    return net, class_names, device


try:
    model, class_names, device = load_model()
except Exception as e:
    st.error(f"Could not load '{MODEL_PATH}'. Make sure it is in the same folder as app.py.\n\n{e}")
    st.stop()

C2G = np.array([GROUP_NAMES.index(CLASS_TO_GROUP.get(c, "Built-up")) for c in class_names])

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


metrics = load_metrics()


def predict_probs(tensors, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors), bs):
            x = torch.stack(tensors[i : i + bs]).to(device)
            out.append(F.softmax(model(x), dim=1).cpu())
    return torch.cat(out).numpy()


def gradcam(tensor, class_idx=None):
    acts, grads = [], []
    layer = model.layer3[-1]

    def fwd_hook(_, __, output):
        acts.append(output)
        output.register_hook(lambda g: grads.append(g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        out = model(tensor)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        out[0, class_idx].backward()
    finally:
        handle.remove()
    a, g = acts[0].detach(), grads[0].detach()
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=tensor.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), class_idx


def cam_overlay(img, cam, alpha=0.5, size=256):
    base = img.resize((size, size), Image.BICUBIC)
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize((size, size), Image.BICUBIC)
    return Image.blend(base, heat, alpha)


@st.cache_data(show_spinner=False)
def scan_bytes(img_bytes, tile, target_size=None, max_tiles=1600):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if target_size:
        img = img.resize(tuple(target_size), Image.LANCZOS)
    W, H = img.size
    tile = int(min(max(tile, 8), W, H))
    rows, cols = H // tile, W // tile
    if rows * cols > max_tiles:
        tile = int(math.ceil(tile * math.sqrt(rows * cols / max_tiles)))
        tile = min(tile, W, H)
        rows, cols = H // tile, W // tile
    crop = img.crop((0, 0, cols * tile, rows * tile))
    tensors = [
        transform(crop.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)))
        for r in range(rows)
        for c in range(cols)
    ]
    probs = predict_probs(tensors)
    return {
        "cls": probs.argmax(1).reshape(rows, cols),
        "conf": probs.max(1).reshape(rows, cols),
        "tile": tile,
        "crop": np.array(crop),
    }


def landcover_overlay(crop, group_grid, alpha=0.5):
    h, w = crop.shape[:2]
    colors = np.zeros((*group_grid.shape, 3), np.uint8)
    for gi, g in enumerate(GROUP_NAMES):
        colors[group_grid == gi] = GROUP_COLORS[g]
    over = Image.fromarray(colors).resize((w, h), Image.NEAREST)
    return Image.blend(Image.fromarray(crop), over, alpha)


def change_overlay(crop, lost, gained, tile, alpha=0.6):
    arr = crop.astype(np.float32).copy()
    for mask, col in ((lost, (255, 40, 40)), (gained, (0, 230, 160))):
        up = np.kron(mask.astype(np.uint8), np.ones((tile, tile), np.uint8)).astype(bool)
        arr[up] = (1 - alpha) * arr[up] + alpha * np.array(col, np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def composition_counts(group_grid):
    return np.bincount(group_grid.ravel(), minlength=len(GROUP_NAMES))


def get_bytes(key, demo_path, label):
    has_demo = os.path.exists(demo_path)
    options = (["Use demo scene"] if has_demo else []) + ["Upload an image"]
    mode = st.radio(label, options, horizontal=True, key=f"{key}_mode")
    if mode == "Upload an image":
        f = st.file_uploader(
            label, type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"{key}_up", label_visibility="collapsed"
        )
        return f.getvalue() if f else None
    with open(demo_path, "rb") as f:
        return f.read()


with st.sidebar:
    st.header("🌍 About this project")
    st.write(
        "Detects forest, farmland, water and built-up land from satellite imagery, "
        "and flags **forest loss & replacement drivers** between two dates."
    )
    if metrics:
        st.metric("Validation accuracy", f"{metrics['best_val_acc']:.2f}%")
        st.metric("Macro F1-score", f"{metrics['macro_f1']:.3f}")
    st.caption(f"Running on: **{'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}**")
    st.markdown("**Colour legend**")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.divider()
    st.caption("Model: ResNet18 (ImageNet-pretrained, fine-tuned) · Data: EuroSAT / Sentinel-2, 10 classes")

tab_classify, tab_scan, tab_change, tab_perf, tab_about = st.tabs(
    ["🔍 Classify a patch", "🗺️ Scan a large image", "📉 Change detection", "📊 Model performance", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# TAB 1: Classify a single patch (+ probabilities + Grad-CAM)
# ----------------------------------------------------------------------------
with tab_classify:
    sample_files = []
    if os.path.isdir(SAMPLES_DIR):
        sample_files = sorted(
            f for f in os.listdir(SAMPLES_DIR) if f.endswith(".png") and not f.startswith("scene_")
        )

    left, right = st.columns([1, 1.25], gap="large")
    image, truth = None, None
    with left:
        options = (["Sample patch"] if sample_files else []) + ["Upload"]
        src = st.radio("Image source", options, horizontal=True, key="cls_src")
        if src == "Upload":
            f = st.file_uploader("Upload a satellite patch", type=["png", "jpg", "jpeg", "tif", "tiff"], key="cls_up")
            if f is not None:
                image = Image.open(f).convert("RGB")
        else:
            idx = st.selectbox(
                "Choose a validation sample",
                range(len(sample_files)),
                format_func=lambda i: f"Sample #{i + 1} ({os.path.splitext(sample_files[i])[0]})",
                key="cls_sample",
            )
            truth = os.path.splitext(sample_files[idx])[0]
            image = Image.open(os.path.join(SAMPLES_DIR, sample_files[idx])).convert("RGB")
        if image is not None:
            show_image(image.resize((256, 256), Image.BICUBIC), caption="Input satellite patch (64×64 native)")

    with right:
        if image is None:
            st.info("👈 Upload a satellite patch or pick a sample to run the model.")
        else:
            tensor = transform(image)
            t0 = time.perf_counter()
            probs = predict_probs([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000

            top = int(probs.argmax())
            cls = class_names[top]
            group = CLASS_TO_GROUP.get(cls, "Built-up")
            conf = float(probs[top]) * 100
            st.markdown(
                f"<div class='card' style='border-left:8px solid {rgb(group)}'>"
                f"<div class='big'>{cls}</div>"
                f"Land-cover group: <b>{group}</b> &nbsp;|&nbsp; Confidence: <b>{conf:.1f}%</b> "
                f"&nbsp;|&nbsp; Inference: <b>{infer_ms:.0f} ms</b></div>",
                unsafe_allow_html=True,
            )
            if truth is not None:
                if truth == cls:
                    st.success(f"✅ Matches the ground-truth validation label ({truth}).")
                else:
                    st.warning(f"Ground-truth label is {truth}.")
            if conf < 60:
                st.warning(
                    "Low confidence. The image may look different from the 64×64 Sentinel-2 patches "
                    "this model was trained on."
                )

            dfp = pd.DataFrame({"Class": class_names, "Probability (%)": probs * 100}).sort_values("Probability (%)")
            fig = px.bar(
                dfp,
                x="Probability (%)",
                y="Class",
                orientation="h",
                text=dfp["Probability (%)"].map(lambda v: f"{v:.1f}%"),
            )
            fig.update_traces(
                marker_color=[rgb(CLASS_TO_GROUP.get(c, "Built-up")) for c in dfp["Class"]], textposition="outside"
            )
            fig.update_layout(height=360, margin=dict(l=0, r=20, t=10, b=0), xaxis_range=[0, 115])
            show_chart(fig)

    if image is not None:
        st.subheader("🔥 Why did the AI decide this? (Grad-CAM Explainability)")
        st.caption(
            "Warm colours highlight the exact spatial regions in `layer3` that drove the ResNet18 prediction—"
            "turning a deep neural network into an explainable ecological tool."
        )
        cam, cidx = gradcam(transform(image).unsqueeze(0).to(device))
        c1, c2 = st.columns(2)
        with c1:
            show_image(image.resize((256, 256), Image.BICUBIC), caption="Original Satellite Patch")
        with c2:
            show_image(cam_overlay(image, cam), caption=f"Grad-CAM Activation Map for '{class_names[cidx]}'")

# ----------------------------------------------------------------------------
# TAB 2: Scan a large image -> land-cover map + forest cover %
# ----------------------------------------------------------------------------
with tab_scan:
    st.write(
        "Upload a larger satellite image. It is split into tiles, each tile is classified in batches, "
        "and the result is drawn as a colour-coded land-cover map."
    )
    c_in, c_set = st.columns([2, 1])
    with c_in:
        scan_data = get_bytes("scan", f"{SAMPLES_DIR}/scene_after.png", "Satellite image")
    with c_set:
        scan_tile = st.slider("Tile size (pixels)", 32, 256, 64, 16, key="scan_tile")
        scan_mpp = st.number_input(
            "Ground resolution (metres per pixel)", 0.5, 100.0, 10.0, 0.5, key="scan_mpp",
            help="Sentinel-2 RGB is 10 m/pixel. Used to estimate area in hectares.",
        )

    if scan_data is None:
        st.info("Upload an image or use the demo scene to start.")
    else:
        with st.spinner("Classifying tiles..."):
            res = scan_bytes(scan_data, scan_tile)
        gg = C2G[res["cls"]]
        counts = composition_counts(gg)
        total = int(counts.sum())
        tile_ha = (res["tile"] * scan_mpp) ** 2 / 10000
        overlay = landcover_overlay(res["crop"], gg)

        a, b = st.columns([1.4, 1], gap="large")
        with a:
            o1, o2 = st.columns(2)
            with o1:
                show_image(Image.fromarray(res["crop"]), caption="Original Scene")
            with o2:
                show_image(overlay, caption="AI Land-Cover Segmentation Map")
            st.markdown(legend_html(), unsafe_allow_html=True)
        with b:
            m1, m2 = st.columns(2)
            m1.metric("Forest cover", f"{100 * counts[FOREST] / total:.1f}%")
            m2.metric("Forest area", f"{counts[FOREST] * tile_ha:,.1f} ha")
            pie = go.Figure(
                go.Pie(
                    labels=GROUP_NAMES,
                    values=counts,
                    hole=0.55,
                    sort=False,
                    marker=dict(colors=[rgb(g) for g in GROUP_NAMES]),
                )
            )
            pie.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            show_chart(pie)
        st.caption(f"{res['cls'].shape[0]}×{res['cls'].shape[1]} = {total} tiles of {res['tile']} px each.")

        rows_, cols_ = res["cls"].shape
        df = pd.DataFrame(
            {
                "row": np.repeat(np.arange(rows_), cols_),
                "col": np.tile(np.arange(cols_), rows_),
                "class": [class_names[i] for i in res["cls"].ravel()],
                "group": [GROUP_NAMES[i] for i in gg.ravel()],
                "confidence_%": (res["conf"].ravel() * 100).round(1),
            }
        )
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download land-cover map (PNG)", to_png_bytes(overlay), "landcover_map.png", "image/png")
        d2.download_button("⬇️ Download tile results (CSV)", df.to_csv(index=False), "tile_results.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 3: Change detection (Deforestation + Carbon Impact + Specific Drivers)
# ----------------------------------------------------------------------------
with tab_change:
    st.write(
        "Compare **two images of the same place** taken at different dates. "
        "The AI maps land cover in both, highlights where forest was lost or regrew, and identifies what replaced it."
    )
    cb, ca = st.columns(2)
    with cb:
        st.markdown("**🕰️ Earlier image (before)**")
        before_data = get_bytes("bef", f"{SAMPLES_DIR}/scene_before.png", "Before image")
    with ca:
        st.markdown("**🛰️ Later image (after)**")
        after_data = get_bytes("aft", f"{SAMPLES_DIR}/scene_after.png", "After image")
    s1, s2, s3 = st.columns(3)
    chg_tile = s1.slider("Tile size (pixels)", 32, 256, 64, 16, key="chg_tile")
    chg_mpp = s2.number_input("Ground resolution (m/pixel)", 0.5, 100.0, 10.0, 0.5, key="chg_mpp")
    carbon_dens = s3.number_input(
        "Biomass carbon density (tC/ha)", 50.0, 300.0, 140.0, 10.0,
        help="Typical dense forest stores ~140 tonnes of Carbon per hectare (1 tC ≈ 3.67 tCO₂)."
    )

    if before_data is None or after_data is None:
        st.info("Provide both images (or use the demo scenes) to run the analysis.")
    else:
        size = Image.open(io.BytesIO(before_data)).size
        with st.spinner("Analysing both images..."):
            rb = scan_bytes(before_data, chg_tile)
            ra = scan_bytes(after_data, chg_tile, target_size=size)
        if rb["cls"].shape != ra["cls"].shape:
            st.error("The two images could not be aligned. Please use images covering the same area.")
        else:
            bg, ag = C2G[rb["cls"]], C2G[ra["cls"]]
            forest_b, forest_a = bg == FOREST, ag == FOREST
            lost, gained = forest_b & ~forest_a, ~forest_b & forest_a
            n = bg.size
            fb, fa = int(forest_b.sum()), int(forest_a.sum())
            tile_ha = (rb["tile"] * chg_mpp) ** 2 / 10000
            lost_ha = lost.sum() * tile_ha
            gained_ha = gained.sum() * tile_ha
            co2_tons = lost_ha * carbon_dens * 3.67
            loss_pct = 100 * lost.sum() / fb if fb else 0.0

            if fb == 0:
                st.info("No forest was detected in the earlier image, so there is nothing to lose.")
            elif loss_pct >= 10:
                st.error(
                    f"🚨 **Severe deforestation alert:** {loss_pct:.1f}% of the baseline forest "
                    f"({lost_ha:,.1f} ha) was cleared, releasing an estimated **{co2_tons:,.0f} tCO₂**."
                )
            elif loss_pct >= 2:
                st.warning(
                    f"⚠️ **Moderate forest loss:** {loss_pct:.1f}% of the baseline forest "
                    f"({lost_ha:,.1f} ha) was cleared (~{co2_tons:,.0f} tCO₂)."
                )
            else:
                st.success("✅ No significant forest loss detected.")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Forest before", f"{100 * fb / n:.1f}%")
            k2.metric("Forest after", f"{100 * fa / n:.1f}%", f"{100 * (fa - fb) / n:+.1f} pts")
            k3.metric("Forest lost", f"{lost_ha:,.1f} ha")
            k4.metric("Forest regrown", f"{gained_ha:,.1f} ha")
            k5.metric("Est. CO₂ emissions", f"{co2_tons:,.0f} tCO₂")

            chg_img = change_overlay(ra["crop"], lost, gained, ra["tile"])
            i1, i2, i3 = st.columns(3)
            with i1:
                show_image(landcover_overlay(rb["crop"], bg), caption="Before: land cover")
            with i2:
                show_image(landcover_overlay(ra["crop"], ag), caption="After: land cover")
            with i3:
                show_image(chg_img, caption="Change map (red = forest lost, green = regrowth)")

            g1, g2 = st.columns(2)
            comp = pd.DataFrame(
                {
                    "Group": GROUP_NAMES * 2,
                    "Image": ["Before"] * 4 + ["After"] * 4,
                    "Share (%)": list(100 * composition_counts(bg) / n) + list(100 * composition_counts(ag) / n),
                }
            )
            figc = px.bar(
                comp, x="Group", y="Share (%)", color="Image", barmode="group",
                color_discrete_map={"Before": "#6c757d", "After": "#20c997"}
            )
            figc.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="Land-cover share (%)")
            with g1:
                show_chart(figc)
            with g2:
                # Fine-grained 10-class EuroSAT driver attribution
                fine_conv = []
                after_cls_grid = ra["cls"]
                total_lost_tiles = int(lost.sum())
                for ci, cname in enumerate(class_names):
                    if cname == "Forest":
                        continue
                    k = int(((after_cls_grid == ci) & lost).sum())
                    if k:
                        fine_conv.append({
                            "Replacement Class (Driver)": cname,
                            "Group": CLASS_TO_GROUP.get(cname, "Built-up"),
                            "Tiles": k,
                            "Area (ha)": round(k * tile_ha, 1),
                            "Share of Loss (%)": round(100.0 * k / max(total_lost_tiles, 1), 1),
                        })
                st.markdown("**🔍 What replaced the lost forest? (Exact EuroSAT Class Drivers)**")
                if fine_conv:
                    df_conv = pd.DataFrame(fine_conv).sort_values("Tiles", ascending=False)
                    stretch(st.dataframe, df_conv, hide_index=True)
                else:
                    st.write("No forest conversion detected.")
            st.markdown(legend_html(), unsafe_allow_html=True)

            dl1, dl2 = st.columns(2)
            # ---  REAL-WORLD IMPACT & RANGER DISPATCH ---
            st.divider()
            st.subheader("🛡️ Field Response & Ecological Recovery Plan")
            
            trees_needed = int(lost_ha * 1100)  # ~1,100 saplings per hectare
            cars_equiv = int(co2_tons / 4.6)    # ~4.6 tCO2 per car per year
            carbon_value_usd = int(co2_tons * 15) # ~$15/tCO2 voluntary carbon market

            r1, r2, r3 = st.columns(3)
            r1.metric("🌱 Saplings Needed to Restore", f"{trees_needed:,} trees", "At 1,100 trees/ha")
            r2.metric("🚗 Annual Car Emissions Equiv.", f"{cars_equiv:,} cars/yr", "EPA standard")
            r3.metric("💰 Carbon Credit Value Lost", f"${carbon_value_usd:,} USD", "@ $15 / tCO₂e")

            # Identify Top 5 Priority Patrol Sectors (Lost forest tiles sorted by AI confidence)
            lost_rows, lost_cols = np.where(lost)
            if len(lost_rows) > 0:
                patrol_list = []
                for r, c in zip(lost_rows, lost_cols):
                    new_cls = class_names[ra["cls"][r, c]]
                    conf_val = float(ra["conf"][r, c]) * 100
                    patrol_list.append({
                        "Sector ID": f"Grid R{r+1}-C{c+1}",
                        "Pixel Box (x, y)": f"({c * ra['tile']}, {r * ra['tile']})",
                        "Detected Encroachment": new_cls,
                        "AI Confidence (%)": round(conf_val, 1),
                        "Priority": "🚨 CRITICAL" if new_cls in ["Industrial", "Highway", "Residential"] else "⚠️ HIGH"
                    })
                df_patrol = pd.DataFrame(patrol_list).sort_values(
                    by=["Priority", "AI Confidence (%)"], ascending=[True, False]
                )
                st.markdown("**📍 Automated Ranger Dispatch Queue (Top Priority Sectors)**")
                stretch(st.dataframe, df_patrol.head(8), hide_index=True)

                # Official Incident Report Text
                top_driver = fine_conv[0]["Replacement Class (Driver)"] if fine_conv else "Unknown"
                report_txt = (
                    f"==========================================================\n"
                    f"       FORESTWATCH AI - DEFORESTATION INCIDENT BRIEF      \n"
                    f"==========================================================\n"
                    f"Severity Status     : {'SEVERE ALERT' if loss_pct >= 10 else 'MODERATE ALERT'}\n"
                    f"Baseline Forest     : {100 * fb / n:.1f}% of scanned region\n"
                    f"Current Forest      : {100 * fa / n:.1f}% of scanned region\n"
                    f"Total Canopy Lost   : {lost_ha:,.1f} hectares ({loss_pct:.1f}% loss)\n"
                    f"Primary Driver      : {top_driver}\n"
                    f"Est. CO2 Released   : {co2_tons:,.0f} tCO2e (~{cars_equiv:,} cars/year)\n"
                    f"Restoration Target  : {trees_needed:,} saplings required\n"
                    f"Top Patrol Sector   : {df_patrol.iloc[0]['Sector ID']} ({df_patrol.iloc[0]['Detected Encroachment']})\n"
                    f"==========================================================\n"
                )
                st.download_button(
                    "📄 Download Official Forestry Action Brief (.TXT)",
                    report_txt,
                    "ForestWatch_Incident_Brief.txt",
                    "text/plain",
                )
            dl1.download_button(
                "⬇️ Download Deforestation Alert Map (PNG)",
                to_png_bytes(chg_img),
                "deforestation_change_map.png",
                "image/png",
            )
            audit_summary = {
                "forest_before_pct": round(100 * fb / n, 2),
                "forest_after_pct": round(100 * fa / n, 2),
                "forest_lost_ha": round(lost_ha, 2),
                "forest_regrown_ha": round(gained_ha, 2),
                "estimated_co2_tons": round(co2_tons, 1),
                "drivers": fine_conv,
            }
            dl2.download_button(
                "⬇️ Download Change Audit Report (JSON)",
                json.dumps(audit_summary, indent=2),
                "deforestation_audit.json",
                "application/json",
            )

# ----------------------------------------------------------------------------
# TAB 4: Model performance (proof the accuracy is real)
# ----------------------------------------------------------------------------
with tab_perf:
    if not metrics:
        st.info("Run the training cell in Colab to create 'metrics.json', then restart the app to see charts here.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Validation accuracy", f"{metrics['best_val_acc']:.2f}%")
        p2.metric("Macro F1-score", f"{metrics['macro_f1']:.3f}")
        p3.metric("Training images", f"{metrics['train_samples']:,}")
        p4.metric("Validation images", f"{metrics['val_samples']:,}")
        st.caption(f"{metrics['arch']} · {metrics['epochs']} epochs · validation images were never used for training.")

        hist = metrics["history"]
        ep = list(range(1, len(hist["train_acc"]) + 1))
        h1, h2 = st.columns(2)
        fa_ = go.Figure()
        fa_.add_trace(go.Scatter(x=ep, y=hist["train_acc"], name="Train", mode="lines+markers"))
        fa_.add_trace(go.Scatter(x=ep, y=hist["val_acc"], name="Validation", mode="lines+markers"))
        fa_.update_layout(title="Accuracy (%)", xaxis_title="Epoch", height=320, margin=dict(l=0, r=0, t=40, b=0))
        fl_ = go.Figure()
        fl_.add_trace(go.Scatter(x=ep, y=hist["train_loss"], name="Train", mode="lines+markers"))
        fl_.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="Validation", mode="lines+markers"))
        fl_.update_layout(title="Loss", xaxis_title="Epoch", height=320, margin=dict(l=0, r=0, t=40, b=0))
        with h1:
            show_chart(fa_)
        with h2:
            show_chart(fl_)

        cm = np.array(metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        fig_cm = px.imshow(
            cm_norm, x=metrics["classes"], y=metrics["classes"], text_auto=".2f",
            color_continuous_scale="Greens", labels=dict(x="Predicted", y="Actual", color="Share"), aspect="auto",
        )
        fig_cm.update_layout(title="Confusion matrix (row-normalised)", height=520, margin=dict(l=0, r=0, t=40, b=0))
        show_chart(fig_cm)

        pc = pd.DataFrame(metrics["per_class"]).T.reset_index().rename(
            columns={"index": "Class", "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Images"}
        )
        st.markdown("**Per-class results**")
        stretch(st.dataframe, pc.round(3), hide_index=True)

# ----------------------------------------------------------------------------
# TAB 5: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### The problem
Forests are lost faster than they can be manually monitored. Field surveys are slow and expensive,
and satellite data is abundant but hard to interpret at scale.

### The solution
ForestWatch AI turns raw satellite imagery into **instant, explainable land-cover maps** and
**forest-loss alerts**.

**Pipeline:** satellite image → tiling → ResNet18 classifier → land-cover map → compare two dates → forest-loss alert & carbon impact

### Real-world applications
- Early warning for illegal logging and encroachment
- Verifying reforestation and carbon-offset projects
- Tracking urban and industrial sprawl into forests and farmland
- Fast first-pass screening before sending field teams

### Honest limitations
- Trained on **EuroSAT** (Sentinel-2 RGB patches, 10 m resolution). Accuracy on other sensors, regions or
  resolutions should be validated before real use.
- Change detection compares two classified tile maps, so seasonal changes (for example,
  harvested crops) can look like change. Images should come from the same season.
- Uses RGB only; adding multispectral bands (NDVI) is the natural next step.
        """
    )
