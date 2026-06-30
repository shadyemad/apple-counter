import os
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "0"

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

st.set_page_config(page_title="S24 Apple Counter", page_icon="🍎", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #1a1a1a !important; }
    .stApp { background: linear-gradient(160deg, #e8f5e9 0%, #f1f8e9 40%, #fffde7 100%); }
    .stApp * { font-family: 'Cairo', sans-serif !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1b5e20, #2e7d32) !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label, [data-testid="stSidebar"] .stCheckbox label {
        color: #c8e6c9 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.3) !important; }

    .hero { background: linear-gradient(135deg, #1b5e20, #2e7d32, #43a047);
        padding: 20px 28px; border-radius: 16px; margin-bottom: 18px;
        box-shadow: 0 6px 22px rgba(27,94,32,0.3); }
    .hero h1 { color: #fff !important; margin: 0; font-size: 2em; font-weight: 900; }
    .hero p { color: #c8e6c9 !important; margin: 4px 0 0 0; }

    .panel { background: #fff; padding: 16px; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #2e7d32; margin: 8px 0; }
    .panel h3 { color: #1b5e20 !important; margin: 0 0 6px 0; }
    .panel .num { font-size: 2.5em; font-weight: 900; color: #1b5e20 !important; line-height: 1; }
    .panel .sub { color: #666 !important; font-size: 0.85em; margin: 2px 0; }

    .apple-card { background: #f9fbe7; padding: 8px 12px; border-radius: 8px; margin: 3px 0;
        border-left: 3px solid #8bc34a; display: flex; justify-content: space-between; }
    .apple-card.sm { border-left-color: #ff9800; background: #fff8e1; }
    .apple-card.md { border-left-color: #4caf50; background: #f1f8e9; }
    .apple-card.lg { border-left-color: #2e7d32; background: #e8f5e9; }

    .history-card { background: #fff; padding: 14px; border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin: 6px 0;
        border-left: 4px solid #1976d2; }
    .history-card h4 { color: #0d47a1 !important; margin: 0 0 4px 0; font-size: 0.95em; }
    .history-card .stat { display: inline-block; background: #e3f2fd; padding: 3px 10px;
        border-radius: 6px; margin: 2px 4px 2px 0; font-size: 0.8em; color: #1565c0 !important; }

    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        color: #fff !important; font-weight: 700 !important; }
    .stTabs [aria-selected="false"] { background: #fff !important; color: #333 !important; }

    div[data-testid="stMetric"] { background: #fff; padding: 12px; border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04); border-top: 3px solid #2e7d32; }
    div[data-testid="stMetric"] label { color: #555 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1b5e20 !important; font-weight: 700 !important; }

    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()


def estimate_weight(diameter_cm, apple_type="standard"):
    types = {"standard": 0.55, "small_variety": 0.45, "large_variety": 0.60,
             "golden": 0.50, "red_delicious": 0.58, "green": 0.52}
    alpha = types.get(apple_type, 0.55)
    return alpha * (diameter_cm ** 3)


def get_size(d):
    if d < 6: return "Small", "#ff9800", "sm"
    elif d < 8: return "Medium", "#4caf50", "md"
    else: return "Large", "#2e7d32", "lg"


def draw_boxes(frame, dets, show_w=True, show_id=True):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        c = (0, 200, 0) if d['confidence'] > 0.6 else (0, 165, 255) if d['confidence'] > 0.35 else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        parts = []
        if show_id: parts.append(f"#{d['index']}")
        if show_w: parts.append(f"~{d['weight_g']:.0f}g")
        lbl = " ".join(parts)
        (tw, tth), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1-tth-8), (x1+tw+6, y1-2), c, -1)
        cv2.putText(out, lbl, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return out


def load_history():
    path = "count_history.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_history(entry):
    path = "count_history.json"
    hist = load_history()
    hist.insert(0, entry)
    hist = hist[:50]
    with open(path, 'w') as f:
        json.dump(hist, f, indent=2)


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:14px; background:rgba(255,255,255,0.1);
         border-radius:12px; margin-bottom:14px; border:1px solid rgba(255,255,255,0.2);">
        <span style="font-size:2.2em;">🍎</span>
        <h3 style="color:#fff; margin:3px 0;">S24 Farms</h3>
        <p style="color:#c8e6c9; font-size:0.75em;">Apple Counter Pro</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Detection")
    confidence = st.slider("Confidence", 0.05, 1.00, 0.25, 0.05)
    show_weight = st.checkbox("Show weight on apple", value=True)
    show_id = st.checkbox("Show ID number", value=True)

    st.divider()
    st.markdown("### Calibration")
    apple_type = st.selectbox("Apple Variety", [
        "standard", "small_variety", "large_variety", "golden", "red_delicious", "green"
    ])
    use_manual = st.checkbox("Manual pixel ratio", value=False)
    if use_manual:
        pixel_ratio = st.slider("Pixel Ratio (cm/px)", 0.01, 0.50, 0.07, 0.005,
                                help="Lower = camera closer to apples")
    else:
        pixel_ratio = 0.07
    st.caption(f"Ratio: {pixel_ratio:.4f} cm/px")

    st.divider()
    st.markdown("### Farm Info")
    farm_name = st.text_input("Farm Name", value="My Farm")
    farm_area = st.number_input("Area (feddans)", value=2.0, min_value=0.1, max_value=1000.0, step=0.5)
    total_trees = st.number_input("Total Trees", value=500, min_value=1, max_value=10000)
    price_kg = st.number_input("Price per kg ($)", value=2.0, min_value=0.1, max_value=100.0)

    st.divider()
    st.markdown("### History")
    hist = load_history()
    st.caption(f"Saved sessions: {len(hist)}/50")


# HEADER
st.markdown("""
<div class="hero">
    <h1>Apple Counter Pro</h1>
    <p>Upload - Detect - Count - Track History</p>
</div>
""", unsafe_allow_html=True)

tab_img, tab_vid, tab_hist = st.tabs(["Image Detection", "Video Detection", "History"])


# ===================== IMAGE TAB =====================
with tab_img:
    uploaded = st.file_uploader("Upload orchard image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original is not None:
            results = model(original, conf=confidence, classes=[47])
            detections = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf_val = float(boxes.conf[i])
                    x1, y1, x2, y2 = box.astype(int)
                    bw, bh = x2-x1, y2-y1
                    avg_px = (bw+bh)/2
                    diam = avg_px * pixel_ratio
                    wt = estimate_weight(diam, apple_type)
                    sz, szc, szcls = get_size(diam)
                    detections.append({
                        "index": len(detections)+1, "confidence": conf_val,
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "width_px": int(bw), "height_px": int(bh),
                        "diameter_cm": round(diam, 1), "weight_g": round(wt, 1),
                        "size": sz, "size_color": szc, "size_cls": szcls,
                    })

            annotated = draw_boxes(original, detections, show_weight, show_id)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**ORIGINAL**")
                st.image(original, channels="BGR", use_container_width=True)
            with c2:
                st.markdown(f"**DETECTED: {len(detections)} APPLES**")
                st.image(annotated, channels="BGR", use_container_width=True)

            st.divider()
            if detections:
                avg_conf = np.mean([d['confidence'] for d in detections])
                avg_d = np.mean([d['diameter_cm'] for d in detections])
                avg_w = np.mean([d['weight_g'] for d in detections])
                total_w = sum(d['weight_g'] for d in detections)/1000

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total", len(detections))
                m2.metric("Confidence", f"{avg_conf:.0%}")
                m3.metric("Avg Diameter", f"{avg_d:.1f} cm")
                m4.metric("Avg Weight", f"{avg_w:.0f} g")
                m5.metric("Total Weight", f"{total_w:.2f} kg")

                sizes = defaultdict(int)
                for d in detections: sizes[d['size']] += 1
                s1, s2, s3 = st.columns(3)
                s1.metric("Small (<6cm)", sizes.get("Small", 0))
                s2.metric("Medium (6-8cm)", sizes.get("Medium", 0))
                s3.metric("Large (>8cm)", sizes.get("Large", 0))

                # Save to history
                if st.button("Save This Count to History", type="primary"):
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "image",
                        "farm": farm_name,
                        "count": len(detections),
                        "avg_diameter": round(avg_d, 1),
                        "avg_weight": round(avg_w, 1),
                        "total_weight_kg": round(total_w, 2),
                        "yield_tons": round(total_w * (total_trees / max(len(detections), 1)) / 1000, 2),
                        "revenue": round(total_w * price_kg, 2),
                        "confidence": round(confidence, 2),
                        "pixel_ratio": round(pixel_ratio, 4),
                        "apple_type": apple_type,
                        "sizes": dict(sizes),
                    }
                    save_history(entry)
                    st.success("Saved!")

                # List
                st.divider()
                st.markdown("### Apples")
                for d in detections:
                    _, _, sc = get_size(d['diameter_cm'])
                    st.markdown(f"""
                    <div class="apple-card {sc}">
                        <span><strong>#{d['index']}</strong> | {d['diameter_cm']}cm | ~{d['weight_g']:.0f}g | {d['size']}</span>
                        <span>{d['confidence']:.0%}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("No apples detected. Lower confidence slider.")

    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; background:#fff; border-radius:12px;
             box-shadow:0 2px 6px rgba(0,0,0,0.04);">
            <span style="font-size:3.5em;">📸</span>
            <h3>Upload an orchard image</h3>
        </div>""")


# ===================== VIDEO TAB =====================
with tab_vid:
    uploaded_vid = st.file_uploader("Upload orchard video", type=["mp4", "mov", "avi", "mkv"], key="vid")

    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.video(open(tfile.name, 'rb').read())
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{vw}x{vh}")
        c2.metric("FPS", f"{fps:.0f}")
        c3.metric("Duration", f"{total_frames/fps:.1f}s")

        if st.button("Start Counting", type="primary", use_container_width=True):
            import time
            cap = cv2.VideoCapture(tfile.name)
            counted_ids = set()
            apple_data = []
            fc = 0
            clx = vw // 2

            pb = st.progress(0)
            status = st.empty()
            vid_area = st.empty()
            info_area = st.container()
            t0 = time.time()

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok: break

                res = model.track(frame, persist=True, classes=[47], conf=confidence)
                cv2.line(frame, (clx, 0), (clx, vh), (0, 0, 255), 2)
                cv2.putText(frame, "COUNT", (clx-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                if res[0].boxes.id is not None:
                    boxes = res[0].boxes
                    for i, bid in enumerate(boxes.id.cpu().numpy()):
                        bb = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = bb.astype(int)
                        cx = (x1+x2)/2

                        if bid not in counted_ids and cx >= clx:
                            counted_ids.add(bid)
                            bw, bh = x2-x1, y2-y1
                            avg_px = (bw+bh)/2
                            diam = avg_px * pixel_ratio
                            wt = estimate_weight(diam, apple_type)
                            sz, _, _ = get_size(diam)
                            apple_data.append({
                                "id": int(bid), "frame": fc,
                                "diameter_cm": round(diam, 1),
                                "weight_g": round(wt, 1), "size": sz,
                            })

                        clr = (0, 200, 0) if bid in counted_ids else (255, 140, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                        if bid in counted_ids:
                            for ad in apple_data:
                                if ad['id'] == int(bid):
                                    lbl = f"~{ad['weight_g']:.0f}g"
                                    (tw, tth), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                    cv2.rectangle(frame, (x1, y1-tth-8), (x1+tw+6, y1-2), clr, -1)
                                    cv2.putText(frame, lbl, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                    break

                cv2.putText(frame, f"TOTAL: {len(counted_ids)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 3)
                vid_area.image(frame, channels="BGR", use_container_width=True)

                with info_area:
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("Counted", len(counted_ids))
                    if apple_data:
                        tw_kg = sum(d['weight_g'] for d in apple_data)/1000
                        ic2.metric("Weight", f"{tw_kg:.2f} kg")
                        ic3.metric("Avg", f"{np.mean([d['diameter_cm'] for d in apple_data]):.1f} cm")
                    ic4.metric("Frame", f"{fc}/{total_frames}")

                pb.progress(fc/total_frames)
                status.text(f"Frame {fc}/{total_frames} | Counted: {len(counted_ids)}")
                fc += 1

            cap.release()

            # Final results
            st.divider()
            if apple_data:
                total_count = len(apple_data)
                tw_kg = sum(d['weight_g'] for d in apple_data)/1000
                avg_d = np.mean([d['diameter_cm'] for d in apple_data])
                avg_w = np.mean([d['weight_g'] for d in apple_data])
                sizes = defaultdict(int)
                for d in apple_data: sizes[d['size']] += 1

                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Total Apples", total_count)
                f2.metric("Total Weight", f"{tw_kg:.2f} kg")
                f3.metric("Avg Diameter", f"{avg_d:.1f} cm")
                f4.metric("Avg Weight", f"{avg_w:.0f} g")

                st1, st2, st3 = st.columns(3)
                st1.metric("Small", sizes.get("Small", 0))
                st2.metric("Medium", sizes.get("Medium", 0))
                st3.metric("Large", sizes.get("Large", 0))

                # Yield prediction
                st.divider()
                yield_kg = tw_kg * (total_trees / max(total_count, 1))
                yield_tons = yield_kg / 1000
                revenue = yield_kg * price_kg
                y1, y2, y3 = st.columns(3)
                y1.metric("Farm Yield (tons)", f"{yield_tons:.2f}")
                y2.metric("Farm Revenue", f"${revenue:,.2f}")
                y3.metric("Per Feddan", f"{yield_tons/farm_area:.2f} tons" if farm_area > 0 else "N/A")

                # Save
                if st.button("Save This Count to History", type="primary", key="save_vid"):
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "video",
                        "farm": farm_name,
                        "count": total_count,
                        "frames": fc,
                        "avg_diameter": round(float(avg_d), 1),
                        "avg_weight": round(float(avg_w), 1),
                        "total_weight_kg": round(tw_kg, 2),
                        "yield_tons": round(yield_tons, 2),
                        "revenue": round(revenue, 2),
                        "confidence": round(confidence, 2),
                        "pixel_ratio": round(pixel_ratio, 4),
                        "apple_type": apple_type,
                        "sizes": dict(sizes),
                    }
                    save_history(entry)
                    st.success("Saved to history!")
            else:
                st.warning("No apples counted.")
    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; background:#fff; border-radius:12px;
             box-shadow:0 2px 6px rgba(0,0,0,0.04);">
            <span style="font-size:3.5em;">🎬</span>
            <h3>Upload orchard video</h3>
        </div>""")


# ===================== HISTORY TAB =====================
with tab_hist:
    st.markdown("### Count History (Last 50 Sessions)")
    hist = load_history()

    if not hist:
        st.info("No history yet. Count apples and save results.")
    else:
        # Summary
        total_counts = sum(h['count'] for h in hist)
        total_weight = sum(h['total_weight_kg'] for h in hist)
        avg_count = np.mean([h['count'] for h in hist])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sessions", len(hist))
        m2.metric("Total Apples Counted", total_counts)
        m3.metric("Total Weight", f"{total_weight:.1f} kg")
        m4.metric("Avg per Session", f"{avg_count:.0f}")

        st.divider()

        for h in hist:
            ts = h.get('timestamp', 'Unknown')
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = ts

            sizes = h.get('sizes', {})
            sizes_str = " | ".join([f"{k}: {v}" for k, v in sizes.items()]) if sizes else "N/A"

            st.markdown(f"""
            <div class="history-card">
                <h4>{date_str} - {h.get('farm', 'Unknown')} ({h.get('type', 'N/A')})</h4>
                <span class="stat">Apples: {h['count']}</span>
                <span class="stat">Weight: {h['total_weight_kg']:.2f} kg</span>
                <span class="stat">Yield: {h.get('yield_tons', 0):.2f} tons</span>
                <span class="stat">Revenue: ${h.get('revenue', 0):,.2f}</span>
                <span class="stat">Sizes: {sizes_str}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            with open("count_history.json", 'w') as f:
                json.dump([], f)
            st.rerun()
