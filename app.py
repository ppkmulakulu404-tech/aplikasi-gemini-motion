import streamlit as st
import cv2
import json
import tempfile
import os
import sys

# --- INISIALISASI ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    status_ok = True
except:
    status_ok = False

def analyze_motion(data_gerakan, fps):
    """Fungsi untuk mengubah koordinat menjadi kalimat aksi"""
    sequences = []
    chunk_size = int(fps * 2)  # Analisa per 2 detik
    
    for i in range(0, len(data_gerakan), chunk_size):
        chunk = data_gerakan[i:i + chunk_size]
        if not chunk: break
        
        start_time = i // fps
        end_time = (i + len(chunk)) // fps
        time_str = f"{start_time}-{end_time}s"
        
        # Ambil frame awal dan akhir di chunk ini untuk bandingkan posisi
        first = chunk[0]['data']
        last = chunk[-1]['data']
        
        action = "casual stance" # Default
        
        # LOGIKA SEDERHANA UNTUK DETEKSI AKSI
        # 1. Deteksi Tangan ke Dada (Hand to chest)
        # Titik 15/16 (tangan) mendekati titik 11/12 (bahu/dada)
        if abs(last[15]['y'] - last[11]['y']) < 0.1:
            action = "hand gently moves toward chest area"
        
        # 2. Deteksi Melangkah (Step forward)
        # Terlihat dari perubahan skala bahu (makin lebar = makin dekat/maju)
        width_start = abs(first[11]['x'] - first[12]['x'])
        width_end = abs(last[11]['x'] - last[12]['x'])
        if width_end > width_start + 0.02:
            action = "steps slightly forward while shifting weight"
            
        # 3. Deteksi Menoleh/Berputar (Body turn)
        # Jarak horizontal hidung (0) ke bahu berubah drastis
        nose_pos = last[0]['x']
        shoulder_center = (last[11]['x'] + last[12]['x']) / 2
        if abs(nose_pos - shoulder_center) > 0.05:
            action = "small body turn with subtle head movement"
            
        # 4. Deteksi Tangan ke bawah (Adjust t-shirt)
        if last[15]['y'] > last[23]['y']: # Tangan di bawah pinggul
            action = "hand adjusts bottom of t-shirt naturally"

        sequences.append({"time": time_str, "action": action})
    
    return sequences

# --- UI STREAMLIT ---
st.set_page_config(page_title="AI Motion to JSON Prompt", layout="wide")
st.title("🎬 Video to Smart AI Prompt")

if not status_ok:
    st.error("MediaPipe Error. Pastikan instalasi benar.")
    st.stop()

uploaded_file = st.file_uploader("Upload video referensi gerakan", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    data_points = []
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        st.info("Sedang membaca gerakan...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > 300: break # Batasi 10 detik biar cepat
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img)
            
            if results.pose_landmarks:
                pts = {i: {'x': lm.x, 'y': lm.y} for i, lm in enumerate(results.pose_landmarks.landmark)}
                data_points.append({'frame': frame_idx, 'data': pts})
            frame_idx += 1
            
    cap.release()
    
    # Generate Smart Prompt
    motion_seq = analyze_motion(data_points, fps)
    
    final_json = {
        "type": "image_to_video",
        "duration": frame_idx // fps,
        "aspect_ratio": "9:16",
        "style": "active lifestyle cinematic, ultra realistic, 4K",
        "subject": {
            "description": "modern young man wearing a plain t-shirt",
            "emotion": "cool, relaxed, approachable"
        },
        "motion_sequence": motion_seq,
        "output": { "resolution": "4K", "fps": fps }
    }

    st.success("Analisa Selesai!")
    st.json(final_json)
    
    st.download_button(
        label="📥 Download JSON untuk Kling",
        data=json.dumps(final_json, indent=2),
        file_name="kling_prompt.json",
        mime="application/json"
    )
    
    os.unlink(tfile.name)