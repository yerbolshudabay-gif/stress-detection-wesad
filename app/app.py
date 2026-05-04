import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─── Настройка страницы ───
st.set_page_config(
    page_title="Детектор стресса",
    page_icon="🧠",
    layout="wide"
)

# ─── Загрузка модели ───
@st.cache_resource
def load_model():
    model    = joblib.load('../models/xgb_model.pkl')
    scaler   = joblib.load('../models/scaler.pkl')
    features = joblib.load('../models/features.pkl')
    return model, scaler, features

model, scaler, features = load_model()

# ─── Заголовок ───
st.title("🧠 Детектор стресса по данным носимых сенсоров")
st.markdown("**WESAD Dataset | XGBoost | F1-Score: 0.725**")
st.divider()

# ─── Сайдбар — ввод данных ───
st.sidebar.header("⌚ Данные сенсоров")
st.sidebar.markdown("Введите показания браслета и нагрудного датчика:")

# Нормальные диапазоны значений
ranges = {
    'EDA_wrist':  (0.0,   20.0,  1.5,   "мкСм"),
    'TEMP_wrist': (25.0,  40.0,  33.0,  "°C"),
    'ACC_wrist':  (55.0,  80.0,  63.0,  "мг"),
    'EDA_chest':  (0.0,   20.0,  2.0,   "мкСм"),
    'EMG':        (-0.5,  0.0,  -0.02,  "мВ"),
    'TEMP_chest': (25.0,  40.0,  37.0,  "°C"),
    'ACC_chest':  (0.5,   6.0,   1.0,   "мг"),
}

ru_names = {
    'EDA_wrist':  'ЭДА запястье',
    'TEMP_wrist': 'Температура запястье',
    'ACC_wrist':  'Акселерометр запястье',
    'EDA_chest':  'ЭДА грудь',
    'EMG':        'ЭМГ (мышцы)',
    'TEMP_chest': 'Температура грудь',
    'ACC_chest':  'Акселерометр грудь',
}

# Пресеты
st.sidebar.markdown("---")
preset = st.sidebar.radio("🎯 Быстрый пресет:", 
                           ["Ручной ввод", "😌 Спокойствие", "😰 Стресс"])

preset_calm = {
    'EDA_wrist': 0.8, 'TEMP_wrist': 34.5, 'ACC_wrist': 62.0,
    'EDA_chest': 1.2, 'EMG': -0.01, 'TEMP_chest': 37.2, 'ACC_chest': 0.85
}
preset_stress = {
    'EDA_wrist': 8.5, 'TEMP_wrist': 31.0, 'ACC_wrist': 68.0,
    'EDA_chest': 12.0, 'EMG': -0.18, 'TEMP_chest': 36.5, 'ACC_chest': 3.5
}

input_values = {}
st.sidebar.markdown("---")

hints = {
    'EDA_wrist':  'потоотделение кожи',
    'TEMP_wrist': 'температура кожи запястья',
    'ACC_wrist':  'движение руки',
    'EDA_chest':  'потоотделение груди',
    'EMG':        'напряжение мышц',
    'TEMP_chest': 'температура кожи груди',
    'ACC_chest':  'движение корпуса',
}

for feat in features:
    min_v, max_v, default, unit = ranges[feat]
    
    if preset == "😌 Спокойствие":
        default = preset_calm[feat]
    elif preset == "😰 Стресс":
        default = preset_stress[feat]
    
    input_values[feat] = st.sidebar.slider(
        f"{ru_names[feat]} ({unit}) — {hints[feat]}",
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(default),
        step=0.01
    )

# ─── Предсказание ───
input_df = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

prob_calm   = probability[0]
prob_stress = probability[1]

# ─── Основной экран ───
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Состояние", 
              "😰 СТРЕСС" if prediction == 1 else "😌 СПОКОЙСТВИЕ")

with col2:
    st.metric("Вероятность стресса", f"{prob_stress*100:.1f}%")

with col3:
    st.metric("Вероятность спокойствия", f"{prob_calm*100:.1f}%")

st.divider()

# ─── Графики ───
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Вероятности")
    fig_prob = go.Figure(go.Bar(
        x=["😌 Спокойствие", "😰 Стресс"],
        y=[prob_calm, prob_stress],
        marker_color=["#2ecc71", "#e74c3c"],
        text=[f"{prob_calm*100:.1f}%", f"{prob_stress*100:.1f}%"],
        textposition="auto"
    ))
    fig_prob.update_layout(
        yaxis_range=[0, 1],
        yaxis_title="Вероятность",
        height=350
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_right:
    st.subheader("🌡️ Показания сенсоров")
    
    # Нормализуем значения для радар-чарта
    normalized = []
    for feat in features:
        min_v, max_v, _, _ = ranges[feat]
        norm = (input_values[feat] - min_v) / (max_v - min_v)
        normalized.append(norm)
    
    labels = [ru_names[f] for f in features]
    
    fig_radar = go.Figure(go.Scatterpolar(
        r=normalized + [normalized[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)' if prediction == 1 else 'rgba(46, 204, 113, 0.2)',
        line_color='#e74c3c' if prediction == 1 else '#2ecc71',
        name='Показания'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=350
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ─── Feature Importance ───
st.subheader("🔍 Вклад признаков в предсказание")

importance = model.feature_importances_
imp_df = pd.DataFrame({
    'Признак': [ru_names[f] for f in features],
    'Важность': importance
}).sort_values('Важность', ascending=True)

fig_imp = px.bar(imp_df, x='Важность', y='Признак', 
                  orientation='h',
                  color='Важность',
                  color_continuous_scale='RdYlGn')
fig_imp.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_imp, use_container_width=True)

# ─── Футер ───
st.divider()
st.markdown("*WESAD Dataset | XGBoost Classifier | LOSO Cross-Validation | F1=0.725*")