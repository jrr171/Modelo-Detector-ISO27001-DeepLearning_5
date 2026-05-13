"""
Streamlit Web App — Evaluador de Madurez en Seguridad de la Información
Tesis: Modelo de Evaluación De la Madurez en Seguridad de la Información
Usando Simulador para la Detección de Incumplimiento de Requisitos
en una Empresa de Inteligencia Comercial en el Sector Comercio Exterior
 
Gráficos incluidos:
  1. Medidor (gauge) de madurez global
  2. Radar de dominios ISO 27001
  3. Barras comparativas: riesgo vs seguro por dominio
  4. Desglose de componentes de score (stacked bar)
  5. Distribución de eventos por dominio (pie)
  6. Mapa de calor de tasa de riesgo por dominio
  7. Escala de madurez tipo semáforo (progress)
  8. Sunburst de eventos clasificados
  9. Histograma de niveles por dominio
"""
 
import sys, io, json, tempfile, os, math
from pathlib import Path
 
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
 
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
 
from analyzer.log_parser       import LogParser
from analyzer.event_classifier import EventClassifier
from analyzer.maturity_scorer  import MaturityScorer
from analyzer.report_generator import export_html, export_json
from rules.iso27001_controls   import MATURITY_LEVELS, ISO27001_DOMAINS
 
# ────────────────────────────────────────────────────────────────────────────
# Paleta de colores corporativa (tesis)
# ────────────────────────────────────────────────────────────────────────────
C = {
    "primary":   "#1565C0",
    "secondary": "#0D47A1",
    "success":   "#2E7D32",
    "warning":   "#F57F17",
    "danger":    "#C62828",
    "level": {
        0: "#B71C1C", 1: "#D32F2F", 2: "#F57C00",
        3: "#FBC02D", 4: "#388E3C", 5: "#1B5E20",
    },
    "domains": [
        "#1565C0","#6A1B9A","#00695C","#E65100","#4527A0","#00838F",
    ],
}
 
def level_color(lvl): return C["level"].get(lvl, "#555")
 
def score_color(s):
    if s >= 81: return C["level"][5]
    if s >= 61: return C["level"][4]
    if s >= 41: return C["level"][3]
    if s >= 21: return C["level"][2]
    if s >  0:  return C["level"][1]
    return C["level"][0]
 
def hex_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"
 
 
# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluador de Madurez ISO 27001 | Comercio Exterior",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main-title  { font-size:2.1rem; font-weight:800; color:#0D47A1; letter-spacing:-0.5px; }
  .subtitle    { font-size:.95rem; color:#546E7A; margin-bottom:1rem; }
  .section-hdr { font-size:1.2rem; font-weight:700; color:#1565C0;
                 border-left:4px solid #1565C0; padding-left:10px; margin:24px 0 12px; }
  .kpi-card    { background:#F8FAFF; border:1px solid #BBDEFB; border-radius:12px;
                 padding:16px 20px; text-align:center; }
  .kpi-val     { font-size:2rem; font-weight:800; }
  .kpi-lbl     { font-size:.8rem; color:#78909C; font-weight:600; letter-spacing:.5px; }
  .finding     { background:#FFF3E0; border-left:4px solid #FF6F00;
                 border-radius:6px; padding:8px 14px; margin-bottom:6px; font-size:.9rem; }
  .rec         { background:#E8F5E9; border-left:4px solid #388E3C;
                 border-radius:6px; padding:8px 14px; margin-bottom:6px; font-size:.9rem; }
  .chart-box   { background:#fff; border:1px solid #E3EAF5; border-radius:12px; padding:16px; }
  footer       { text-align:center; color:#90A4AE; font-size:.78rem; margin-top:40px; }
</style>
""", unsafe_allow_html=True)
 
# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡 ISO 27001 Maturity")
    st.markdown("**Modelo COBIT — 6 Niveles**")
    for i in range(6):
        info = MATURITY_LEVELS[i]
        lo, hi = info["range"]
        rng = f"{lo}–{hi}%" if i > 0 else "0%"
        st.markdown(
            f"<div style='padding:5px 8px;margin-bottom:4px;border-radius:6px;"
            f"background:{level_color(i)}22;border-left:3px solid {level_color(i)};'>"
            f"<b style='color:{level_color(i)}'>Nivel {i}</b> · {rng}<br>"
            f"<span style='font-size:.8em;color:#555'>{info['name']}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("ISO/IEC 27001:2013 · COBIT 5 · NTP ISO/IEC 27001:2008")
    st.caption("Comercio Exterior — Tesis 2025")
 
# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡 Evaluador de Madurez en Seguridad de la Información</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detección de Incumplimiento de Requisitos ISO 27001 mediante análisis de logs · Empresa de Inteligencia Comercial · Sector Comercio Exterior</div>', unsafe_allow_html=True)
 
# ────────────────────────────────────────────────────────────────────────────
# Input tabs
# ────────────────────────────────────────────────────────────────────────────
tab_up, tab_demo, tab_paste = st.tabs(["📁 Subir archivos", "🧪 Demo Comercio Exterior", "📋 Pegar texto"])
 
entries, source_label = [], ""
 
with tab_up:
    st.markdown("**Formatos soportados:** Apache/Nginx `.log`, Linux syslog/auth.log, Windows Event Log `.csv`, JSON `.json`, `.gz`")
    uploaded = st.file_uploader("Arrastra tus archivos de log aquí", type=["log","txt","csv","json","gz"], accept_multiple_files=True)
    if uploaded:
        with tempfile.TemporaryDirectory() as d:
            for f in uploaded:
                (Path(d) / f.name).write_bytes(f.read())
            parser = LogParser()
            entries = parser.parse_path(d)
            source_label = f"{len(uploaded)} archivo(s)"
            st.success(f"✅ {parser.stats['parsed_ok']:,} eventos leídos de {len(uploaded)} archivo(s)")
 
with tab_demo:
    st.info("Logs simulados de una empresa de Comercio Exterior (declaraciones DUA, ERP aduanero, portal de importaciones, SIEM, Active Directory).")
    if st.button("▶ Ejecutar análisis con logs demo", type="primary"):
        sdir = ROOT / "samples"
        sample_files = list(sdir.glob("sample_*.log")) + list(sdir.glob("sample_*.csv"))
        if not sample_files:
            import subprocess
            subprocess.run([sys.executable, str(sdir / "generate_samples.py")], check=True)
        parser = LogParser()
        entries = parser.parse_path(str(sdir))
        source_label = "Logs Demo — Comercio Exterior"
        st.success(f"✅ {parser.stats['parsed_ok']:,} eventos procesados")
        st.session_state.update({"entries": entries, "source": source_label})
 
with tab_paste:
    pasted = st.text_area("Pega el contenido de tu log:", height=180,
        placeholder="Jan  1 10:00:00 srv sshd[1234]: Failed password for root from 10.0.0.1 port 22 ssh2")
    if st.button("▶ Analizar texto", type="primary") and pasted.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
            tf.write(pasted); tf_path = tf.name
        parser = LogParser()
        entries = parser.parse_path(tf_path)
        os.unlink(tf_path)
        source_label = "Texto pegado"
        st.success(f"✅ {len(entries):,} eventos leídos")
 
if not entries and "entries" in st.session_state:
    entries = st.session_state["entries"]
    source_label = st.session_state.get("source","")
 
# ────────────────────────────────────────────────────────────────────────────
# ANÁLISIS Y GRÁFICOS
# ────────────────────────────────────────────────────────────────────────────
if not entries:
    st.divider()
    st.markdown("### ¿Cómo usar esta herramienta?")
    st.markdown("""
1. **Sube tus logs** o usa el botón **Demo** para ver un ejemplo inmediato.
2. La herramienta clasifica los eventos según los **6 dominios ISO 27001**.
3. Calcula el **nivel de madurez COBIT (0–5)** con gráficos detallados.
4. Descarga el reporte en **HTML o JSON** para tu tesis.
    """)
    for i, (key, dom) in enumerate(ISO27001_DOMAINS.items()):
        with st.expander(f"{dom.id} — {dom.name}  (peso {dom.weight:.0%})"):
            st.caption(dom.description)
    st.stop()
 
# Pipeline
with st.spinner("Clasificando eventos y calculando madurez…"):
    domain_stats = EventClassifier().classify(entries)
    result = MaturityScorer().score(domain_stats)
 
lvl      = result.overall_level
lvl_info = MATURITY_LEVELS[lvl]
lc       = level_color(lvl)
domains  = list(result.domain_scores.values())
dom_names = [d.domain_name for d in domains]
 
# ── KPIs ──────────────────────────────────────────────────────────────────────
st.divider()
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (f"{result.overall_score:.1f}/100", "SCORE GLOBAL", lc),
    (f"Nivel {lvl}", lvl_info["name"][:16], lc),
    (f"{result.total_events:,}", "EVENTOS TOTALES", C["primary"]),
    (f"{result.total_risk_events:,}", "EVENTOS DE RIESGO", C["danger"]),
    (f"{result.total_domains_active}/{len(domain_stats)}", "DOMINIOS ACTIVOS", C["success"]),
    (f"{result.total_risk_events/max(result.total_events,1):.1%}", "TASA DE RIESGO", C["warning"]),
]
for col, (val, lbl, color) in zip([c1,c2,c3,c4,c5,c6], kpis):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-val" style="color:{color}">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )
 
# ════════════════════════════════════════════════════════
# FILA 1: Gauge + Radar
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📊 Resultado Global</div>', unsafe_allow_html=True)
col_gauge, col_radar = st.columns([1, 1.2])
 
# ── GRÁFICO 1: Gauge / Medidor de madurez ────────────────────────────────────
with col_gauge:
    st.markdown("#### 🎯 Medidor de Nivel de Madurez")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result.overall_score,
        delta={"reference": 60, "valueformat":".1f", "suffix":" pts"},
        title={"text": f"<b>Nivel {lvl} — {lvl_info['name']}</b><br><span style='font-size:.8em;color:#555'>{source_label}</span>", "font":{"size":15}},
        number={"suffix": " / 100", "font":{"size":36, "color": lc}},
        gauge={
            "axis": {"range":[0,100], "tickwidth":1, "tickcolor":"#333",
                     "tickvals":[0,20,40,60,80,100],
                     "ticktext":["0\nNivel 0","20\nNivel 1","40\nNivel 2","60\nNivel 3","80\nNivel 4","100\nNivel 5"]},
            "bar":  {"color": lc, "thickness":0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range":[0,20],  "color":"#FFCDD2"},
                {"range":[20,40], "color":"#FFE0B2"},
                {"range":[40,60], "color":"#FFF9C4"},
                {"range":[60,80], "color":"#C8E6C9"},
                {"range":[80,100],"color":"#A5D6A7"},
            ],
            "threshold": {"line":{"color":lc,"width":4}, "thickness":0.75, "value":result.overall_score},
        }
    ))
    fig_gauge.update_layout(height=320, margin=dict(l=20,r=20,t=60,b=10), paper_bgcolor="white")
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(f'<div style="background:{lc}18;border:1px solid {lc}44;border-radius:8px;padding:10px 14px;font-size:.88em;color:#333">'
                f'<b style="color:{lc}">ℹ {lvl_info["name"]}</b><br>{lvl_info["description"]}</div>', unsafe_allow_html=True)
 
# ── GRÁFICO 2: Radar / Spider de dominios ISO 27001 ──────────────────────────
with col_radar:
    st.markdown("#### 🕸 Radar de Dominios ISO 27001")
    scores_radar = [d.raw_score for d in domains]
    labels_radar = [f"A.{ISO27001_DOMAINS[d.domain_key].id.split('A.')[1]}<br>{d.domain_name}" for d in domains]
 
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores_radar + [scores_radar[0]],
        theta=labels_radar + [labels_radar[0]],
        fill="toself",
        fillcolor=hex_rgba(C["primary"], 0.2),
        line=dict(color=C["primary"], width=2.5),
        name="Score por dominio",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>",
    ))
    # Añadir anillo de referencia nivel 3 (60 pts)
    fig_radar.add_trace(go.Scatterpolar(
        r=[60]*len(labels_radar) + [60],
        theta=labels_radar + [labels_radar[0]],
        mode="lines",
        line=dict(color="#FBC02D", width=1.5, dash="dot"),
        name="Referencia Nivel 3 (60 pts)",
        hoverinfo="skip",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9),
                            gridcolor="#E8EAF6", tickvals=[20,40,60,80,100]),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        height=360,
        margin=dict(l=60, r=60, t=40, b=60),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
# ════════════════════════════════════════════════════════
# FILA 2: Barras comparativas + Desglose componentes
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📋 Análisis por Dominio ISO 27001</div>', unsafe_allow_html=True)
col_bar1, col_bar2 = st.columns(2)
 
# ── GRÁFICO 3: Barras riesgo vs seguro por dominio ────────────────────────────
with col_bar1:
    st.markdown("#### ⚠ Eventos de Riesgo vs Seguros por Dominio")
    dom_keys = list(domain_stats.keys())
    dom_names_short = [d.domain_name.replace("Seguridad en ","Seg. ").replace("Gestión de ","Gest. ")[:24] for d in domains]
    safe_counts = [domain_stats[k].indicator_events for k in dom_keys]
    risk_counts = [domain_stats[k].risk_events      for k in dom_keys]
 
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Eventos Seguros", x=dom_names_short, y=safe_counts,
        marker_color=hex_rgba(C["success"], 0.8),
        hovertemplate="<b>%{x}</b><br>Eventos seguros: %{y}<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        name="Eventos de Riesgo", x=dom_names_short, y=risk_counts,
        marker_color=hex_rgba(C["danger"], 0.8),
        hovertemplate="<b>%{x}</b><br>Eventos de riesgo: %{y}<extra></extra>",
    ))
    fig_bar.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10,r=10,t=20,b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
        yaxis=dict(title="N° eventos", gridcolor="#F0F0F0"),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
 
# ── GRÁFICO 4: Desglose de componentes del score (stacked horizontal bar) ─────
with col_bar2:
    st.markdown("#### 🔬 Desglose del Score por Componente")
    comps = ["Presencia de Logs","Efectividad de Controles","Ajuste Severidad","Cobertura"]
    comp_keys = ["logging_presence","control_effectiveness","severity_adjustment","coverage_bonus"]
    comp_colors = [C["primary"],"#00897B","#FB8C00","#8E24AA"]
 
    fig_stack = go.Figure()
    for comp, key, color in zip(comps, comp_keys, comp_colors):
        vals = [max(0, d.breakdown.get(key, 0)) for d in domains]
        fig_stack.add_trace(go.Bar(
            name=comp, y=dom_names_short, x=vals,
            orientation="h", marker_color=hex_rgba(color, 0.8),
            hovertemplate=f"<b>%{{y}}</b><br>{comp}: %{{x:.1f}} pts<extra></extra>",
        ))
    fig_stack.update_layout(
        barmode="stack", height=320,
        margin=dict(l=10,r=10,t=20,b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
        xaxis=dict(title="Puntos", range=[0,100], gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig_stack, use_container_width=True)
 
# ════════════════════════════════════════════════════════
# FILA 3: Score barras + Pie distribución
# ════════════════════════════════════════════════════════
col_scores, col_pie = st.columns([1.4, 1])
 
# ── GRÁFICO 5: Score por dominio (barras horizontales con colores de nivel) ───
with col_scores:
    st.markdown("#### 📊 Score y Nivel por Dominio")
    sorted_domains = sorted(domains, key=lambda d: d.raw_score)
    bar_colors  = [level_color(d.level) for d in sorted_domains]
    bar_names   = [f"{d.domain_name} ({d.clause.split('–')[0].strip()})" for d in sorted_domains]
    bar_scores  = [d.raw_score for d in sorted_domains]
    bar_levels  = [f"Nivel {d.level} — {d.level_name}" for d in sorted_domains]
 
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(
        y=bar_names, x=bar_scores, orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}" for s in bar_scores],
        textposition="outside",
        customdata=bar_levels,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<br>%{customdata}<extra></extra>",
    ))
    # Líneas de referencia de niveles
    for threshold, label, color in [(20,"Nivel 1","#D32F2F"),(40,"Nivel 2","#F57C00"),(60,"Nivel 3","#FBC02D"),(80,"Nivel 4","#388E3C")]:
        fig_h.add_vline(x=threshold, line_dash="dot", line_color=color, line_width=1.5,
                        annotation_text=label, annotation_position="top",
                        annotation_font=dict(size=9, color=color))
    fig_h.update_layout(
        height=340, margin=dict(l=10,r=60,t=30,b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(range=[0,110], title="Score (0–100)", gridcolor="#F0F0F0"),
        showlegend=False,
    )
    st.plotly_chart(fig_h, use_container_width=True)
 
# ── GRÁFICO 6: Pie distribución de eventos por dominio ────────────────────────
with col_pie:
    st.markdown("#### 🥧 Distribución de Eventos por Dominio")
    pie_vals  = [domain_stats[d.domain_key].total_events for d in domains]
    pie_names = [d.domain_name.replace("Seguridad en ","Seg. ").replace("Gestión de ","Gest. ")[:22] for d in domains]
    fig_pie = go.Figure(go.Pie(
        labels=pie_names, values=pie_vals,
        marker=dict(colors=C["domains"], line=dict(color="white", width=2)),
        hole=0.45,
        hovertemplate="<b>%{label}</b><br>Eventos: %{value:,}<br>%{percent}<extra></extra>",
        textinfo="percent+label",
        textfont=dict(size=10),
        pull=[0.05 if domain_stats[d.domain_key].risk_events/max(domain_stats[d.domain_key].total_events,1) > 0.3 else 0 for d in domains],
    ))
    fig_pie.update_layout(
        height=340, margin=dict(l=10,r=10,t=30,b=30),
        paper_bgcolor="white",
        annotations=[dict(text=f"<b>{result.total_events:,}</b><br>eventos", x=0.5, y=0.5,
                          font_size=12, showarrow=False)],
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)
 
# ════════════════════════════════════════════════════════
# FILA 4: Heatmap de riesgo + Sunburst
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">🔥 Mapa de Riesgo y Estructura de Eventos</div>', unsafe_allow_html=True)
col_heat, col_sun = st.columns(2)
 
# ── GRÁFICO 7: Heatmap tasa de riesgo ────────────────────────────────────────
with col_heat:
    st.markdown("#### 🌡 Mapa de Calor — Tasa de Riesgo por Dominio")
    categories = ["Tasa Riesgo %","Score (inv.)","Eventos Críticos","Cobertura IPs"]
    dom_short = [d.domain_name.replace("Seguridad en ","").replace("Gestión de ","")[:18] for d in domains]
 
    heat_data = []
    for d in domains:
        ds = domain_stats[d.domain_key]
        rrate   = round(ds.risk_rate * 100, 1)
        inv_sc  = round(100 - d.raw_score, 1)
        crit    = min(100, ds.critical_events * 10)
        cov_ips = min(100, len(ds.unique_ips) * 5)
        heat_data.append([rrate, inv_sc, crit, cov_ips])
 
    df_heat = pd.DataFrame(heat_data, index=dom_short, columns=categories)
 
    fig_heat = go.Figure(go.Heatmap(
        z=df_heat.values.tolist(),
        x=categories, y=dom_short,
        colorscale=[
            [0.0,"#E8F5E9"],[0.25,"#FFF9C4"],[0.5,"#FFE0B2"],
            [0.75,"#FFCDD2"],[1.0,"#B71C1C"],
        ],
        text=[[f"{v:.0f}" for v in row] for row in df_heat.values.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Nivel<br>riesgo", tickfont=dict(size=9)),
    ))
    fig_heat.update_layout(
        height=330, margin=dict(l=10,r=10,t=20,b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickangle=-15, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("🔴 Rojo = mayor riesgo/exposición · 🟢 Verde = menor riesgo · Valores en escala 0–100")
 
# ── GRÁFICO 8: Sunburst eventos ───────────────────────────────────────────────
with col_sun:
    st.markdown("#### 🌞 Estructura Jerárquica de Eventos")
    sun_ids, sun_labels, sun_parents, sun_vals, sun_colors = [], [], [], [], []
 
    sun_ids.append("root"); sun_labels.append("Total\nEventos"); sun_parents.append("")
    sun_vals.append(result.total_events); sun_colors.append(C["primary"])
 
    for i, (key, d) in enumerate(zip(list(domain_stats.keys()), domains)):
        ds = domain_stats[key]
        if ds.total_events == 0: continue
        did = f"dom_{key}"
        sun_ids.append(did); sun_labels.append(d.domain_name.replace("Seguridad en ","Seg.\n").replace("Gestión de ","Gest.\n")[:20])
        sun_parents.append("root"); sun_vals.append(ds.total_events); sun_colors.append(C["domains"][i % len(C["domains"])])
 
        if ds.indicator_events > 0:
            sun_ids.append(f"{did}_ok"); sun_labels.append("Seguros")
            sun_parents.append(did); sun_vals.append(ds.indicator_events); sun_colors.append("#66BB6A")
        if ds.risk_events > 0:
            sun_ids.append(f"{did}_risk"); sun_labels.append("Riesgo")
            sun_parents.append(did); sun_vals.append(ds.risk_events); sun_colors.append("#EF5350")
 
    fig_sun = go.Figure(go.Sunburst(
        ids=sun_ids, labels=sun_labels, parents=sun_parents, values=sun_vals,
        marker=dict(colors=sun_colors, line=dict(width=1.5, color="white")),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Eventos: %{value:,}<extra></extra>",
        textfont=dict(size=10),
        insidetextorientation="radial",
    ))
    fig_sun.update_layout(
        height=350, margin=dict(l=0,r=0,t=10,b=10),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_sun, use_container_width=True)
    st.caption("🟢 Verde = eventos seguros · 🔴 Rojo = eventos de riesgo · Por dominio ISO 27001")
 
# ════════════════════════════════════════════════════════
# FILA 5: Histograma de niveles + Progresión
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📈 Distribución de Niveles y Análisis de Brechas</div>', unsafe_allow_html=True)
col_hist, col_prog = st.columns([1, 1.2])
 
# ── GRÁFICO 9: Histograma distribución de niveles por dominio ────────────────
with col_hist:
    st.markdown("#### 📊 Distribución de Dominios por Nivel COBIT")
    level_names = [f"Nivel {i}\n{MATURITY_LEVELS[i]['name'][:12]}" for i in range(6)]
    level_counts = [sum(1 for d in domains if d.level == i) for i in range(6)]
    level_pcts   = [c/len(domains)*100 for c in level_counts]
    bar_c        = [level_color(i) for i in range(6)]
 
    fig_hist = go.Figure(go.Bar(
        x=level_names, y=level_counts,
        marker_color=bar_c,
        text=[f"{p:.0f}%<br>({c} dom.)" for p,c in zip(level_pcts,level_counts)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Dominios: %{y}<br>%{text}<extra></extra>",
    ))
    fig_hist.update_layout(
        height=300, margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(title="N° de dominios", dtick=1, gridcolor="#F0F0F0", range=[0, len(domains)+0.5]),
        xaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
 
# ── GRÁFICO 10: Análisis de brecha — distancia a nivel 5 ─────────────────────
with col_prog:
    st.markdown("#### 🚀 Análisis de Brecha — Distancia al Nivel 5 (100 pts)")
    target = 100
    gap_names  = [d.domain_name.replace("Seguridad en ","Seg. ").replace("Gestión de ","Gest. ")[:26] for d in domains]
    gap_actual = [d.raw_score for d in domains]
    gap_needed = [max(0, target - d.raw_score) for d in domains]
 
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        name="Score actual", y=gap_names, x=gap_actual, orientation="h",
        marker_color=[level_color(d.level) for d in domains],
        hovertemplate="<b>%{y}</b><br>Score actual: %{x:.1f}<extra></extra>",
    ))
    fig_gap.add_trace(go.Bar(
        name="Brecha al Nivel 5", y=gap_names, x=gap_needed, orientation="h",
        marker_color="#ECEFF1",
        marker_line=dict(color="#B0BEC5", width=1),
        hovertemplate="<b>%{y}</b><br>Brecha: %{x:.1f} pts<extra></extra>",
    ))
    fig_gap.update_layout(
        barmode="stack", height=310,
        margin=dict(l=10,r=10,t=20,b=50),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        xaxis=dict(title="Puntos", range=[0,100], gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig_gap, use_container_width=True)
    st.caption(f"Brecha global al Nivel 5: **{100-result.overall_score:.1f} pts** — Score actual: {result.overall_score:.1f}/100")
 
# ════════════════════════════════════════════════════════
# Hallazgos y Recomendaciones
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">🚨 Hallazgos Críticos y Recomendaciones</div>', unsafe_allow_html=True)
col_find, col_rec = st.columns(2)
 
with col_find:
    st.markdown("#### ⚠ Hallazgos Críticos")
    if result.critical_findings:
        for f in result.critical_findings:
            st.markdown(f'<div class="finding">⚠ {f}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ Sin hallazgos críticos.")
 
with col_rec:
    st.markdown("#### 💡 Recomendaciones")
    for i, rec in enumerate(result.recommendations, 1):
        st.markdown(f'<div class="rec">{i}. {rec}</div>', unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════
# Tabla de resumen detallado
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📋 Tabla Resumen por Dominio</div>', unsafe_allow_html=True)
table_data = []
for key, d in result.domain_scores.items():
    ds = domain_stats[key]
    table_data.append({
        "Dominio": d.domain_name,
        "Cláusula": d.clause.split("–")[0].strip(),
        "Peso": f"{d.weight:.0%}",
        "Score": f"{d.raw_score:.1f}",
        "Nivel": f"{d.level} — {d.level_name}",
        "Total Eventos": ds.total_events,
        "Riesgo": ds.risk_events,
        "Tasa Riesgo": f"{ds.risk_rate:.1%}",
        "IPs Únicas": len(ds.unique_ips),
        "Usuarios": len(ds.unique_users),
    })
df_table = pd.DataFrame(table_data).sort_values("Score", ascending=False)
st.dataframe(df_table, use_container_width=True, hide_index=True)
 
# ════════════════════════════════════════════════════════
# Descargas
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">💾 Exportar Resultados</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)
 
with dl1:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
        export_html(result, source_label, tf.name)
        html_bytes = Path(tf.name).read_bytes(); os.unlink(tf.name)
    st.download_button("⬇ Descargar Reporte HTML", data=html_bytes,
        file_name="reporte_madurez_iso27001.html", mime="text/html", use_container_width=True, type="primary")
 
with dl2:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        export_json(result, tf.name)
        json_bytes = Path(tf.name).read_bytes(); os.unlink(tf.name)
    st.download_button("⬇ Descargar Datos JSON", data=json_bytes,
        file_name="resultado_madurez_iso27001.json", mime="application/json", use_container_width=True)
 
st.markdown(f"""
<footer>
  🛡 Evaluador de Madurez en Seguridad de la Información · ISO/IEC 27001:2013 · COBIT 5 · NTP ISO/IEC 27001:2008<br>
  Fuente analizada: <b>{source_label}</b> · Eventos procesados: <b>{result.total_events:,}</b>
</footer>
""", unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════════════
# ███████  SECCIÓN DEEP LEARNING  ████████████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════
 
st.divider()
st.markdown(
    '<div class="section-hdr" style="font-size:1.4rem;color:#6A1B9A;">'
    '🧠 Análisis con Deep Learning — Autoencoder + LSTM + MLP</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Los **3 modelos de Deep Learning** se entrenan en tiempo real sobre los "
    "logs analizados y enriquecen la evaluación de madurez con detección de "
    "anomalías, patrones temporales y clasificación neuronal."
)
 
# ── Arquitectura visual ────────────────────────────────────────────────────────
with st.expander("📐 Ver arquitectura de los modelos", expanded=False):
    arch_cols = st.columns(3)
    arch_info = [
        ("🔵 Autoencoder",      "63 → 32 → 16 → **8** → 16 → 32 → 63",
         "Reconstruye eventos normales.\nAlta pérdida = ANOMALÍA.",
         ["Entrada (63)","Dense 32","Dense 16","Bottleneck 8","Dense 16","Dense 32","Salida (63)"],
         "#1565C0"),
        ("🟣 LSTM Bidireccional","(20×13) → BiLSTM(32) → LSTM(16) → Dense(8) → **sigmoid**",
         "Analiza secuencias de 20 eventos.\nDetecta patrones de ataque temporal.",
         ["Seq (20,13)","BiLSTM 32","LSTM 16","Dense 8","Prob amenaza"],
         "#6A1B9A"),
        ("🟠 MLP Clasificador",  "24 → 64 → 32 → 16 → **softmax(6)**",
         "Clasifica el nivel de madurez\nISO 27001 (0–5) directamente.",
         ["Features (24)","Dense 64","Dense 32","Dense 16","Softmax (6)"],
         "#E65100"),
    ]
    for col, (title, arch, desc, layers_list, color) in zip(arch_cols, arch_info):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
            # Mini diagrama de capas
            for i, lyr in enumerate(layers_list):
                is_bottleneck = "8" in lyr or "sigmoid" in lyr or "Softmax" in lyr
                bg = color if is_bottleneck else color + "33"
                fc = "white" if is_bottleneck else color
                st.markdown(
                    f'<div style="background:{bg};color:{fc};border:1px solid {color};'
                    f'border-radius:6px;padding:4px 8px;text-align:center;'
                    f'margin-bottom:4px;font-size:.82em;font-weight:600">{lyr}</div>',
                    unsafe_allow_html=True,
                )
                if i < len(layers_list) - 1:
                    st.markdown(f'<div style="text-align:center;color:{color};margin:-2px 0">▼</div>',
                                unsafe_allow_html=True)
 
# ── Entrenamiento ──────────────────────────────────────────────────────────────
st.markdown("#### ⚙ Entrenamiento")
dl_col1, dl_col2, dl_col3, dl_col4 = st.columns([1,1,1,1])
with dl_col1:
    ae_epochs   = st.slider("Épocas Autoencoder",   5, 50, 25, 5)
with dl_col2:
    lstm_epochs = st.slider("Épocas LSTM",           5, 40, 20, 5)
with dl_col3:
    mlp_epochs  = st.slider("Épocas MLP Clasificador", 10, 60, 35, 5)
with dl_col4:
    st.markdown("<br>", unsafe_allow_html=True)
    run_dl = st.button("🚀 Entrenar y Analizar con DL", type="primary", use_container_width=True)
 
if run_dl or "dl_result" in st.session_state:
    if run_dl:
        # Importar aquí para no ralentizar el arranque
        from ml.dl_pipeline import DLPipeline
        from rules.iso27001_controls import MATURITY_LEVELS as ML
 
        prog_bar = st.progress(0, text="Inicializando modelos…")
 
        @st.cache_resource(show_spinner=False)
        def get_pipeline():
            return DLPipeline()
 
        pipeline = get_pipeline()
        pipeline._trained = False   # forzar reentrenamiento con nuevos hiperparámetros
 
        prog_bar.progress(10, text="🔵 Entrenando Autoencoder…")
        pipeline.autoencoder = __import__('ml.autoencoder_model', fromlist=['LogAutoencoder']).LogAutoencoder()
        pipeline.autoencoder.fit(entries, epochs=ae_epochs, verbose=0)
 
        prog_bar.progress(40, text="🟣 Entrenando LSTM Bidireccional…")
        from ml.lstm_model import LSTMThreatDetector
        from ml.dl_pipeline import _separate_normal_attack, _augment_attack_entries
        pipeline.lstm = LSTMThreatDetector()
        pipeline.lstm.extractor = pipeline.autoencoder.extractor
        pipeline.lstm.extractor._fitted = True
        normal_e, attack_e = _separate_normal_attack(entries)
        if len(attack_e) < 30:
            attack_e = _augment_attack_entries(attack_e, normal_e)
        pipeline.lstm.fit(normal_e, attack_e, epochs=lstm_epochs, verbose=0)
 
        prog_bar.progress(70, text="🟠 Entrenando MLP Clasificador…")
        from ml.maturity_classifier import MaturityClassifier
        pipeline.classifier = MaturityClassifier()
        pipeline.classifier.fit(epochs=mlp_epochs, verbose=0)
        pipeline._trained = True
 
        prog_bar.progress(90, text="📊 Calculando predicciones…")
        dl_res = pipeline.run(entries, domain_stats, result)
        st.session_state["dl_result"]  = dl_res
        st.session_state["dl_pipeline"] = pipeline
        prog_bar.progress(100, text="✅ Listo")
        prog_bar.empty()
    else:
        dl_res   = st.session_state["dl_result"]
 
    from rules.iso27001_controls import MATURITY_LEVELS as ML
 
    # ── KPIs Deep Learning ────────────────────────────────────────────────────
    st.markdown("---")
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis_dl = [
        (f"{dl_res.anomaly_rate:.1f}%",      "TASA DE ANOMALÍAS (AE)",    "#1565C0"),
        (f"{dl_res.threat_level['mean_threat_prob']:.1%}", "PROB. AMENAZA MEDIA (LSTM)", "#6A1B9A"),
        (f"Nivel {dl_res.dl_predicted_level}","NIVEL DL (MLP)",            "#E65100"),
        (f"{dl_res.dl_confidence:.1f}%",      "CONFIANZA MLP",             "#00695C"),
        ("✅ Sí" if dl_res.agreement else "⚠ No", "ACUERDO REGLAS vs DL", "#2E7D32" if dl_res.agreement else "#C62828"),
    ]
    for col,(val,lbl,color) in zip([k1,k2,k3,k4,k5], kpis_dl):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-val" style="color:{color}">{val}</div>'
                f'<div class="kpi-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
 
    # ════════════════════════════════════════════════════════
    # DL FILA 1: Curvas de entrenamiento
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr" style="color:#6A1B9A">📉 Curvas de Entrenamiento</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
 
    def plot_loss_curve(train_loss, val_loss, train_acc, val_acc, title, color):
        epochs_ax = list(range(1, len(train_loss)+1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_ax, y=train_loss, name="Train Loss",
            line=dict(color=color, width=2), mode="lines"))
        if val_loss:
            fig.add_trace(go.Scatter(x=epochs_ax, y=val_loss[:len(epochs_ax)], name="Val Loss",
                line=dict(color=color, width=2, dash="dot"), mode="lines"))
        if train_acc:
            fig2_ax = go.Scatter(x=epochs_ax, y=[a*max(train_loss) for a in train_acc],
                name="Train Acc (escalada)", line=dict(color="#FFA726", width=1.5, dash="dash"),
                mode="lines", yaxis="y2", visible="legendonly")
            fig.add_trace(fig2_ax)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color=color)),
            height=240, margin=dict(l=10,r=10,t=40,b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
            xaxis=dict(title="Época", gridcolor="#F0F0F0"),
            yaxis=dict(title="Pérdida", gridcolor="#F0F0F0"),
        )
        return fig
 
    with tc1:
        st.markdown("**🔵 Autoencoder**")
        fig = plot_loss_curve(dl_res.ae_train_loss, dl_res.ae_val_loss, [], [], "Pérdida AE (MSE)", "#1565C0")
        st.plotly_chart(fig, use_container_width=True)
        sm = dl_res.ae_summary
        st.caption(f"Parámetros: {sm['parameters']:,} · Épocas: {sm['epochs_trained']} · Loss final: {sm['final_train_loss']}")
 
    with tc2:
        st.markdown("**🟣 LSTM Bidireccional**")
        fig = plot_loss_curve(dl_res.lstm_train_loss, dl_res.lstm_val_loss,
                               dl_res.lstm_train_acc, dl_res.lstm_val_acc,
                               "Pérdida LSTM (Binary CE)", "#6A1B9A")
        st.plotly_chart(fig, use_container_width=True)
        sm = dl_res.lstm_summary
        acc = f"{sm['final_val_accuracy']:.1%}" if sm.get('final_val_accuracy') else "N/A"
        st.caption(f"Parámetros: {sm['parameters']:,} · Épocas: {sm['epochs_trained']} · Acc val: {acc}")
 
    with tc3:
        st.markdown("**🟠 MLP Clasificador**")
        fig = plot_loss_curve(dl_res.mlp_train_loss, dl_res.mlp_val_loss,
                               dl_res.mlp_train_acc, dl_res.mlp_val_acc,
                               "Pérdida MLP (Categorical CE)", "#E65100")
        st.plotly_chart(fig, use_container_width=True)
        sm = dl_res.mlp_summary
        acc = f"{sm['final_val_accuracy']:.1%}" if sm.get('final_val_accuracy') else "N/A"
        st.caption(f"Parámetros: {sm['parameters']:,} · Épocas: {sm['epochs_trained']} · Acc val: {acc}")
 
    # ════════════════════════════════════════════════════════
    # DL FILA 2: Autoencoder — distribución de errores + anomalías
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr" style="color:#1565C0">🔵 Autoencoder — Detección de Anomalías</div>', unsafe_allow_html=True)
    ae1, ae2 = st.columns(2)
 
    with ae1:
        st.markdown("#### Distribución del Error de Reconstrucción")
        scores_norm = dl_res.anomaly_scores
        normal_scores = scores_norm[~dl_res.is_anomaly]
        anom_scores   = scores_norm[dl_res.is_anomaly]
 
        fig_hist_ae = go.Figure()
        if len(normal_scores):
            fig_hist_ae.add_trace(go.Histogram(
                x=normal_scores, name="Eventos Normales",
                marker_color=hex_rgba("#2E7D32", 0.7), nbinsx=40,
                hovertemplate="Score: %{x:.1f}<br>Eventos: %{y}<extra>Normal</extra>",
            ))
        if len(anom_scores):
            fig_hist_ae.add_trace(go.Histogram(
                x=anom_scores, name="Anomalías Detectadas",
                marker_color=hex_rgba("#C62828", 0.7), nbinsx=40,
                hovertemplate="Score: %{x:.1f}<br>Eventos: %{y}<extra>Anomalía</extra>",
            ))
        fig_hist_ae.add_vline(x=50, line_dash="dash", line_color="#F57F17",
                               annotation_text="Umbral (P95)", line_width=2)
        fig_hist_ae.update_layout(
            barmode="overlay", height=280,
            margin=dict(l=10,r=10,t=10,b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(title="Score de Anomalía (0–100)", gridcolor="#F0F0F0"),
            yaxis=dict(title="N° eventos", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_hist_ae, use_container_width=True)
 
    with ae2:
        st.markdown("#### Timeline de Anomalías Detectadas")
        step = max(1, len(scores_norm) // 200)
        idx_plot = list(range(0, len(scores_norm), step))
        scores_plot = scores_norm[idx_plot]
        colors_plot = ["#C62828" if s >= 50 else "#2E7D32" for s in scores_plot]
 
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=idx_plot, y=scores_plot.tolist(),
            mode="markers", name="Score por evento",
            marker=dict(color=colors_plot, size=4, opacity=0.7),
            hovertemplate="Evento #%{x}<br>Score: %{y:.1f}<extra></extra>",
        ))
        fig_time.add_hline(y=50, line_dash="dash", line_color="#F57F17",
                            annotation_text="Umbral anomalía")
        fig_time.update_layout(
            height=280, margin=dict(l=10,r=10,t=10,b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="N° evento", gridcolor="#F0F0F0"),
            yaxis=dict(title="Score anomalía (0–100)", range=[0,105], gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_time, use_container_width=True)
 
    st.info(
        f"🔵 **Autoencoder:** {dl_res.anomaly_rate:.1f}% de eventos clasificados como "
        f"anomalías ({int(dl_res.is_anomaly.sum()):,} de {len(dl_res.is_anomaly):,}). "
        f"Umbral automático (P95): {dl_res.autoencoder_threshold:.6f}"
    )
 
    # ════════════════════════════════════════════════════════
    # DL FILA 3: LSTM — probabilidades de amenaza
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr" style="color:#6A1B9A">🟣 LSTM — Detección Temporal de Amenazas</div>', unsafe_allow_html=True)
    ls1, ls2 = st.columns(2)
 
    with ls1:
        st.markdown("#### Probabilidad de Amenaza por Ventana de 20 Eventos")
        tp = dl_res.threat_probs
        step2 = max(1, len(tp)//150)
        tp_plot = tp[::step2]
        col_tp = ["#C62828" if p>=0.75 else "#F57F17" if p>=0.5 else "#2E7D32" for p in tp_plot]
 
        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Bar(
            x=list(range(len(tp_plot))), y=tp_plot.tolist(),
            marker_color=col_tp, name="Prob. amenaza",
            hovertemplate="Ventana %{x}<br>Prob: %{y:.3f}<extra></extra>",
        ))
        fig_lstm.add_hline(y=0.75, line_dash="dash", line_color="#C62828",
                            annotation_text="Alto riesgo")
        fig_lstm.add_hline(y=0.50, line_dash="dot",  line_color="#F57F17",
                            annotation_text="Riesgo medio")
        fig_lstm.update_layout(
            height=280, margin=dict(l=10,r=10,t=10,b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Ventana temporal", gridcolor="#F0F0F0"),
            yaxis=dict(title="Probabilidad", range=[0,1.05], gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_lstm, use_container_width=True)
 
    with ls2:
        st.markdown("#### Distribución de Niveles de Amenaza")
        tl = dl_res.threat_level
        labels_t = ["🟢 Bajo (<50%)", "🟡 Medio (50–75%)", "🔴 Alto (>75%)"]
        vals_t   = [tl["pct_low_threat"], tl["pct_medium_threat"], tl["pct_high_threat"]]
        fig_donut = go.Figure(go.Pie(
            labels=labels_t, values=vals_t,
            marker=dict(colors=["#2E7D32","#F57F17","#C62828"],
                        line=dict(color="white", width=2)),
            hole=0.5,
            hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
            textinfo="percent+label", textfont=dict(size=11),
        ))
        fig_donut.update_layout(
            height=280, margin=dict(l=10,r=10,t=10,b=30),
            paper_bgcolor="white", showlegend=False,
            annotations=[dict(text=f"{tl['mean_threat_prob']:.1%}<br>media", x=0.5, y=0.5,
                              font_size=12, showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
 
    st.info(
        f"🟣 **LSTM:** Prob. amenaza máxima detectada: **{tl['max_threat_prob']:.1%}** · "
        f"Ventanas de alto riesgo: **{tl['pct_high_threat']:.1f}%** · "
        f"Secuencias analizadas: **{tl['total_sequences']:,}**"
    )
 
    # ════════════════════════════════════════════════════════
    # DL FILA 4: MLP — predicción de madurez + comparativa
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr" style="color:#E65100">🟠 MLP — Clasificación de Nivel de Madurez</div>', unsafe_allow_html=True)
    ml1, ml2 = st.columns(2)
 
    with ml1:
        st.markdown("#### Probabilidades por Nivel de Madurez (MLP)")
        probs_dict = dl_res.dl_probabilities
        niveles_lbl = [f"Nivel {i}\n{ML[i]['name'][:10]}" for i in range(6)]
        probs_vals  = [probs_dict.get(i, 0) for i in range(6)]
        bar_col_mlp = [level_color(i) for i in range(6)]
 
        fig_mlp = go.Figure(go.Bar(
            x=niveles_lbl, y=probs_vals,
            marker_color=bar_col_mlp,
            text=[f"{v:.1f}%" for v in probs_vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Probabilidad: %{y:.1f}%<extra></extra>",
        ))
        pred_lvl = dl_res.dl_predicted_level
        fig_mlp.add_vline(x=pred_lvl, line_color=level_color(pred_lvl),
                           line_width=3, line_dash="dash",
                           annotation_text=f"▲ Predicción: Nivel {pred_lvl}",
                           annotation_font_color=level_color(pred_lvl))
        fig_mlp.update_layout(
            height=300, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title="Probabilidad (%)", range=[0,105], gridcolor="#F0F0F0"),
            xaxis=dict(tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig_mlp, use_container_width=True)
 
    with ml2:
        st.markdown("#### 🆚 Comparativa: Sistema de Reglas vs Deep Learning")
        rule_lvl  = dl_res.rule_based_level
        dl_lvl    = dl_res.dl_predicted_level
        adj_score = dl_res.dl_adjusted_score
 
        compare_data = {
            "Método": ["Sistema de Reglas\n(ISO 27001)", "MLP — Deep Learning\n(Clasificador neuronal)", "Score Ajustado DL\n(con penalización AE)"],
            "Nivel":  [rule_lvl, dl_lvl, int(adj_score / 20)],
            "Score":  [dl_res.rule_based_score, dl_res.dl_confidence, adj_score],
            "Color":  [level_color(rule_lvl), level_color(dl_lvl), level_color(int(adj_score/20))],
        }
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=compare_data["Método"],
            y=compare_data["Score"],
            marker_color=compare_data["Color"],
            text=[f"Nivel {l}<br>{s:.1f}" for l,s in zip(compare_data["Nivel"], compare_data["Score"])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
        ))
        acuerdo_txt = "✅ Ambos métodos coinciden" if dl_res.agreement else "⚠ Métodos difieren — revisar"
        acuerdo_color = "#2E7D32" if dl_res.agreement else "#C62828"
        fig_comp.update_layout(
            height=300, margin=dict(l=10,r=10,t=20,b=60),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title="Score / Confianza (%)", range=[0,115], gridcolor="#F0F0F0"),
            xaxis=dict(tickfont=dict(size=9)),
            annotations=[dict(text=acuerdo_txt, x=0.5, y=-0.25, xref="paper", yref="paper",
                              font=dict(color=acuerdo_color, size=12), showarrow=False)],
            showlegend=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
 
    # ── Tabla resumen modelos DL ───────────────────────────────────────────────
    st.markdown("#### 📋 Resumen de los Modelos Entrenados")
    sm_ae   = dl_res.ae_summary
    sm_lstm = dl_res.lstm_summary
    sm_mlp  = dl_res.mlp_summary
    df_models = pd.DataFrame([
        {"Modelo": "🔵 Autoencoder",
         "Arquitectura": sm_ae.get("architecture",""),
         "Parámetros": f"{sm_ae.get('parameters',0):,}",
         "Épocas": sm_ae.get("epochs_trained",""),
         "Loss final (train)": sm_ae.get("final_train_loss",""),
         "Loss final (val)":   sm_ae.get("final_val_loss",""),
         "Métrica clave": f"Tasa anomalías: {dl_res.anomaly_rate:.1f}%"},
        {"Modelo": "🟣 LSTM Bidireccional",
         "Arquitectura": sm_lstm.get("architecture",""),
         "Parámetros": f"{sm_lstm.get('parameters',0):,}",
         "Épocas": sm_lstm.get("epochs_trained",""),
         "Loss final (train)": sm_lstm.get("final_train_loss",""),
         "Loss final (val)":   sm_lstm.get("final_val_loss",""),
         "Métrica clave": f"Prob. amenaza media: {dl_res.threat_level['mean_threat_prob']:.1%}"},
        {"Modelo": "🟠 MLP Clasificador",
         "Arquitectura": sm_mlp.get("architecture",""),
         "Parámetros": f"{sm_mlp.get('parameters',0):,}",
         "Épocas": sm_mlp.get("epochs_trained",""),
         "Loss final (train)": sm_mlp.get("final_train_loss",""),
         "Loss final (val)":   sm_mlp.get("final_val_loss",""),
         "Métrica clave": f"Nivel predicho: {dl_res.dl_predicted_level} ({dl_res.dl_confidence:.1f}% confianza)"},
    ])
    st.dataframe(df_models, use_container_width=True, hide_index=True)
 
    st.success(
        f"🧠 **Análisis Deep Learning completado** · "
        f"Total parámetros entrenados: "
        f"**{sm_ae.get('parameters',0)+sm_lstm.get('parameters',0)+sm_mlp.get('parameters',0):,}** · "
        f"Score ajustado por DL: **{dl_res.dl_adjusted_score:.1f}/100**"
    )
else:
    st.info("👆 Configura las épocas y presiona **'Entrenar y Analizar con DL'** para activar el análisis neuronal.")
