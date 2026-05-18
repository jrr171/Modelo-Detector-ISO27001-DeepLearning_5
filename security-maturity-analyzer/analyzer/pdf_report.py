"""
PDF Report Generator
Genera un reporte PDF profesional del análisis de madurez ISO 27001.
Usa fpdf2 (puro Python, sin dependencias de sistema) + matplotlib para gráficos.
"""

import io
import math
import tempfile
import os
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from fpdf import FPDF
from fpdf.enums import XPos, YPos

def _clean(text: str) -> str:
    """Strip non-latin-1 characters for fpdf2 compatibility."""
    replacements = {
        '—':'-','–':'-','×':'x','·':'.',
        '◄':'<<','►':'>>','▲':'^','▼':'v',
        '•':'-','’':"'",'“':'"','”':'"',
        '‘':"'",'…':'...',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return ''.join(c if ord(c) < 256 else '?' for c in text)


from analyzer.maturity_scorer  import MaturityResult
from analyzer.event_classifier import DomainStats
from rules.iso27001_controls   import MATURITY_LEVELS, ISO27001_DOMAINS


# ?? Color palette ?????????????????????????????????????????????????????????????
LEVEL_COLORS_HEX = {
    0: "#C62828", 1: "#D32F2F", 2: "#F57C00",
    3: "#F9A825", 4: "#388E3C", 5: "#1B5E20",
}
LEVEL_COLORS_RGB = {
    0: (198,40,40),  1: (211,47,47),  2: (245,124,0),
    3: (249,168,37), 4: (56,142,60),  5: (27,94,32),
}
PRIMARY_RGB  = (21, 101, 192)
DARK_RGB     = (33, 33, 33)
GRAY_RGB     = (84, 110, 122)
LIGHT_RGB    = (227, 242, 253)


def _hex_to_rgb(hx: str):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))


def _score_color(s: float):
    if s >= 81: return LEVEL_COLORS_RGB[5]
    if s >= 61: return LEVEL_COLORS_RGB[4]
    if s >= 41: return LEVEL_COLORS_RGB[3]
    if s >= 21: return LEVEL_COLORS_RGB[2]
    if s > 0:   return LEVEL_COLORS_RGB[1]
    return LEVEL_COLORS_RGB[0]


# ?? Chart generators ??????????????????????????????????????????????????????????

def _make_radar_chart(result: MaturityResult) -> str:
    domains     = list(result.domain_scores.values())
    scores      = [d.raw_score for d in domains]
    labels      = [d.domain_name.replace("Seguridad en ","Seg.\n").replace("Gestión de ","Gest.\n")[:20] for d in domains]
    n           = len(scores)
    angles      = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles_c    = angles + [angles[0]]
    scores_c    = scores + [scores[0]]

    lc  = LEVEL_COLORS_HEX.get(result.overall_level, "#555")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Reference rings
    for val, col in [(20,"#EF5350"),(40,"#FF9800"),(60,"#FDD835"),(80,"#66BB6A")]:
        ax.plot(angles_c, [val]*n+[val], linestyle=":", color=col, linewidth=1.0, alpha=0.6)

    ax.plot(angles_c, scores_c, "o-", linewidth=2.5, color=lc)
    ax.fill(angles_c, scores_c, alpha=0.25, color=lc)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9, fontfamily="DejaVu Sans")
    ax.set_ylim(0, 100)
    ax.set_yticks([20,40,60,80,100])
    ax.set_yticklabels(["20","40","60","80","100"], fontsize=8)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    for angle, val, d in zip(angles, scores, domains):
        ax.text(angle, val+6, f"{val:.0f}", ha="center", va="bottom",
                fontsize=9, color=lc, fontweight="bold")

    ax.set_title(
        f"Nivel {result.overall_level} - {result.overall_level_name}\n"
        f"Score global: {result.overall_score:.1f}/100",
        fontsize=11, fontweight="bold", color=lc, pad=20,
    )
    fig.patch.set_facecolor("white")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def _make_bar_chart(result: MaturityResult, domain_stats: Dict[str, DomainStats]) -> str:
    domains = list(result.domain_scores.values())
    names   = [d.domain_name.replace("Seguridad en ","Seg. ").replace("Gestión de ","Gest. ")[:22] for d in domains]
    scores  = [d.raw_score for d in domains]
    colors  = [LEVEL_COLORS_HEX.get(d.level,"#555") for d in domains]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.6)
    for val, bar in zip(scores, bars):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}", va="center", ha="left", fontsize=9, fontweight="bold")
    for x, lbl, col in [(20,"N1","#EF5350"),(40,"N2","#FF9800"),(60,"N3","#FDD835"),(80,"N4","#66BB6A")]:
        ax.axvline(x, color=col, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(x+0.5, -0.7, lbl, fontsize=7.5, color=col)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (0-100)", fontsize=10)
    ax.set_title("Score por Dominio ISO 27001", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def _make_risk_chart(domain_stats: Dict[str, DomainStats]) -> str:
    keys   = list(ISO27001_DOMAINS.keys())
    safe   = [domain_stats[k].indicator_events for k in keys]
    risk   = [domain_stats[k].risk_events for k in keys]
    names  = [ISO27001_DOMAINS[k].name.replace("Seguridad en ","Seg. ").replace("Gestión de ","Gest. ")[:20] for k in keys]

    x   = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x-w/2, safe, w, label="Eventos seguros",   color="#2E7D32", alpha=0.85, edgecolor="white")
    ax.bar(x+w/2, risk, w, label="Eventos de riesgo", color="#C62828", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("N° de eventos", fontsize=10)
    ax.set_title("Eventos Seguros vs. de Riesgo por Dominio", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white"); fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); return tmp.name


# ?? PDF Class ?????????????????????????????????????????????????????????????????

class MaturityPDF(FPDF):
    def __init__(self, source_label: str):
        super().__init__()
        self.source_label = source_label
        self.set_auto_page_break(auto=True, margin=15)


    def normalize_text(self, text):
        """Auto-clean non-latin1 chars before any PDF output."""
        return super().normalize_text(_clean(str(text)))

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*PRIMARY_RGB)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 2)
        self.cell(0, 8, "Evaluador de Madurez en Seguridad de la Información ISO 27001", align="L")
        self.set_xy(-30, 2)
        self.cell(20, 8, f"Pág. {self.page_no()}", align="R")
        self.set_text_color(*DARK_RGB)
        self.ln(14)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_fill_color(*PRIMARY_RGB)
        self.rect(0, 285, 210, 12, "F")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(200, 220, 255)
        self.set_xy(10, 287)
        self.cell(0, 6, "ISO/IEC 27001:2013 * COBIT 5 * NTP ISO/IEC 27001:2008 * Deep Learning: Autoencoder + LSTM + MLP", align="L")

    def section_title(self, text: str):
        self.set_fill_color(*LIGHT_RGB)
        self.set_draw_color(*PRIMARY_RGB)
        self.set_text_color(*PRIMARY_RGB)
        self.set_font("Helvetica", "B", 12)
        self.set_line_width(0.8)
        self.cell(0, 9, text, border="B", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK_RGB)
        self.set_line_width(0.2)
        self.ln(3)

    def body_text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*DARK_RGB)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def colored_cell(self, text: str, w: float, h: float, rgb, bold=False, center=False):
        self.set_fill_color(*rgb)
        self.set_text_color(255, 255, 255) if sum(rgb) < 400 else self.set_text_color(*DARK_RGB)
        self.set_font("Helvetica", "B" if bold else "", 9)
        self.cell(w, h, text, fill=True, border=1,
                  align="C" if center else "L",
                  new_x=XPos.RIGHT, new_y=YPos.LAST)
        self.set_text_color(*DARK_RGB)


# ?? Main generator ????????????????????????????????????????????????????????????

def generate_pdf(
    result: MaturityResult,
    domain_stats: Dict[str, DomainStats],
    source_label: str,
    action_plan: List[dict],
) -> bytes:
    """Generate complete PDF report and return as bytes."""

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lvl = result.overall_level
    lvl_info = MATURITY_LEVELS[lvl]
    lc_rgb = LEVEL_COLORS_RGB[lvl]

    # Pre-generate charts
    radar_path = _make_radar_chart(result)
    bar_path   = _make_bar_chart(result, domain_stats)
    risk_path  = _make_risk_chart(domain_stats)

    pdf = MaturityPDF(source_label)

    # ?? COVER PAGE ?????????????????????????????????????????????????????????????
    pdf.add_page()
    # Header bar
    pdf.set_fill_color(*PRIMARY_RGB)
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_xy(0, 5)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 15, "REPORTE DE MADUREZ EN SEGURIDAD", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(210, 8, "Evaluador ISO 27001 * COBIT * Deep Learning", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(210, 8, "Empresa de Inteligencia Comercial * Sector Comercio Exterior", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*DARK_RGB)

    # Big level badge
    pdf.ln(12)
    pdf.set_fill_color(*lc_rgb)
    pdf.set_font("Helvetica", "B", 48)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(60)
    pdf.cell(90, 25, f"NIVEL {lvl}", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(60)
    pdf.cell(90, 10, lvl_info["name"], fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*DARK_RGB)
    pdf.ln(6)

    # Score + metadata table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(*LIGHT_RGB)
    fields = [
        ("Score Global",        f"{result.overall_score:.1f} / 100 puntos"),
        ("Nivel COBIT",         f"Nivel {lvl} - {lvl_info['name']}"),
        ("Eventos procesados",  f"{result.total_events:,}"),
        ("Eventos de riesgo",   f"{result.total_risk_events:,}  ({result.total_risk_events/max(result.total_events,1)*100:.1f}%)"),
        ("Dominios activos",    f"{result.total_domains_active} de 6 dominios ISO 27001"),
        ("Fuente analizada",    _clean(source_label[:60])),
        ("Fecha del reporte",   now),
    ]
    for label, value in fields:
        pdf.set_fill_color(*LIGHT_RGB)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 8, f"  {label}", fill=True, border=1, new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 8, _clean(f"  {value}"), fill=True, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Description
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY_RGB)
    pdf.multi_cell(0, 5, lvl_info["description"])
    pdf.set_text_color(*DARK_RGB)

    # Level scale
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Escala de madurez COBIT:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    for i in range(6):
        info = MATURITY_LEVELS[i]
        lo, hi = info["range"]
        rng = f"{lo}-{hi}%" if i > 0 else "0%"
        rgb = LEVEL_COLORS_RGB[i]
        marker = "  << NIVEL ACTUAL" if i == lvl else ""
        pdf.set_fill_color(*rgb)
        pdf.set_font("Helvetica", "B" if i == lvl else "", 9)
        pdf.set_text_color(255, 255, 255) if sum(rgb) < 450 else pdf.set_text_color(*DARK_RGB)
        pdf.cell(190, 6.5, f"  Nivel {i} * {rng} * {info['name']}{marker}",
                 fill=True, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK_RGB)

    # ?? PAGE 2: CHARTS ?????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.section_title("1. Radar de Madurez por Dominio ISO 27001")
    pdf.body_text(
        "El gráfico radar muestra el perfil de madurez de la organización en los 6 dominios "
        "de control ISO/IEC 27001:2013. Cada vértice representa el score (0-100) de un dominio. "
        "Los anillos discontinuos indican los umbrales de los niveles COBIT 1 al 4."
    )
    pdf.image(radar_path, x=30, w=150)
    pdf.ln(4)

    pdf.section_title("2. Score por Dominio - Barras Horizontales")
    pdf.body_text(
        "Las barras muestran el score individual de cada dominio. "
        "Las líneas verticales discontinuas indican los umbrales de nivel COBIT."
    )
    pdf.image(bar_path, x=10, w=185)

    # ?? PAGE 3: DOMAIN TABLE + RISK CHART ??????????????????????????????????????
    pdf.add_page()
    pdf.section_title("3. Tabla Detallada por Dominio ISO 27001")

    # Table header
    col_w = [55, 20, 28, 22, 22, 22, 21]
    headers = ["Dominio", "Peso", "Score", "Nivel", "Eventos", "Riesgo", "% Riesgo"]
    pdf.set_fill_color(*PRIMARY_RGB)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 8, h, fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
    pdf.ln()
    pdf.set_text_color(*DARK_RGB)

    for i, (key, ds) in enumerate(result.domain_scores.items()):
        raw_ds = domain_stats[key]
        risk_pct = raw_ds.risk_rate * 100
        rgb_row = LIGHT_RGB if i % 2 == 0 else (245, 245, 245)
        rgb_lvl = LEVEL_COLORS_RGB[ds.level]

        pdf.set_fill_color(*rgb_row)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(col_w[0], 7.5, f"  {ds.domain_name[:30]}", fill=True, border=1, new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.cell(col_w[1], 7.5, f"{ISO27001_DOMAINS[key].weight:.0%}", fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        # Score cell with level color
        pdf.set_fill_color(*rgb_lvl)
        pdf.set_text_color(255, 255, 255) if sum(rgb_lvl) < 450 else pdf.set_text_color(*DARK_RGB)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(col_w[2], 7.5, f"{ds.raw_score:.1f}", fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_fill_color(*rgb_row); pdf.set_text_color(*DARK_RGB); pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(col_w[3], 7.5, f"Nv. {ds.level}", fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.cell(col_w[4], 7.5, str(raw_ds.total_events), fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.cell(col_w[5], 7.5, str(raw_ds.risk_events), fill=True, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        risk_rgb = (198,40,40) if risk_pct>30 else (230,81,0) if risk_pct>10 else (46,125,50)
        pdf.set_fill_color(*risk_rgb)
        pdf.set_text_color(255,255,255) if risk_pct>10 else pdf.set_text_color(*DARK_RGB)
        pdf.set_font("Helvetica","B",8.5)
        pdf.cell(col_w[6], 7.5, f"{risk_pct:.1f}%", fill=True, border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK_RGB)

    pdf.ln(8)
    pdf.section_title("4. Eventos Seguros vs. Eventos de Riesgo")
    pdf.body_text("Comparativa del volumen de eventos seguros e indicadores de riesgo por cada dominio ISO 27001.")
    pdf.image(risk_path, x=10, w=185)

    # ?? PAGE 4: CRITICAL FINDINGS + ACTION PLAN ????????????????????????????????
    pdf.add_page()

    if result.critical_findings:
        pdf.section_title("5. Hallazgos Críticos")
        pdf.set_fill_color(255, 235, 238)
        for finding in result.critical_findings:
            pdf.set_fill_color(255, 235, 238)
            pdf.set_font("Helvetica", "", 9.5)
            pdf.cell(5, 6.5, "", new_x=XPos.RIGHT, new_y=YPos.LAST)
            pdf.multi_cell(0, 6.5, _clean(f"!  {finding}"), fill=True)
            pdf.ln(1)
        pdf.ln(4)

    pdf.section_title("6. Plan de Acción Prioritizado")
    pdf.body_text("Acciones ordenadas de mayor a menor urgencia según el score de cada dominio.")
    pdf.ln(2)

    for item in action_plan:
        effort_rgb = {"Bajo":(46,125,50), "Medio":(230,81,0), "Alto":(198,40,40)}.get(item["effort"], GRAY_RGB)
        pdf.set_fill_color(*LIGHT_RGB)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"  {item['priority']}. {item['domain']}  -  Score: {item['score']:.1f}/100",
                 fill=True, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for action in item["actions"]:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(6, 6, "", new_x=XPos.RIGHT, new_y=YPos.LAST)
            pdf.multi_cell(180, 6, f"-  {action}")
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(*effort_rgb)
        pdf.set_text_color(255, 255, 255) if sum(effort_rgb) < 400 else pdf.set_text_color(*DARK_RGB)
        pdf.cell(40, 6, f"  Esfuerzo: {item['effort']}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK_RGB)
        pdf.ln(4)

    # ?? PAGE 5: RECOMMENDATIONS ????????????????????????????????????????????????
    pdf.add_page()
    pdf.section_title("7. Recomendaciones por Nivel de Madurez")
    pdf.body_text(f"Recomendaciones específicas para avanzar desde el Nivel {lvl} ({lvl_info['name']}) hacia el siguiente nivel:")
    pdf.ln(2)
    for i, rec in enumerate(result.recommendations, 1):
        pdf.set_fill_color(*LIGHT_RGB)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(8, 7, f" {i}.", fill=True, new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 7, _clean(f" {rec}"), fill=True)
        pdf.ln(1)

    pdf.ln(10)
    pdf.set_fill_color(*PRIMARY_RGB)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(200, 220, 255)
    pdf.multi_cell(0, 5,
        f"Reporte generado el {now}  |  Fuente: {_clean(source_label[:80])}\n"
        "ISO/IEC 27001:2013 * COBIT 5 * NTP ISO/IEC 27001:2008 * "
        "Deep Learning: Autoencoder + Detector de Secuencias + MLP Clasificador",
        align="C",
    )

    # Cleanup temp files
    for p in [radar_path, bar_path, risk_path]:
        try: os.unlink(p)
        except: pass

    return bytes(pdf.output())
