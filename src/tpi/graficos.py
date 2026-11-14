"""Estilo gráfico común (notebook, informe y app) con la paleta validada de referencia.

Reglas: colores categóricos en orden fijo (nunca ciclados), una sola escala Y por panel,
marcas finas, grilla recesiva, texto en tintas neutras (nunca del color de la serie).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from tpi.config import FIGURES_DIR, ROOT

# Paleta categórica (modo claro), en orden fijo. Para scatter/superposiciones: máx. 3 series.
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
CATEGORICA = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO]
TEXTO, TEXTO_2, GRILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SECUENCIAL = "Blues"
DIVERGENTE = "RdBu_r"  # dos polos + blanco en el centro


def estilo() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "axes.edgecolor": TEXTO_2,
            "axes.labelcolor": TEXTO,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
            "grid.color": GRILLA,
            "grid.linewidth": 0.6,
            "xtick.color": TEXTO_2,
            "ytick.color": TEXTO_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


# Figuras de exploración del notebook: regenerables, fuera de git
FIGURAS_NOTEBOOK = ROOT / "notebooks" / "figuras"

# Informe en formato paper a dos columnas (A4, márgenes de 1,9 cm, columnsep de 0,6 cm)
ANCHO_COLUMNA = 3.27  # pulgadas (8,3 cm)
ANCHO_PAGINA = 6.77  # pulgadas (17,2 cm)
ESTILO_PAPER = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def guardar(fig: plt.Figure, nombre: str) -> Path:
    """Guarda una figura de exploración en notebooks/figuras/<nombre>.png (regenerable, no se
    versiona). Las figuras del informe se guardan con `guardar_paper`."""
    FIGURAS_NOTEBOOK.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS_NOTEBOOK / f"{nombre}.png"
    fig.savefig(ruta)
    return ruta


def guardar_paper(fig: plt.Figure, nombre: str) -> Path:
    """Guarda una figura del informe como PDF vectorial en report/figures/<nombre>.pdf."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / f"{nombre}.pdf"
    fig.savefig(ruta)
    fig.savefig(FIGURES_DIR / f"{nombre}.png", dpi=300)  # para las slides (beamer) y el README
    return ruta
