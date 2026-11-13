"""Regenera el bloque de resultados del README desde models/resultados.json.

Uso:  python scripts/cifras_readme.py          (reescribe el bloque entre los marcadores)
      python scripts/cifras_readme.py --check  (sale con código 1 si el README está desactualizado)

Así ninguna cifra del README se escribe a mano: `tests/test_cifras.py` falla si el bloque no
coincide con lo que genera el notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
README = RAIZ / "README.md"
RESULTADOS = RAIZ / "models" / "resultados.json"
INICIO, FIN = "<!-- cifras:inicio -->", "<!-- cifras:fin -->"


def n(x: float, d: int = 2) -> str:
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def ic(lo: float, hi: float, d: int = 2) -> str:
    return f"[{n(lo, d)}; {n(hi, d)}]"


def bloque(r: dict) -> str:
    f4 = r["fase4"]
    o = f4["or_intra"]
    b = f4["robustez"]["bootstrap"]
    neg, rea = f4["negocio_segmentos"], f4["reasignacion"]
    bench = o["R = 1 (aleatorio) | ajustado por Z"]
    adj = o["R = 0 (algoritmo) | ajustado por Z"]
    ing = o["R = 0 (algoritmo) | sin ajuste"]
    return f"""{INICIO}
| Estimación del efecto del top-5 sobre la reserva (OR dentro de la búsqueda) | OR | IC 95 % |
|---|---|---|
| **Orden aleatorio, ajustado (benchmark causal)** | **{n(bench["OR intra"])}** | {ic(*bench["OR IC95"])} |
| Orden del algoritmo, ajustado por todos los confusores observados | {n(adj["OR intra"])} | {ic(*adj["OR IC95"])} |
| Orden del algoritmo, comparación ingenua | {n(ing["OR intra"])} | {ic(*ing["OR IC95"])} |

- **El top-5 sí causa reservas.**
- **Con datos del algoritmo, el efecto se sobreestima aun después de ajustar:** cociente de OR
  {n(b["cociente_OR"]["mediana"])} (IC 95 % bootstrap por búsqueda
  {ic(b["cociente_OR"]["IC95_inf"], b["cociente_OR"]["IC95_sup"])}, {f4["robustez"]["B"]} réplicas); en
  diferencia de riesgos, cociente de efectos marginales {n(b["cociente_AME"]["mediana"])}
  {ic(b["cociente_AME"]["IC95_inf"], b["cociente_AME"]["IC95_sup"])}. El ranker usa información que no
  observamos.
- **Para quién rinde el slot (orden aleatorio, LinearDML):**
  {n(neg["barato y bien ubicado"]["reservas incrementales / 1.000"], 1)} reservas incrementales cada
  1.000 exposiciones con hoteles baratos y bien ubicados, contra
  {n(neg["caro y mal ubicado"]["reservas incrementales / 1.000"], 1)} con hoteles caros y mal ubicados.
  Reasignar 1.000 exposiciones suma ≈ {n(rea["usd_por_1000"], 0)} USD con el gasto mediano.
{FIN}"""


def actualizar(check: bool = False) -> bool:
    r = json.loads(RESULTADOS.read_text("utf-8"))
    texto = README.read_text("utf-8")
    i, f = texto.index(INICIO), texto.index(FIN) + len(FIN)
    nuevo = texto[:i] + bloque(r) + texto[f:]
    if check:
        return nuevo == texto
    README.write_text(nuevo, "utf-8")
    return True


if __name__ == "__main__":
    ok = actualizar(check="--check" in sys.argv)
    if not ok:
        print("README desactualizado: correr `python scripts/cifras_readme.py`")
    sys.exit(0 if ok else 1)
