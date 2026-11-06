"""
PURPOSE: Reanálisis de la encuesta de escucha de la tesis (15 personas x 10 ejemplos x 2 clips, puntajes
         1-5; la fila "Respuesta esperada" es el HC del sistema por clip). Verifica el TPRc de la tesis y
         lo recalcula tomando el ejemplo como unidad (binomial por ítem, Spearman sobre 10 ítems).
         Entrada: la hoja exportada a CSV (Respuestas_de_formulario_1.csv).
         Uso: python tools/harmonic_tiv/survey_reanalysis.py <csv>
CHANGELOG:
  - 2026-11-06: Creación (rama harmonic-tiv, informe de auditoría).
"""
import csv
import sys
import numpy as np
from scipy.stats import binomtest, chi2_contingency, spearmanr

rows = list(csv.reader(open(sys.argv[1], encoding="utf-8")))
raters = []
for r in rows[1:20]:
    try:
        float(r[0])
    except (ValueError, IndexError):
        continue
    raters.append([float(x) for x in r[1:21]])
A = np.array(raters)                               # (15, 20)
expected = None
for r in rows:
    if r and r[0].startswith("Respuesta espe"):
        expected = np.array([float(x) for x in r[1:21]])
        break
print("raters:", A.shape[0], "clips:", A.shape[1])
c = A[:, 0::2] - A[:, 1::2]                        # (15, 10) clip1 - clip2 per rater/example
d = expected[0::2] - expected[1::2]                # (10,) system HC1 - HC2
print("d (system):", d)
# thesis contingency (rater x example responses, ties dropped)
dp, cp = d[None, :] > 0, c > 0
cn = c < 0
tab = np.array([[np.sum(~dp & cn), np.sum(~dp & cp)], [np.sum(dp & cn), np.sum(dp & cp)]])
print("contingency [[dn&cn, dn&cp],[dp&cn, dp&cp]]:", tab.tolist(), " ties:", int(np.sum(c == 0)), "of", c.size)
agree = tab[0, 0] + tab[1, 1]
print(f"TPRc = {agree}/{tab.sum()} = {agree / tab.sum():.3f}")
chi2, p, _, _ = chi2_contingency(tab, correction=False)
print(f"chi2 (as in thesis, pseudo-replicated) = {chi2:.2f}, p = {p:.2g}")
# per-item analysis: the unit is the example
mean_c = c.mean(axis=0)
share_c1 = (c > 0).sum(0) / np.maximum((c != 0).sum(0), 1)
print("\nper example: d, mean listener diff, share preferring clip1 (no ties), agrees?")
for e in range(10):
    print(f"  ex{e + 1:2d} d={d[e]:+6.1f}  mean c={mean_c[e]:+.2f}  pref clip1={share_c1[e]:.2f}  agree={np.sign(mean_c[e]) == np.sign(d[e])}")
k = int(np.sum(np.sign(mean_c) == np.sign(d)))
n = int(np.sum(mean_c != 0))
print(f"items where majority agrees with system: {k}/{n}; exact binomial one-sided p = {binomtest(k, n, 0.5, alternative='greater').pvalue:.3f}")
print(f"Spearman(d, mean listener diff) over 10 items: {spearmanr(d, mean_c).correlation:.3f} (p={spearmanr(d, mean_c).pvalue:.3f})")
# rater-level consistency: per rater, agreement rate
rate = [(np.sign(c[u]) == np.sign(d))[c[u] != 0].mean() for u in range(c.shape[0])]
print("per-rater agreement with system (no ties):", np.round(rate, 2), "median", np.median(rate).round(2))
# leave-one-item-out robustness of TPRc
loo = []
for e in range(10):
    m = np.ones(10, bool); m[e] = False
    cc, dd = c[:, m], d[m]
    nz = cc != 0
    loo.append((np.sign(cc) == np.sign(dd)[None, :])[nz].mean())
print("TPRc leaving one example out: min %.3f max %.3f" % (min(loo), max(loo)))
