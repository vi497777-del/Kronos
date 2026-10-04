"""Backtest rodante de NKE con Kronos-base (preentrenado o ajustado).
Para cada ventana: contexto de 400 dias, se predicen 20 dias y se compara la direccion con la real."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, yfinance as yf, torch
from model import Kronos, KronosTokenizer, KronosPredictor
torch.manual_seed(0)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "NeoQuasar/Kronos-base"
L, P, S, NWIN = int(sys.argv[2]) if len(sys.argv) > 2 else 400, 20, 10, 16
pred = KronosPredictor(Kronos.from_pretrained(MODEL), KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"),
                       device="cpu", max_context=512)
raw = yf.download("NKE", period="7y", interval="1d", progress=False, auto_adjust=False)
raw.columns = raw.columns.get_level_values(0)
df = pd.DataFrame({"timestamps": raw.index, "open": raw.Open.values, "high": raw.High.values,
                   "low": raw.Low.values, "close": raw.Close.values, "volume": raw.Volume.values})
df["amount"] = df.volume * df[["open", "high", "low", "close"]].mean(axis=1)
cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
ends = [len(df) - P - k * P for k in range(NWIN)][::-1]   # ventanas sin solapamiento en el periodo de prediccion
xs = [df.iloc[e - L:e] for e in ends]
out = pred.predict_batch([x[cols] for x in xs], [x.timestamps.reset_index(drop=True) for x in xs],
                         [df.timestamps.iloc[e:e + P].reset_index(drop=True) for e in ends],
                         pred_len=P, T=1.0, top_p=0.9, sample_count=S, verbose=False)
rows = []
for e, o in zip(ends, out):
    c0, real, pr = df.close.iloc[e - 1], df.close.iloc[e + P - 1], o.close.iloc[-1]
    rows.append(dict(inicio=df.timestamps.iloc[e].date(), ultimo=round(c0, 2), real=round(real, 2), pred=round(pr, 2),
                     real_pct=round((real / c0 - 1) * 100, 1), pred_pct=round((pr / c0 - 1) * 100, 1),
                     ctx_pct=round((c0 / df.close.iloc[e - L] - 1) * 100, 1), ctx_media=round(df.close.iloc[e - L:e].mean(), 2)))
r = pd.DataFrame(rows)
r["acierta_dir"] = np.sign(r.real_pct) == np.sign(r.pred_pct)
print(MODEL, "contexto", L); print(r.to_string(index=False))
print(f"\npredice subida: {(r.pred_pct > 0).mean():.0%} | sube de verdad: {(r.real_pct > 0).mean():.0%} | "
      f"acierto de direccion: {r.acierta_dir.mean():.0%} | corr(pred,real): {np.corrcoef(r.pred_pct, r.real_pct)[0,1]:.2f} | "
      f"cambio medio pred {r.pred_pct.mean():+.1f}% vs real {r.real_pct.mean():+.1f}% | error abs medio {(r.pred_pct - r.real_pct).abs().mean():.1f} pts")
