import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, yfinance as yf, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor
torch.manual_seed(0)
pred = KronosPredictor(Kronos.from_pretrained(sys.argv[1]),
                       KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"), device="cpu", max_context=512)
raw = yf.download("NKE", period="3y", interval="1d", progress=False, auto_adjust=False)
raw.columns = raw.columns.get_level_values(0)
df = pd.DataFrame({"timestamps": raw.index, "open": raw.Open.values, "high": raw.High.values,
                   "low": raw.Low.values, "close": raw.Close.values, "volume": raw.Volume.values})
df["amount"] = df.volume * df[["open","high","low","close"]].mean(axis=1)
cols = ['open','high','low','close','volume','amount']; L, P, N = 400, 20, 30

def sim(end):  # N trayectorias independientes usando contexto que termina en 'end'
    x = df.iloc[end-L:end]; xt = x.timestamps.reset_index(drop=True)
    yt = df.timestamps.iloc[end:end+P].reset_index(drop=True) if end < len(df) else \
         pd.Series(pd.bdate_range(df.timestamps.iloc[-1] + pd.Timedelta(days=1), periods=P))
    out = pred.predict_batch([x[cols]]*N, [xt]*N, [yt]*N, pred_len=P, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    return yt, np.array([o.close.values for o in out])

fig, axs = plt.subplots(2, 1, figsize=(11, 9))
for ax, end, title in [(axs[0], len(df)-P, "Backtest: simulacion vs real (ultimos 20 dias)"),
                       (axs[1], len(df), "Simulacion de los proximos 20 dias habiles")]:
    yt, paths = sim(end)
    h = df.iloc[end-100:end]
    ax.plot(h.timestamps, h.close, color='#333', lw=2, label='Historico')
    for p in paths: ax.plot(yt, p, color='#2a6fdb', lw=0.8, alpha=0.25)
    q10, q50, q90 = np.percentile(paths, [10, 50, 90], axis=0)
    ax.fill_between(yt, q10, q90, color='#2a6fdb', alpha=0.15, label='Rango 10-90%')
    ax.plot(yt, q50, color='#2a6fdb', lw=2, label=f'Mediana ({N} trayectorias)')
    if end < len(df): ax.plot(yt, df.close.iloc[end:end+P].values, color='#d9480f', lw=2, label='Real')
    ax.set_title(title, loc='left'); ax.set_ylabel('USD'); ax.grid(alpha=0.2); ax.legend(frameon=False)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    print(title, "| mediana final %.2f  p10 %.2f  p90 %.2f  P(sube)=%.0f%%" %
          (q50[-1], q10[-1], q90[-1], 100*(paths[:,-1] > df.close.iloc[end-1]).mean()))
fig.suptitle(f"NKE - {sys.argv[3]}, Monte Carlo (T=1.0, top_p=0.9)", x=0.06, ha='left', fontsize=14)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
