"""Ajuste fino del predictor Kronos-base con velas diarias de acciones de EE. UU. (CPU).
El tokenizer se mantiene congelado. Los datos de entrenamiento terminan en CUTOFF para
que el backtest de NKE posterior a esa fecha no contenga fuga de informacion."""
import sys, time, random; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch, yfinance as yf
from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

TICKERS = ["NKE", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC", "XOM", "CVX",
           "JNJ", "PFE", "MRK", "KO", "PEP", "MCD", "SBUX", "WMT", "TGT", "COST", "HD", "LOW", "DIS", "NFLX",
           "INTC", "AMD", "CSCO", "ORCL", "IBM", "BA", "CAT", "GE", "MMM", "UPS", "FDX", "VFC", "LULU", "UAA",
           "DECK", "SKX", "TPR", "PVH", "RL", "CROX", "ADDYY", "FL", "ULTA"]
CUTOFF = "2026-09-04"          # inicio del backtest de NKE (excluido del entrenamiento)
LOOKBACK, PRED, CLIP = 200, 20, 5.0
STEPS, BATCH, LR = 400, 16, 1e-5
SAVE = "finetune_us/kronos_base_us_daily"
COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.set_num_threads(4)

def load(t):
    r = yf.download(t, period="15y", interval="1d", progress=False, auto_adjust=True)
    if r.empty: return None
    r.columns = r.columns.get_level_values(0)
    d = pd.DataFrame({"timestamps": r.index, "open": r.Open.values, "high": r.High.values, "low": r.Low.values,
                      "close": r.Close.values, "volume": r.Volume.values}).dropna()
    d["amount"] = d.volume * d[["open", "high", "low", "close"]].mean(axis=1)
    d = d[d.timestamps < CUTOFF].reset_index(drop=True)
    return d if len(d) > LOOKBACK + PRED + 50 else None

data = {t: d for t in TICKERS if (d := load(t)) is not None}
print(f"{len(data)} series, {sum(len(d) for d in data.values())} velas")

W = LOOKBACK + PRED
def sample(split):
    t = random.choice(list(data)); d = data[t]
    n = len(d) - W; cut = int(n * 0.9)  # ultimo 10% de cada serie para validacion
    s = random.randint(0, cut) if split == "train" else random.randint(cut + 1, n)
    w = d.iloc[s:s + W]
    x = w[COLS].values.astype(np.float32)
    m, sd = x[:LOOKBACK].mean(0), x[:LOOKBACK].std(0)
    x = np.clip((x - m) / (sd + 1e-5), -CLIP, CLIP)
    st = calc_time_stamps(w.timestamps.reset_index(drop=True)).values.astype(np.float32)
    return x, st

def batch(split):
    xs, ss = zip(*[sample(split) for _ in range(BATCH)])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ss))

tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
for p in tok.parameters(): p.requires_grad_(False)

def step_loss(x, st):
    with torch.no_grad():
        s0, s1 = tok.encode(x, half=True)
    logits = model(s0[:, :-1], s1[:, :-1], st[:, :-1, :])
    return model.head.compute_loss(logits[0], logits[1], s0[:, 1:], s1[:, 1:])[0]

random.seed(123); val_batches = [batch("val") for _ in range(10)]; random.seed(0)
def validate():
    model.eval()
    with torch.no_grad(): v = np.mean([step_loss(x, s).item() for x, s in val_batches])
    model.train(); return v

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05, div_factor=10)
best = validate(); print(f"val loss inicial {best:.4f}"); model.save_pretrained(SAVE)
t0 = time.time(); model.train()
for i in range(1, STEPS + 1):
    loss = step_loss(*batch("train"))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sch.step()
    if i % 100 == 0:
        v = validate()
        print(f"paso {i}/{STEPS} train {loss.item():.4f} val {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best: best = v; model.save_pretrained(SAVE); print("  guardado")
print(f"mejor val loss {best:.4f}")
