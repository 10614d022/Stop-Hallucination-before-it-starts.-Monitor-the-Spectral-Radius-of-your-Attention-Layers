# Spectral Instability — Full Clean Version

這是乾淨完整版，保留兩條主線：

1. **公式／理論主線**
   - arXiv 論文主稿（LaTeX）
   - 公式推導總表
   - 定理 / 引理 / 猜想分級
   - WTRED 與 Transformer attention 的跨域對齊

2. **最小實驗主線**
   - 最小 A/B 實驗
   - 譜 proxy
   - clamp
   - survival 匯總

## 目錄

- `paper/main.tex`：可直接編譯的論文主稿
- `paper/FORMULAS.md`：完整公式生成總表
- `paper/THEOREM_MAP.md`：Theorem / Lemma / Conjecture 地圖
- `main.py`：最小實驗入口
- `spectral/`：probe / clamp / collapse metrics
- `experiments/run_ab.py`：A/B 主流程
- `analysis/summary.py`：匯總
- `analysis/plot_survival.py`：survival 圖

## 最短使用

```bash
pip install -r requirements.txt
python main.py --model gpt2 --runs 3 --prompt "Paris is the capital of"
```
