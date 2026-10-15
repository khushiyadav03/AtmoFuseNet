# AtmoFuseNet Research Paper — Actionable TODO Checklist

This document tracks every placeholder and user-actionable item in the IEEE conference paper `paper/main.tex` and bibliography `paper/references.bib`.

---

## 1. Author and Institutional Affiliations
- [ ] **Paper Title Block (`paper/main.tex:21`)**:
  - Replace `\textcolor{red}{[TODO: author names/affiliations]}` with author names, departments, universities/organizations, cities, countries, and email addresses according to IEEE conference format.

---

## 2. Research Contribution Framing
- [ ] **Introduction Contributions (`paper/main.tex:49`)**:
  - Choose and refine the narrative positioning:
    - **Option A (Reproduction & Critical Code Audit)**: Framing as an independent reproduction study of physics-guided graph ODEs with deep architectural audit revealing ODE parameter persistence challenges.
    - **Option B (Model Extension & Gated Fusion)**: Framing as an algorithmic extension analyzing the gated fusion of diffusion and advective wind fields.
    - **Option C (Benchmark & Regional Evaluation)**: Framing as an empirical benchmark on urban atmospheric monitoring networks.

---

## 3. Dataset Characteristics and Partitioning
- [ ] **Data Split Statistics (`paper/main.tex:162`)**:
  - Fill in the exact observation period (e.g., date ranges: month/year to month/year).
  - Fill in exact sample counts for `train.npz` (e.g., 2,208 hourly sequences), `val.npz` (e.g., 318 sequences), and `test.npz` (e.g., 598 sequences).
  - Note missing value imputation strategy (if applicable).

---

## 4. Baseline Benchmark Results (Table II)
- [ ] **Table II Baseline Performance Metrics (`paper/main.tex:188-193`)**:
  - Run or cite external baseline benchmark figures for horizons 2h, 4h, 8h, 16h, 24h across MAE, MAPE, RMSE:
    - [ ] **Historical Average (HA)**
    - [ ] **Standard GRU / LSTM**
    - [ ] **Diffusion Convolutional Recurrent Neural Network (DCRNN)**
    - [ ] **Graph WaveNet**
    - [ ] **AirPhyNet (original baseline paper values)**
  *(Note: AtmoFuseNet (Ours) row is already filled with genuine evaluated numbers: 18.51, 22.98, 29.29, 37.05, 42.85 MAE).*

---

## 5. Ablation Study Sweep (Table III)
- [ ] **Table III Physical Transport Ablation (`paper/main.tex:204-207`)**:
  - Execute `run_experiments.ps1` (or `run_experiments.sh`) across all 4 filter variants with `--exp_repeat 5`:
    - [ ] **No Physics (`unkP`)**: Multi-layer perceptron without spatial graphs.
    - [ ] **Diffusion Only (`diff`)**: *(Already populated from current checkpoint)*.
    - [ ] **Advection Only (`adv`)**: Wind-derived dynamic adjacency matrix.
    - [ ] **Full Gated Fusion (`diff_adv`)**: Unified diffusion + advection gated integration.
  - Run `python make_tables.py logs/ablations/atmofusenet_results_<variant>.txt` to automatically generate the LaTeX rows.

---

## 6. Computational Environment and Hardware
- [ ] **Experimental Hardware Setup (`paper/main.tex:173`)**:
  - Specify CPU model (e.g., Intel / AMD, clock speed, cores), GPU model (e.g., NVIDIA RTX / A100 if evaluated on GPU), RAM capacity, and operating system.

---

## 7. Additional Visualizations
- [ ] **Station-Level Time Series Plot (`paper/main.tex:219`)**:
  - Generate a 24-hour prediction vs. ground truth trajectory curve for representative urban vs. suburban monitoring stations using `models/atmofusenet_predictions.npz`.
- [ ] **Error Growth vs. Continuous Horizon**:
  - Plot MAE as a smooth function of continuous ODE horizon $\tau \in [0, 1)$.
- [ ] **Number of Function Evaluations (NFE)**:
  - Record and plot Dormand-Prince NFE counts across different tolerance settings (`atol`, `rtol`).

---

## 8. Bibliography and Literature Verification
- [ ] **STDEN Citation (`paper/references.bib:67`)**:
  - Verify complete publication venue and volume/pages for the STDEN reference (`ji2022stden`).
- [ ] Ensure all referenced conference proceedings adhere to IEEE citation guidelines without missing DOI/volume/pages.

---

## 9. Final Proofing and Page Constraints
- [ ] Confirm compliance with conference page limit (e.g., 6 or 8 pages).
- [ ] Review overfull horizontal box warnings upon inserting author details.
