# AtmoFuseNet Codebase Audit Report

This document presents a comprehensive technical audit of the AtmoFuseNet (AirPhyNet fork) codebase. Per the audit instructions, **no training or model logic fixes have been applied in this phase**. Each finding details the exact file, line number, operational behavior, impact, and proposed remediation.

---

## Executive Summary

Two major architectural bugs were uncovered during empirical and static code analysis:
1. **Critical Bug in ODE Layer Training (Item A)**: In `AtmoFuseNetModel.forward()`, the `ODEFunc` and `DiffeqSolver` objects are instantiated freshly inside `forward()`. Consequently, their graph convolution parameters are re-initialized with fresh Xavier random weights on every forward call. Crucially, after batch 0, these dynamically created parameters **never enter the Adam optimizer**. Therefore, the graph convolution weights inside the neural ODE are never optimized during training; only the GRU encoder and latent MLP projection are updated.
2. **Critical Bug in Adjacency Geographic Calculations (Item B)**: In `utils.get_adjacency_matrix()`, `dest_lat` is erroneously assigned from `station_df['longitude'][dest]`. This causes geographic bearing computations to receive longitude as latitude (~116.4° N, an invalid coordinate beyond the North Pole), completely corrupting directional attributes in the advection branch.

---

## Detailed Audit Findings

### Finding A: ODEFunc / DiffeqSolver Dynamic Re-creation & Optimizer Parameter Disconnect
* **File**: [atmofusenet_model.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_model.py#L97-L101), [atmofusenet_supervisor.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_supervisor.py#L149-L152), [ode_func.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/ode_func.py#L18-L25)
* **Lines**: `atmofusenet_model.py:97-101`, `atmofusenet_supervisor.py:149-152`, `ode_func.py:18-25`
* **What the code does**:
  1. In `AtmoFuseNetModel.__init__()`, neither `odefunc` nor `diffeq_solver` is initialized.
  2. Inside `AtmoFuseNetModel.forward()`, `odefunc = ODEFunc(...)` and `self.diffeq_solver = DiffeqSolver(odefunc, ...)` are instantiated anew on **every single forward call**.
  3. Inside `ODEFunc`, graph convolution weights are created lazily using `LayerParams`, calling `nn.init.xavier_normal_()` on uninitialized memory tensors.
  4. In `atmofusenet_supervisor.py`, the training loop contains an ad-hoc workaround:
     ```python
     if batches_seen == 0:
         optimizer = torch.optim.Adam(self.atmofusenet_model.parameters(), lr=base_lr, eps=epsilon)
     ```
* **Empirical Verification (Random Tensor Probe Script)**:
  A controlled probe script was executed tracking parameter addresses (`id(p)`) and gradient delivery:
  * **Initial Model Parameters**: 8 tensors (GRU weights/biases: 4, `hiddens_to_z0` weights/biases: 4). Total params: 13,388.
  * **After Forward Pass 1 (`batches_seen == 0`)**: 16 tensors (8 encoder tensors + 4 ODE gconv tensors + 2 flow_net tensors + 2 gated_fusion tensors). Total params: 37,869. The Adam optimizer is re-created with these 16 tensors.
  * **Backward Pass 1**: Gradients are populated for encoder (8) and ODE gconv (4).
  * **After Forward Pass 2 (`batches_seen == 1`)**:
    * All 8 encoder parameters retain identical memory addresses (`same id as fwd 1? True`).
    * **All 8 ODE parameters are newly created tensors with completely different memory addresses (`same id as fwd 1? False`)**:
      * `diffeq_solver.odefunc.gconv_weight_(12, 64)`: id changed from `2244385196368` to `2244390501232`.
      * `diffeq_solver.odefunc.gconv_biases_64`: id changed from `2244389973808` to `2244390501408`.
      * `diffeq_solver.odefunc.gconv_weight_(192, 4)`: id changed from `2246021431600` to `2244390702608`.
      * `diffeq_solver.odefunc.gconv_biases_4`: id changed from `2246021432048` to `2244390236048`.
    * **Are these new ODE tensors tracked by the optimizer? `in optimizer? False` for every single ODE parameter.**
* **Why it matters**:
  On batch 1, 2, ..., $N$, `optimizer.step()` updates the discarded, unreferenced weight tensors from batch 0. The current ODE weights used to compute predictions receive gradients during `loss.backward()`, but are **never updated by `optimizer.step()`** because the optimizer knows nothing about them. On the next batch, they are garbage-collected and replaced with newly generated random weights.
  * **Conclusion**: **Only the GRU encoder (`encoder_z0`) is being trained.** The neural ODE functions as an untrained, randomly initialized reservoir with fresh weights per forward pass.
* **Suggested Fix**:
  Persist `ODEFunc` and `DiffeqSolver` as persistent submodules on `AtmoFuseNetModel.__init__()`. Static weights should be registered during model construction, allowing PyTorch's `optimizer` to track and update them across all training epochs.

---

### Finding B: Destination Latitude Assigned from Longitude in Adjacency Graph
* **File**: [utils.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/utils.py#L72)
* **Line**: 72
* **What the code does**:
  ```python
  src_lat, src_lon = station_df['latitude'][src], station_df['longitude'][src]
  dest_lat, dest_lon = station_df['longitude'][dest], station_df['longitude'][dest]
  ```
  `dest_lat` reads from `station_df['longitude'][dest]` instead of `station_df['latitude'][dest]`.
* **Why it matters**:
  Beijing station coordinates are approximately $39.9^\circ \text{ N}$ latitude and $116.4^\circ \text{ E}$ longitude. Setting `dest_lat` to $116.4$ injects an impossible latitude ($> 90^\circ$ North). When passed to `get_bearing(src_lat, src_lon, dest_lat, dest_lon)`, the trigonometric formula:
  $$\Delta \text{lon} = \text{lon}_2 - \text{lon}_1 \approx 116.4 - 116.4 = 0$$
  computes completely invalid bearing angles ($\sim 0^\circ$ or distorted azimuths) for all 1,190 directed edges. These corrupted bearings form the edge attribute tensor `edge_attr[:, 1]` fed directly to the advection branch.
* **Suggested Fix**:
  Change line 72 of `utils.py` to:
  ```python
  dest_lat, dest_lon = station_df['latitude'][dest], station_df['longitude'][dest]
  ```

---

### Finding C: Unthresholded Dense Adjacency Matrix
* **File**: [utils.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/utils.py#L50-L51)
* **Lines**: 50–51
* **What the code does**:
  ```python
  # Apply threshold to adjacency matrix
  adj_mx = dist_mx.copy()
  ```
  Despite the code comment stating `"Apply threshold to adjacency matrix"`, no thresholding operation (such as Gaussian kernel thresholding $\exp(-d^2 / \sigma^2) \ge \epsilon$) is applied.
* **Why it matters**:
  Every pair of distinct stations $(i, j)$ with $i \ne j$ receives an edge with weight $1 / d_{ij} > 0$. For 35 stations, this produces a fully connected complete graph of $35 \times 34 = 1,190$ directed edges. A dense graph incurs maximal computational complexity in graph convolutions and eliminates spatial sparsity priors.
* **Suggested Fix**:
  Apply a spatial distance cutoff (e.g., standard thresholding: `adj_mx[dist_mx < threshold] = 0.0`) or calibrate a Gaussian distance kernel if graph sparsity is desired.

---

### Finding D: Dead Hyperparameters in Configuration
* **File**: [config.yaml](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/config.yaml#L34-L42), [atmofusenet_supervisor.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_supervisor.py#L22)
* **Lines**: `config.yaml:34, 35, 41, 42`; `atmofusenet_supervisor.py:22`
* **What the code does**:
  1. `weight_decay: 0.0005`: Configured in YAML, but `torch.optim.Adam` in `atmofusenet_supervisor.py` lines 118 & 151 only receives `(self.atmofusenet_model.parameters(), lr=base_lr, eps=epsilon)`. `weight_decay` is never passed to Adam (defaulting to 0.0).
  2. `dropout: 0`: Configured in YAML, but no dropout layers (`nn.Dropout`) exist anywhere in `AtmoFuseNetModel`, `Encoder_z0_RNN`, or `ODEFunc`.
  3. `max_grad_norm: 5`: In `atmofusenet_supervisor.py:22`, `self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)` is stored as an instance attribute, but `torch.nn.utils.clip_grad_norm_` is **never called** in the training loop. Gradients are stepped without clipping.
  4. `min_learning_rate: 2.0e-06`: Configured in YAML, but `torch.optim.lr_scheduler.MultiStepLR` does not support minimum learning rate clamping, and the value is never referenced in code.
* **Why it matters**:
  Reporting these values in research papers misrepresents the actual training regularization.
* **Suggested Fix**:
  Either pass `weight_decay` to Adam and invoke `torch.nn.utils.clip_grad_norm_(self.atmofusenet_model.parameters(), self.max_grad_norm)` before `optimizer.step()`, or document in the paper that $L_2$ weight decay and gradient clipping are inactive.

---

### Finding E: Filter Type Configuration & Ablation Execution
* **File**: [config.yaml](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/config.yaml#L27), [ode_func.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/ode_func.py#L86-L178), [atmofusenet_supervisor.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_supervisor.py#L61-L70)
* **Lines**: `config.yaml:27`; `ode_func.py:86-178`
* **What the code does**:
  The default configuration ships with `filter_type: diff`. Four variants exist in code:
  1. `diff` (Diffusion Only): Computes ChebNet convolution on static normalized Laplacian $L_{\text{diff}}$. Gated fusion is bypassed.
  2. `adv` (Advection Only): Computes dynamic edge weights from projected wind speeds, constructs dynamic Laplacians per batch, and evaluates advection graph convolutions.
  3. `diff_adv` (Full Physics Fusion): Computes both diffusion and advection representations, fusing them via `GatedFusionModel`: $H = z \odot X_{\text{diff}} + (1 - z) \odot X_{\text{adv}}$.
  4. `unkP` (Unknown Physics / No Physics): Bypasses graph ODE entirely, using a 2-layer MLP (`create_net`) as the ODE derivative function.
* **How to run each variant**:
  Currently, `main.py` does not take a `--filter_type` argument. Users must either:
  * Modify `filter_type` directly in `config.yaml`, or
  * Pass dedicated configuration files: `python main.py --config_filename configs/config_diff_adv.yaml`.
* **Suggested Fix**:
  Add `--filter_type` to `argparse` in `main.py` to allow CLI ablation sweeps.

---

### Finding F: Performance Bottleneck in Advection Graph Construction
* **File**: [ode_func.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/ode_func.py#L108-L128)
* **Lines**: 108–128 (and 141–163)
* **What the code does**:
  In `ODEFunc.__init__` for `adv` and `diff_adv`:
  ```python
  adj_mx_adv = torch.zeros(batch_size, num_nodes, num_nodes)
  for batch_index in range(batch_size):
      for edge_id in range(num_edges):
          src_node = edge_index[0, edge_id]
          tgt_node = edge_index[1, edge_id]
          adj_mx_adv[batch_index, src_node, tgt_node] = edge_weight[batch_index, edge_id]
  ...
  for i in range(batch_size):
      adj_mx_new = adj_mx_adv[i]
      supports_adv.append(calculate_scaled_laplacian(adj_mx_new.detach().numpy()))
  ```
* **Performance Cost Benchmark**:
  Benchmarked with batch size 32, 35 nodes, and 1,190 edges:
  * Python double `for`-loop (38,080 iterations per batch): **1,363 ms** per batch.
  * SciPy CPU Laplacian conversion (32 matrices per batch): **58 ms** per batch.
  * **Total overhead per batch**: **1,421 ms (~1.42 seconds)**.
  * **Cumulative overhead per epoch (69 batches)**: **98.1 seconds** spent purely in Python loops on CPU!
  * **Gradient Truncation**: Notice `adj_mx_new.detach().numpy()` breaks the PyTorch computation graph, preventing `flow_net` from receiving backpropagated gradients through the Laplacian.
* **Suggested Fix**:
  Vectorize tensor placement using native PyTorch operations (`adj_mx_adv[:, edge_index[0], edge_index[1]] = edge_weight`) in $< 1\text{ ms}$, and perform normalized Laplacian calculations in native PyTorch on GPU without detaching.

---

### Finding G: Deprecated APIs and In-Place Tensor Mutations
* **File**: [metrics.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/metrics.py#L4), [atmofusenet_supervisor.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_supervisor.py#L172-L182)
* **Lines**: `metrics.py:4, 13, 22`, `atmofusenet_supervisor.py:172, 182`
* **What the code does**:
  1. **In-place target mutation**: In `metrics.py`, line 4 (`y_true[y_true < 1e-4] = 0`), line 13, and line 22 mutate the input tensor `y_true` in place. Because `_compute_loss_eval()` calls `masked_mae_loss`, `masked_mape_loss`, and `masked_rmse_loss` sequentially, subsequent functions operate on previously mutated arrays. In `evaluate_more()`, `y_true` is a slice of the dataset tensor, causing permanent in-place data mutation.
  2. **Deprecated Scheduler Call**: Line 182 calls `lr_scheduler.get_lr()[0]`, which raises `UserWarning: To get the last learning rate computed by the scheduler, please use get_last_lr()`.
  3. **Scheduler Step Ordering**: Line 172 calls `lr_scheduler.step()` before the subsequent iteration's `optimizer.step()`, raising PyTorch warnings regarding skipped learning rate milestones.
* **Suggested Fix**:
  Use non-mutating clone masks: `y_true_clean = torch.where(y_true < 1e-4, torch.zeros_like(y_true), y_true)`. Replace `lr_scheduler.get_lr()[0]` with `lr_scheduler.get_last_lr()[0]`.

---

### Finding H: Non-Reproducibility Risks (Absence of Random Seeding)
* **File**: Whole project ([main.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/main.py), [utils.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/utils.py), [atmofusenet_supervisor.py](file:///c:/Users/Khushi/Downloads/Deep%20Learning%20Project/AirPhyNet/atmofusenet_supervisor.py))
* **What the code does**:
  Zero random seeds are set across the codebase:
  * No `torch.manual_seed()`
  * No `torch.cuda.manual_seed_all()`
  * No `np.random.seed()`
  * No `random.seed()`
  * `DataLoader` shuffles batches via `np.random.permutation(self.size)` with non-deterministic state.
  * In `Encoder_z0_RNN`, `sample_standard_gaussian()` generates latent states from unseeded distributions.
  * `atmofusenet_supervisor.py` imports `from random import SystemRandom`, which uses OS-level non-deterministic entropy.
* **Why it matters**:
  Every training execution produces substantially varying loss trajectories, making empirical validation impossible to reproduce without manual seed locking.
* **Suggested Fix**:
  Add a centralized seed utility (e.g., `seed_everything(seed=42)`) in `main.py` setting Python, NumPy, PyTorch CPU, and CUDA random seeds.

---

## What is Actually Used in Training

The table below contrasts the hyperparameters declared in `config.yaml` against the **actual runtime values** applied by the code:

| Hyperparameter / Component | Config Declared | Actually Applied in Code | Status / Note |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | AtmoFuseNet (PGDE) | GRU Encoder + Dynamic ODE | ODE weights re-initialized per batch |
| **Active Trained Weights** | Full model (37,869) | **Encoder Only (13,388)** | ODE weights never updated by Adam after batch 0 |
| **Stations ($N$)** | Implicit | **35** | `data/station.csv` |
| **Input Sequence Length** | 24 | **24 steps** (24 hours) | Hourly PM2.5, meteorological vars |
| **Prediction Horizon** | 24 | **24 steps** (24 hours) | Multi-step horizon |
| **Input Feature Dim to GRU**| 6 | **1** (PM2.5 only) | Temp, pressure, humidity discarded by encoder |
| **Wind Feature Dim to ODE** | Implicit | **2** (WS, WD) | Passed to `flow_net` |
| **Encoder Architecture** | GRU | **1-layer GRU (64 hidden)** | Followed by Linear(64 $\to$ 50) $\to$ Tanh $\to$ Linear(50 $\to$ 8) |
| **Latent ODE Dim** | 4 | **4** | $\mu, \sigma \in \mathbb{R}^4$ |
| **Trajectory Samples ($S$)**| 3 | **3** | Monte Carlo trajectories |
| **ODE Solver** | dopri5 | **Dormand-Prince (dopri5)** | `torchdiffeq.odeint` |
| **ODE Tolerances** | atol=1e-5, rtol=1e-5 | **atol=1e-5, rtol=1e-5** | Adaptive step-size |
| **GCN Chebyshev Order ($K$)**| 2 | **2** | Scaled Laplacian powers |
| **Filter Type** | diff | **diff** (Diffusion only) | Advection & gated fusion inactive |
| **Adjacency Construction** | Inverse distance | **Dense (1,190 edges)** | No threshold applied |
| **Optimizer** | Adam | **Adam** | Hardcoded |
| **Base Learning Rate** | 0.0005 | **0.0005** | Applied |
| **Optimizer Epsilon** | 1.0e-3 | **0.001** | Overrides default 1e-8 |
| **Weight Decay ($L_2$)** | 0.0005 | **0.0 (None)** | **Dead hyperparameter** (not passed to Adam) |
| **Dropout** | 0 | **None** | **Dead hyperparameter** (no dropout in model) |
| **Gradient Clipping** | max_grad_norm=5 | **None** | **Dead hyperparameter** (never called) |
| **LR Scheduler** | steps=[20,30,40,50], ratio=0.1 | **MultiStepLR** | Decays by $0.1\times$ at epochs 20, 30, 40, 50 |
| **Early Stopping** | patience=20 | **Patience = 20 epochs** | Triggers at Epoch 26 based on validation MAE |
| **Loss Function** | masked MAE | **Masked MAE** | Evaluated on inverse-scaled PM2.5 |
| **Random Seed** | None | **None (Non-deterministic)**| Unseeded |
