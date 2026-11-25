# 酶动力学绝对值预测工程实现报告

版本：v1.0。日期：2026-11-25。本文记录 `kinetics_abs` 的实际实现、已执行验证与未完成阻塞。所有指标均来自本机真实运行；未完成的全量实验不作为结果报告。

## 1. 范围与主线

- 主任务：`log10(kcat)`，单位 `s^-1`。
- 可选独立任务：`log10(Km)`，单位 `mM`；数据适配与审计已完成，未训练 Km 模型。
- `kcat/Km`：未实现。CatPred 未提供已核验的同酶、同目标底物、匹配条件配对标签。
- CatPred 主实验关闭实测温度/pH 数值输入与排序损失；保留亲本、变体、测量分组字段接口。
- 实现入口：`python -m kinetics_abs.cli {adapt,audit,build-features,train,evaluate,predict}`。

## 2. 环境与锁定

实测环境为 CPU-only：

- Python 3.10.12
- torch 2.14.1+cpu；CUDA 不可用
- transformers 4.45.2（从 4.46.3 降级以兼容 ESM tokenizer）
- esm 3.2.1.post1，rxnfp 0.1.0，rdkit 2026.9.2
- numpy 2.2.6，pandas 2.3.3，scipy 1.15.3，scikit-learn 1.7.2，PyYAML 6.0.4，tqdm 4.70.1，httpx 0.28.1

锁定文件：`requirements_kinetics_abs.txt`。torch 需从 CPU index 安装。模型 checkpoint 信息由 `train.py` 写入 `checkpoint_best.pt` / `checkpoint_last.pt` 的 `config` 与 `experiment_summary.json`；本报告不内嵌大文件。

## 3. 数据适配与审计（P1）

CatPred 处理表路径：`/home/ubuntu/data/CatPred-DB`。适配输出：

- `data/adapted/kcat_unified.csv`
- `data/adapted/Km_unified.csv`

审计输出：`data/audit/kcat/*`、`data/audit/Km/*`（含 `audit_report.json`、`audit_report.md`、`audit_failures.csv`）。

| 任务 | rows | unique_sequence_ids | unique_chemical_contexts | unique_reactions | label_missing_rows | structure_local_existing_rows |
|---|---:|---:|---:|---:|---:|---:|
| kcat | 23,197 | 7,183 | 10,853 | 12,290 | 0 | 0 |
| Km | 41,174 | 12,355 | 10,535 | 0 | 0 | 0 |

官方 split 修正后行数：

- kcat：train 18,789 / val 2,088 / test 2,320
- Km：train 33,350 / val 3,706 / test 4,118

适配要点：`sample_id` 稳定；`sequence_id`/`chemical_id`/`reaction_id` 为内容 hash；kcat 化学上下文取 `reactant_smiles`，缺失时回退 reaction 左侧；Km 目标底物取 `substrate_smiles`；split 列按处理表位置索引合并并修正 `_split` 后缀冲突；不把 `log10kcat_max` 与同一行 `value`/`log10_value` 混用。

## 4. 特征生成与缓存验证（P2-P3）

小样本冒烟集：`data/adapted/kcat_smoke_unified.csv`，42 行（train 24 / val 12 / test 6），33 个唯一序列、31 个化学上下文、31 个反应、42 行 reaction 可用。

冒烟特征构建结果：`features_cache/kcat_smoke/index/feature_index.csv`

- protein / chemical / reaction：status `ok` 均为 42，failure 均为 0
- structure：`disabled` 42（本地无坐标文件，审计 `structure_local_existing_rows=0`）
- 缓存前缀：`seq_`、`chem_`、`rxn_`、`struct_`；同一 `cache_root` 复用。

已验证真实预训练特征路径：ESMC-600M 残基/全局、ChemBERTa-77M-MTR 384 维 + MACCS 166 位 = 550 维、RXNFP `bert_pretrained` 256 维 CLS。超过长度输入记失败，不静默截断；RXNFP token 长度守卫已实现。

ESM 兼容修复：`protein_features._patch_esmc_tokenizer_for_transformers()` 解决 `EsmSequenceTokenizer` 特殊 token 只读 property 与 transformers 4.45.2 初始化冲突。

## 5. 模型与训练/推理路径（P4-P5）

实现模块：`data_adapters.py`、`audit_data.py`、`protein_features.py`、`chemical_features.py`、`reaction_features.py`、`structure_features.py`、`build_features.py`、`dataset.py`、`model.py`、`train.py`、`evaluate.py`、`predict.py`、`cli.py`、`configs/*.yaml`。

模型要点：

- B0：ESMC global 1152 + ChemBERTa/MACCS 550，MLP。
- B1：B0 + RXNFP 256 + 可用性标记。
- A1/A4/A4-fixed：融合模型，条件化查询数分别为 1/4/4，A4-fixed 不用 s/r 生成查询条件。
- G/GQ：A4 + 2 层残基图更新；G 不按质量控制，GQ 使用边质量与门控残差。
- GQ-no-global / GQ-no-rxn：完整模型消融。
- 融合维度：`concat(p_global,p_local,s,r,p_local*s,c)=1344`，head `1344→512→256→128→1`。
- 结构图：序列边 + 空间 kNN（k≤16，≤12 Å），RBF 16、gap bucket、边类型、端点质量/PAE 缺失中性处理；`edge_index.numel()==0` 时图分支原样返回，不会因缺结构崩溃，但也不产生结构收益。

已修复并验证：

- baseline/fusion 输出形状 `[B,1]`。
- 验证/测试预测 metadata 按样本展开。
- checkpoint 使用 `weights_only=False` 兼容 numpy RNG state；`--resume` 已跑通。
- `predict` 已对冒烟集输出 42 行预测。
- 合成张量测试覆盖 `graph_mode = none/uncontrolled/quality` 前向与结构图构建。

冒烟训练均为 CPU、单 seed 42、小 epoch，仅用于路径验证，不构成科学结论：

| 实验 | val n | RMSE | MAE | PCC | SCC | R² |
|---|---:|---:|---:|---:|---:|---:|
| kcat_smoke_b0 | 12 | 1.6979 | 1.4704 | -0.1310 | 0.0490 | -0.2184 |
| kcat_smoke_b1 | 12 | 1.4999 | 1.2697 | 0.7297 | 0.6993 | 0.0491 |
| kcat_smoke_a1 | 12 | 2.0179 | 1.8030 | -0.0571 | -0.2937 | -0.7210 |
| kcat_smoke_gq | 12 | 2.0404 | 1.8461 | -0.5628 | -0.7133 | -0.7597 |

## 6. 全量实验配置（P6-P7，未运行）

已生成 configs：`kcat_b0.yaml`、`kcat_b1.yaml`、`kcat_a1.yaml`、`kcat_a4.yaml`、`kcat_a4_fixed.yaml`、`kcat_g.yaml`、`kcat_gq.yaml`、`kcat_gq_no_global.yaml`、`kcat_gq_no_rxn.yaml`。均指向 `data/adapted/kcat_unified.csv` 与 `features_cache/kcat_full/index/feature_index.csv`，official split，seeds `[42, 2026, 3407]`，max_epochs 200，effective batch 128。CPU 全量训练未执行。

建议执行顺序：

```bash
# 1) 复用/补齐全量 kcat 特征缓存（protein/chemical/reaction；structure 由后续配置处理）
python -m kinetics_abs.cli build-features --config configs/kcat_a4.yaml --device cpu
# 2) 生成带 structure_status 的索引；若仍无坐标，G/GQ 结构分支为空操作
python -m kinetics_abs.cli build-features --config configs/kcat_gq.yaml --device cpu
# 3) 训练示例（GPU/集群上把 training.device 改为 cuda）
python -m kinetics_abs.cli train --config configs/kcat_gq.yaml --output-root results_abs
# 4) 恢复训练
python -m kinetics_abs.cli train --config configs/kcat_gq.yaml --output-root results_abs --resume
# 5) 预测
python -m kinetics_abs.cli predict --config configs/kcat_gq.yaml \
  --checkpoint results_abs/kcat_gq/fold0_seed42/checkpoints/checkpoint_best.pt \
  --unified-csv data/adapted/kcat_unified.csv \
  --feature-index-csv features_cache/kcat_full/index/feature_index.csv \
  --output-csv predictions.csv --device cpu
```

OOF 聚合入口：`python -m kinetics_abs.cli evaluate --oof-files <oof_csv...> --output-dir <dir>`。

## 7. 阻塞与未完成项

- 无 CUDA；ESMC-600M/ChemBERTa/RXNFP 全量编码（7,183 唯一 kcat 序列、10,853 化学上下文、12,290 反应）在 CPU 上未运行，预计需要 GPU 或长时间 CPU 作业。
- 结构坐标缺失：审计 `structure_local_existing_rows=0`；G/GQ 真实结构收益需要 AlphaFold/PDB manifest 或坐标目录后重跑 `structure_features` 与 `build-features`。
- CatPred 为汇总标签，逐测量温度/pH 与聚合标签对应关系未核验，因此主实验不使用实测条件数值。
- 未做严格可比测量组，因此排序损失未启用。
- kcat/Km 配对标签未核验，未实现差值分支。
- `cluster10` split protocol 已在 dataset 层支持，但全量折级训练/OOF 未运行。

## 8. 交付物与可复现性

代码、配置、测试与本报告已纳入 git；大体积派生物（`data/`、`features_cache/`、`models_abs/`、`results_abs*/`、checkpoint）被 `.gitignore` 排除，可由上述 CLI 重新生成。小样本真实缓存与冒烟结果保留在工作区路径中，供检查。

测试：`tests/test_kinetics_abs.py` 覆盖指标、结构图构建、三种 graph mode 前向；本机无 pytest，已用等价函数调用验证通过。
