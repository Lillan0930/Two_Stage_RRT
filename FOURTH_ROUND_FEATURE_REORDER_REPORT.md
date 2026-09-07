# 第四轮诊断与修复报告 — 旧 .pt 重排可行性

> 项目：CAMELYON16（C16）WSI 二分类
> 路径：`/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT`
> 日期：2026-09-03
>
> **本轮目标**：不重跑 CTransPath，尝试从旧 `.pt` 恢复 row→patch coordinate 映射并直接重排为 coordinate-sorted。
> **唯一动作**：测试时枚举源 JPEG 文件名（`rglob('*')` 不排序），恢复 permutation，重排旧 `.pt`，与第三轮已重提取的 sorted 特征逐行对比。

---

## 1. Permutation Recovery Validation

用第三轮已经重新提取并保存 metadata 的 `normal_001` / `tumor_001`（HE+PR 各一）做严格验证。
对旧 `.pt`：先复现原 feature extraction 的 patch 枚举顺序（HE 用原始 `C16-HE/{slide}` 符号链接目录、
PR 用原始 `Results-PR/{slide}` 目录，均位于 `/home/Public/data_hdd0/lillan/C16/`），解析 `(row,col)`，
建立 permutation，重排旧 `.pt`，与 sorted 特征逐行比较。

| Slide | Modality | Old→New mean cosine | max_abs_diff | PASS/FAIL |
| ----- | -------- | ------------------: | -----------: | --------- |
| normal_001 | HE | 0.6968 | 1.876e+00 | **FAIL** |
| normal_001 | PR | 0.7204 | 1.594e+00 | **FAIL** |
| tumor_001  | HE | 0.6401 | 2.096e+00 | **FAIL** |
| tumor_001  | PR | 0.7029 | 2.064e+00 | **FAIL** |

要求判据是 `mean cosine > 0.999`，实测只有 **0.64–0.72**，远未达标；`max_abs_diff` 高达 ~1.5–2.1（而非 ≈0），
说明重排后每一行对应的是**错误的 patch**，permutation 没有恢复成功。

**附加诊断（证明问题不在“顺序是否能被重排”，而在“顺序本身已丢失”）：**

对 4 个 `(slide, modality)`，用**逐 tile 精确匹配**（tile 级 cosine）从旧 `.pt` 与 sorted 特征之间反解出**真实 permutation**：

| 检查 | 结果 |
|------|------|
| 旧 `.pt` 是否为 sorted 特征的纯置换 | **是**（tile 级 matched cosine = **1.000000**，min/max 均为 1.0） |
| patch 分组是否 16 行/tile 连续 | **是**（normal 395/395、tumor 1488/1488 全连续） |
| 真实 permutation 是否 bijection | **是**（unique targets = P/P） |
| 真实旧顺序 vs `rglob` 复现顺序 命中率 | **0.0000–0.0027**（= 1/P，纯随机） |
| 真实旧顺序 vs 字母序 命中率 | **0.0000–0.0025**（纯随机） |
| 真实旧顺序 vs 数值 `(row,col)` 序 命中率 | **0.0000–0.0025**（纯随机） |

真实旧顺序对行/列坐标**无任何单调或分组规律**（前 30 个 patch 坐标即为散乱值，如 `(57,107),(64,112),(65,102),(50,84),…`），
这是典型 ext4 `htree` 目录的 readdir（哈希）顺序特征。

---

## 2. Recovery Conclusion

```
PERMUTATION RECOVERY FAILED
```

**根因（已严格定位）：**

1. 旧 `.pt` 的 row 顺序 = 原特征提取时 `Path(wsi_dir).rglob('*')` 返回的**文件系统 readdir 顺序**（ext4 htree 按文件名哈希排列，外观随机、且与具体文件系统的哈希种子绑定）。
2. 数据盘从 `data_hdd0`（原挂载 `/media/kemove/data_hdd0`，现已改挂 `/home/Public/data_hdd0`）被**复制/迁移**到 `SANDISK ELE`，且原始目录的 readdir 状态也已变化——即使对当前 `/home/Public/data_hdd0/lillan/C16/C16-HE/{slide}` 原目录重跑 `rglob('*')`，也**无法复现**当初的顺序。
3. 因此「重新枚举文件名 → 得到旧顺序 → 建 permutation」这条路**不可行**：文件名集合还在（坐标集合完全一致），但**顺序信息已随磁盘复制永久丢失**。

这与本轮任务开头强调的警告一致：*“不要因为当前目录里还能看到相同 filename，就直接假定 permutation 一定可以恢复。”* —— 文件名可见，但顺序不可恢复。

---

## 3. Full Dataset Reorder

未执行。验证（§1）未通过，按规则**立即停止**，不批量生成新 `.pt`。

```
Total:  399 slides（159 normal + 111 tumor + 129 test）
PASS:   0
FAIL:   —（未批量执行）
```

---

## 4. HE/PR Coordinate Alignment

**未实现。** 因旧 `.pt` 的 row 顺序无法恢复，无法把 HE/PR 重排到 `HE[i] ↔ PR[i] ↔ 同一 (row,col)`。

补充说明（来自第三轮已证实的结论）：**源数据 HE/PR 在坐标上是对齐的**（`Results-PR/{slide}` 与 `C16_raw/…/valA/{slide}_*.jpeg` 坐标集合逐行一致），
只是旧 `.pt` 在提取时用未排序 `rglob` 破坏了这个对齐。要对齐，必须重新提取。

---

## 5. Data Integrity

未做全量重排，故无重排前后一致性可报告。但本轮已通过第三轮 sorted 特征**正面确认**：

> 旧 `.pt` 与第三轮重提取的 sorted 特征是**同一个 feature 集合的纯置换**（tile 级 cosine = 1.000000、patch 分组 16 行连续、bijection），
> 即旧 `.pt` 的 feature **数值本身没有错**，错的是 **row 顺序**。这一点对第五轮很重要——说明问题可修复，修复方式就是按正确顺序重新提取。

---

## 6. 最终回答

### Q1：能否可靠从旧 `.pt` 恢复当初的 patch row mapping？

**不能。** 旧 `.pt` 内不含任何 row↔coordinate 元数据；row 顺序由原文件系统的 readdir 哈希顺序决定，该顺序在磁盘复制/迁移后**不可复现**（对 `rglob`/字母序/数值序的命中率均 ≈ 1/P，纯随机）。

### Q2：是否无需重新运行 CTransPath？

**否。** 顺序信息已丢失，无法通过重排旧 `.pt` 恢复；**必须重新运行 CTransPath**（以 `sorted()` 数值坐标序提取）才能得到 coordinate-sorted 特征。

### Q3：重排后 HE / PR 是否实现严格 `HE[i] ↔ PR[i] ↔ same (row,col)`？

**否（重排这条路走不通）。** 只有重新提取（HE 与 PR 都按数值 `(row,col)` 排序）才能实现。第三轮对 2 张 WSI 的 sorted 重提取已经证明这条路径可行且对齐（diag 余弦 gap +0.14~0.17）。

### Q4：新的 token 顺序是否按照数值 `(row,col)` 空间顺序排列？

**本轮未生成任何新特征**（验证失败即停止）。第三轮重提取的 `*_sorted.pt`（normal_001/tumor_001 各 HE/PR）确为数值 `(row,col)` 序——这证明正确的排序管线已存在，只是尚未覆盖全量数据。

### Q5：是否已经具备用新 feature 重新训练（HE-only / PR-only / Two-RRT+Concat / Two-RRT+CR-MSA）的条件？

**不具备。** 需要先按数值 `(row,col)` 排序，对全部 399 个 WSI 的 HE + PR **重新提取特征**（复用第三轮 `recheck_c16_alignment.py` 的 sorted 管线，扩展到全量），并落地到
`C16_HE_features_coord_sorted/`、`C16_PR_features_coord_sorted/`，之后才具备用对齐后的特征重新训练/重做因果消融的条件。

---

## 结论与下一步建议

本轮结论一句话：**旧 `.pt` 无法靠重排修复，必须重新提取。** 但这同时是一个**确定性很强、可执行**的修复：

- 证据链完整：旧 `.pt` = sorted 特征的纯置换（提取完全确定）→ 唯一损坏的是顺序 → 顺序是文件系统 readdir 哈希序、随磁盘复制丢失。
- 修复路径明确：第三轮的 `sorted()` 提取管线已经正确（对齐、数值坐标序），只需把它的范围从 2 张 WSI 扩展到全部 399 张（HE + PR）。
- 源 JPEG 均可访问：`/media/kemove/SANDISK ELE/C16-raw&translation/`（HE `C16_raw/…/valA`，PR `Results-PR/{slide}`），原始盘亦挂载于 `/home/Public/data_hdd0`。

**建议（第五轮）**：全量重提取 HE/PR 特征（数值 `(row,col)` 排序、含 `{features, filenames, coords}` 元数据），生成 coordinate-sorted 的 `.pt`，随后在**对齐后的**特征上重做因果消融与重新评估 PR 的增量贡献。

---

## 附录：本轮产物

| 文件 | 用途 |
|------|------|
| `feature_extract/reorder_c16_features.py` | 复现 rglob 顺序 → 建 permutation → 重排 → 对比 sorted（验证模式） |
| `results/reorder_validation.json` | 4 个 (slide,modality) 的重排 cosine / max_abs_diff |
| `results/recheck_alignment/recheck_alignment_report.json` | 第三轮 sorted 重提取的对齐证据（复用为验证 ground-truth） |

**本轮未修改** RRT / CR-MSA / ABMIL / 训练结构 / split；未重新提取全量特征；未生成任何新 `.pt`。
