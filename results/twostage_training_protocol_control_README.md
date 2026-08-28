# Two-stage 训练协议受控实验

修复 per-epoch sampler（persistent_workers=False）+ LR 变量隔离。3 条件 × 3 seeds（42/123/456），270/129 test-as-val，全部从头随机初始化，无 pretrained。

## AUC 表格

| seed | HE samplerfix | Two-stage diff LR | Two-stage unified LR |
|---|---|---|---|
| 42 | 0.8508 | 0.7296 | 0.8776 |
| 123 | 0.8936 | 0.6980 | 0.8528 |
| 456 | 0.8038 | 0.7200 | 0.8212 |
| **mean** | 0.8494 | 0.7159 | 0.8505 |
| **std** | 0.0367 | 0.0132 | 0.0231 |

## paired ΔAUC (mean ± std)

- Two-stage diff LR − HE samplerfix = -0.1335 ± 0.0465
- Two-stage unified LR − HE samplerfix = 0.0011 ± 0.0299
- Two-stage unified LR − Two-stage diff LR = 0.1347 ± 0.0239

## best_epoch

```
{
  "he_rrt_samplerfix_lr1e4": {
    "42": 5,
    "123": 33,
    "456": 4
  },
  "twostage_r4_noepeg_samplerfix_diff_lr": {
    "42": 12,
    "123": 0,
    "456": 17
  },
  "twostage_r4_noepeg_samplerfix_unified_lr1e4": {
    "42": 14,
    "123": 23,
    "456": 15
  }
}
```

## 实际 optimizer LR（每活跃模块）

```
{
  "he_rrt_samplerfix_lr1e4": {
    "he_projection": 0.0001,
    "he_rrt": 0.0001,
    "abmil": 0.0001
  },
  "twostage_r4_noepeg_samplerfix_diff_lr": {
    "he_projection": 1e-05,
    "pr_projection": 1e-05,
    "he_rrt": 1e-05,
    "pr_rrt": 1e-05,
    "stage2": 2e-05,
    "abmil": 2e-05
  },
  "twostage_r4_noepeg_samplerfix_unified_lr1e4": {
    "he_projection": 0.0001,
    "pr_projection": 0.0001,
    "he_rrt": 0.0001,
    "pr_rrt": 0.0001,
    "stage2": 0.0001,
    "abmil": 0.0001
  }
}
```

## 协议验证（verify 脚本输出）

- 预训练权重加载：**False** (initialization=random_from_scratch)
- gradient norms（forward/backward）:
```
{
  "he_projection": 3.162626,
  "pr_projection": 4.554284,
  "he_rrt": 4.582856,
  "pr_rrt": 5.2498,
  "stage2": 2.565485,
  "abmil": 3.741285
}
```
- relative parameter updates（optimizer.step 后）:
```
{
  "he_projection": 0.00252399,
  "pr_projection": 0.00252686,
  "he_rrt": 0.00320263,
  "pr_rrt": 0.00293463,
  "stage2": 0.00320096,
  "abmil": 0.00196671
}
```
- per-epoch sampler MD5: epoch0=172a85acdfec…, epoch1=3eacc4f34b9f…, repeat=172a85acdfec…, train_persistent_workers=False
- patch count/dim 一致性: patch_count_mismatch=0, feature_dim_mismatch=0

## 附加工件

- `results/twostage_training_protocol_verify.json` — 7 项协议检查输出
- 每个 seed 目录含 `result.json` / `test_predictions.csv` / `protocol_check.json` / `config.json` / `ckpt/best_model.pt` / `logs/run.log`

