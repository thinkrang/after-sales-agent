# 核销规则离线回放

数据集 voucher-redemption-synthetic-v1 包含 18 份合成快照。两版使用相同的证据和固定判定时间。
参考标签由模拟复核事实预先给出，指标仅描述本样本集。

| 指标 | 基线 | 候选 |
| --- | ---: | ---: |
| 规则 | voucher-v1 | voucher-v2 |
| 时间容差 | 300 秒 | 120 秒 |
| 与参考标注一致 | 15/18 | 15/18 |
| 误提示核销证据吻合 | 2/18 | 0/18 |
| 漏提示核销证据吻合 | 1/18 | 3/18 |
| 初始进入人工复核 | 17/18 | 17/18 |

核查建议变化 4/18。初始分流变化 0/18。
抽查样本 R03、R04、R05、R06、R07。

| 快照 | 场景 | 两端时差 | 基线建议与原因 | 候选建议与原因 | 参考标注 |
| --- | --- | ---: | --- | --- | --- |
| R01 | 时间边界 | 0 秒 | 核销记录吻合 · matching_redemption | 核销记录吻合 · matching_redemption | 核销记录吻合 · 模拟审核已确认同一次核销，两个来源的时间完全相同。 |
| R02 | 时间边界 | 120 秒 | 核销记录吻合 · matching_redemption | 核销记录吻合 · matching_redemption | 核销记录吻合 · 模拟审核已确认同一次核销，商家时钟偏移正好 120 秒。 |
| R03 | 时间边界 | 121 秒 | 核销记录吻合 · matching_redemption | 需要人工核查 · redemption_time_mismatch | 核销记录吻合 · 模拟 POS 原始单和到店凭证确认是同一笔，商家时钟偏移 121 秒。 |
| R04 | 时间边界 | 180 秒 | 核销记录吻合 · matching_redemption | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · 模拟审核发现商家把另一桌的记录关联到相同流水号，当前证据不足以确认。 |
| R05 | 时间边界 | 240 秒 | 核销记录吻合 · matching_redemption | 需要人工核查 · redemption_time_mismatch | 核销记录吻合 · 模拟原始 POS 单据证明同一顾客兑换，商家时钟慢 4 分钟。 |
| R06 | 时间边界 | 300 秒 | 核销记录吻合 · matching_redemption | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · 模拟审核发现 POS 批量同步时复用了旧流水号，仍需人工核对。 |
| R07 | 时间边界 | 301 秒 | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · redemption_time_mismatch | 核销记录吻合 · 模拟平台原始记录和商家日志均确认同一次核销，商家本地时钟快 301 秒。 |
| R08 | 流水不一致 | 0 秒 | 需要人工核查 · redemption_reference_mismatch | 需要人工核查 · redemption_reference_mismatch | 需要人工核查 · 平台和商家的流水指向不同交易，现有记录不足以确认消费。 |
| R09 | 来源缺失 | — | 需要人工核查 · missing_source | 需要人工核查 · missing_source | 需要人工核查 · 只有平台核销记录，缺商家记录。 |
| R10 | 状态不一致 | — | 需要人工核查 · status_mismatch | 需要人工核查 · status_mismatch | 需要人工核查 · 双方对是否核销的记录互相冲突。 |
| R11 | 未核销且有效 | — | 支持申请退款 · unused_and_valid | 支持申请退款 · unused_and_valid | 支持申请退款 · 两份未核销记录一致，模拟券仍有效。 |
| R12 | 有效期 | — | 需要人工核查 · expired_voucher | 需要人工核查 · expired_voucher | 需要人工核查 · 到期时点已不能按未核销且有效给出建议。 |
| R13 | 有效期 | 60 秒 | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · 核销流水出现在券到期之后，需人工核查。 |
| R14 | 时间有效性 | 1 秒 | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · redemption_time_mismatch | 需要人工核查 · 两份流水都晚于本次判定时间，当前不能确认。 |
| R15 | 时间有效性 | — | 需要人工核查 · invalid_timestamp | 需要人工核查 · invalid_timestamp | 需要人工核查 · 商家记录未提供时区，无法对齐两个来源。 |
| R16-before | V200 补传前 | — | 需要人工核查 · status_mismatch | 需要人工核查 · status_mismatch | 需要人工核查 · 平台显示核销、商家仍显示未使用，模拟审核等待补传。 |
| R16-after | V200 补传后 | 60 秒 | 核销记录吻合 · matching_redemption | 核销记录吻合 · matching_redemption | 核销记录吻合 · 商家补传原始 POS 记录，经模拟审核确认流水与平台记录对应。 |
| R17 | 到店异常 | — | 支持申请退款 · unused_and_valid | 支持申请退款 · unused_and_valid | 支持申请退款 · 两边显示未核销，用户在店遇到无法接待，仍需人工处理到店异议。 |
