# Factorized Voice-Style Dataset Plan

本文件只描述后续数据需求。本阶段不生成数据、不筛选大规模音频，也不训练模型。

## 1. 数据来源

使用原始 VoiceDesign 作为候选生成器：

```text
clean voice_prompt + style_prompt + text -> K candidate wavs
```

`voice_prompt` 应只描述相对稳定的说话人/音色属性，`style_prompt` 应只描述本次韵律与表达控制。数据表必须分别保存二者，不能只保存拼接后的字符串。每条数据还应保存 `sample_id`、`voice_id`、`style_id`、`text_id`、随机种子、生成配置、checkpoint revision 和音频路径。

## 2. 多源标注

候选音频后续可通过互补信号标注：

- **Qwen3.5-Omni**：仅作为 style adherence auxiliary judge；保存严格 JSON、模型快照和原始响应。
- **ECAPA/CAM++**：衡量同一 voice condition 下的 speaker consistency；不由 Omni 替代。
- **Objective acoustics**：speaking rate、F0、energy、pause、duration，并针对具体 style 使用有方向性的相对指标。
- **ASR**：转写并计算 CER/内容正确性；不由 style judge 替代。
- **Quality checks**：silence、clipping、RMS、duration、异常长停顿及解码失败。

所有自动标签都应保留缺失值和错误状态，不能把 API/解析失败当作低质量语音标签。

## 3. Preference pair

在相同 `(voice_prompt, style_prompt, text)` condition 内构造候选对：

```text
chosen   = speaker consistency 高 + style 符合 + 内容正确 + 音质正常
rejected = speaker drift 或 style 不符 或 内容错误 或音质异常
```

优先选择只有一个主要失败原因的 rejected，避免模型无法归因。保存 chosen/rejected 的各维度分数、阈值版本和 pair provenance。训练/验证/测试应按 voice condition 和 text 做防泄漏划分。

## 4. 后续训练阶段

```text
Stage 1: SFT warm-up for adapter / LoRA
Stage 2: DPO or ORPO preference tuning
Stage 3: optional GRPO only after reward is stable
```

Stage 1 应先只训练新 adapter（或受控 LoRA），冻结 Talker 主体并验证 legacy regression。Stage 2 只有在 preference pair 人工抽检通过后进行。Stage 3 只有在 reward calibration、对抗探针和跨域稳定性充分验证后考虑。

本阶段不执行以上任何训练步骤。
