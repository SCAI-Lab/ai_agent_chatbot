# 自动化实验运行器使用指南

## 快速开始

### 1. 启动Ollama服务（首次运行或需要特定配置）

```bash
# 设置环境变量启用Flash Attention和KV Cache
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

# 启动Ollama服务
ollama serve
```

**注意**: 在另一个终端窗口运行实验，保持Ollama服务运行。

### 2. 运行实验

```bash
cd /home/user/ai_agent/chatbot/tests/prompts4test/test
python3 run_experiments.py --rounds 5 10 20
```

## 核心概念

### 实验设计
本工具用于评估：**在固定的测试数据上，不同的分块大小N对处理时间和准确性的影响**

```
固定测试数据 (chats/mock_user/)
        ↓
    按 N 轮分块处理
        ↓
存储到 MemoBase (随机 User ID)
        ↓
    评估准确率和时间
```

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| **测试数据源** | 固定路径，所有实验共用 | `chats/mock_user/` |
| **MemoBase User ID** | 每次实验随机生成4位字符串 | `x7k2`, `a3d9`, `m5p1` |
| **N (rounds)** | 每个chunk包含的对话轮数（变量） | 5, 10, 20 |

**为什么这样设计？**
- ✅ 固定数据源 → 保证实验可比性
- ✅ 随机User ID → 确保实验独立性，不互相影响
- ✅ 只变化N → 准确评估分块大小的影响

## 使用方法

### 基本命令

```bash
# 测试多个N值
python3 run_experiments.py --rounds 5 10 20

# 测试更广泛的N值范围
python3 run_experiments.py --rounds 3 5 7 10 15 20 30

# 自定义实验前缀
python3 run_experiments.py --rounds 5 10 --user-prefix mytest

# 跳过评估（仅测试性能）
python3 run_experiments.py --rounds 5 10 20 --skip-eval
```

### 单独运行extract.py

```bash
# 使用默认随机User ID
python3 extract.py --rounds-per-chunk 5

# 指定User ID（用于调试）
python3 extract.py --rounds-per-chunk 5 --user-id test123
```

## 输出说明

### 运行时输出

```
================================================================================
AUTOMATED EXPERIMENT RUNNER
================================================================================
Configurations: Flash Attention=1, KV Cache=q8_0
Chat data source: chats/mock_user/
Rounds to test: [5, 10, 20]
================================================================================
================================================================================
Warming up model...
================================================================================
✓ Model warmed up and ready

================================================================================
EXPERIMENT 1/3: N=5 rounds per chunk
Experiment ID: exp_n5_x7k2
MemoBase User ID: x7k2
================================================================================

  [1/3] Running extraction (N=5)...
      Processing all chat files in chats/mock_user/

MemoBase User ID: x7k2
Reading chat data from: chats/mock_user/
Rounds per chunk: 5

[Session 1/5] File: chats/mock_user/1.json
  Chunks in this session: 2
  Chunk 1: 5 user turns, 5 assistant replies [Overall: 1/33]
  Chunk 2: 3 user turns, 3 assistant replies [Overall: 2/33]
  Session processing time: 2.45s

[Session 2/5] File: chats/mock_user/2.json
  Chunks in this session: 1
  Chunk 1: 5 user turns, 5 assistant replies [Overall: 3/33]
  Session processing time: 1.23s

[... Sessions 3-5 ...]

✓ All 5 sessions processed (33 chunks total)
Cost time(s) 12.34

  ✓ Extraction completed in 12.34s
  [2/3] Running evaluation...
  ✓ Evaluation completed:
    - Precision: 0.892
    - Recall: 0.856
    - F1 Score: 0.874
    - Redundancy: 0.110
    - TP=145, FP=18, FN=24
  [3/3] Recording Ollama metrics...
  ✓ Ollama metrics recorded:
    - TTFT: 0.123s
    - Prompt eval: 1.234s
    - Eval duration: 0.567s
    - Total duration: 2.456s

✓ Experiment 1/3 completed
  - Experiment ID: exp_n5_x7k2
  - MemoBase User ID: x7k2
```

### 最终汇总

```
================================================================================
ALL EXPERIMENTS COMPLETED
================================================================================

📊 Results Summary (Data source: chats/mock_user/):
   User ID |     N |   Duration |     TTFT | Precision |    Recall |        F1
------------------------------------------------------------------------------------------
      x7k2 |     5 |     12.34s |   0.123s |     0.892 |     0.856 |     0.874
      a3d9 |    10 |     15.67s |   0.145s |     0.910 |     0.875 |     0.892
      m5p1 |    20 |     22.89s |   0.178s |     0.925 |     0.890 |     0.907

📁 Detailed results saved to: results/experiment_summary_20250121_143156.json
```

## 记录的指标

### 时间指标
- **duration**: 提取总时间（秒）
- **ttft**: Time To First Token - 首个token生成时间
- **prompt_eval_duration**: 提示词评估时间
- **eval_duration**: 生成评估时间
- **total_duration**: Ollama总处理时间

### 准确性指标
- **precision**: 精确率 - 提取信息的准确性
- **recall**: 召回率 - 提取信息的完整性
- **f1**: F1分数 - 精确率和召回率的调和平均
- **redundancy_rate**: 冗余率 - 无关信息比例
- **tp**: True Positives - 正确提取的信息数量
- **fp**: False Positives - 错误提取的信息数量
- **fn**: False Negatives - 遗漏的信息数量

## 结果文件

### JSON格式

位置: `results/experiment_summary_<timestamp>.json`

```json
{
  "bench_n5_x7k2": {
    "experiment_id": "bench_n5_x7k2",
    "rounds": 5,
    "memobase_user_id": "x7k2",
    "chat_data_source": "chats/mock_user/",
    "duration": 12.34,
    "ttft": 0.123,
    "prompt_eval_duration": 1.234,
    "eval_duration": 0.567,
    "total_duration": 2.456,
    "precision": 0.892,
    "recall": 0.856,
    "f1": 0.874,
    "redundancy_rate": 0.110,
    "tp": 145,
    "fp": 18,
    "fn": 24
  }
}
```

## 目录结构

```
test/
├── chats/
│   ├── mock_user/          # 固定的测试数据源（所有实验共用）
│   │   ├── 1.json         # 对话文件
│   │   ├── 2.json
│   │   ├── 3.json
│   │   ├── 4.json
│   │   └── 5.json
│   ├── ground_truth/       # 标准答案
│   │   ├── 1.txt
│   │   └── ...
│   └── output/            # 评估输出
├── logs/                  # Ollama日志
├── results/               # 实验结果JSON
├── run_experiments.py     # 主实验运行器
├── extract.py            # 信息提取脚本
└── evaluate.py           # 评估脚本
```

## 实验流程详解

### 单个实验的完整流程

以 N=5 为例：

1. **生成随机User ID**: `x7k2`
2. **配置Ollama**: 启用Flash Attention和KV Cache
3. **读取数据**: 从 `chats/mock_user/` 读取所有5个JSON文件
4. **分块处理**: 每5轮对话分成一个chunk
5. **插入MemoBase**: 使用User ID `x7k2` 存储
6. **评估**: 计算precision、recall、F1等指标
7. **记录性能**: TTFT、处理时间等

### 多个实验的对比

| 实验 | User ID | 数据源 | N值 | 分块方式 | 结果 |
|------|---------|--------|-----|----------|------|
| 1 | x7k2 | mock_user | 5 | 每5轮一块 | P=0.892, F1=0.874 |
| 2 | a3d9 | mock_user | 10 | 每10轮一块 | P=0.910, F1=0.892 |
| 3 | m5p1 | mock_user | 20 | 每20轮一块 | P=0.925, F1=0.907 |

**关键点：**
- ✅ 相同数据源 → 结果可比
- ✅ 不同User ID → 实验独立
- ✅ 不同N值 → 准确评估影响

## 结果分析

### 分析N值的影响

```python
import json
import matplotlib.pyplot as plt

# 读取结果
with open('results/experiment_summary_xxx.json') as f:
    data = json.load(f)

# 提取数据
rounds = [v['rounds'] for v in data.values()]
f1_scores = [v['f1'] for v in data.values()]
durations = [v['duration'] for v in data.values()]

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(rounds, f1_scores, marker='o')
ax1.set_xlabel('Rounds per Chunk (N)')
ax1.set_ylabel('F1 Score')
ax1.set_title('Accuracy vs N')

ax2.plot(rounds, durations, marker='o', color='orange')
ax2.set_xlabel('Rounds per Chunk (N)')
ax2.set_ylabel('Duration (s)')
ax2.set_title('Processing Time vs N')

plt.tight_layout()
plt.savefig('experiment_analysis.png')
```

### 找到最佳N值

```python
# 计算效率分数（考虑时间和准确性）
for exp_id, metrics in data.items():
    efficiency = metrics['f1'] / metrics['duration']
    print(f"N={metrics['rounds']:2d}: "
          f"F1={metrics['f1']:.3f}, "
          f"Time={metrics['duration']:6.2f}s, "
          f"Efficiency={efficiency:.4f}")

# 输出示例：
# N= 5: F1=0.874, Time= 12.34s, Efficiency=0.0708
# N=10: F1=0.892, Time= 15.67s, Efficiency=0.0569
# N=20: F1=0.907, Time= 22.89s, Efficiency=0.0396
```

## 常见问题

### Q: 为什么每次实验要用不同的MemoBase User ID？
**A:** 确保每个实验都是独立的、干净的。如果使用相同的User ID，MemoBase会累积之前的数据，后面的实验会受到前面的影响。

### Q: 所有实验都处理相同的数据吗？
**A:** 是的！所有实验都处理 `chats/mock_user/` 里的5个JSON文件。唯一的区别是分块大小N。

### Q: 如何知道实验是否正确运行？
**A:** 检查以下几点：
- ✅ 输出显示 `Reading chat data from: chats/mock_user/`
- ✅ 每个实验的MemoBase User ID不同
- ✅ 处理了所有5个JSON文件
- ✅ 不同N值的分块数量不同

### Q: 实验结果如何比较？
**A:** 因为使用相同的测试数据，可以直接比较：
- N=5 vs N=10 vs N=20 的F1分数
- 找到准确性和速度的最佳平衡点

## 系统要求

### 必需
- Python 3.10+
- Ollama 已安装
- 模型 qwen2.5:7b-instruct 已下载
- sudo 权限（用于停止/启动Ollama服务）

### Python依赖
```bash
pip install memobase rich httpx
```

### 检查环境
```bash
# 检查Ollama
which ollama
ollama list | grep qwen2.5

# 检查权限
sudo -v

# 检查Python版本
python3 --version
```

## 故障排查

### Ollama启动失败
```bash
# 手动启动测试
ollama serve

# 检查端口
curl http://127.0.0.1:11434/api/tags
```

### 权限错误
```bash
# 验证sudo权限
sudo systemctl status ollama

# 授予脚本执行权限
chmod +x run_experiments.py extract.py evaluate.py
```

### 模型未找到
```bash
# 下载模型
ollama pull qwen2.5:7b-instruct

# 验证
ollama list
```

### 依赖缺失
```bash
# 安装所有依赖
pip install memobase rich httpx

# 或使用requirements.txt（如果有）
pip install -r requirements.txt
```

## 高级用法

### 修改Ollama配置

编辑 `run_experiments.py` 中的环境变量：

```python
env = {
    "OLLAMA_FLASH_ATTENTION": "1",      # 启用Flash Attention
    "OLLAMA_KV_CACHE_TYPE": "q8_0",     # KV Cache量化类型
    # 其他可选配置：
    # "OLLAMA_NUM_GPU": "1",
    # "OLLAMA_MAX_LOADED_MODELS": "1",
}
```

### 更换测试数据

修改 `CHAT_DATA_DIR` 常量：

```python
# run_experiments.py 和 extract.py
CHAT_DATA_DIR = "my_custom_data"  # 指向 chats/my_custom_data/
```

### 重复实验验证

```bash
# 运行3次实验验证稳定性
for i in {1..3}; do
    python3 run_experiments.py --rounds 5 10 20 --user-prefix run${i}
done
```

## 命令行参数

### run_experiments.py

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--rounds` | 要测试的N值列表（必需） | - | `--rounds 5 10 20` |
| `--user-prefix` | 实验ID前缀 | `exp` | `--user-prefix bench` |
| `--skip-eval` | 跳过评估步骤 | False | `--skip-eval` |

### extract.py

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--user-id` | MemoBase User ID | 随机生成 | `--user-id test123` |
| `--rounds-per-chunk` | 每个chunk的轮数 | 5 | `-r 10` |
| `--project_url` | MemoBase服务URL | `http://localhost:8019` | `-u http://localhost:8019` |
| `--project_token` | API访问令牌 | `secret` | `-t mytoken` |
| `--skip-profile` | 跳过获取用户画像 | False | `--skip-profile` |

## 完整示例

### 1. 基础实验
```bash
python3 run_experiments.py --rounds 5 10 20
```

### 2. 大范围测试
```bash
python3 run_experiments.py --rounds 3 5 7 10 15 20 30 40 50
```

### 3. 重复验证
```bash
python3 run_experiments.py --rounds 10 --user-prefix verify1
python3 run_experiments.py --rounds 10 --user-prefix verify2
python3 run_experiments.py --rounds 10 --user-prefix verify3
# 对比三次结果的稳定性
```

### 4. 性能测试（无评估）
```bash
python3 run_experiments.py --rounds 5 10 20 --skip-eval
```

## 相关文件

- `run_experiments.py` - 自动化实验运行器主脚本
- `extract.py` - 信息提取脚本
- `evaluate.py` - 评估脚本
- `../../README.md` - 完整测试套件文档
