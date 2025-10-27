1. 10.21 重新加载了原始prompt。在没有flash attention和kv cache的情况下，重新跑模型的evaluate。   ✅

如果跑一个结果和之前一模一样，说明之前用的是原prompt ❌

2. 用原prompt跑flash-attention和kv-cache下N=5的qwen，评估表现 

3.现在做的事有意义吗？应该先从产品角度考虑，不然做prompt和N的测试意义不大，最多快一点点，还会牺牲稳定性

4.flash- attention和kv-cache在1k多点的场景下加速不多


10.22和shre谈话：

1.夜间的处理时间可以很长

2.需要好看一点的evaluation指标

3.仔细看一下火影忍者的仓库

4.情感分析更好的实现

5.用一点RAG成分，类似naruto或者trump chatbot，可以把人物的对话总结成带时间戳（metadata）的总结（或者甚至用原记录），用于丰富人物背景故事。

6.长期记忆要控制增长，也许可以直接用滑动窗口