20260323 面试记录

工作职责:
1. 基于海量文本或销售数据进行挖掘分析，提取高质量特征与话术，支持销售人员培训、知识库补充、销售策略制定及文本内容优化等场景，提供 AI 能力支撑。
2. 参与的技术方向包括但不限于：
    ○ 文本挖掘：对用户文本进行画像构建、关键词抽取、高质量话术提炼等。
    ○ 文本质量评估：结合业务标准构建评价指标体系并训练自动评分模型。
    ○ 文本分类：如情感分析、销售流程分环节分类等。
    ○ 权益发放与增益模型研发：提升营销资源投放效率。
    ○ 大模型应用与定制：熟练应用和调优大语言模型（如 ChatGPT、Qwen、DeepSeek等）以赋能销售场景，例如话术生成、客户互动自动化、智能总结与推荐等。
任职资格:
1. 计算机相关专业本科及以上学历，经验不限，能力优先。
2. 扎实的编程能力，良好的数据结构和算法基础。
3. 对 NLP 技术有深入理解，熟练掌握序列标注、文本分类、文本评价等任务的主流方法。
4. 对机器学习方法掌握熟练，有营销类策略（如增益模型、A/B 测试设计）经验者优先。

---


上海-大连路地铁站 双休，975-985的样子,甲方

---

1. 做一个强化学习的Agent怎么做的?
2. 强化学习里面参数 $G$, $\epsilon$, $\beta$ (0.01-KL惩罚系数)调大调小有什么影响
3. 奖励函数怎么写的?
4. 多分类任务为什么不用bert?
5. 最近看了什么论文,稍微讲一下
6. 多轮对话怎么做的?
7. 上下文工程怎么做的?
8. 长期记忆,短期记忆怎么做的?
9. 怎么做的工程管理


---



太久没做算法题目了,我之后一定要把这个记住,妈的


- 最长回文子串
```python
def longestPalindrome(s: str) -> str:
    n = len(s)
    if n < 2:
        return s
    
    # 初始化 DP 表，单个字符本身一定是回文
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True
        
    max_len = 1
    start = 0
    
    # L 是子串长度，从 2 开始逐渐增加
    for L in range(2, n + 1):
        for i in range(n):
            j = L + i - 1  # 结束索引
            if j >= n:
                break
                
            if s[i] == s[j]:
                if j - i < 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            
            # 记录最长长度和起始位置
            if dp[i][j] and L > max_len:
                max_len = L
                start = i
                
    return s[start:start + max_len]
```

- 二分类查找数组

input = [1, 6, 56, 78, 99]

target = 6

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        # 取中间索引，(left + right) // 2 在极端情况下可能溢出
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid  # 找到目标，返回下标
        elif nums[mid] < target:
            left = mid + 1  # 目标在右边，忽略左半部分
        else:
            right = mid - 1  # 目标在左边，忽略右半部分
            
    return -1  # 没找到
```
