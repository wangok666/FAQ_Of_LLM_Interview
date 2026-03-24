1. 长期记忆,短期记忆怎么做的?
2. Skills解释
3. openclaw是什么,架构怎么样?
4. agent开发有哪些范式,用过哪些,什么情况用什么范式?
5. LangGraph搭建多 agent 的系统 怎么做的?
6. langgraph为什么选择它?有其他的了解使用过吗?
7. 反思机制的有点和不足,什么场景下面用
8. 调研分析型agent怎么搭建的? 
9. langgraph里面不是有个human in the loop的设计吗?其设计是怎么样的?
10. agent 项目框架怎么设计呢?
11. 议价流程到底怎么做?
12. 服务并发怎么做的?数量是多少?
13. 整个 Agent 项目你觉得还有什么待优化的点


---

## 1. 在agent搭建里面,尤其是司机货主撮合的agent里面,怎么做长期记忆,短期记忆

### 短期记忆 (Short-term Memory)

短期记忆主要负责维护**当前会话**的状态，确保 Agent 理解当前货源的议价背景、装货细节和司机的即时需求。

#### 实现方式
* **滑动窗口 (Sliding Window):** 仅保留最近的 $N$ 轮对话。这在处理高频沟通（如反复议价）时能有效控制 Token 成本，防止模型因上下文过长而产生幻觉。
* **概要缓存 (Summary Buffer):** 对历史对话进行分段摘要。Agent 会将之前的沟通要点（如：司机已报底价、要求下午3点前装货）总结为关键信息，带入后续对话。
* **回答信息**: 存储在向量数据库里面

### 长期记忆 (Long-term Memory)

长期记忆用于沉淀**用户画像、历史偏好和业务知识**，是实现“个性化推荐”的关键。

### 实现方式
* **结构化用户画像 (Profile Store):**
    * 利用关系型数据库 (PostgreSQL) 存储静态属性（车型、吨位、证件状态）。
    * **动态属性：** 记录司机的价格敏感度（如：经常接受低于市场价10%的单子）。
* **知识图谱 (Knowledge Graph):** 存储复杂的物流关系（如：某工业区禁行策略、常去装货点的排队规律）。


## 2. 大模型的skill是什么?Skills解释,给出例子. 和mcp工具有什么区别

大模型的 Skill 与 MCP（Model Context Protocol，模型上下文协议） 是 AI Agent 能力扩展中的两种核心机制，二者并非替代关系，而是互补关系，共同提升模型的实用性。

Skill 本质上是将完成某一类任务所需的领域知识、操作流程、最佳实践、提示词指令以及可选的工具/脚本打包成一个可复用的“能力包”。它最早由 Anthropic 公司在 Claude 模型中提出，并逐渐演变为开放标准，被广泛应用于各种 Agent 开发框架中。

### Skill 文件夹结构
```
code-reviewer/
├── SKILL.md                  # 核心文件（必需）
├── references/               # 可选：参考文档
│   └── code-review-checklist.md
├── scripts/                  # 可选：辅助脚本（若平台支持沙箱执行）
│   └── static_analysis.py
└── assets/                   # 可选：静态资源（如模板图片）
    └── review-template.png
```

### 完整 SKILL.md 内容

直接复制以下内容保存为 `SKILL.md` 即可使用

```markdown
---
name: code-reviewer
description: 当用户要求审查代码、检查代码质量、找出潜在问题或提出改进建议时，自动加载此 Skill。适用于任何编程语言的代码审查任务。
version: 1.0.0
allowed_tools: ["filesystem", "terminal"]  # 可调用 MCP 工具读取文件或运行分析
required_context: ["project_structure", "coding_standards"]
tags: [code-review, quality-assurance, security, best-practices]
---

# 代码审查 Skill（Code Review Skill）

## 技能概述
本 Skill 使 AI 成为专业的代码审查专家，按照行业标准（安全性、性能、可维护性、可读性）系统审查代码，避免随意评论或遗漏关键问题。审查过程必须严格遵循以下 SOP（标准操作程序），确保输出客观、专业、一致。

## 使用触发条件
- 用户输入包含：“审查代码”“code review”“检查这段代码”“找 bug”“优化代码”“安全性检查”等关键词。
- 用户上传/粘贴代码文件或目录时。

## 审查流程（必须严格按顺序执行）
1. **理解上下文（Context Understanding）**  
   - 首先阅读整个代码及其所在文件/项目结构。  
   - 识别语言、框架、主要功能和业务意图。  
   - 如果需要，使用 MCP 工具读取相关文件（如 requirements.txt、README）。

2. **执行静态分析**  
   - 检查常见问题：语法错误、安全漏洞（SQL 注入、XSS、命令注入等）、性能瓶颈、内存泄漏风险。  
   - （可选）调用 scripts/static_analysis.py 执行自动化扫描。

3. **分维度审查**（使用以下检查清单）
   - **安全性（Security）**：  
     - 输入验证、认证授权、敏感数据处理、第三方库漏洞。  
   - **性能与效率（Performance）**：  
     - 时间/空间复杂度、循环优化、数据库查询效率。  
   - **可读性与可维护性（Readability & Maintainability）**：  
     - 命名规范、注释完整性、代码重复、单一职责原则。  
   - **代码风格与规范（Style & Standards）**：  
     - 符合 PEP8（Python）、ESLint（JS）或其他语言标准。  
   - **测试与鲁棒性（Testing & Robustness）**：  
     - 边缘案例处理、异常捕获、单元测试覆盖建议。

4. **提出改进建议**  
   - 每条问题必须给出：  
     - 问题描述 + 严重程度（Critical / High / Medium / Low） + 具体位置（行号或函数名） + 改进代码示例。  
   - 优先级排序：先修复 Critical 问题。

5. **生成最终报告**  
   - 使用固定模板输出（见下文）。  
   - 总结整体评分（满分 10 分）及改进优先级列表。

## 输出格式模板（必须使用）
\```
# 代码审查报告

## 基本信息
- 文件/模块： [文件名]
- 语言： [语言]
- 审查时间： [当前时间]
- 整体评分： X/10

## 发现问题（按严重程度排序）
1. **严重程度：Critical**  
   问题：...  
   位置：...  
   建议：...

## 改进建议
- 立即修复：...
- 建议优化：...

## 总结与推荐
- 优点：...
- 下一步行动：...
\```

## 参考资源
- references/code-review-checklist.md（可按需加载详细检查清单）
- 常见漏洞参考：OWASP Top 10、CWE 列表

## 注意事项
- 永远保持客观、中性语气，不得使用“绝对不行”“必须”等主观词语。
- 若代码过长，分模块审查并标注。
- 若发现无法判断的问题，明确说明“需更多上下文”并建议用户补充。
- 审查完成后，询问用户是否需要重构代码或生成测试用例。

**此 Skill 已完成。请严格按照以上流程执行。**
```

### 如何使用此 Skill
1. 将整个 `code-reviewer` 文件夹打包为 ZIP 或直接放入支持 Skills 的目录。
2. 在 Claude、Cursor 等工具中上传/安装该 Skill。
3. 直接输入：“请审查以下代码” + 粘贴代码，即可自动触发。
4. 可结合 MCP 工具实现真实文件读取和脚本执行，进一步提升自动化程度。


## 3. openclaw是什么,架构怎么样?

```
+-------------------------------+     +-------------------------------+
|       客户端层 (Surfaces)     |<--->|       渠道适配层 (Channels)   |
|  WhatsApp / Telegram / Slack  |     |  Baileys / grammY / discord.js |
|  Web UI / CLI / macOS App     |     |  iMessage / Discord / Matrix   |
|  iOS/Android Nodes (摄像头等) |     +-------------------------------+
+-------------------------------+                 |
                                                ▼
                                     +-------------------------------+
                                     |      Gateway 控制平面层       |
                                     |   (WebSocket Server)   |
                                     |  • 消息路由 / 会话隔离        |
                                     |  • 访问控制 / 插件加载        |
                                     |  • 事件驱动循环 (消息+Cron+Webhook) |
                                     +-------------------------------+
                                                |
                                                ▼
                                     +-------------------------------+
                                     |     Agent Runtime (Pi Core)   |
                                     |   • 上下文组装 (感知层)       |
                                     |   • LLM 推理循环 (决策层)     |
                                     |   • 工具调用 & 执行 (执行层)  |
                                     |   • 技能/能力注册 (Capabilities)|
                                     |   • 反思校验 & 状态持久化     |
                                     +-------------------------------+
                                                |
                          +---------------------+---------------------+
                          |                                           |
                    [ Memory 系统 ]                            [ Tools / Skills ]
                    • 会话历史日志 (JSON append-only)          • 浏览器 / Bash / 文件 / Canvas
                    • 向量数据库 (SQLite + BM25)             • 插件扩展 / 沙箱 (Docker)
                    • 自动压缩总结 (compaction)              • 权限策略 / 多代理路由
```

## 4. agent开发有哪些范式,什么情况用什么范式?

在目前的 AI Agent（智能体）开发领域，主流的开发范式经历了从“单体简单推理”到“复杂多机协作”的演进。

###  ReAct 范式 (Reasoning + Acting)
这是最经典、最基础的单 Agent 范式。它模仿人类的思考方式：先思考（Thought），再行动（Action），然后观察结果（Observation），循环往复。

* **核心逻辑：** `Thought -> Action -> Observation -> Thought...`
* **适用场景：**
    * **简单工具调用：** 如查询天气、搜索实时新闻并总结。
    * **动态环境：** 下一步行动高度依赖于上一步返回的结果。
* **优点：** 逻辑直观，实现简单（如 LangChain 的 `ZeroShotAgent`）。
* **缺点：** 容易陷入死循环（Loop），且对于复杂任务容易“走丢”（忘记最终目标）。

### 计划与执行范式 (Plan-and-Execute)
这种范式将“规划”与“执行”解耦。先由一个大脑生成完整的任务清单，再由执行器逐一完成。

* **核心逻辑：** `Planner (生成 Plan) -> Executor (按顺序执行) -> Re-planner (根据结果修正计划)`。
* **适用场景：**
    * **长流程任务：** 如“帮我调研 A 行业，写一份报告并发送邮件给老板”。
    * **需要稳定性的场景：** 避免 Agent 在执行过程中因为一步报错而全盘崩溃。
* **优点：** 目标感强，效率高（部分任务可以并行执行）。
* **缺点：** 对 Planner 的能力要求极高，一旦初始计划错误且没有重规划机制，结果会很糟。

### 反思与自我修正范式 (Reflexion / Self-Correction)
在这种范式下，Agent 不会直接给出最终答案，而是先生成一个草稿，然后自我检查或由另一个“审查 Agent”打分，不断迭代直至达标。

* **核心逻辑：** `Generate -> Critique -> Revise`。
* **适用场景：**
    * **代码生成：** 生成代码后运行单元测试，根据错误信息自动修复。
    * **高质量文案创作：** 不断润色语言风格。
    * **科学推理：** 验证数学推导的严谨性。
* **优点：** 显著提升输出质量，减少幻觉。
* **缺点：** 消耗 Token 较多，响应延迟高。

### 多智能体协作范式 (Multi-Agent Systems, MAS)
通过引入不同的角色（Role-playing）让多个 Agent 协作。例如：一个产品经理 Agent、一个程序员 Agent、一个测试员 Agent。

* **核心逻辑：** `SOP (标准作业程序) + Role Play + Messaging`。
* **适用场景：**
    * **复杂软件工程：** 如 MetaGPT 自动生成整个项目的代码。
    * **对抗与辩论：** 通过两个 Agent 互怼来挖掘问题的深度。
    * **大型分工任务：** 需要不同专业知识（如法律、金融、技术）交叉协作的场景。
* **优点：** 模块化程度高，能处理极高复杂度的任务。
* **缺点：** 编排复杂（Orchestration），通信开销大，容易出现“群体迷失”。


## 5. LangGraph搭建多 agent 的系统,怎么做的?

用 LangGraph 搭建多 Agent 系统，核心思想是将 Agent 抽象为图（Graph）中的节点（Nodes），将 Agent 之间的通信和切换逻辑抽象为边（Edges），并利用 State（状态） 在它们之间传递信息。

如果说传统的 LangChain 是“链式”的，那么 LangGraph 就是“网状”的，它通过循环控制解决了复杂多智能体协作中“谁该下一步行动”的问题。


```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
```

```python
config = {"configurable": {"thread_id": "user_123"}}
graph.invoke(input_data, config)
```
