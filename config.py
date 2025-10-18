# config.py
class Config:
    # ========== AI 智能体模型配置 ==========
    POOR_AGENT_MODEL = "llama3.2-vision:11b"
    RICH_AGENT_MODEL = "llama3.2-vision:11b"

    OLLAMA_BASE_URL = "http://localhost:11434/api/generate"  # Ollama 推理服务地址

    # ========== 模拟环境配置 ==========
    WIDTH = 30                 # 网格宽度
    HEIGHT = 30                # 网格高度
    NUM_POOR = 180             # 穷人数量
    NUM_RICH = 20              # 富人数量
    TRANSPARENT = True         # 穷人是否能看到富人财富（影响上下文构建）
    SEED = 42                  # 随机种子，确保实验可复现
    RICH_RATIO = 0.1  # 富人占总人口比例（用于 num_agents 模式）

    # ========== 智能体参数 ==========
    MAX_MEMORY_SIZE = 10       # 每个智能体最多保留多少条记忆
    MAX_MEMORY_EVENTS = 3      # 构建 prompt 时纳入的最近事件数量
    SIMULATION_STEPS = 5       # 总模拟步数（由外部主控循环控制）

    # ========== 并发控制 ==========
    MAX_CONCURRENT_OLLAMA_REQUESTS = 1  # 限制同时进行的 AI 请求数量，防止本地 overload

    # ========== 调度器设置 ==========
    AGENTS_PER_STEP = 200                # 每步最大处理代理数
    SHUFFLE_AGENT_ORDER = True         # 是否在每步内打乱代理顺序
    INTERLEAVE_AGENT_TYPES = True      # 是否交叉调度穷人与富人，防止类别偏置
    DEBUG_MODE = False  # 👉 新增此项，用于全局调试开关

    # ========== 日志与保存 ==========
    LOG_LEVEL = "INFO"                 # 控制日志输出等级
    SAVE_EVENTS = True                 # 是否保存事件日志（如雇佣、抗议、生产等）
    SAVE_INTERACTIONS = True           # 是否保存 AI 对话记录（prompt + response）

    ENABLE_PERSONALITY = True  # 是否开启人格特征（Big Five）影响行为
    ENABLE_MEMORY_CONTEXT = True  # 是否启用智能体记忆上下文
    USE_SIMPLE_MODEL_NAME = False  # 是否切换到更轻量模型如 llama3:8b
