import logging
import random
from collections import defaultdict
from itertools import zip_longest
from config import Config

logger = logging.getLogger(__name__)

class CustomScheduler:
    def __init__(self, model, agents_per_step=None):
        self.model = model
        self.steps = 0
        self.agents = []
        self.agent_types = defaultdict(list)
        self.agent_pointer = 0

        # 每步调度执行的 agent 数量，-1 表示全部执行
        self.agents_per_step = agents_per_step if agents_per_step is not None else getattr(Config, "AGENTS_PER_STEP", -1)
        self.shuffle_order = getattr(Config, "SHUFFLE_AGENT_ORDER", False)
        self.interleave = getattr(Config, "INTERLEAVE_AGENT_TYPES", False)

    def add_agent(self, agent):
        self.agents.append(agent)
        self.agent_types[type(agent).__name__].append(agent)

    def get_all_agents(self):
        return list(self.agents)

    def get_agents_by_type(self, agent_type):
        return list(self.agent_types.get(agent_type, []))

    def sample_agents(self, count):
        total_agents = len(self.agents)
        if total_agents == 0:
            return []

        # 全部执行或执行数量大于总数，返回全部
        if count == -1 or count >= total_agents:
            selected = self.agents[:]
        else:
            # 轮流选择 agents，避免重复或遗漏
            start = self.agent_pointer
            end = start + count
            if end > total_agents:
                selected = self.agents[start:] + self.agents[:end - total_agents]
                self.agent_pointer = end - total_agents
            else:
                selected = self.agents[start:end]
                self.agent_pointer = end % total_agents

        # 是否打乱执行顺序
        if self.shuffle_order:
            random.shuffle(selected)

        # 是否交错富人和穷人执行
        if self.interleave:
            poor_agents = [a for a in selected if type(a).__name__.lower() == "pooragent"]
            rich_agents = [a for a in selected if type(a).__name__.lower() == "richagent"]
            interleaved = []
            for p, r in zip_longest(poor_agents, rich_agents):
                if p is not None:
                    interleaved.append(p)
                if r is not None:
                    interleaved.append(r)
            selected = interleaved

        return selected

    async def run_step(self):
        # 调度器步数加一（内部维护）
        self.steps += 1
        if hasattr(self.model, "steps"):
            self.model.steps = self.steps  # 同步模型步数

        logger.info(f"=== 🌀 Scheduler Step {self.steps} | Total Agents: {len(self.agents)} ===")

        selected_agents = self.sample_agents(self.agents_per_step)
        if not selected_agents:
            logger.warning(f"⚠️ Step {self.steps}: 没有 agent 被选中执行")
            return

        # 异步依次执行选中 agent 的 step() 协程
        for agent in selected_agents:
            try:
                await agent.step()
            except Exception as e:
                logger.error(f"[调度器错误] Agent {agent.unique_id} 执行失败: {e}", exc_info=True)
