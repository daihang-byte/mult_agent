# model.py
import os                      # 用于文件/目录操作（保存结果）
import random                  # 随机数工具（初始化随机位置、财富等）
import asyncio                 # 异步工具（并发调用 AI、关闭会话等）
import logging                 # 日志模块
import pandas as pd            # 用于将结果写成 CSV 表格
import numpy as np             # 数值计算（平均值等）
from typing import List, Dict # 类型注解
import mesa                    # Mesa 框架主包（模型/调度器等）
from mesa.space import MultiGrid  # 网格空间（支持多智能体放在格子上）
from config import Config      # 配置常量（你项目里的 Config）
from agents import PoorAgent, RichAgent, BaseAgent  # 引入智能体类
from market import WealthMarket  # 引入市场/事件记录器
from scheduler import CustomScheduler  # 自定义调度器（支持异步调度）

logger = logging.getLogger(__name__)  # 获取模块级 logger，用于记录模块日志


class WealthModel(mesa.Model):  # 定义模型类，继承 Mesa 的 Model
    def __init__(self, num_poor=None, num_rich=None, num_agents=None,
                 rich_ratio=Config.RICH_RATIO,
                 width=Config.WIDTH, height=Config.HEIGHT,
                 transparent=Config.TRANSPARENT, seed=Config.SEED,
                 experiment_id="UNKNOWN"):
        super().__init__()  # 调用父类构造器，完成必要初始化

        # 智能体数量配置
        if num_agents is not None:
            self.num_rich = num_rich if num_rich is not None else max(1, int(num_agents * rich_ratio))
            self.num_poor = num_poor if num_poor is not None else num_agents - self.num_rich
        else:
            self.num_poor = num_poor or Config.NUM_POOR
            self.num_rich = num_rich or Config.NUM_RICH

        self.max_workers_per_rich = max(1, int(self.num_poor / max(1, self.num_rich)))

        random.seed(seed)
        np.random.seed(seed)

        self.experiment_id = experiment_id
        self.model_id = f"Model_{random.randint(10000, 99999)}"
        self.transparent = transparent
        self.grid = MultiGrid(width, height, torus=True)
        self.market = WealthMarket(self)
        self.running = True
        self.current_id = 0
        self.steps = 0
        self.agents: List[BaseAgent] = []
        self.agent_dict: Dict[str, BaseAgent] = {}
        self.schedule = CustomScheduler(self)  # 你自己的调度器，支持异步

        # 异步请求并发控制信号量，传给所有智能体，用于限制同时调用 Ollama AI 接口数
        self.ollama_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_OLLAMA_REQUESTS)
        self.agents_per_step = Config.AGENTS_PER_STEP

        self.total_rich_output = 0
        self.total_poor_earnings = 0

        self.rich_agents: List[RichAgent] = []
        self.poor_agents: List[PoorAgent] = []

        self._create_agents(width, height)
        self._initial_hiring()
        self.update_totals()

        logger.info(
            f"🧪 初始化模型 {self.model_id}（实验 {self.experiment_id}） | 透明={self.transparent} | "
            f"穷人={self.num_poor} | 富人={self.num_rich} | 每位富人最多雇佣 {self.max_workers_per_rich} 人"
        )

        self.model_data: List[dict] = []
        self.agent_data: List[dict] = []

    def _create_agents(self, width: int, height: int) -> None:
        for _ in range(self.num_poor):
            pos = (self.random.randrange(width), self.random.randrange(height))
            # 传入信号量 ollama_semaphore，方便智能体异步调用限流
            agent = PoorAgent(f"Poor_{self.next_id()}", self, pos,
                              transparency=self.transparent,
                              ollama_semaphore=self.ollama_semaphore)
            agent.wealth = round(random.uniform(90, 110), 2)
            agent.moral_level = 1 if random.random() < 0.9 else 0
            agent.satisfaction = 0.5
            agent.effort = 0.0
            agent.last_output = 0.0
            agent.protesting = False
            agent.employed = False
            agent.wage = None
            agent.reservation_wage = None
            agent.last_action = None

            self.poor_agents.append(agent)
            self._register_agent(agent)

        for _ in range(self.num_rich):
            pos = (self.random.randrange(width), self.random.randrange(height))
            agent = RichAgent(f"Rich_{self.next_id()}", self, pos,
                              transparency=self.transparent,
                              ollama_semaphore=self.ollama_semaphore)
            agent.wealth = round(random.uniform(110, 130), 2)
            agent.moral_level = 1 if random.random() < 0.9 else 0
            agent.satisfaction = 0.5
            agent.effort = 0.0
            agent.last_output = 0.0
            agent.protesting = False
            agent.employed = False
            agent.wage = None
            agent.reservation_wage = None
            agent.last_action = None

            self.rich_agents.append(agent)
            self._register_agent(agent)

    def _register_agent(self, agent: BaseAgent):
        self.agents.append(agent)
        self.agent_dict[agent.unique_id] = agent
        self.schedule.add_agent(agent)
        self.grid.place_agent(agent, agent.pos)

    def _initial_hiring(self):
        rich_index = 0
        total_assigned = 0
        for poor in self.poor_agents:
            for _ in range(len(self.rich_agents)):
                rich = self.rich_agents[rich_index % len(self.rich_agents)]
                rich_index += 1
                if len(rich.workers) < self.max_workers_per_rich:
                    poor.employed = True
                    poor.employer = rich
                    poor.wage = round(random.uniform(2.0, 4.0), 2)
                    rich.workers.append(poor)
                    rich.wealth -= poor.wage
                    poor.wealth += poor.wage
                    self.market.record_event("initial_hire", f"{poor.unique_id} 雇佣于 {rich.unique_id}", wage=poor.wage)
                    total_assigned += 1
                    break
        logger.info(f"📋 初始雇佣完成 | 配对数: {total_assigned} / {len(self.poor_agents)}")

    async def step(self):
        self.steps += 1
        logger.info(f"🌀 Step {self.steps} 开始")

        # 异步运行调度器的异步 step，调用所有智能体异步 step 方法
        await self.schedule.run_step()

        self.update_totals()
        self.collect_data()

        latest = self.model_data[-1]
        logger.info(
            f"📊 Step {latest['step']} | Gini={latest['Gini']:.4f}, "
            f"贫富比={latest['Wealth_Ratio']:.2f}, "
            f"总产出={latest['Total_Rich_Production']:.2f}, "
            f"总工资={latest['Total_Poor_Earnings']:.2f}"
        )

    def update_totals(self):
        self.total_rich_output = 0
        self.total_poor_earnings = 0
        for rich in self.rich_agents:
            output = sum(getattr(worker, "last_output", 0) for worker in rich.workers)
            wages = sum(getattr(worker, "wage", 0) for worker in rich.workers)
            rich.last_output = output - wages
            self.total_rich_output += rich.last_output
            self.total_poor_earnings += wages
            if rich.last_output > 0:
                rich.wealth += rich.last_output

    def collect_data(self):
        self.model_data.append({
            "step": self.steps,
            "Gini": self.calculate_gini(),
            "Avg_Poor_Wealth": self.avg_poor_wealth(),
            "Avg_Rich_Wealth": self.avg_rich_wealth(),
            "Wealth_Ratio": self.wealth_ratio(),
            "Total_Wealth": self.total_wealth(),
            "Total_Rich_Production": round(self.total_rich_output, 2),
            "Total_Poor_Earnings": round(self.total_poor_earnings, 2),
            "transparent": self.transparent,
            "experiment": self.experiment_id,
        })

        for agent in self.schedule.get_all_agents():
            neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=False, radius=2)
            self.agent_data.append({
                "step": self.steps,
                "agent_id": agent.unique_id,
                "type": type(agent).__name__,
                "wealth": agent.wealth,
                "moral_level": getattr(agent, "moral_level", None),
                "wage": getattr(agent, "wage", None),
                "employed": getattr(agent, "employed", None),
                "reservation_wage": getattr(agent, "reservation_wage", None),
                "protesting": getattr(agent, "protesting", False),
                "last_action": getattr(agent, "last_action", None),
                "neighbors": [n.unique_id for n in neighbors],
            })

    def calculate_gini(self):
        wealths = [a.wealth for a in self.schedule.get_all_agents()]
        if not wealths or sum(wealths) == 0:
            return 0
        sorted_w = sorted(wealths)
        n = len(sorted_w)
        cum_wealth = sum(sorted_w)
        gini = sum((i + 1) * w for i, w in enumerate(sorted_w))
        return 2 * gini / (n * cum_wealth) - (n + 1) / n

    def avg_poor_wealth(self):
        poor = self.schedule.get_agents_by_type("PoorAgent")
        return np.mean([a.wealth for a in poor]) if poor else 0

    def avg_rich_wealth(self):
        rich = self.schedule.get_agents_by_type("RichAgent")
        return np.mean([a.wealth for a in rich]) if rich else 0

    def wealth_ratio(self):
        avg_poor = self.avg_poor_wealth()
        avg_rich = self.avg_rich_wealth()
        return avg_rich / avg_poor if avg_poor else 0

    def total_wealth(self):
        return sum(a.wealth for a in self.schedule.get_all_agents())

    def save_results(self, path: str):
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(self.model_data).to_csv(os.path.join(path, "model_data.csv"), index=False)
        pd.DataFrame(self.agent_data).to_csv(os.path.join(path, "agent_data.csv"), index=False)
        if Config.SAVE_EVENTS or Config.SAVE_INTERACTIONS:
            self.market.save_to_csv(path)

    async def close_sessions(self):
        # 异步关闭所有 agent 的 Ollama 会话，防止资源泄漏
        tasks = [agent.close() for agent in self.schedule.get_all_agents()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ 关闭 {self.schedule.get_all_agents()[i].unique_id} 会话失败: {result}")
        logger.info("✅ 所有 Agent 会话关闭完毕")
