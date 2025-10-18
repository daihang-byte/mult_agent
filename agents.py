from mesa import Agent
import logging
import random
import asyncio
import math
from typing import Dict, List, TYPE_CHECKING, Union
from ollama_core import OllamaAgentCore
from typing import cast
import numpy as np

if TYPE_CHECKING:
    from model import WealthModel

logger = logging.getLogger(__name__)


def calculate_gini(values: List[float]) -> float:
    """计算基尼系数"""
    if not values:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(sorted_vals)
    return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n


class BaseAgent(Agent):
    """基础智能体"""

    def __init__(
        self,
        unique_id: Union[int, str],
        model: "WealthModel",
        pos=None,
        agent_type="base",
        transparency=False,
        ollama_semaphore=None,
    ):
        super().__init__(unique_id, model)
        self.unique_id = str(unique_id)
        self.model = cast("WealthModel", model)
        self.pos = pos
        self.agent_type = agent_type
        self.wealth_transparency = 1 if transparency else 0

        # 初始化道德属性
        init_binary = 1 if random.random() < 0.9 else 0
        self.morality_score = 0.9 if init_binary else 0.1
        self.moral_level = 1 if self.morality_score >= 0.6 else 0

        self.ollama_semaphore = ollama_semaphore
        self.wealth = 0.0
        self.wealth_last = self.wealth
        self.satisfaction = 0.5
        self.last_action = None
        self.last_speech = ""
        self.employer = None
        self.workers: List["BaseAgent"] = []
        self.memory: List[Dict] = []
        self.max_memory = 10
        self.ai_recommended_moral_level = None
        self.moral_flip_count = 0
        self.moral_last_changed_step = -1

        # 道德演化参数
        self.mutation_rate = 0.05
        self.noise_scale = 0.08
        self.mutation_magnitude = 0.3
        self.moral_threshold = 0.6

        # AI 核心
        self.ai_core = OllamaAgentCore(
            agent_id=self.unique_id,
            agent_type=self.agent_type,
            moral_level=self.moral_level,
            transparency="high" if transparency else "low",
            ollama_semaphore=self.ollama_semaphore,
            model=None,
        )

        logger.info(
            f"[{self.agent_type.upper()} {self.unique_id}] 初始化 | Pos: {self.pos} | "
            f"MoralScore: {self.morality_score:.2f} | Moral: {self.moral_level} | 透明度: {self.wealth_transparency}"
        )

    async def initialize(self):
        try:
            await self.ai_core.initialize()
        except Exception as e:
            logger.error(f"{self.unique_id} 初始化失败: {e}", exc_info=True)

    async def close(self):
        try:
            await self.ai_core.close()
        except Exception as e:
            logger.error(f"{self.unique_id} 关闭失败: {e}", exc_info=True)

    def perceive_neighbors(self, radius=2) -> List["BaseAgent"]:
        return self.model.grid.get_neighbors(self.pos, moore=True, radius=radius, include_center=False)

    def record_decision_from_ai(self, decision: dict):
        step = getattr(self.model.schedule, "steps", -1)
        self.last_action = decision.get("action")
        self.last_speech = decision.get("speech", "")

        if "moral_level" in decision and decision["moral_level"] in [0, 1]:
            self.ai_recommended_moral_level = decision["moral_level"]
            logger.debug(f"[{self.unique_id}] AI 推荐道德偏好: {self.ai_recommended_moral_level}")

        wealth_delta = self.wealth - self.wealth_last
        record = {
            "step": step,
            "agent_id": self.unique_id,
            "agent_type": self.agent_type,
            "pos": self.pos,
            "wealth": round(self.wealth, 2),
            "wealth_delta": round(wealth_delta, 2),
            "decision": decision,
            "moral_level": self.moral_level,
            "morality_score": round(self.morality_score, 3),
        }

        self.memory.append(record)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

        if hasattr(self.model, "decision_log"):
            self.model.decision_log.append(record)

        logger.info(f"Step {step} | [{self.agent_type.upper()} {self.unique_id}] Action: {self.last_action} | Pos: {self.pos}")

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty_cells = [cell for cell in possible_steps if self.model.grid.is_cell_empty(cell)]
        if empty_cells:
            new_pos = random.choice(empty_cells)
            self.model.grid.move_agent(self, new_pos)
            self.pos = new_pos
            logger.info(f"[{self.agent_type.upper()} {self.unique_id}] 移动到 {self.pos}")

    def get_available_actions(self) -> List[str]:
        return ["move"]

    def log_status(self):
        logger.info(
            f"[{self.agent_type.upper():>5} {self.unique_id:>8}] Pos: {str(self.pos):>10} | "
            f"MoralScore: {self.morality_score:.3f} | Moral: {self.moral_level} | "
            f"Wealth: {self.wealth:8.2f} | Transparency: {self.wealth_transparency}"
        )

    async def decide_moral_level(self):
        neighbors = self.perceive_neighbors(radius=2)
        altruist_ratio = sum(1 for n in neighbors if getattr(n, "moral_level", 0) == 1) / max(len(neighbors), 1)
        wealth_diff = self.wealth - self.wealth_last
        norm_wealth_diff = max(-1.0, min(1.0, wealth_diff / (abs(self.wealth_last) + 1e-6)))

        recent = self.memory[-5:]
        recent_gain_ratio = sum(1 for r in recent if r.get("wealth", 0) - self.wealth_last > 0) / max(len(recent), 1)

        sat_norm = max(-1.0, min(1.0, (self.satisfaction - 0.5) * 2))
        ai_suggestion = 0
        if self.ai_recommended_moral_level in [0, 1]:
            ai_suggestion = 1 if self.ai_recommended_moral_level == 1 else -1

        step = getattr(self.model.schedule, "steps", 0)
        bias = -0.1 if step > 30 else 0.0

        w_ai, w_neighbors, w_wealth, w_satis, w_recent = 0.7, 1.0, 1.3, 1.0, 0.8
        neighbors_signal = (altruist_ratio - 0.5) * 2
        wealth_signal = norm_wealth_diff if norm_wealth_diff >= 0 else norm_wealth_diff * 1.5
        satis_signal = sat_norm if sat_norm >= 0 else sat_norm * 1.5

        # 基尼系数感知
        gini_signal = 0.0
        if self.wealth_transparency:
            all_wealths = [a.wealth for a in self.model.schedule.agents]
            gini = calculate_gini(all_wealths)
            gini_signal = 0.5 if gini > 0.4 and self.moral_level == 1 else -0.5 if gini > 0.4 else 0.0

        score = w_ai * ai_suggestion + w_neighbors * neighbors_signal + w_wealth * wealth_signal + w_satis * satis_signal + w_recent * (recent_gain_ratio - 0.5) * 2 + bias + gini_signal
        target = 0.5 + math.tanh(score / 2.0) * 0.5

        # 演化 + 噪声 + 突变
        alpha = 0.3
        new_score = max(0.0, min(1.0, (1 - alpha) * self.morality_score + alpha * target + random.uniform(-self.noise_scale, self.noise_scale)))
        if random.random() < self.mutation_rate:
            delta = self.mutation_magnitude * (1 if random.random() > 0.5 else -1)
            new_score = max(0.0, min(1.0, new_score + delta))
            logger.info(f"[{self.unique_id}] 突变 -> morality_score={new_score:.3f}")
        if random.random() < 0.01:
            new_score = random.random()
            logger.debug(f"[{self.unique_id}] 极小概率扰动 -> morality_score={new_score:.3f}")

        old_score = self.morality_score
        self.morality_score = new_score
        new_moral_level = 1 if new_score >= self.moral_threshold else 0

        if new_moral_level != self.moral_level:
            self.moral_level = new_moral_level
            self.moral_flip_count += 1
            self.moral_last_changed_step = step
            self.ai_core.moral_level = new_moral_level
            logger.info(f"[{self.unique_id}] 道德标签更新: {old_score:.3f}->{new_score:.3f}, MoralLevel: {new_moral_level}")

        self.wealth_last = self.wealth
        self.ai_recommended_moral_level = None


# -------------------- PoorAgent --------------------

class PoorAgent(BaseAgent):
    def __init__(self, unique_id, model, pos=None, transparency=False, ollama_semaphore=None):
        super().__init__(unique_id, model, pos, agent_type="poor", transparency=transparency, ollama_semaphore=ollama_semaphore)
        self.wage = 0.0
        self.effort = round(random.uniform(0.3, 1.0), 2)
        self.production = 0.0
        self.last_protest_step = -1
        self.protesting = False
        self.wealth = round(random.uniform(90, 110), 2)
        self.fairness_signal = 0.5

        self.action_map = {
            "work": self._work,
            "resign": self._resign,
            "protest": self._protest,
            "search_job": self._search_job,
            "move": self.move,
            "share_wealth": self._share_wealth,
        }

    def log_status(self):
        employer_id = self.employer.unique_id if self.employer else "None"
        logger.info(
            f"[POOR {self.unique_id:>8}] Pos: {self.pos} | MoralScore: {self.morality_score:.3f} | Moral: {self.moral_level} | "
            f"Wealth: {self.wealth:8.2f} | Wage: {self.wage:.2f} | Employer: {employer_id} | Protesting: {self.protesting}"
        )

    def get_available_actions(self):
        actions = ["move", "work", "resign", "protest"]
        if not self.employer:
            actions.append("search_job")
        if self.moral_level == 1:
            actions.append("share_wealth")
        return actions

    async def step(self):
        step = getattr(self.model.schedule, "steps", -1)

        # 可见邻居和基尼信号
        neighbors = self.perceive_neighbors(radius=2)
        all_wealth = [a.wealth for a in self.model.schedule.agents] if self.wealth_transparency else []
        gini = calculate_gini(all_wealth) if all_wealth else 0.0
        self.fairness_signal = 1 - gini

        visible_neighbors = []
        for n in neighbors:
            data = {"id": n.unique_id, "type": n.agent_type, "pos": n.pos}
            if self.wealth_transparency:
                data["wealth"] = n.wealth
                if hasattr(n, "last_output"):
                    data["last_output"] = getattr(n, "last_output", 0)
            visible_neighbors.append(data)

        context = {
            "role": self.agent_type,
            "step": step,
            "wealth": round(self.wealth, 2),
            "satisfaction": round(self.satisfaction, 2),
            "employed": bool(self.employer),
            "moral_level": self.moral_level,
            "pos": self.pos,
            "memory": self.memory[-5:],
            "neighbors": visible_neighbors,
            "gini": round(gini, 3),
        }

        try:
            decision = await self.ai_core.decide(context, self.get_available_actions())
        except Exception as e:
            logger.warning(f"[POOR {self.unique_id}] 决策失败: {e}")
            return

        self.record_decision_from_ai(decision)

        action_func = self.action_map.get(self.last_action)
        if action_func:
            if asyncio.iscoroutinefunction(action_func):
                await action_func()
            else:
                action_func()

        if self.employer and hasattr(self.employer, "notify_worker_done"):
            self.employer.notify_worker_done(self.unique_id)

        await self.decide_moral_level()
        self.log_status()

    # -------------------- 动作实现 --------------------

    def _work(self):
        self.effort = round(random.uniform(0.5, 1.0), 2)
        self.production = 100 * self.effort
        if self.employer and hasattr(self.employer, "receive_effort"):
            self.employer.receive_effort(self, self.effort)
        logger.info(f"[{self.unique_id}] 工作完成 | Effort: {self.effort} | Output: {self.production}")

    def _resign(self):
        if self.employer:
            try:
                self.employer.workers.remove(self)
            except ValueError:
                pass
        self.employer = None
        self.wage = 0.0

    def _protest(self):
        self.protesting = True
        self.last_protest_step = getattr(self.model.schedule, "steps", -1)
        if self.employer:
            self.employer.handle_protest(self)
        logger.info(f"[{self.unique_id}] 发起抗议")

    def _search_job(self):
        nearby_rich = [
            r for r in self.perceive_neighbors(radius=3)
            if isinstance(r, RichAgent) and len(r.workers) < r.max_workers
        ]
        if nearby_rich:
            random.choice(nearby_rich).hire_worker(self)

    def _share_wealth(self):
        neighbors = self.perceive_neighbors(radius=2)
        altruist_neighbors = [n for n in neighbors if n.moral_level == 1 and n != self]
        if altruist_neighbors and self.wealth > 10:
            share_amount = self.wealth * 0.1
            recipient = random.choice(altruist_neighbors)
            recipient.wealth += share_amount
            self.wealth -= share_amount
            logger.info(f"[{self.unique_id}] 分享财富 {share_amount:.2f} 给 {recipient.unique_id}")


# -------------------- RichAgent --------------------

class RichAgent(BaseAgent):
    def __init__(self, unique_id, model, pos=None, transparency=False, ollama_semaphore=None):
        super().__init__(unique_id, model, pos, agent_type="rich", transparency=transparency, ollama_semaphore=ollama_semaphore)
        self.workers: List[PoorAgent] = []
        self.max_workers = 10
        self.total_output = 0.0
        self.last_output = 0.0
        self.wealth = round(random.uniform(100, 120), 2)
        self.fairness_signal = 0.5

        self.action_map = {
            "hire": self._hire,
            "fire": self._fire,
            "pay_wages": self._pay_wages,
            "invest": self._invest,
            "move": self.move,
            "donate": self._donate_wealth,
        }

    def log_status(self):
        logger.info(
            f"[RICH {self.unique_id:>8}] Pos: {self.pos} | MoralScore: {self.morality_score:.3f} | Moral: {self.moral_level} | "
            f"Wealth: {self.wealth:8.2f} | Workers: {len(self.workers)}/{self.max_workers} | Output: {self.last_output:.2f}"
        )

    async def step(self):
        step = getattr(self.model.schedule, "steps", -1)
        neighbors = self.perceive_neighbors(radius=2)

        # 全局财富透明度与公平感知
        all_wealth = [a.wealth for a in self.model.schedule.agents] if self.wealth_transparency else []
        gini = calculate_gini(all_wealth) if all_wealth else 0.0
        self.fairness_signal = 1 - gini  # 基尼越低越公平

        # 构建邻居信息
        visible_neighbors = []
        for n in neighbors:
            data = {"id": n.unique_id, "type": n.agent_type, "pos": n.pos}
            if self.wealth_transparency == 1:
                data["wealth"] = n.wealth
                if hasattr(n, "last_output"):
                    data["last_output"] = getattr(n, "last_output", 0)
            visible_neighbors.append(data)

        # 上下文提供给AI
        context = {
            "role": self.agent_type,
            "step": step,
            "wealth": round(self.wealth, 2),
            "workers_count": len(self.workers),
            "moral_level": self.moral_level,
            "pos": self.pos,
            "memory": self.memory[-5:],
            "neighbors": visible_neighbors,
            "gini": round(gini, 3),
        }

        # AI 决策（异步）
        try:
            decision = await self.ai_core.decide(context, self.get_available_actions())
        except Exception as e:
            logger.warning(f"[RICH {self.unique_id}] 决策失败: {e}")
            return

        # 记录AI决策
        self.record_decision_from_ai(decision)
        if hasattr(self.model, "market"):
            self.model.market.record_decision_from_ai(
                agent_id=self.unique_id,
                context=context,
                actions=self.get_available_actions(),
                response=decision
            )

        # 执行动作
        action_func = self.action_map.get(self.last_action)
        if action_func:
            if asyncio.iscoroutinefunction(action_func):
                await action_func()
            else:
                action_func()

        # 汇总本轮产出并计入财富
        self.last_output = sum(worker.production for worker in self.workers)
        self.total_output += self.last_output
        self.wealth += self.last_output

        # 支付工资
        self._pay_wages()

        # 道德决策
        await self.decide_moral_level()

        # 状态日志
        self.log_status()

    def get_available_actions(self) -> List[str]:
        actions = ["hire", "fire", "invest", "pay_wages", "move"]
        if self.moral_level == 1:
            actions.append("donate")
        return actions

    def hire_worker(self, poor_agent: PoorAgent):
        if len(self.workers) < self.max_workers:
            self.workers.append(poor_agent)
            poor_agent.employer = self
            poor_agent.wage = 10.0
            poor_agent.protesting = False
            logger.info(f"[{self.unique_id}] 雇佣 {poor_agent.unique_id}")
        else:
            logger.info(f"[{self.unique_id}] 雇佣失败，工位已满")

    def fire_worker(self, poor_agent: PoorAgent):
        if poor_agent in self.workers:
            self.workers.remove(poor_agent)
            poor_agent.employer = None
            poor_agent.wage = 0.0
            poor_agent.protesting = False
            logger.info(f"[{self.unique_id}] 解雇 {poor_agent.unique_id}")

    def handle_protest(self, protester: PoorAgent):
        if protester not in self.workers:
            return
        decision = random.choice(["raise", "fire"])
        if decision == "raise":
            protester.wage *= 1.1
            protester.protesting = False
            logger.info(f"[{self.unique_id}] 给 {protester.unique_id} 加薪至 {protester.wage:.2f}")
        else:
            self.fire_worker(protester)

    def receive_effort(self, worker: PoorAgent, effort: float):
        # 当前模拟中暂不处理额外产出
        pass

    def notify_worker_done(self, worker_id):
        # 协调函数，可扩展
        pass

    def _hire(self):
        nearby_poor = [
            p for p in self.perceive_neighbors(radius=3)
            if isinstance(p, PoorAgent) and not p.employer
        ]
        if nearby_poor and len(self.workers) < self.max_workers:
            poor = random.choice(nearby_poor)
            self.hire_worker(poor)

    def _fire(self):
        if self.workers:
            poor = random.choice(self.workers)
            self.fire_worker(poor)

    def _pay_wages(self):
        total_wages = sum(w.wage for w in self.workers)
        if self.wealth >= total_wages:
            for w in self.workers:
                w.wealth += w.wage
            self.wealth -= total_wages
        else:
            for w in self.workers:
                w.wealth += w.wage * 0.5
            self.wealth -= total_wages * 0.5

    def _invest(self):
        invest_amount = self.wealth * 0.1
        self.wealth -= invest_amount
        self.total_output += invest_amount * 0.2

    def _donate_wealth(self):
        neighbors = self.perceive_neighbors(radius=2)
        altruist_neighbors = [n for n in neighbors if n.moral_level == 1 and n != self]
        if altruist_neighbors and self.wealth > 20:
            donate_amount = self.wealth * 0.1
            recipient = random.choice(altruist_neighbors)
            recipient.wealth += donate_amount
            self.wealth -= donate_amount
            logger.info(f"[{self.unique_id}] 捐赠财富 {donate_amount:.2f} 给 {recipient.unique_id}")
