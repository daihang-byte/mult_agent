import json
import asyncio
import logging
import aiohttp
import time
import re
from typing import Optional, List
from config import Config

logger = logging.getLogger(__name__)


def extract_float_wage(text: str, min_wage=0.5, max_wage=10.0) -> Optional[float]:
    numbers = re.findall(r"\d+\.\d+", text or "")
    for num in numbers:
        wage = float(num)
        if min_wage <= wage <= max_wage:
            return wage
    return None


class OllamaAgentCore:
    """
    AI 封装：
      - 负责把上下文拼成 prompt、向 Ollama 请求决策并解析
      - 不直接修改 Agent 的 moral_level（仅作为建议返回）
      - prompt 默认允许自利行为、提示偶尔反常决策，并使用较高 temperature 增加行为多样性
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        moral_level: int = 0,
        transparency: str = "low",
        ollama_semaphore: Optional[asyncio.Semaphore] = None,
        model_name: Optional[str] = None,
        model=None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type.lower()
        self.moral_level = moral_level
        self.transparency = transparency
        self.ollama_semaphore = ollama_semaphore
        self.model_name = model_name or (
            Config.POOR_AGENT_MODEL if self.agent_type == "poor" else Config.RICH_AGENT_MODEL
        )
        self.session = aiohttp.ClientSession()
        self.history: List[str] = []
        self.memory: List[str] = []
        self.model = model
        logger.info(f"[{self.agent_type}][{self.agent_id}] 初始化完成，moral_level={self.moral_level}")

    async def initialize(self):
        logger.info(f"[{self.agent_type}][{self.agent_id}] 执行初始化逻辑")
        await asyncio.sleep(0.05)

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"[{self.agent_type}][{self.agent_id}] 会话已关闭")

    async def decide(self, context: dict, available_actions: List[str]) -> dict:
        """
        向 Ollama 发送 prompt 并解析 JSON 返回
        """
        context.setdefault("fairness_signal", getattr(self, "fairness_signal", 1.0))
        prompt = self._build_prompt(context, available_actions)
        translations = {
            "寻找工作": "search_job", "工作": "work", "辞职": "resign", "抗议": "protest",
            "雇佣": "hire", "解雇": "fire", "涨薪": "give_raise", "评价生产力": "evaluate_productivity",
            "捐赠": "donate", "无所事事": "do_nothing", "移动": "move", "响应抗议": "respond_protest"
        }

        step = getattr(self.model.schedule, "steps", 0) if self.model else -1
        logger.info(f"[{self.agent_type}][{self.agent_id}] Step {step} | 开始决策")

        for attempt in range(1, 4):
            try:
                sem = self.ollama_semaphore or asyncio.Semaphore(1)
                async with sem:
                    start = time.time()
                    raw = await self._send_ollama_request(prompt)
                    parsed = self._parse_json_response(raw)

                    # 翻译中文动作词
                    if "action" in parsed and parsed["action"] in translations:
                        parsed["action"] = translations[parsed["action"]]

                    if not self._validate(parsed, available_actions):
                        raise ValueError(f"非法动作或响应字段缺失: {parsed.get('action')} / keys={parsed.keys()}")

                    duration = time.time() - start
                    logger.info(f"[{self.agent_type}][{self.agent_id}] Step {step} 决策成功 ⏱️{duration:.2f}s")

                    # 历史和记忆
                    self.history.append(json.dumps(parsed, ensure_ascii=False))
                    self.memory.append(f"{parsed.get('speech','')}（动机: {parsed.get('thoughts','')}）")
                    self.history = self.history[-5:]
                    self.memory = self.memory[-5:]

                    self._record(parsed, context)
                    return parsed

            except Exception as e:
                logger.warning(f"[{self.agent_type}][{self.agent_id}] Step {step} 决策失败 第{attempt}次: {e}")
                await asyncio.sleep(1)

        fallback = self._fallback(context)
        self._record(fallback, context)
        return fallback

    def _build_prompt(self, context: dict, actions: List[str]) -> str:
        memory = "\n".join(self.memory[-3:]) or "（无）"
        history = "\n".join(self.history[-3:]) or "（无）"
        fairness = context.get("fairness_signal", 1.0)

        moral_str = "高道德，倾向利他" if self.moral_level == 1 else "低道德，倾向自利"
        agent_role = "穷人" if self.agent_type in ["poor", "pooragent"] else "富人"
        employed_status = "已雇佣，请努力提升收入、满意度。" if context.get("employed") else "失业，请积极找工作。"
        traits_desc = getattr(self, "traits_desc", "（无详细性格描述）")
        action_list = ", ".join([f'"{a}"' for a in actions])

        # 可见信息描述
        visibility = "你能看到所有人的财富信息" if getattr(self, "full_visibility", False) else "你只能看到部分邻居的财富信息"

        system_prompt = f"""
    你是一名{moral_str}的{agent_role}（ID: {self.agent_id}）。
    性格特征: {traits_desc}
    你处在一个由富人和穷人共同组成的经济体中，资源有限且竞争激烈。

    【当前状态】
    - 财富: {context.get("wealth", 0.0)}
    - 满意度: {context.get("satisfaction", 0.5)}
    - 上一步产出: {context.get("last_output", 0.0)}
    - 当前员工数（如适用）: {context.get("num_workers", 0)}
    - 雇佣状态: {employed_status}
    - 信息透明度: {self.transparency} ({visibility})
    - 社会公平指标: {fairness}

    【记忆摘要】（最近 3 条）
    {memory}

    【历史行为】（最近 3 条）
    {history}

    【决策要求】
    1) 根据当前状态和环境信息选择最合理的行动。
    2) 可以权衡自身收益与可能影响他人。
    3) 有 3%~7% 概率做出与道德倾向相反的行为。
    4) 可考虑邻居互动、历史收益、当前财富、雇佣关系、社会稳定性。
    5) 富人可通过招聘、涨薪、裁员等提升产出；穷人可找工作、抗议、捐赠等。
    6) 输出必须严格遵守以下 JSON 格式（无 Markdown/注释）:

    {{
      "action": "动作名称（必须在可用动作列表中）",
      "action_target": null,
      "effort": 0.0,
      "wage": null,
      "speech": "你打算说的话",
      "thoughts": "选择此动作的理由",
      "moral_level": 0 或 1
    }}

    【可选动作】
    {action_list}
    """

        if self.agent_type in ["poor", "pooragent"]:
            system_prompt += """
    \n【提示】（穷人）
    - 当失业或收入不足时，应优先寻找工作。
    - 工资过低或工作条件差时，可选择罢工、抗议或提出加薪。
    - 如果有多份工作机会，需权衡工资、工作时间与满意度。
    - 健康与长期可持续收入比短期暴利更重要。
    """
        elif self.agent_type in ["rich", "richagent"]:
            system_prompt += """
    \n【提示】（富人）
    - 保证利润增长的同时，维持员工积极性，避免罢工或高流失率。
    - 工资过低可能导致员工生产力下降，工资过高则会压缩利润。
    - 可通过涨薪、培训、奖金、增加监督等方式提升产出。
    - 在经济下行时，需灵活裁员、削减成本或探索新市场。
    """
        return system_prompt.strip()

    def _fallback(self, context: dict) -> dict:
        employed = context.get("employed", False)
        satisfaction = context.get("satisfaction", 0.5)
        output = context.get("last_output", 0.0)

        if self.agent_type == "poor":
            if not employed:
                action = "search_job"
                thoughts = "失业状态下，优先寻找工作。"
            elif satisfaction < 0.3:
                action = "protest"
                thoughts = "不满当前工作条件，选择抗议。"
            else:
                action = "work"
                thoughts = "继续工作，维持生计。"

        elif self.agent_type == "rich":
            if context.get("num_workers", 0) == 0:
                action = "hire"
                thoughts = "没有员工，必须招聘。"
            elif output < 5.0:
                action = "evaluate_productivity"
                thoughts = "产出太低，需要评估员工表现。"
            elif satisfaction < 0.4:
                action = "give_raise"
                thoughts = "员工满意度低，考虑涨薪激励。"
            else:
                action = "hire"
                thoughts = "扩大团队规模。"
        else:
            action = "do_nothing"
            thoughts = "角色类型未知，保持当前状态。"

        return {
            "action": action,
            "action_target": None,
            "effort": 0.0,
            "wage": None,
            "speech": "请求失败，采用保守策略。",
            "thoughts": thoughts,
            "fallback": True,
        }

    def _validate(self, resp: dict, allowed: List[str]) -> bool:
        required = {"action", "action_target", "effort", "wage", "speech", "thoughts"}
        return isinstance(resp, dict) and required.issubset(resp.keys()) and resp["action"] in allowed

    def _record(self, decision: dict, context: dict):
        step = getattr(self.model.schedule, "steps", 0) if self.model else -1
        action = decision.get("action")
        speech = decision.get("speech")
        thoughts = decision.get("thoughts")
        effort = decision.get("effort", 0.0)
        wage = decision.get("wage", None)

        if self.model and hasattr(self.model, "market") and hasattr(self.model.market, "record_decision"):
            self.model.market.record_decision(
                step=step,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                action=action,
                speech=speech,
                thoughts=thoughts,
                mood_change=0.0,
                action_target=decision.get("action_target"),
                effort=effort,
                wage=wage,
                context=context,
                timestamp=time.time(),
                executed=decision.get("executed", True),
                failure_reason=decision.get("failure_reason"),
                ai_model=self.model_name,
            )

        logger.info(
            f"[🧠 AI决策][{self.agent_type}][{self.agent_id}][Step {step}] "
            f"ACTION={action} | SPEAK=\"{speech}\" | EFFORT={effort} | WAGE={wage} | THOUGHTS=\"{thoughts}\" | MORAL_LEVEL={decision.get('moral_level', 'N/A')}"
        )

    async def _send_ollama_request(self, prompt: str) -> str:
        url = Config.OLLAMA_BASE_URL or "http://localhost:11434/v1/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9, "num_ctx": 4096},
        }
        async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            result = await resp.json()
            raw_text = result.get("response") or result.get("output") or ""
            if isinstance(raw_text, list):
                raw_text = " ".join(raw_text)
            return raw_text.strip()

    def _parse_json_response(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw.strip())
        except Exception:
            match = re.search(r"\{(?:[^{}]|(?R))*}", raw)
            if match:
                candidate = match.group(0)
                try:
                    return json.loads(candidate)
                except Exception as e:
                    logger.warning(f"[解析错误] 提取到 JSON 但解析失败: {e}")
                    return {}
            else:
                logger.warning("[解析错误] 未能从模型输出中解析 JSON")
                return {}
