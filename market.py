# market.py
import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class WealthMarket:
    def __init__(self, model):
        self.model = model
        self.decisions = []
        self.interactions = []
        self.events = []
        self.conversations = []
        self.agent_states = []

    def _safe_step(self):
        if not isinstance(self.model.steps, int) or self.model.steps < 0:
            logger.warning(f"[⚠️ 记录跳过] 当前 model.steps = {self.model.steps} 非法，跳过记录")
            return False
        return True

    def record_decision(
        self,
        agent_id,
        action,
        speech="",
        thoughts="",
        mood_change=0.0,
        executed=True,
        **kwargs,
    ):
        if not self._safe_step():
            return

        step = self.model.steps
        agent = self.model.agent_dict.get(agent_id)
        moral_level = getattr(agent, "moral_level", None)
        pos = getattr(agent, "pos", None)
        wage = getattr(agent, "wage", None)
        effort = getattr(agent, "effort", None)
        production = getattr(agent, "production", None)

        record = {
            "timestamp": pd.Timestamp.now(),
            "step": step,
            "agent_id": agent_id,
            "agent_type": type(agent).__name__ if agent else "Unknown",
            "action": action or "wait",
            "speech": speech or "",
            "thoughts": thoughts or "",
            "mood_change": mood_change,
            "executed": executed,
            "wage": wage,
            "effort": effort,
            "production": production,
            "position": pos,
            "moral_level": moral_level,
            "memory": kwargs.pop("memory", None),
            "fallback": kwargs.pop("fallback", False),
            "success": kwargs.pop("success", True),
        }

        if "failure_reason" in kwargs:
            record["executed"] = False
            record["failure_reason"] = kwargs.pop("failure_reason")

        record.update(kwargs)
        self.decisions.append(record)

        logger.debug(f"[记录决策] Step {step} | Agent {agent_id} 执行动作: {action}")

    def record_decision_from_ai(self, agent_id, context, actions, response):
        if not self._safe_step():
            return

        agent = self.model.agent_dict.get(agent_id)
        moral_level = getattr(agent, "moral_level", None)
        memory = getattr(agent, "memory", None)

        self.record_decision(
            agent_id=agent_id,
            action=response.get("action", "wait"),
            speech=response.get("speech", ""),
            thoughts=response.get("thoughts", ""),
            mood_change=response.get("mood_change", 0.0),
            ai_model=response.get("model"),
            failure_reason=response.get("failure_reason"),
            fallback=response.get("fallback", False),
            success=not response.get("fallback", False),
            moral_level=moral_level,
            memory=memory,
            available_actions=",".join(actions) if isinstance(actions, list) else actions,
        )

        self.conversations.append({
            "step": self.model.steps,
            "agent_id": agent_id,
            "agent_type": type(agent).__name__ if agent else "Unknown",
            "transparent": getattr(self.model, "transparent", False),
            "input_context": context,
            "actions": actions,
            "ai_response": response,
            "fallback": response.get("fallback", False),
            "success": not response.get("fallback", False),
            "memory": memory,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def record_event(self, event_type, description, **kwargs):
        if not self._safe_step():
            return

        step = self.model.steps
        agent_id = kwargs.get("agent_id", None)
        agent = self.model.agent_dict.get(agent_id)
        event = {
            "step": step,
            "type": event_type,
            "description": description or "",
            "agent_id": agent_id,
            "agent_type": type(agent).__name__ if agent else "Unknown",
            "timestamp": pd.Timestamp.now(),
        }

        event.update(kwargs)
        self.events.append(event)

    def record_production(self, agent_id, amount, **kwargs):
        try:
            amount = float(amount)
        except Exception:
            logger.warning(f"[产出异常] agent {agent_id} 的产出 {amount} 非法，将记录为 0.0")
            amount = 0.0

        anomaly = amount == 0.0

        self.record_event(
            event_type="production",
            description=f"{agent_id} 生产总价值 {amount:.2f}",
            agent_id=agent_id,
            value=amount,
            anomaly=anomaly,
            note=kwargs.get("note", "可能因情绪低落或缺勤导致") if anomaly else None,
        )

    def record_earning(self, agent_id, amount):
        self.record_event(
            event_type="earning",
            description=f"{agent_id} 获得工资 {amount:.2f}",
            agent_id=agent_id,
            value=amount,
        )

    def record_agent_state(self):
        if not self._safe_step():
            return

        for agent in self.model.schedule.get_all_agents():
            record = {
                "step": self.model.steps,
                "agent_id": agent.unique_id,
                "agent_type": type(agent).__name__,
                "satisfaction": getattr(agent, "satisfaction", None),
                "employed": bool(agent.employer) if hasattr(agent, "employer") else None,
                "employer_id": getattr(agent.employer, "unique_id", None) if getattr(agent, "employer", None) else None,
                "wealth": getattr(agent, "wealth", None),
                "wage": getattr(agent, "wage", None),
                "production": getattr(agent, "production", None),
                "effort": getattr(agent, "effort", None),
                "pos": getattr(agent, "pos", None),
                "moral_level": getattr(agent, "moral_level", None),
                "transparent": getattr(self.model, "transparent", False),
                "gini": self.model.calculate_gini() if hasattr(self.model, "calculate_gini") else None,
                "timestamp": pd.Timestamp.now(),
            }
            self.agent_states.append(record)

    def save_to_csv(self, path: str):
        os.makedirs(path, exist_ok=True)

        def save_df(df, filename):
            df.to_csv(os.path.join(path, filename), index=False)
            logger.info(f"保存 {filename} 共 {len(df)} 条")

        if self.decisions:
            df = pd.DataFrame(self.decisions)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            save_df(df, "decisions.csv")

        if self.events:
            df = pd.DataFrame(self.events)
            save_df(df, "events.csv")

        if self.agent_states:
            df = pd.DataFrame(self.agent_states)
            save_df(df, "agent_states.csv")

        if self.conversations:
            with open(os.path.join(path, "conversations.json"), "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 conversations.json 共 {len(self.conversations)} 条")
