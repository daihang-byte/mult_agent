import os
import json
import random
import logging
from typing import Optional
from model import WealthModel

logger = logging.getLogger(__name__)

async def run_single_experiment(
    experiment_id: str,
    transparent: bool,
    num_steps: int = 20,
    seed: Optional[int] = None,
    num_agents: int = 200,
    rich_ratio: float = 0.1,
) -> None:
    if seed is not None:
        random.seed(seed)

    logger.info(f"🚀 启动实验 {experiment_id}（透明模式={transparent}，种子={seed}）")

    # 根据富人比例计算各类智能体数量
    num_rich = max(1, int(num_agents * rich_ratio))
    num_poor = max(1, num_agents - num_rich)

    # 初始化模型（宽度高度由config或默认值控制）
    model = WealthModel(
        num_poor=num_poor,
        num_rich=num_rich,
        transparent=transparent,
        seed=seed,
        experiment_id=experiment_id
    )

    result_dir = os.path.join("results", f"experiment_{experiment_id}")
    os.makedirs(result_dir, exist_ok=True)

    try:
        for step in range(num_steps):
            logger.info(f"🕒 [{experiment_id}] === 时间步 {step + 1} 开始 ===")
            await model.step()

        # 保存结果
        model.save_results(result_dir)

        # 保存实验元信息
        meta = {
            "experiment_id": experiment_id,
            "transparent": transparent,
            "seed": seed,
            "steps": num_steps,
            "num_rich": num_rich,
            "num_poor": num_poor,
        }
        with open(os.path.join(result_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✅ 实验 {experiment_id} 完成 ✅ 结果路径：{result_dir}")

    except Exception as e:
        logger.error(f"❌ 实验 {experiment_id} 执行失败: {e}", exc_info=True)

    finally:
        await model.close_sessions()
