import asyncio
import logging
import os
import argparse
import glob
import sys
import platform
from experiment import run_single_experiment

def setup_logging(log_file: str = "logs/experiment.log") -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def print_environment_info() -> None:
    print("Python 可执行路径:", sys.executable)
    print("操作系统:", platform.system(), platform.release())
    try:
        user = os.getlogin()
    except Exception:
        user = "未知用户"
    print("当前用户:", user)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行财富分配实验模拟")
    parser.add_argument("--runs", type=int, default=1, help="每种模式的实验次数")
    parser.add_argument("--steps", type=int, default=40, help="每轮模拟步数")
    parser.add_argument("--agents", type=int, default=80, help="总智能体数")
    parser.add_argument("--rich-ratio", type=float, default=0.1, help="富人所占比例 (0-1)")
    parser.add_argument("--auto-plot", action="store_true", help="是否自动生成图表（未启用）")
    parser.add_argument("--debug", action="store_true", help="打印环境信息用于调试")
    return parser.parse_args()

async def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🎬 启动实验流程")
    logger.info(
        f"参数: 每种模式运行 {args.runs} 次，每轮 {args.steps} 步，总人数 {args.agents}，富人占比 {args.rich_ratio:.2f}"
    )

    if args.debug:
        print_environment_info()

    try:
        for i in range(args.runs):
            await run_single_experiment(
                experiment_id=f"T_{i}",
                transparent=True,
                num_steps=args.steps,
                seed=i + 1000,
                num_agents=args.agents,
                rich_ratio=args.rich_ratio,
            )
            await run_single_experiment(
                experiment_id=f"F_{i}",
                transparent=False,
                num_steps=args.steps,
                seed=i + 2000,
                num_agents=args.agents,
                rich_ratio=args.rich_ratio,
            )

    except Exception as e:
        logger.error(f"实验运行失败: {e}", exc_info=True)
        sys.exit(1)

    logger.info("✅ 所有实验完成！输出目录结构如下：")
    for path in sorted(glob.glob("results/experiment_*")):
        logger.info(f"  - {path}")

if __name__ == "__main__":
    asyncio.run(main())
