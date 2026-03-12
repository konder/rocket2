#!/usr/bin/env python3
"""
Progress Logger - 通用进度日志系统

用于基线测试、微调训练等长时间任务的进度追踪。

Usage:
    from progress_logger import ProgressLogger
    
    logger = ProgressLogger(log_file="progress.log", task_name="Fine-tuning")
    logger.log("process", 1, 100, info={"loss": 3.2})
    logger.log("process", 50, 100, info={"loss": 1.5, "epoch": 1})
    logger.close()

日志格式:
    【Fine-tuning】process 1/100, [info: loss: 3.2]
    【Fine-tuning】process 50/100, [info: loss: 1.5, epoch: 1]
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any


class ProgressLogger:
    """通用进度日志记录器"""
    
    def __init__(
        self,
        log_file: str,
        task_name: str = "Task",
        append: bool = True,
        also_print: bool = True
    ):
        """
        初始化进度日志记录器
        
        Args:
            log_file: 日志文件路径
            task_name: 任务名称（显示在日志中）
            append: 是否追加模式（True=追加，False=覆盖）
            also_print: 是否同时打印到控制台
        """
        self.log_file = log_file
        self.task_name = task_name
        self.also_print = also_print
        self.start_time = time.time()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # 打开文件
        mode = "a" if append else "w"
        self.file = open(log_file, mode, encoding="utf-8")
        
        # 写入开始标记
        self._write_line(f"=== {task_name} started at {datetime.now().isoformat()} ===")
    
    def log(self, action: str, current: int, total: int, info: Dict[str, Any] = None):
        """
        记录进度
        
        Args:
            action: 动作类型（如 "process", "epoch", "task"）
            current: 当前进度
            total: 总数
            info: 附加信息字典
        """
        # 构建信息字符串
        info_str = ""
        if info:
            info_parts = [f"{k}: {v}" for k, v in info.items()]
            info_str = ", ".join(info_parts)
        
        # 构建日志行
        timestamp = datetime.now().strftime("%H:%M:%S")
        if info_str:
            line = f"【{self.task_name}】{action} {current}/{total}, [info: {info_str}]"
        else:
            line = f"【{self.task_name}】{action} {current}/{total}"
        
        line = f"[{timestamp}] {line}"
        
        self._write_line(line)
    
    def log_event(self, event: str, details: Optional[str] = None):
        """
        记录事件（不包含进度）
        
        Args:
            event: 事件描述
            details: 详细信息
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        if details:
            line = f"[{timestamp}] 【{self.task_name}】{event} - {details}"
        else:
            line = f"[{timestamp}] 【{self.task_name}】{event}"
        
        self._write_line(line)
    
    def _write_line(self, line: str):
        """写入一行日志"""
        self.file.write(line + "\n")
        self.file.flush()  # 立即刷新，确保可以实时查看
        
        if self.also_print:
            print(line)
    
    def elapsed_time(self) -> str:
        """返回已用时间"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def close(self):
        """关闭日志文件"""
        elapsed = self.elapsed_time()
        self._write_line(f"=== {self.task_name} completed in {elapsed} ===")
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# 便捷函数
def create_logger(log_file: str, task_name: str = "Task") -> ProgressLogger:
    """创建进度日志记录器"""
    return ProgressLogger(log_file, task_name)


# 示例用法
if __name__ == "__main__":
    import random
    
    # 示例1: 训练进度
    with ProgressLogger("logs/training_progress.log", "Fine-tuning") as logger:
        for epoch in range(1, 3):
            logger.log_event(f"Epoch {epoch} started")
            for batch in range(1, 101):
                loss = 3.0 - (epoch - 1) * 0.5 - batch * 0.01 + random.random() * 0.1
                if batch % 10 == 0:
                    logger.log("process", batch, 100, info={"loss": f"{loss:.4f}", "epoch": epoch})
            logger.log_event(f"Epoch {epoch} completed", f"avg_loss: {loss:.4f}")
    
    # 示例2: 基线测试
    with ProgressLogger("logs/benchmark_progress.log", "Benchmark") as logger:
        tasks = ["mine_coal", "collect_wood", "build_bridge"]
        for i, task in enumerate(tasks, 1):
            logger.log("task", i, len(tasks), info={"name": task})
            # 模拟测试
            success = random.choice([True, False])
            logger.log_event(f"Task {task}", "success" if success else "failed")