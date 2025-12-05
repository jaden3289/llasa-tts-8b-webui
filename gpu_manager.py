"""
GPU Resource Manager - 智能显存管理系统

功能：
1. 懒加载：首次请求时加载模型到 GPU
2. 即用即卸：任务完成后立即转移到 CPU
3. 自动监控：空闲超时后自动释放资源

状态转换：
未加载 ──首次(20-30s)──> GPU ──任务完成(2s)──> CPU ──新请求(2-5s)──> GPU
  ↑                                                    ↓
  └──────────────超时/手动释放(1s)─────────────────────┘
"""

import torch
import threading
import time
import logging
from typing import Callable, Optional, Dict, Any
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GPUResourceManager:
    """GPU 资源管理器 - 懒加载 + 即用即卸"""

    def __init__(self, idle_timeout: int = 60):
        """
        初始化 GPU 资源管理器

        Args:
            idle_timeout: 空闲超时时间（秒），默认 60 秒
        """
        self.idle_timeout = idle_timeout
        self.model_on_gpu = None      # GPU 上的模型
        self.model_on_cpu = None      # CPU 缓存的模型
        self.last_use_time = 0        # 最后使用时间
        self.lock = threading.Lock()  # 线程锁
        self.running = False          # 监控线程运行状态
        self.monitor_thread = None    # 监控线程
        self.load_func = None         # 模型加载函数
        self.model_name = "Model"     # 模型名称（用于日志）

        logger.info(f"🔧 GPU 资源管理器初始化完成，空闲超时: {idle_timeout} 秒")

    def start_monitor(self):
        """启动监控线程"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("🚀 GPU 监控线程已启动")

    def stop_monitor(self):
        """停止监控线程"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("🛑 GPU 监控线程已停止")

    def _monitor_loop(self):
        """监控线程主循环"""
        while self.running:
            time.sleep(30)  # 每 30 秒检查一次

            with self.lock:
                if self.model_on_gpu is not None:
                    idle_time = time.time() - self.last_use_time

                    # 超时自动卸载
                    if idle_time > self.idle_timeout:
                        logger.info(
                            f"⏱️  {self.model_name} 空闲 {idle_time:.1f} 秒，"
                            f"超过阈值 {self.idle_timeout} 秒，自动卸载到 CPU"
                        )
                        self._move_to_cpu()

    def get_model(
        self,
        load_func: Callable,
        model_name: str = "Model",
        force_reload: bool = False
    ):
        """
        获取模型（懒加载逻辑）

        Args:
            load_func: 模型加载函数，返回加载好的模型
            model_name: 模型名称（用于日志显示）
            force_reload: 是否强制重新加载

        Returns:
            加载好的模型（在 GPU 上）
        """
        with self.lock:
            self.load_func = load_func
            self.model_name = model_name
            self.last_use_time = time.time()

            # 情况1: 模型已在 GPU 上
            if self.model_on_gpu is not None and not force_reload:
                logger.info(f"✅ {model_name} 已在 GPU 上，直接返回")
                return self.model_on_gpu

            # 情况2: 模型在 CPU 上，快速转移到 GPU
            if self.model_on_cpu is not None and not force_reload:
                logger.info(f"📤 {model_name} 在 CPU 上，正在转移到 GPU...")
                start_time = time.time()

                self.model_on_gpu = self._move_to_gpu(self.model_on_cpu)
                self.model_on_cpu = None  # 释放 CPU 缓存

                elapsed = time.time() - start_time
                logger.info(f"✅ {model_name} 已转移到 GPU，耗时 {elapsed:.2f} 秒")

                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()

                return self.model_on_gpu

            # 情况3: 首次加载，从磁盘加载到 GPU
            logger.info(f"🔄 首次加载 {model_name}，请稍候...")
            start_time = time.time()

            self.model_on_gpu = load_func()

            elapsed = time.time() - start_time
            logger.info(f"✅ {model_name} 加载完成，耗时 {elapsed:.2f} 秒")

            # 显示显存使用情况
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(
                        f"   GPU {i}: 已分配 {mem_allocated:.2f} GB, "
                        f"已保留 {mem_reserved:.2f} GB"
                    )

            return self.model_on_gpu

    def force_offload(self):
        """
        立即卸载：任务完成后立即调用
        将模型从 GPU 转移到 CPU，释放显存（2-5秒）
        """
        with self.lock:
            if self.model_on_gpu is not None:
                logger.info(f"📥 正在卸载 {self.model_name} 到 CPU...")
                start_time = time.time()

                self._move_to_cpu()

                elapsed = time.time() - start_time
                logger.info(
                    f"✅ {self.model_name} 已卸载到 CPU，耗时 {elapsed:.2f} 秒"
                )

                # 显示释放后的显存
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        logger.info(f"   GPU {i}: 剩余占用 {mem_allocated:.2f} GB")

    def force_release(self):
        """
        完全释放：长期不用时调用
        清空 GPU 和 CPU 缓存（1秒）
        """
        with self.lock:
            logger.info(f"🗑️  正在完全释放 {self.model_name}...")

            self.model_on_gpu = None
            self.model_on_cpu = None

            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"✅ {self.model_name} 已完全释放")

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态

        Returns:
            状态字典
        """
        with self.lock:
            idle_time = time.time() - self.last_use_time if self.last_use_time > 0 else 0

            # 确定模型位置
            if self.model_on_gpu is not None:
                location = "GPU"
            elif self.model_on_cpu is not None:
                location = "CPU"
            else:
                location = "未加载"

            # 获取 GPU 显存信息
            gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"GPU_{i}"] = {
                        "allocated_gb": round(
                            torch.cuda.memory_allocated(i) / 1024**3, 2
                        ),
                        "reserved_gb": round(
                            torch.cuda.memory_reserved(i) / 1024**3, 2
                        ),
                        "total_gb": round(
                            torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                        )
                    }

            return {
                "model_name": self.model_name,
                "location": location,
                "idle_time_seconds": round(idle_time, 1),
                "idle_timeout_seconds": self.idle_timeout,
                "monitor_running": self.running,
                "gpu_memory": gpu_memory
            }

    def update_timeout(self, new_timeout: int):
        """更新空闲超时时间"""
        with self.lock:
            old_timeout = self.idle_timeout
            self.idle_timeout = new_timeout
            logger.info(
                f"⚙️  空闲超时已更新: {old_timeout} 秒 → {new_timeout} 秒"
            )

    def _move_to_cpu(self):
        """内部方法：将模型从 GPU 转移到 CPU"""
        if self.model_on_gpu is None:
            return

        # 转移到 CPU
        self.model_on_cpu = self.model_on_gpu.cpu()
        self.model_on_gpu = None

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_to_gpu(self, model):
        """内部方法：将模型从 CPU 转移到 GPU"""
        return model.cuda()


class MultiModelGPUManager:
    """
    多模型 GPU 管理器
    用于管理多个模型（如 Llasa-8B, XCodec2, WhisperModel）
    """

    def __init__(self, idle_timeout: int = 60):
        """
        初始化多模型管理器

        Args:
            idle_timeout: 空闲超时时间（秒）
        """
        self.managers: Dict[str, GPUResourceManager] = {}
        self.idle_timeout = idle_timeout
        self.global_lock = threading.Lock()

        logger.info(f"🔧 多模型 GPU 管理器初始化完成")

    def register_model(self, model_name: str) -> GPUResourceManager:
        """
        注册一个模型

        Args:
            model_name: 模型名称

        Returns:
            该模型的 GPU 管理器
        """
        with self.global_lock:
            if model_name not in self.managers:
                manager = GPUResourceManager(idle_timeout=self.idle_timeout)
                manager.start_monitor()
                self.managers[model_name] = manager
                logger.info(f"📝 已注册模型: {model_name}")

            return self.managers[model_name]

    def get_manager(self, model_name: str) -> Optional[GPUResourceManager]:
        """获取指定模型的管理器"""
        return self.managers.get(model_name)

    def offload_all(self):
        """卸载所有模型到 CPU"""
        logger.info("📥 正在卸载所有模型到 CPU...")
        for name, manager in self.managers.items():
            manager.force_offload()
        logger.info("✅ 所有模型已卸载到 CPU")

    def release_all(self):
        """完全释放所有模型"""
        logger.info("🗑️  正在完全释放所有模型...")
        for name, manager in self.managers.items():
            manager.force_release()
        logger.info("✅ 所有模型已完全释放")

    def get_all_status(self) -> Dict[str, Any]:
        """获取所有模型的状态"""
        status = {}
        for name, manager in self.managers.items():
            status[name] = manager.get_status()
        return status

    def update_all_timeout(self, new_timeout: int):
        """更新所有模型的空闲超时时间"""
        for manager in self.managers.values():
            manager.update_timeout(new_timeout)

    def stop_all(self):
        """停止所有监控线程"""
        logger.info("🛑 正在停止所有监控线程...")
        for manager in self.managers.values():
            manager.stop_monitor()
        logger.info("✅ 所有监控线程已停止")


# 全局实例（单例模式）
_global_manager: Optional[MultiModelGPUManager] = None


def get_global_manager(idle_timeout: int = 60) -> MultiModelGPUManager:
    """
    获取全局多模型管理器（单例）

    Args:
        idle_timeout: 空闲超时时间（秒）

    Returns:
        全局多模型管理器实例
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = MultiModelGPUManager(idle_timeout=idle_timeout)
    return _global_manager


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("GPU 资源管理器测试")
    print("=" * 60)

    # 模拟模型加载函数
    def load_dummy_model():
        import time
        print("正在加载模型...")
        time.sleep(2)  # 模拟加载时间

        class DummyModel:
            def __init__(self):
                self.data = torch.randn(1000, 1000).cuda()

            def cpu(self):
                self.data = self.data.cpu()
                return self

            def cuda(self):
                self.data = self.data.cuda()
                return self

        return DummyModel()

    # 测试单模型管理器
    print("\n测试 1: 单模型管理器")
    manager = GPUResourceManager(idle_timeout=10)
    manager.start_monitor()

    # 首次加载
    print("\n1. 首次加载...")
    model = manager.get_model(load_dummy_model, "DummyModel")
    print(f"状态: {manager.get_status()}")

    # 立即卸载
    print("\n2. 立即卸载...")
    manager.force_offload()
    print(f"状态: {manager.get_status()}")

    # 再次获取（从 CPU 快速恢复）
    print("\n3. 再次获取...")
    model = manager.get_model(load_dummy_model, "DummyModel")
    print(f"状态: {manager.get_status()}")

    # 完全释放
    print("\n4. 完全释放...")
    manager.force_release()
    print(f"状态: {manager.get_status()}")

    manager.stop_monitor()

    # 测试多模型管理器
    print("\n\n测试 2: 多模型管理器")
    multi_manager = get_global_manager(idle_timeout=10)

    # 注册多个模型
    llasa_manager = multi_manager.register_model("Llasa-8B")
    codec_manager = multi_manager.register_model("XCodec2")

    # 加载模型
    print("\n1. 加载 Llasa-8B...")
    llasa_model = llasa_manager.get_model(load_dummy_model, "Llasa-8B")

    print("\n2. 加载 XCodec2...")
    codec_model = codec_manager.get_model(load_dummy_model, "XCodec2")

    # 查看所有状态
    print("\n3. 所有模型状态:")
    import json
    print(json.dumps(multi_manager.get_all_status(), indent=2, ensure_ascii=False))

    # 卸载所有
    print("\n4. 卸载所有模型...")
    multi_manager.offload_all()

    # 再次查看状态
    print("\n5. 卸载后状态:")
    print(json.dumps(multi_manager.get_all_status(), indent=2, ensure_ascii=False))

    multi_manager.stop_all()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
