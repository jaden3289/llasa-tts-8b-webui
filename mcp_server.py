"""
Llasa-TTS-8B MCP 服务器

Model Context Protocol - 程序化访问接口
使用 FastMCP 框架
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from xcodec2.modeling_xcodec2 import XCodec2Model
from fastmcp import FastMCP

# 导入 GPU 管理器
from gpu_manager import get_global_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 初始化 MCP 服务器
# ============================================================================

mcp = FastMCP("Llasa-TTS-8B")

# GPU 管理器（全局单例）
gpu_manager = get_global_manager(
    idle_timeout=int(os.getenv('GPU_IDLE_TIMEOUT', 600))
)

# 模型路径配置
LLASA_MODEL_PATH = os.getenv('LLASA_MODEL_PATH', 'HKUSTAudio/Llasa-8B')
XCODEC_MODEL_PATH = os.getenv('XCODEC_MODEL_PATH', 'HKUSTAudio/xcodec2')
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', 'Systran/faster-whisper-large-v3')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')

# 全局变量
whisper_model = None
llasa_tokenizer = None

# 输出目录
OUTPUT_DIR = Path(os.getenv('OUTPUTS_DIR', './outputs'))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 模型加载函数
# ============================================================================

def load_llasa_model():
    """加载 Llasa-8B 模型"""
    logger.info(f"🔄 加载 Llasa-8B: {LLASA_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        LLASA_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()
    return model


def load_xcodec_model():
    """加载 XCodec2 模型"""
    logger.info(f"🔄 加载 XCodec2: {XCODEC_MODEL_PATH}")
    model = XCodec2Model.from_pretrained(XCODEC_MODEL_PATH, device_map='cuda')
    model.eval()
    return model


def get_whisper_model():
    """获取 Whisper 模型"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"🔄 加载 Whisper: {WHISPER_MODEL_PATH}")
        whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=WHISPER_DEVICE)
    return whisper_model


def get_tokenizer():
    """获取 Tokenizer"""
    global llasa_tokenizer
    if llasa_tokenizer is None:
        logger.info(f"🔄 加载 Tokenizer: {LLASA_MODEL_PATH}")
        llasa_tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL_PATH)
    return llasa_tokenizer


# ============================================================================
# 辅助函数
# ============================================================================

def ids_to_speech_tokens(speech_ids):
    """将 speech IDs 转换为 token 字符串"""
    return [f"<|s_{sid}|>" for sid in speech_ids]


def extract_speech_ids(speech_tokens_str):
    """从 token 字符串提取 speech IDs"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            speech_ids.append(int(num_str))
    return speech_ids


# ============================================================================
# MCP 工具函数
# ============================================================================

@mcp.tool()
def generate_speech(
    audio_path: str,
    ref_text: str,
    target_text: str,
    output_path: Optional[str] = None,
    system_prompt: str = "Convert the text to speech",
    sample_rate: int = 24000,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    penalty: float = 1.2,
    random_seed: int = 49
) -> Dict[str, Any]:
    """
    生成语音克隆（TTS）

    Args:
        audio_path: 参考音频文件路径
        ref_text: 参考音频的文本内容
        target_text: 要生成的文本内容
        output_path: 输出文件路径（可选，默认自动生成）
        system_prompt: 系统提示词
        sample_rate: 采样率（16000, 22050, 24000, 28000, 32000）
        temperature: 温度参数（0.0-1.5）
        top_k: Top-K 采样（1-100）
        top_p: Nucleus 采样（0.0-1.0）
        penalty: 重复惩罚（0.0-2.0）
        random_seed: 随机种子

    Returns:
        包含生成结果的字典
    """
    try:
        logger.info(f"🎙️ 开始生成语音: {target_text[:50]}...")

        # 验证参数
        if not Path(audio_path).exists():
            return {
                'status': 'error',
                'error': f'参考音频文件不存在: {audio_path}'
            }

        if not ref_text.strip() or not target_text.strip():
            return {
                'status': 'error',
                'error': '参考文本和目标文本不能为空'
            }

        # 设置随机种子
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # 加载和处理音频
        waveform, sr = torchaudio.load(audio_path)

        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        prompt_wav = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sample_rate
        )(waveform)

        input_text = ref_text + ' ' + target_text

        # 获取模型管理器
        codec_manager = gpu_manager.register_model("XCodec2")
        llasa_manager = gpu_manager.register_model("Llasa-8B")

        try:
            # === 阶段 1: 编码音频 ===
            codec_model = codec_manager.get_model(load_xcodec_model, "XCodec2")

            with torch.no_grad():
                vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav)
                vq_code_prompt = vq_code_prompt[0, 0, :]
                speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

            codec_manager.force_offload()
            logger.info("✅ 音频编码完成")

            # === 阶段 2: 生成语音 token ===
            llasa_model = llasa_manager.get_model(load_llasa_model, "Llasa-8B")
            tokenizer = get_tokenizer()

            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            chat = [
                {"role": "user", "content": system_prompt + ":" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors='pt',
                continue_final_message=True
            ).cuda()

            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            with torch.no_grad():
                outputs = llasa_model.generate(
                    input_ids,
                    max_length=2048,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=penalty
                )

            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

            llasa_manager.force_offload()
            logger.info("✅ 语音 token 生成完成")

            # === 阶段 3: 解码音频 ===
            codec_model = codec_manager.get_model(load_xcodec_model, "XCodec2")

            with torch.no_grad():
                gen_wav = codec_model.decode_code(speech_tokens)
                gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]

            codec_manager.force_offload()
            logger.info("✅ 音频解码完成")

            # 保存音频
            if output_path is None:
                import time
                output_path = str(OUTPUT_DIR / f"generated_{int(time.time())}.wav")

            output_path = str(Path(output_path).absolute())
            torchaudio.save(output_path, gen_wav[0].cpu(), sample_rate)

            logger.info(f"🎉 语音生成成功: {output_path}")

            return {
                'status': 'success',
                'output_path': output_path,
                'sample_rate': sample_rate,
                'duration_seconds': gen_wav.shape[-1] / sample_rate
            }

        except Exception as e:
            # 确保异常时卸载模型
            codec_manager.force_offload()
            llasa_manager.force_offload()
            raise e

    except Exception as e:
        error_msg = f"语音生成失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': error_msg
        }


@mcp.tool()
def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    转录音频（ASR）

    Args:
        audio_path: 音频文件路径

    Returns:
        包含转录结果的字典
    """
    try:
        if not Path(audio_path).exists():
            return {
                'status': 'error',
                'error': f'音频文件不存在: {audio_path}'
            }

        logger.info(f"🎤 开始转录音频: {audio_path}")

        whisper = get_whisper_model()
        segments, info = whisper.transcribe(
            audio=audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700)
        )

        text = ""
        for segment in segments:
            text += segment.text + "\n"

        logger.info(f"✅ 转录完成: {len(text)} 字符")

        return {
            'status': 'success',
            'text': text.strip(),
            'language': info.language if hasattr(info, 'language') else 'unknown'
        }

    except Exception as e:
        error_msg = f"转录失败: {str(e)}"
        logger.error(error_msg)
        return {
            'status': 'error',
            'error': error_msg
        }


@mcp.tool()
def get_gpu_status() -> Dict[str, Any]:
    """
    获取 GPU 状态

    Returns:
        GPU 状态信息
    """
    try:
        status = gpu_manager.get_all_status()
        return {
            'status': 'success',
            'data': status
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def offload_gpu() -> Dict[str, str]:
    """
    手动卸载所有模型到 CPU（释放 GPU 显存）

    Returns:
        操作结果
    """
    try:
        logger.info("📥 手动卸载所有模型到 CPU...")
        gpu_manager.offload_all()
        return {
            'status': 'success',
            'message': '所有模型已卸载到 CPU，GPU 显存已释放'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def release_gpu() -> Dict[str, str]:
    """
    完全释放所有模型（清空 GPU 和 CPU 缓存）

    Returns:
        操作结果
    """
    try:
        logger.info("🗑️  完全释放所有模型...")
        gpu_manager.release_all()
        return {
            'status': 'success',
            'message': '所有模型已完全释放'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def update_gpu_timeout(timeout_seconds: int) -> Dict[str, str]:
    """
    更新 GPU 空闲超时时间

    Args:
        timeout_seconds: 超时时间（秒）

    Returns:
        操作结果
    """
    try:
        if timeout_seconds < 0:
            return {
                'status': 'error',
                'error': '超时时间必须大于等于 0'
            }

        gpu_manager.update_all_timeout(timeout_seconds)
        return {
            'status': 'success',
            'message': f'GPU 空闲超时已更新为 {timeout_seconds} 秒'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 启动 Llasa-TTS-8B MCP 服务器")
    logger.info("=" * 60)
    logger.info(f"📦 可用工具:")
    logger.info("   • generate_speech    - 生成语音克隆")
    logger.info("   • transcribe_audio   - 转录音频")
    logger.info("   • get_gpu_status     - 获取 GPU 状态")
    logger.info("   • offload_gpu        - 卸载模型到 CPU")
    logger.info("   • release_gpu        - 完全释放模型")
    logger.info("   • update_gpu_timeout - 更新超时时间")
    logger.info("=" * 60)
    logger.info("")

    # 运行 MCP 服务器
    mcp.run()
