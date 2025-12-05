"""
Llasa-TTS-8B API 服务器

提供 RESTful API 接口，使用 GPU 智能管理器
支持 Swagger 文档
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flasgger import Swagger, swag_from
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from xcodec2.modeling_xcodec2 import XCodec2Model

# 导入 GPU 管理器
from gpu_manager import get_global_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Flask 应用初始化
# ============================================================================

app = Flask(__name__)
CORS(app)  # 启用 CORS

# Swagger 配置
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Llasa-TTS-8B API",
        "description": "语音克隆 / TTS API with GPU Smart Management",
        "version": "1.0.0"
    },
    "basePath": "/api",
    "schemes": ["http", "https"]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# ============================================================================
# 全局变量
# ============================================================================

# GPU 管理器
gpu_manager = get_global_manager(
    idle_timeout=int(os.getenv('GPU_IDLE_TIMEOUT', 600))
)

# 模型路径
LLASA_MODEL_PATH = os.getenv('LLASA_MODEL_PATH', 'HKUSTAudio/Llasa-8B')
XCODEC_MODEL_PATH = os.getenv('XCODEC_MODEL_PATH', 'HKUSTAudio/xcodec2')
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', 'Systran/faster-whisper-large-v3')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')

# 模型实例（懒加载）
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
    logger.info(f"正在加载 Llasa-8B 模型: {LLASA_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        LLASA_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()
    return model


def load_xcodec_model():
    """加载 XCodec2 模型"""
    logger.info(f"正在加载 XCodec2 模型: {XCODEC_MODEL_PATH}")
    model = XCodec2Model.from_pretrained(XCODEC_MODEL_PATH, device_map='cuda')
    model.eval()
    return model


def get_whisper_model():
    """获取 Whisper 模型（非 GPU 管理）"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"正在加载 Whisper 模型: {WHISPER_MODEL_PATH}")
        whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=WHISPER_DEVICE)
    return whisper_model


def get_tokenizer():
    """获取 Tokenizer（非 GPU 管理）"""
    global llasa_tokenizer
    if llasa_tokenizer is None:
        logger.info(f"正在加载 Tokenizer: {LLASA_MODEL_PATH}")
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


def tts_process(
    audio_path: str,
    ref_text: str,
    target_text: str,
    system_prompt: str = "Convert the text to speech",
    sample_rate: int = 24000,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    penalty: float = 1.2,
    do_sample: bool = True,
    random_seed: int = 49
) -> Optional[str]:
    """
    TTS 处理流程（使用 GPU 管理器）

    Returns:
        生成的音频文件路径
    """
    try:
        # 设置随机种子
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # 1. 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # 立体声转单声道
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 重采样
        prompt_wav = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sample_rate
        )(waveform)

        input_text = ref_text + ' ' + target_text

        # 2. 获取模型（懒加载）
        codec_manager = gpu_manager.register_model("XCodec2")
        llasa_manager = gpu_manager.register_model("Llasa-8B")

        try:
            # 获取 XCodec2
            codec_model = codec_manager.get_model(load_xcodec_model, "XCodec2")

            # 编码音频
            with torch.no_grad():
                vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav)
                vq_code_prompt = vq_code_prompt[0, 0, :]
                speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

            # 立即卸载 Codec
            codec_manager.force_offload()

            # 获取 Llasa-8B
            llasa_model = llasa_manager.get_model(load_llasa_model, "Llasa-8B")
            tokenizer = get_tokenizer()

            # 准备输入
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

            # 生成语音 token
            with torch.no_grad():
                outputs = llasa_model.generate(
                    input_ids,
                    max_length=2048,
                    eos_token_id=speech_end_id,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=penalty
                )

            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

            # 立即卸载 Llasa
            llasa_manager.force_offload()

            # 重新获取 Codec 进行解码
            codec_model = codec_manager.get_model(load_xcodec_model, "XCodec2")

            # 解码音频
            with torch.no_grad():
                gen_wav = codec_model.decode_code(speech_tokens)
                gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]

            # 立即卸载 Codec
            codec_manager.force_offload()

            # 保存音频
            output_path = OUTPUT_DIR / f"output_{int(time.time())}.wav"
            torchaudio.save(
                str(output_path),
                gen_wav[0].cpu(),
                sample_rate
            )

            logger.info(f"✅ 音频生成成功: {output_path}")
            return str(output_path)

        except Exception as e:
            # 确保异常时也卸载模型
            codec_manager.force_offload()
            llasa_manager.force_offload()
            raise e

    except Exception as e:
        logger.error(f"TTS 处理失败: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# API 端点
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查
    ---
    tags:
      - System
    responses:
      200:
        description: 系统正常
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })


@app.route('/api/gpu/status', methods=['GET'])
def get_gpu_status():
    """
    获取 GPU 状态
    ---
    tags:
      - GPU Management
    responses:
      200:
        description: GPU 状态信息
    """
    status = gpu_manager.get_all_status()
    return jsonify(status)


@app.route('/api/gpu/offload', methods=['POST'])
def offload_gpu():
    """
    手动卸载所有模型到 CPU
    ---
    tags:
      - GPU Management
    responses:
      200:
        description: 卸载成功
    """
    gpu_manager.offload_all()
    return jsonify({
        'status': 'success',
        'message': '所有模型已卸载到 CPU'
    })


@app.route('/api/gpu/release', methods=['POST'])
def release_gpu():
    """
    完全释放所有模型
    ---
    tags:
      - GPU Management
    responses:
      200:
        description: 释放成功
    """
    gpu_manager.release_all()
    return jsonify({
        'status': 'success',
        'message': '所有模型已完全释放'
    })


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    转录音频（ASR）
    ---
    tags:
      - TTS
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: 音频文件
    responses:
      200:
        description: 转录结果
    """
    if 'audio' not in request.files:
        return jsonify({'error': '未上传音频文件'}), 400

    audio_file = request.files['audio']

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        whisper = get_whisper_model()
        segments, info = whisper.transcribe(
            audio=tmp_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700)
        )

        text = ""
        for segment in segments:
            text += segment.text + "\n"

        return jsonify({
            'text': text.strip(),
            'language': info.language if hasattr(info, 'language') else None
        })

    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


@app.route('/api/tts', methods=['POST'])
def tts_generate():
    """
    生成语音
    ---
    tags:
      - TTS
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: 参考音频文件
      - name: ref_text
        in: formData
        type: string
        required: true
        description: 参考音频的文本
      - name: target_text
        in: formData
        type: string
        required: true
        description: 要生成的文本
      - name: system_prompt
        in: formData
        type: string
        required: false
        description: 系统提示词
      - name: sample_rate
        in: formData
        type: integer
        required: false
        description: 采样率（默认 24000）
      - name: temperature
        in: formData
        type: number
        required: false
        description: Temperature（默认 1.0）
      - name: top_k
        in: formData
        type: integer
        required: false
        description: Top-K（默认 50）
      - name: top_p
        in: formData
        type: number
        required: false
        description: Top-P（默认 0.9）
      - name: penalty
        in: formData
        type: number
        required: false
        description: 重复惩罚（默认 1.2）
      - name: random_seed
        in: formData
        type: integer
        required: false
        description: 随机种子（默认 49）
    responses:
      200:
        description: 生成的音频文件
      400:
        description: 参数错误
      500:
        description: 生成失败
    """
    # 参数验证
    if 'audio' not in request.files:
        return jsonify({'error': '未上传参考音频'}), 400

    ref_text = request.form.get('ref_text', '').strip()
    target_text = request.form.get('target_text', '').strip()

    if not ref_text or not target_text:
        return jsonify({'error': '缺少必需的文本参数'}), 400

    # 获取参数
    system_prompt = request.form.get('system_prompt', 'Convert the text to speech')
    sample_rate = int(request.form.get('sample_rate', 24000))
    temperature = float(request.form.get('temperature', 1.0))
    top_k = int(request.form.get('top_k', 50))
    top_p = float(request.form.get('top_p', 0.9))
    penalty = float(request.form.get('penalty', 1.2))
    random_seed = int(request.form.get('random_seed', 49))

    # 保存音频
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        tmp_audio_path = tmp.name

    try:
        # 处理 TTS
        output_path = tts_process(
            audio_path=tmp_audio_path,
            ref_text=ref_text,
            target_text=target_text,
            system_prompt=system_prompt,
            sample_rate=sample_rate,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            penalty=penalty,
            random_seed=random_seed
        )

        if output_path and Path(output_path).exists():
            return send_file(
                output_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'generated_{int(time.time())}.wav'
            )
        else:
            return jsonify({'error': '音频生成失败'}), 500

    finally:
        # 清理临时文件
        Path(tmp_audio_path).unlink(missing_ok=True)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('API_SERVER_PORT', 7861))

    logger.info(f"🚀 启动 Llasa-TTS-8B API 服务器")
    logger.info(f"📊 访问地址: http://0.0.0.0:{port}")
    logger.info(f"📖 API 文档: http://0.0.0.0:{port}/apidocs")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
