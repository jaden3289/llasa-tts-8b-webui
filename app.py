import os
# 设置使用 GPU 1 和 2（用户指定 1 号和 2 号 GPU 比较空闲）
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '1,2')

from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
import torch
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio
import gradio as gr
import numpy
import random

"""
# Llasa-8B 下载需要 hf 的 token，下面是获取 token 和登录，如果需要就去除掉注释
# 建议直接将模型下载到本地加载
import os
api_key = os.getenv("HF_TOKEN")

from huggingface_hub import login
login(token=api_key)

# 没魔法就用这个镜像
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
"""


def set_seed(seed=49):
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# FastWhisper 配置
print("🔄 正在加载 FastWhisper 模型...")
fastwhisper_path = "Systran/faster-whisper-large-v3"
fastwhisper_model = WhisperModel(fastwhisper_path, device="cpu")
language = None
print("✅ FastWhisper 模型加载完成!")

def fastwhisper_asr_file(audio_file):
    if audio_file is None:
        gr.Warning("请先上传参考音频!")
        return ""
    text = ""
    try:
        segments, info = fastwhisper_model.transcribe(
            audio=audio_file,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            language=language
        )
        for segment in segments:
            text += segment.text + "\n"
        if not text.strip():
            gr.Warning("未能识别出音频中的文字，请检查音频质量或手动输入文本")
    except Exception as e:
        gr.Error(f"转录失败: {str(e)}")
        print(f"FastWhisper 错误: {e}")
    return text.strip()


# Llasa-8B 模型配置
print("🔄 正在加载 XCodec2 模型...")
model_path = "HKUSTAudio/xcodec2"
Codec_model = XCodec2Model.from_pretrained(model_path, device_map='cuda')
Codec_model.eval()
print("✅ XCodec2 模型加载完成!")

print("🔄 正在加载 Llasa-8B 模型 (这可能需要几分钟)...")
llasa_8b = 'HKUSTAudio/Llasa-8B'
tokenizer = AutoTokenizer.from_pretrained(llasa_8b)
llasa_model = AutoModelForCausalLM.from_pretrained(
    llasa_8b,
    torch_dtype=torch.float16,
    device_map='cuda'
)
llasa_model.eval()
print("✅ Llasa-8B 模型加载完成!")

# 打印 GPU 信息
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 GPU 信息:")
    print(f"   可用 GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"   当前使用的 GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}\n")
else:
    print("⚠️  警告: 未检测到 CUDA GPU!")


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def tts(sample_audio_path, sample_text, system_prompt_text, target_text,
        sample_rate_input=24000, penalty=1.2, temperature=1.0, top_k=50,
        top_p=0.9, do_sample=True, random_seed=49):

    # 参数验证
    if sample_audio_path is None:
        gr.Warning("请先上传参考音频!")
        return None
    if not sample_text or not sample_text.strip():
        gr.Warning("请输入参考音频的文本内容!")
        return None
    if not target_text or not target_text.strip():
        gr.Warning("请输入要生成的文字!")
        return None

    set_seed(random_seed)
    progress = gr.Progress()

    try:
        progress(0, '🎵 加载音频...')
        waveform, sample_rate = torchaudio.load(sample_audio_path)

        # 检查音频长度
        audio_duration = len(waveform[0]) / sample_rate
        if audio_duration > 20:
            gr.Warning(f"参考音频较长 ({audio_duration:.1f}秒)，建议使用 15-20 秒的音频以获得最佳效果")

        # 立体声转单声道
        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform

        prompt_wav = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=sample_rate_input
        )(waveform_mono)

        progress(0.2, '📝 处理文本...')
        if len(target_text) > 300:
            gr.Warning("要生成的文字太长，已自动截断到 300 个字符")
            target_text = target_text[:300]

        input_text = sample_text + ' ' + target_text

        with torch.no_grad():
            progress(0.4, '🔊 编码音频...')
            vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0, 0, :]
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            chat = [
                {"role": "user", "content": system_prompt_text + ":" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors='pt',
                continue_final_message=True
            )
            input_ids = input_ids.cuda()
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            progress(0.6, '🎙️ 生成语音中...')
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

            progress(0.8, '🎧 解码音频...')
            gen_wav = Codec_model.decode_code(speech_tokens)
            gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]

            progress(1, '✅ 生成完成!')
            return (sample_rate_input, gen_wav[0, 0, :].cpu().numpy())

    except Exception as e:
        gr.Error(f"生成失败: {str(e)}")
        print(f"TTS 生成错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 自定义 CSS 样式
custom_css = """
#main-container {
    max-width: 1400px;
    margin: auto;
}

.header-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

.sub-header {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 2em;
}

.info-box {
    background-color: #f0f7ff;
    border-left: 4px solid #3b82f6;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.warning-box {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.section-header {
    font-size: 1.5em;
    font-weight: bold;
    color: #1f2937;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    padding-bottom: 0.5em;
    border-bottom: 2px solid #e5e7eb;
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

footer {
    margin-top: 3em;
    text-align: center;
    color: #999;
}
"""

# 构建界面
with gr.Blocks(title="LLASA-8B TTS WebUI") as app:
    with gr.Column(elem_id="main-container"):
        # 标题
        gr.HTML("""
            <div class="header-text">🎙️ LLASA-8B TTS WebUI</div>
            <div class="sub-header">
                基于 Llasa-8B 的高质量语音合成系统 | 支持中英文混合语音克隆
            </div>
        """)

        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 🚀 快速开始

            1. **上传参考音频**：上传一段 15-20 秒的清晰语音（.wav 格式）
            2. **转录或输入文本**：使用自动转录功能，或手动输入参考音频中的文字
            3. **输入目标文本**：填写你想要生成的语音内容（最多 300 字）
            4. **调整参数**（可选）：根据需要调整高级参数
            5. **点击生成**：等待模型生成语音

            ### ⚙️ 参数说明

            - **Temperature**：控制生成的随机性（越低越稳定，越高越多样）
            - **Top K/Top P**：采样策略参数
            - **Penalty**：重复惩罚系数
            - **Sample Rate**：输出音频的采样率

            ### 💡 提示

            - 参考音频质量越高，生成效果越好
            - 建议使用清晰、无背景噪音的音频
            - 文本内容应与参考音频的语言风格一致
            """)

        # 高级设置区域
        with gr.Accordion("⚙️ 高级设置", open=False):
            with gr.Row():
                with gr.Column():
                    random_seed = gr.Number(
                        label="🎲 随机种子",
                        value=49,
                        minimum=0,
                        maximum=10000000,
                        step=1,
                        info="固定种子可以确保结果可复现"
                    )
                    sample_rate = gr.Dropdown(
                        choices=[16000, 22050, 24000, 28000, 32000],
                        value=24000,
                        label="🎵 采样率 (Hz)",
                        info="建议使用 24000 Hz"
                    )
                    do_sample = gr.Checkbox(
                        label="🎯 启用采样",
                        value=True,
                        info="关闭后将使用贪婪解码"
                    )

                with gr.Column():
                    temperature = gr.Slider(
                        label="🌡️ Temperature",
                        value=1.0,
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        info="控制生成的随机性"
                    )
                    top_k = gr.Slider(
                        label="📊 Top K",
                        value=50,
                        minimum=1,
                        maximum=100,
                        step=1,
                        info="Top-K 采样参数"
                    )
                    top_p = gr.Slider(
                        label="📈 Top P",
                        value=0.9,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="Nucleus 采样参数"
                    )
                    penalty = gr.Slider(
                        label="🔄 重复惩罚",
                        value=1.2,
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        info="惩罚重复内容"
                    )

        gr.HTML('<div class="section-header">📂 步骤 1: 上传参考音频</div>')
        gr.HTML("""
            <div class="info-box">
                ℹ️ <strong>提示：</strong>请上传 15-20 秒的 .wav 格式音频文件，确保音质清晰、无背景噪音
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                ref_audio_input = gr.Audio(
                    label="🎵 参考音频文件",
                    type="filepath",
                    sources=["upload"]
                )

            with gr.Column(scale=1):
                ref_text_input = gr.Textbox(
                    label="📝 参考音频的文本内容",
                    lines=5,
                    placeholder="点击「自动转录」按钮，或手动输入参考音频中说的内容...",
                    info="必须与参考音频内容一致"
                )
                transcribe_btn = gr.Button(
                    "🎤 自动转录（使用 FastWhisper）",
                    variant="secondary",
                    size="lg"
                )

        transcribe_btn.click(
            fn=fastwhisper_asr_file,
            inputs=[ref_audio_input],
            outputs=[ref_text_input]
        )

        gr.HTML('<div class="section-header">✍️ 步骤 2: 输入生成内容</div>')

        system_prompt_text_input = gr.Textbox(
            label="💬 系统提示词",
            lines=1,
            value="Convert the text to speech",
            info="通常不需要修改"
        )

        gen_text_input = gr.Textbox(
            label="📄 要生成的文字内容",
            lines=6,
            placeholder="在这里输入你想要生成的语音内容（最多 300 字符）...",
            info="支持中英文混合输入"
        )

        generate_btn = gr.Button(
            "🎙️ 生成语音",
            variant="primary",
            size="lg"
        )

        gr.HTML('<div class="section-header">🎧 步骤 3: 生成结果</div>')

        audio_output = gr.Audio(
            label="🔊 生成的语音",
            type="numpy"
        )

        generate_btn.click(
            fn=tts,
            inputs=[
                ref_audio_input,
                ref_text_input,
                system_prompt_text_input,
                gen_text_input,
                sample_rate,
                penalty,
                temperature,
                top_k,
                top_p,
                do_sample,
                random_seed
            ],
            outputs=[audio_output],
        )

        # 示例
        with gr.Accordion("📚 查看示例", open=False):
            gr.Examples(
                examples=[
                    [
                        None,
                        "这是一段参考音频的文本内容。",
                        "Convert the text to speech",
                        "你好，欢迎使用 Llasa 语音合成系统。",
                    ]
                ],
                inputs=[
                    ref_audio_input,
                    ref_text_input,
                    system_prompt_text_input,
                    gen_text_input,
                ],
                label="示例输入"
            )

        # 页脚
        gr.HTML("""
            <footer>
                <p>💻 Powered by <strong>Llasa-8B</strong> |
                🔬 Model: <a href="https://huggingface.co/HKUSTAudio/Llasa-8B" target="_blank">HKUSTAudio/Llasa-8B</a> |
                📝 License: MIT</p>
                <p style="font-size: 0.9em; color: #999;">
                    ⚡ GPU: {gpu_info}
                </p>
            </footer>
        """.format(
            gpu_info=f"使用 GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
            if torch.cuda.is_available() else "未检测到 GPU"
        ))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 启动 Llasa-8B TTS WebUI")
    print("="*60)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
