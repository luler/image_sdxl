import io
import json
import os

import dotenv
import gradio as gr
import requests
import retrying
from PIL import Image


@retrying.retry(stop_max_attempt_number=3, wait_fixed=1000)
def translate(text):
    with requests.get('https://lingva.thedaviddelta.com/api/v1/auto/en/' + text) as response:
        text = ''
        if response.status_code == 200:
            data = response.json()
            text = data['translation']
        return text


def get_image_content(text, guidance):
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + os.getenv('CLOUDFLARE_AI_TOKEN'),
    }
    data = {
        "guidance": guidance,
        "num_steps": 20,
        "prompt": text,
        "strength": 1
    }
    with requests.post('https://api.cloudflare.com/client/v4/accounts/' + os.getenv(
            'CLOUDFLARE_AI_ACCOUNT_ID') + '/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0', data=json.dumps(data),
                       headers=headers) as response:
        return response.content


@retrying.retry(stop_max_attempt_number=3, wait_fixed=1000)
def sdxl(text, guidance):
    return Image.open(io.BytesIO(get_image_content(text, guidance)))


def dosomething(text, guidance):
    return sdxl(translate(text), guidance)


# 这里是主程序的代码
if __name__ == "__main__":
    # 加载配置
    dotenv.load_dotenv()

    ii = gr.Textbox(label='图片描述')
    iii = gr.Slider(label='文本描述的遵循程度', minimum=1, maximum=20, value=7.5, step=0.1)
    oo = gr.Image(label='生成的图片')

    demo = gr.Interface(dosomething, inputs=[ii, iii], outputs=oo, title='基于Stable Diffusion模型的图片生成')

    demo.launch(server_port=7860)
