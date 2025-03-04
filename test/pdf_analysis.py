import fitz
import openai
import os

from openai.types.chat import ChatCompletionChunk, ChatCompletion
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = os.getenv("MODEL")


class OpenAIWithoutAuth(openai.OpenAI):
    def __init__(self, base_url):
        super().__init__(api_key="", base_url=base_url)

    @property
    def auth_headers(self):
        return {}


def extract_text_from_pdf(pdf_path):
    """从 PDF 文件中提取文本"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def analyze_pdf_with_openai(pdf_text: str, prompt: str) -> ChatCompletion:
    client = OpenAIWithoutAuth(base_url=OPENAI_API_BASE)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个擅长分析 PDF 文档的助手。"},
            {"role": "user", "content": f"{prompt}\n\n{pdf_text}"},
        ],
        stream=False,
    )

    return response


def analyze_pdf_with_openai_stream(pdf_text: str, prompt: str):
    client = OpenAIWithoutAuth(base_url=OPENAI_API_BASE)

    # 设置 stream 为 True
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个擅长分析 PDF 文档的助手。"},
            {"role": "user", "content": f"{prompt}\n\n{pdf_text}"},
        ],
        stream=True,
    )

    # 迭代生成器，逐步获取响应内容
    for chunk in response:
        chunk: ChatCompletionChunk
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            if content.startswith("<think>"):
                print("深度思考:")
                content = content.replace("<think>", "")
            if content.endswith("</think>"):
                print("思考结束!")
                content = content.replace("</think>", "")
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)
    analyze_pdf_with_openai_stream(pdf_text=text, prompt="请总结这份 PDF 的主要内容")
