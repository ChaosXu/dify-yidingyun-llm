import fitz
import openai
import os

from openai.types.chat import ChatCompletion
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


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)
    result = analyze_pdf_with_openai(pdf_text=text, prompt="请总结这份 PDF 的主要内容")
    print("PDF 结果：")
    print(result.choices[0].message.content)
