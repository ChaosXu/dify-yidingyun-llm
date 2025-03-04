curl -i -X POST http://localhost:9999/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-R1-Distill-Llama-70B",
  "messages": [
    {
      "role": "system",
      "content": "翻译用户输入内容为英语"
    },
    {
      "role": "user",
      "content":"你好世界"
    }
  ]
}'