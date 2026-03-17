
<|vision_start|>
<|image_pad|>（或多个连续的 <|image_pad|>，数量取决于图像被切分成多少个 patch 组）
<|vision_end|>

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [
    {"role": "user", "content": [
        {"type": "image"},  # 或 "image": "url/path"
        {"type": "text", "text": "Describe this capacitor image."}
    ]},
    {"role": "assistant", "content": "..."}
]

# 只看文本部分（不 tokenize）
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(text)

# 这一步正确处理了图像占位符的插入
text = processor.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False
)
```