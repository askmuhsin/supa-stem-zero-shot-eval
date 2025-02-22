# OpenAI Chat Completion Parameters Guide

## Essential Parameters (Most Used)

### messages [Required]
```python
messages=[
    {"role": "user", "content": "Your question here"},
    {"role": "assistant", "content": "Previous response"},
]
```
- List of messages comprising the conversation
- Roles: "user", "assistant", "system"
- Always required for chat completions

### model [Required]
```python
model="gpt-3.5-turbo"  # or "gpt-4", etc.
```
- Specifies which model to use
- Affects capability and cost

### temperature [Optional, Default: 1.0]
```python
temperature=0.7
```
- Range: 0-2
- Lower (0.1-0.4): More focused, deterministic
- Higher (0.7-1.0): More creative, varied
- Most used range: 0.7-0.8

### max_tokens [Optional]
```python
max_tokens=150
```
- Maximum length of response
- Affects cost and response length
- Default varies by model

## Advanced Parameters (Situational Use)

### top_p [Optional, Default: 1.0]
```python
top_p=0.9
```
- Alternative to temperature
- Range: 0-1
- Lower values = more focused

### n [Optional, Default: 1]
```python
n=3
```
- Number of completions to generate
- Useful for getting multiple responses

### stream [Optional, Default: False]
```python
stream=True
```
- Stream responses token by token
- Good for real-time applications

### presence_penalty [Optional, Default: 0]
```python
presence_penalty=0.6
```
- Range: -2.0 to 2.0
- Positive values encourage new topics

### frequency_penalty [Optional, Default: 0]
```python
frequency_penalty=0.6
```
- Range: -2.0 to 2.0
- Reduces repetition in responses

## Example Usage

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Write a short story"}
    ],
    temperature=0.7,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

## Best Practices

1. Start with essential parameters only
2. Adjust temperature first for output control
3. Use max_tokens to manage costs
4. Add advanced parameters only when needed
5. Keep system messages concise
