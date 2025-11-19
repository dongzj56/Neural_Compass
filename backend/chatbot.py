import requests
import json

API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
API_KEY = "1a8abe2a-4da6-47de-a7b8-6548a2533ceb" 

def get_llm_response(user_message: str, image_base64: str = None) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    user_content = []
    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_base64 
            }
        })
    user_content.append({"type": "text", "text": user_message})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "1、你是一个AI助手，专门用于回答阿尔茨海默病、轻度认知障碍、老年疾病等问题；2、请简洁明了地回答所有问题，不要使用加粗格式。3、你的核心功能是辅助受试者填写表格，比如有不懂的你要回答，此外，要为受试者分析结果，并回答使用者的一些疑问"}]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    data = {
        "model": "doubao-seed-1-6-251015",
        "max_completion_tokens": 1024,
        "reasoning_effort": "medium",
        "messages": messages
    }
    try:
        resp = requests.post(API_URL, headers=headers, data=json.dumps(data))
        resp.raise_for_status() 
        resp_json = resp.json()
        message = resp_json["choices"][0]["message"]
        return message.get("content", "")
    except requests.exceptions.RequestException as e:
        return f"抱歉，与大模型通信时发生错误: {e}"
    except KeyError as e:
        return "抱歉，无法解析大模型的响应。"
    except Exception as e:
        return "抱歉，发生未知错误。"
