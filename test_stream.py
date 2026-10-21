
import json
import httpx

url = "http://127.0.0.1:8002/chat/stream"

data = {
    "thread_id": "stream_test_001",
    "message": "店铺支持多久无理由退货？"
}

print("开始测试 SSE 流式接口...\n")

with httpx.stream(
    "POST",
    url,
    json=data,
    timeout=120.0
) as response:

    print("HTTP 状态码：", response.status_code)
    response.raise_for_status()

    event = ""

    for line in response.iter_lines():

        # 获取事件名称
        if line.startswith("event:"):
            event = line[6:].strip()

        # 获取事件数据
        elif line.startswith("data:"):

            payload = json.loads(line[5:].strip())

            if event == "status":
                print(
                    "\n[状态]",
                    payload.get("text", "")
                )

            elif event == "token":
                print(
                    payload.get("text", ""),
                    end="",
                    flush=True
                )

            elif event == "done":
                print("\n\n[回答完成]")

            elif event == "error":
                print(
                    "\n[错误]",
                    payload.get("message", "")
                )