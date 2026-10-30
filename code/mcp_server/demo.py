"""MCP 서버 직접 사용해 보기(터미널 대화). 실행: .venv-mcp/bin/python3 mcp_server/demo.py

평가와 같은 호스트(gpt-6-luna, Responses API)가 MCP 서버(mcp_server/server.py, FT 리랭커 전체 학습본,
도구 find_device·search_manual·search_official_web)를 stdio로 띄워 질문에 답한다. 도구 호출(이름·인자)을 화면에 같이 보여 준다.
입력: 처음에 기기 모델코드(예: SM-S741N, SM-R630), 그다음 질문을 계속 입력. 빈 줄 = 종료, '/기기' = 기기 바꾸기.
기기 이름 입력은 지원하지 않는다. 입력 코드와 그 코드의 시리즈를 그대로 고객 기기 문구에 쓴다(DC.find_by_code).
비용: 질문 1개 약 $0.01 안팎(외부 검색 호출 시 증가). 첫 검색은 모델 로드로 20~60초 걸릴 수 있음.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from rag import settings as C  # noqa: E402
from rag import create_answer as A  # noqa: E402


async def main():
    from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, set_tracing_disabled
    from agents.items import ToolCallItem
    from agents.mcp import MCPServerStdio
    from openai import AsyncOpenAI
    import devices as DC

    C.load_env_file()
    set_tracing_disabled(True)
    groups = DC.load_groups()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GALAXY_MCP_")}
    server = MCPServerStdio(name="galaxy-manual", cache_tools_list=True, client_session_timeout_seconds=600,
                            params={"command": sys.executable, "args": [str(ROOT / "mcp_server/server.py")],
                                    "env": env, "cwd": str(ROOT)})
    async with server:
        ins = getattr(server.server_initialize_result, "instructions", None) or server.session.instructions
        tools = [t.name for t in await server.list_tools()]
        print(f"MCP 서버 연결됨. 도구: {tools}\n")
        agent = Agent(name="galaxy-support", instructions=ins, mcp_servers=[server],
                      model=OpenAIResponsesModel(model=A.GEN_MODEL, openai_client=AsyncOpenAI()),
                      model_settings=ModelSettings(max_tokens=4000, store=False, response_include=["reasoning.encrypted_content"]))
        desc = None
        while True:
            if desc is None:
                q = input("기기 모델코드 (예: SM-S741N, 빈 줄 = 종료): ").strip()
                if not q:
                    break
                dev = DC.find_by_code(groups, q.upper())
                if dev is None:
                    print("  카탈로그에 없는 모델코드입니다. 기기 모델코드(예: SM-S741N, SM-R630)를 확인해 다시 입력해 주세요.")
                    continue
                desc = DC.model_desc(dev)
                print(f"  → {desc}\n")
                continue
            q = input(f"[{desc}] 질문 (빈 줄 = 종료, /기기 = 기기 바꾸기): ").strip()
            if not q:
                break
            if q == "/기기":
                desc = None
                continue
            res = await Runner.run(agent, f"[고객 기기] {desc}\n\n[고객 질문]\n{q}", max_turns=8)
            for it in res.new_items:
                if isinstance(it, ToolCallItem):
                    raw = it.raw_item
                    name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else "?")
                    args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "")
                    print(f"  🔧 {name}({args})")
            p = A.parse_output(str(res.final_output or ""))
            print(f"\n  [근거 충분: {p['sufficient']}]  [사용 근거: {p['evidence']}]\n")
            print(p["body"] if p["parse_ok"] else res.final_output)
            print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
