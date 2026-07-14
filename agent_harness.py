"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ดึงฟังก์ชันที่เราทำสำเร็จแล้วจาก sales_logger มาใช้งานจริง # Session 2.5 - Agent Harness completed
from sales_logger import append_to_sheet, send_notification

TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """TODO 1: ส่ง cmd ไป Gemini พร้อม TOOL_SCHEMA ขอให้ตอบเป็น JSON {tool, args}"""
    # ตรวจสอบ API Key
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("ไม่พบ GEMINI_API_KEY ใน Environment")

    # ตั้งค่าเรียกใช้งาน Gemini Client ด้วย SDK ตัวใหม่ล่าสุด
    client = genai.Client(api_key=key)

    # วางระบบ System Instruction บังคับให้ AI ทำตัวเป็น Router ตัดสินใจเลือกเครื่องมือและห้ามหลุดนอกกรอบ
    system_instruction = (
        "คุณคือระบบแยกแยะเจตนาและแปลงคำสั่งของผู้ใช้ (Intent Router) "
        "หน้าที่ของคุณคือการวิเคราะห์คำสั่งของผู้ใช้ แล้วตัดสินใจเลือกใช้หนึ่งในเครื่องมือที่มีอยู่จาก TOOL_SCHEMA "
        f"ข้อมูลวันที่ปัจจุบัน: {datetime.now().strftime('%Y-%m-%d')}\n"
        "คุณต้องตอบกลับเป็นข้อความโครงสร้าง JSON ที่มีคีย์เป็น 'tool' และ 'args' เท่านั้น ห้ามเขียนอธิบายใดๆ ทั้งสิ้น\n"
        "ตัวอย่างผลลัพธ์:\n"
        '{"tool": "log_sale", "args": {"menu": "นมหมีฮอกไกโด", "qty": 2, "price": 65.0}}'
    )

    try:
        # สั่งงานโมเดลยอดนิยม gemini-2.5-flash บังคับคืนค่าเป็นประเภท JSON โครงสร้างแท้ๆ
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"คำสั่งผู้ใช้: {cmd}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )

        # ถอดรหัสข้อความ String JSON ออกมาเป็น Python Dictionary
        result = json.loads(response.text.strip())

        if "tool" not in result or "args" not in result:
            raise ValueError(
                "โครงสร้าง JSON ผลลัพธ์ไม่ถูกต้อง ขาดคีย์ 'tool' หรือ 'args'")

        return result

    except Exception as e:
        raise RuntimeError(f"การแปลงคำสั่งด้วย Gemini ผิดพลาด: {e}")


def dispatch_tool(tool_call: dict) -> str:
    """TODO 2: เรียก tool ตาม tool_call["tool"] ด้วย args จริง"""
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_sale":
        # ดึงค่าอาร์กิวเมนต์ที่สกัดได้ไปรันฟังก์ชันหลังบ้านของเราจริง
        menu = args.get("menu")
        qty = int(args.get("qty", 1))
        price = float(args.get("price", 0))

        row_result = append_to_sheet(menu=menu, qty=qty, price=price)
        total = row_result["total"]

        # ส่งแจ้งเตือนผ่านบอท Telegram ต่อยอดจากแล็บเดิม
        msg = f"บันทึก {menu} x{qty} = {total} บาท"
        send_notification(msg)

        return f"บันทึกสำเร็จ ยอดรวม {total} บาท"

    elif tool_name == "query_sales":
        # พาร์ทการคิวรีข้อมูล (ในแล็บสเต็ปนี้อาจจะส่งคำตอบ Mock กลับก่อน หรือดึงค่าจริงตามเงื่อนไขโจทย์)
        target_date = args.get("date")
        return f"ข้อมูลยอดขายของวันที่ {target_date} (Mock Result): รวมยอดขายทัังหมด 450 บาท"

    elif tool_name == "send_alert":
        message = args.get("message", "")
        send_notification(message)
        return "ส่งแจ้งเตือนสำเร็จ"

    else:
        raise RuntimeError(f"ไม่รู้จักเครื่องมือ: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")

    try:
        # สั่งรันชุดคิดวิเคราะห์ของ LLM (TODO 3)
        tool_call = parse_command(args.cmd)
        print(f"[LLM]   tool={tool_call['tool']} args={tool_call['args']}")

        # สั่งสลับไปทำงานฟังก์ชันจริงหลังบ้าน
        result = dispatch_tool(tool_call)
        print(f"[TOOL] {tool_call['tool']} {result}")
        print(f"[USER] ← {result}")

    except Exception as exc:
        print(f"[ERROR] ระบบพังล้มเหลว: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
