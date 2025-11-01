from pathlib import Path
from typing import List
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import hashlib
import json
import re

# ==== CONFIG ====
MODEL_CANDIDATES = ["gpt-5-pro", "gpt-5", "gpt-4o", "gpt-4.1"]
INPUT_MD = "Minimal Viable Product Plan.md"
OUTPUT_MD = "Minimal_Viable_Product_Plan_KR.md"
CHECKPOINT_JSON = ".translation_progress.json"
MAX_CHARS_PER_CHUNK = 14000  # keep comfortably below context limits
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "You are a meticulous translation engine for software documentation.\n"
    "Goal: Translate ALL content to **natural Korean (Hangul)** in a **casual developer tone** "
    "without leaving any English, while preserving the original Markdown structure exactly "
    "(headings, lists, tables, code fences, footnotes, links, images, inline code, etc.).\n\n"
    "CRITICAL:\n"
    "- Translate EVERYTHING: prose, headings, tables, inline code content, code comments, "
    "  docstrings, identifiers, filenames, UI strings.\n"
    "- Keep the same Markdown structure and fencing. Do not add or remove blocks.\n"
    "- Do NOT include any English in the output unless it is a code symbol that has no Korean equivalent.\n"
    "- Keep link targets (URLs) unchanged, but translate link text.\n"
    "- Keep code fence languages the same (e.g., ```python stays ```python), but translate identifiers/comments/strings.\n"
    "- Maintain spacing and indentation.\n"
    "- No extra commentary; output ONLY the translated Markdown for the provided chunk.\n"
)

USER_INSTRUCTIONS_TEMPLATE = """\
다음의 마크다운 조각을 **자연스러운 개발자 한국어**로 번역해 주세요.
요구 사항:
- 영어 원문은 제거하고 **한국어만** 남겨요.
- 마크다운 구조(헤딩, 목록, 표, 인라인 코드, 링크, 이미지, 각주, 코드블록 등)는 **그대로 유지**해요.
- 코드 블록 내부도 **식별자/주석/문자열까지 모두 한글화**해요. (실행 가능성은 고려하지 않아요.)
- 링크 URL은 바꾸지 말고, 링크 텍스트만 번역해요.
- 코드 펜스 언어 표기는 유지해요. (예: ```python)
- 추가 설명 없이 **번역된 마크다운만** 출력해요.

---
{chunk}
---
"""

client = OpenAI()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def md_heading_split(text: str) -> List[str]:
    """
    Split by top-level and second-level headings first to keep sections coherent.
    Then re-pack to honor MAX_CHARS_PER_CHUNK.
    """
    # Split on lines that start with one or more '#' followed by a space
    pieces = re.split(r"(?m)^(?=#)", text)
    # Repack to size
    chunks, buf = [], ""
    for p in pieces:
        if not p:
            continue
        if len(buf) + len(p) <= MAX_CHARS_PER_CHUNK:
            buf += p
        else:
            if buf.strip():
                chunks.append(buf)
            if len(p) <= MAX_CHARS_PER_CHUNK:
                buf = p
            else:
                # If a single piece is still too big, split by subsections or paragraphs
                buf = ""
                sub = paragraph_split(p)
                for s in sub:
                    if len(buf) + len(s) <= MAX_CHARS_PER_CHUNK:
                        buf += s
                    else:
                        if buf.strip():
                            chunks.append(buf)
                        buf = s
    if buf.strip():
        chunks.append(buf)
    return chunks

def paragraph_split(piece: str) -> List[str]:
    # Split by double newlines as a fallback
    paras = piece.split("\n\n")
    out, buf = [], ""
    for para in paras:
        if len(buf) + len(para) + 2 <= MAX_CHARS_PER_CHUNK:
            buf += (("\n\n" if buf else "") + para)
        else:
            if buf.strip():
                out.append(buf)
            if len(para) <= MAX_CHARS_PER_CHUNK:
                buf = para
            else:
                # Hard split long paragraph
                for i in range(0, len(para), MAX_CHARS_PER_CHUNK):
                    out.append(para[i:i+MAX_CHARS_PER_CHUNK])
                buf = ""
    if buf.strip():
        out.append(buf)
    return out

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def load_progress(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}, "order": []}

def save_progress(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def translate_chunk(chunk_text: str) -> str:
    user_prompt = USER_INSTRUCTIONS_TEMPLATE.format(chunk=chunk_text)

    last_err = None
    for model_name in MODEL_CANDIDATES:
        try:
            # Prefer Responses API for modern models (works for gpt-5/gpt-4o).
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            return resp.output[0].content[0].text

        except Exception as e:
            # If a model isn’t permitted (e.g., 400 unsupported_value), try next.
            last_err = e
            print(f"⚠️ Model {model_name} failed: {e}. Trying next…")
            continue

    # If all attempts failed, raise the last error
    raise last_err

def main():
    src = Path(INPUT_MD)
    dst = Path(OUTPUT_MD)
    ckpt = Path(CHECKPOINT_JSON)

    original = read_text(src)
    chunks = md_heading_split(original)

    progress = load_progress(ckpt)
    if not progress["order"]:
        progress["order"] = [hash_chunk(c) for c in chunks]
        save_progress(ckpt, progress)

    translated_sections = []
    for idx, chunk in enumerate(chunks):
        h = hash_chunk(chunk)
        if h in progress["completed"]:
            print(f"✅ Skipping {idx+1}/{len(chunks)} (already done)")
            translated = progress["completed"][h]
        else:
            print(f"🔄 Translating {idx+1}/{len(chunks)}")
            translated = translate_chunk(chunk)
            # Basic sanity: preserve code fence count
            def fence_count(s: str) -> int:
                return len(re.findall(r"(?m)^```", s))
            if fence_count(chunk) != fence_count(translated):
                # If fence counts differ, try a quick second attempt
                print("⚠️ Code fence count mismatch. Retrying once...")
                translated = translate_chunk(chunk)

            progress["completed"][h] = translated
            save_progress(ckpt, progress)

        translated_sections.append(translated)

    final_text = "\n\n".join(translated_sections).strip()
    write_text(dst, final_text)
    print(f"\n✅ Done. Saved: {dst}")
    print(f"🧾 Progress file: {ckpt} (you can delete it after verifying the result)")

if __name__ == "__main__":
    main()