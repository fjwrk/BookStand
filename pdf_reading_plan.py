import argparse
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
import subprocess
import time
from langdetect import detect_langs, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 設定
PDF_PATH = None
MINUTES_PER_DAY = 20  # デフォルト: 1日あたりの読書時間（分）
START_DATE = datetime.today()


# PDFページごとのテキスト抽出と文字数カウント
def get_pdf_page_texts_and_counts(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    page_char_counts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        page_texts.append(text)
        page_char_counts.append(len(text))
    return page_texts, page_char_counts


def detect_language_of_document(page_texts, sample_chars=2000):
    DetectorFactory.seed = 0
    sample = ""
    for t in page_texts:
        if t and t.strip():
            sample += t + "\n"
        if len(sample) >= sample_chars:
            break
    if not sample:
        return None
    try:
        langs = detect_langs(sample)
        if not langs:
            return None
        # 最も確度の高い言語を返す
        return langs[0].lang
    except Exception:
        return None


def normalize_paragraphs(text):
    """テキストの行分割を解除して意味のある段落単位に整形する。

    - 空行は段落境界とみなす
    - 行末がハイフンならハイフンを削除して単語を連結
    - その他は空白で連結して段落を構成する
    """
    if not text:
        return []
    text = text.replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    paras = []
    cur = ""
    for line in lines:
        if not line:
            if cur:
                paras.append(cur.strip())
                cur = ""
            continue
        if not cur:
            cur = line
            continue
        # ハイフンによる継続（単語分断）
        if cur.endswith("-"):
            cur = cur[:-1] + line
            continue
        # 普通はスペースで連結（大きな段落を作る）
        cur = cur + " " + line
    if cur:
        paras.append(cur.strip())
    return paras


def chunk_paragraphs(paragraphs, max_chars=800):
    """段落リストを指定文字数以下のチャンク（段落リスト）に分割する。"""
    if not paragraphs:
        return []
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        plen = len(p)
        if cur_len + plen <= max_chars or not cur:
            cur.append(p)
            cur_len += plen
        else:
            chunks.append(cur)
            cur = [p]
            cur_len = plen
    if cur:
        chunks.append(cur)
    return chunks


def translate_pages_to_japanese(page_texts, out_path="translations.md"):
    # 旧: googletrans を使った翻訳（非公式）。ここでは MarianMT を利用する関数を別に用意しています。
    print("translate_pages_to_japanese: 非推奨（ローカルMarianMTを使用してください）")


def translate_pages_marian(
    page_texts,
    out_path="translations.md",
    model_id="Helsinki-NLP/opus-mt-en-ja",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    # Determine encoder max length from model config (fallback to 512)
    encoder_max_len = getattr(model.config, "max_position_embeddings", None)
    if encoder_max_len is None:
        encoder_max_len = getattr(model.config, "max_length", 512)
    try:
        encoder_max_len = int(encoder_max_len)
    except Exception:
        encoder_max_len = 512
    # cap to a reasonable upper bound
    encoder_max_len = min(encoder_max_len, 1024)

    translated_pages = ["" for _ in page_texts]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Translations (Japanese)\n\n")
        for idx, text in enumerate(page_texts):
            if not text or not text.strip():
                continue
            # 段落単位に正規化してからチャンク化する
            paragraphs = normalize_paragraphs(text)
            para_chunks = chunk_paragraphs(paragraphs, max_chars=800)
            translated_parts = []
            for chunk_paras in para_chunks:
                chunk_text = "\n\n".join(chunk_paras)
                try:
                    inputs = tokenizer(
                        chunk_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=encoder_max_len,
                    )
                    input_ids = inputs.get("input_ids")
                    attention_mask = inputs.get("attention_mask")
                    if input_ids is None:
                        raise ValueError("tokenizer returned no input_ids")
                    input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    try:
                        outs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=256,
                            num_beams=4,
                            early_stopping=True,
                        )
                    except IndexError as ie:
                        print(
                            f"ページ{idx + 1}チャンクでIndexError: {ie}; input_ids.shape={tuple(input_ids.shape)}"
                        )
                        continue
                    except Exception:
                        try:
                            outs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=128,
                                num_beams=2,
                                early_stopping=True,
                            )
                        except Exception as e2:
                            print(
                                f"ページ{idx + 1}チャンクの生成で失敗: {e2} — 当該チャンクをスキップします。"
                            )
                            continue
                    translated = tokenizer.decode(outs[0], skip_special_tokens=True)
                    translated_parts.append(translated)
                except Exception as e:
                    print(
                        f"ページ{idx + 1}のチャンク翻訳でエラー: {e} — 当該チャンクをスキップします。"
                    )
                    continue
            full_trans = "\n\n".join(translated_parts)
            translated_pages[idx] = full_trans
            f.write(f"## Page {idx + 1}\n\n")
            f.write(full_trans + "\n\n")
    print(f"日本語訳を{out_path}に出力しました（モデル: {model_id}）。")
    return translated_pages


# 指定テキストを音読し所要時間を計測（macOS の say を利用）
def tts_and_measure(text, say_rate=None):
    # say に与えると長すぎて失敗する場合があるため、短く区切って実行
    CHUNK = 4000
    start = time.time()
    cmd_base = ["say"]
    if say_rate:
        cmd_base += ["-r", str(int(say_rate))]
    for i in range(0, len(text), CHUNK):
        chunk = text[i : i + CHUNK]
        subprocess.run(cmd_base + [chunk])
    end = time.time()
    return end - start


# サンプルページで1文字あたりの音読秒数を計測
def measure_seconds_per_char(
    page_texts, sample_pages=None, say_rate=None, skip_tts=False
):
    if sample_pages is None:
        # 最初の非空ページを最大3ページサンプルにする
        sample_pages = []
        for idx, t in enumerate(page_texts):
            if t and t.strip():
                sample_pages.append(idx)
            if len(sample_pages) >= 3:
                break
    total_chars = 0
    total_seconds = 0.0
    for i in sample_pages:
        if i < 0 or i >= len(page_texts):
            continue
        text = page_texts[i]
        if not text.strip():
            continue
        print(f"ページ{i + 1}を音読して計測します...")
        if skip_tts:
            print("音読計測をスキップします（デフォルト値を使用）。")
            sec = 0.0
        else:
            sec = tts_and_measure(text, say_rate=say_rate)
        chars = len(text)
        print(f"{chars}文字, {sec:.2f}秒")
        total_chars += chars
        total_seconds += sec
    if total_chars == 0:
        # フォールバック値: 1文字あたり0.5秒
        return 0.5
    # total_seconds may be 0 if skip_tts was True; handle that
    if total_seconds == 0.0:
        return 0.5
    return total_seconds / total_chars


# ページごとの音読所要秒数リスト作成
def estimate_page_seconds(page_char_counts, sec_per_char):
    return [count * sec_per_char for count in page_char_counts]


# 読書計画生成（ページごとの所要時間を考慮）
def generate_reading_plan_by_seconds(page_seconds, minutes_per_day, start_date):
    seconds_per_day = minutes_per_day * 60
    plan = []
    i = 0
    total_pages = len(page_seconds)
    while i < total_pages:
        day = start_date + timedelta(days=len(plan))
        start_page = i + 1
        acc_sec = 0.0
        # 可能な限りその日の時間に収まるページを追加
        while i < total_pages and acc_sec + page_seconds[i] <= seconds_per_day:
            acc_sec += page_seconds[i]
            i += 1
        # 1ページも入らなかった場合は1ページだけ進める（長い章の場合を想定）
        if start_page == i + 1:
            i += 1
        end_page = i
        plan.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "start_page": start_page,
                "end_page": end_page,
            }
        )
    return plan


# Obsidian用ToDo出力
def export_obsidian_todo(plan):
    lines = []
    for item in plan:
        lines.append(
            f"- [ ] {item['date']} ページ{item['start_page']}〜{item['end_page']}"
        )
    return "\n".join(lines)


def write_plan_with_frontmatter(
    plan, pdf_path, minutes_per_day, tts_speed, out_path="reading_plan.md"
):
    created = datetime.now().strftime("%Y-%m-%d")
    front = [
        "---",
        f'title: "Reading Plan for {pdf_path}"',
        f"created: {created}",
        f'pdf: "{pdf_path}"',
        f"minutes_per_day: {minutes_per_day}",
        f"tts_speed: {tts_speed}",
        f"total_pages: {len(plan) and sum([item['end_page'] - item['start_page'] + 1 for item in plan]) or 0}",
        "---",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(front))
        f.write(export_obsidian_todo(plan))
    print(f"Obsidian用計画を{out_path}に出力しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reading plan for a PDF with optional offline translation and TTS settings"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--minutes", type=int, default=20, help="Minutes per day")
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        choices=[1.0, 1.5, 2.0],
        help="TTS playback speed multiplier",
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help="Skip TTS measurement and use fallback speed",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip translation step (do not attempt MarianMT)",
    )
    parser.add_argument(
        "--model-id",
        default="staka/fugumt-en-ja",
        help="MarianMT model id for translation",
    )
    args = parser.parse_args()

    PDF_PATH = args.pdf
    MINUTES_PER_DAY = args.minutes
    START_DATE = datetime.today()

    page_texts, page_char_counts = get_pdf_page_texts_and_counts(PDF_PATH)
    print(f"全{len(page_texts)}ページ、サンプルページで音読速度を計測します。")
    # map tts-speed multiplier to say rate (words per minute)
    default_wpm = 200
    say_rate = int(default_wpm * args.tts_speed)
    sec_per_char = measure_seconds_per_char(
        page_texts, say_rate=say_rate, skip_tts=args.no_measure
    )
    print(f"1文字あたり平均{sec_per_char:.3f}秒 (tts-speed={args.tts_speed}x)")
    page_seconds = estimate_page_seconds(page_char_counts, sec_per_char)
    plan = generate_reading_plan_by_seconds(page_seconds, MINUTES_PER_DAY, START_DATE)
    # write Obsidian plan with frontmatter
    write_plan_with_frontmatter(plan, PDF_PATH, MINUTES_PER_DAY, args.tts_speed)
    # 自動言語判定と（英語なら）MarianMTによるオフライン翻訳
    if args.no_translate:
        translated_pages = None
        print("翻訳をスキップしました（--no-translate）。")
    else:
        lang = detect_language_of_document(page_texts)
        print(f"検出言語: {lang}")
        if lang and lang.startswith("en"):
            try:
                translated_pages = translate_pages_marian(
                    page_texts, out_path="translations.md", model_id=args.model_id
                )
            except Exception as e:
                print("翻訳に失敗しました:", e)
                translated_pages = None
        else:
            translated_pages = None
            print(
                "英語文書ではないか判定できたため、翻訳はスキップしました。必要なら強制翻訳を実行してください。"
            )

    # 各日の読み上げ対象テキストを Markdown に出力
    def export_reading_segments(
        plan, page_texts, translated_pages=None, out_path="reading_segments.md"
    ):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Reading Segments\n\n")
            for item in plan:
                start = item["start_page"] - 1
                end = item["end_page"]
                if start >= end:
                    continue
                f.write(
                    f"## {item['date']} — Pages {item['start_page']}–{item['end_page']}\n\n"
                )
                # 原文
                f.write("### Original\n\n")
                segment_text = "\n\n".join(page_texts[start:end])
                f.write(segment_text + "\n\n")
                # 翻訳があれば書く
                if translated_pages:
                    f.write("### Japanese Translation\n\n")
                    trans_segment = "\n\n".join(
                        [
                            translated_pages[i]
                            for i in range(start, end)
                            if translated_pages[i]
                        ]
                    )
                    f.write(trans_segment + "\n\n")
                f.write("---\n\n")
        print(f"読み上げ用テキストを{out_path}に出力しました。")

    export_reading_segments(plan, page_texts, translated_pages=translated_pages)
