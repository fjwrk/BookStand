# BookStand

PDFから毎日の読書計画を生成し、Obsidianで扱いやすい形式で出力する小さなCLIツールです。

## 機能

- PDFからテキストを抽出し、ページごとの読書時間を推定します。
- 1日あたりの目標分数に合わせた日次の読書計画（Obsidian Tasks形式のTODO出力）を生成します。
- 翻訳はローカルMarianMTまたはリモート翻訳サーバ（`--translate-url`）を選択可能です。
- ページ単位のHTMLビューアを出力します（左に埋め込みPDF、右に英語原文／日本語訳、TTS操作付き）。
- 各PDFごとにタイムスタンプ付きフォルダを作成し、`reading_plan.md`、`md/translations.md`、`md/reading_segments.md`、`pages/pageN.html`、`metadata.json` を出力します。

## Obsidian Tasks互換性

- 出力される `reading_plan.md` は Obsidian Tasks プラグインが認識しやすい日付表記（絵文字）を使用します。タスク例:

```markdown
- [ ] BookStand MyDoc.pdf — Pages 1–1 🛫 2025-12-19 📆 2025-12-19 — [Open HTML](file:///absolute/path/to/pages/page1.html)
```

- 意味:
	- `🛫 YYYY-MM-DD` = スケジュール（開始予定日）
	- `📆 YYYY-MM-DD` = 締切（期限日）

- 出力は絶対 `file://` URI を使っており、Obsidianから直接開けます。

## ステージング / アトミック出力

- 生成はまず `.staging` ディレクトリ内で行われ、すべての生成物（翻訳、HTML、metadata）が揃った段階で最終のタイムスタンプ付きフォルダへアトミックにリネームされます。これにより不完全な出力がObsidianに見えるのを防ぎます。

## インストール

仮想環境を作成して有効化し、依存関係をインストールしてください:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` が無い場合の最低限のパッケージ:

```bash
pip install PyPDF2 langdetect transformers torch requests
```

## 使い方

簡易実行（翻訳なし、TTS計測なし）:

```bash
python pdf_reading_plan.py /path/to/file.pdf --no-translate --no-measure --export-html
```

リモート翻訳サーバ（POST `/translate`）を使う例:

```bash
python pdf_reading_plan.py /path/to/file.pdf --translate-url http://<host>:8000/translate --no-measure --export-html
```

PDFフォルダ内の全ファイルを一括処理する例:

```bash
for f in PDF/*.pdf; do python pdf_reading_plan.py "$f" --translate-url http://<host>:8000/translate --no-measure --export-html; done
```

TTS速度オプション: `1.0`, `1.5`, `2.0`（例: `--tts-speed 1.5`）。`--no-measure` を指定すると既定値（平均速度）を使用します。

## 注意事項

- ローカルMarianMTモデルは大きく、ダウンロードに時間とディスク容量が必要です。
- リモート翻訳サーバを利用する場合、`/translate` エンドポイントが JSON `{text, src_lang, tgt_lang}` を受け取り `{text}` を返す形式を満たしていることを確認してください。

## バージョン

Tool version: 0.2

## ライセンス

MIT
