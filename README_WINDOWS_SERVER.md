Windowsでの翻訳サーバ起動手順
=================================

このドキュメントは、リポジトリ内の `server.py` を Windows 環境で起動するための手順をまとめたものです。

前提
- Python 3.8+ がインストールされていること
- Git リポジトリがローカルにクローン済みであること
- （オプション）GPU を使う場合は CUDA 対応の PyTorch をインストールしておくこと

1) 仮想環境の作成

```powershell
cd \path\to\BookStand
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# または
.\.venv\Scripts\activate.bat    # cmd.exe
```

2) 依存パッケージのインストール

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
# CPUのみで動かす場合、必要に応じて torch の CPU ビルドを明示的に指定
```

3) 必要モデルの準備
- `server.py` は Hugging Face Transformers ベースのモデルをロードします。モデルIDを指定してダウンロードしてください（例: `staka/fugumt-en-ja` や `facebook/mbart-large-50-many-to-many-mmt`）。
- 初回起動時にモデルのダウンロードが行われるため十分なディスクとネットワークが必要です。

4) 環境変数（任意）
- モデルIDやバインド先ポートを環境変数で渡すことができます（`SERVER_MODEL_ID`, `SERVER_PORT` などを `server.py` に合わせて設定）。

5) サーバ起動（開発用）

```powershell
# PowerShell の例: uvicorn を使って起動
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

# 起動ログが出力され、/health と /translate エンドポイントが利用可能になります
```

6) 常駐化/サービス化（Windows サービス）
- プロダクションで常駐させたい場合は NSSM (Non-Sucking Service Manager) などを使って Python コマンドを Windows サービスとして登録してください。

7) 注意点
- 大きなモデルをメモリに載せるとメモリ不足になるため、必要に応じて小さいモデルを使うか CPU/GPU の割り当てに注意してください。
- ファイアウォールやポート開放設定を確認してください。

参考: `server.py` の実装に合わせて環境変数名や引数を適宜調整してください。

---

変更履歴:
- 2025-12-17: 初版作成
