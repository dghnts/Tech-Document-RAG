# Tech Document RAG

技術ドキュメントのURLを解析し、Google Gemini (2.0 Flash) を使用して質問に回答するRAG（Retrieval-Augmented Generation）アプリケーションです。

## 🚀 特徴

- **URL解析**: 指定されたURL（Streamlitドキュメントなど）の内容を読み込み、即座にチャット可能な状態にします。
- **多言語対応**: 日本語と英語の表示切り替えに対応しています。
- **モダンな技術スタック**: LangChain、FAISS、Google Gemini APIを活用。
- **高速な環境管理**: `uv` を使用したスムーズなパッケージ・仮想環境管理。

## 📋 セットアップ

### 1. プリリクエスト
- Python 3.12以上
- [uv](https://docs.astral.sh/uv/)（推奨）
- Google Gemini APIキー

### 2. 環境構築
`uv` を使用して依存関係をインストールし、環境を同期します。
```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 3. アプリケーションの実行
```bash
uv run streamlit run app.py
```

## 🛠 使い方

1. サイドバーで **Language**（言語）を選択します。
2. サイドバーに **Gemini API Key** を入力します。
3. メイン画面の入力欄に解析したいドキュメントの **URL** を入力し、「Analyze Document」（または「ドキュメントを解析」）ボタンを押します。
4. 解析完了後、下のチャット入力欄からドキュメントに関する質問を投げてください。

## 🧰 技術スタック

- **UI**: [Streamlit](https://streamlit.io/)
- **LLM**: [Google Gemini 2.0 Flash](https://ai.google.dev/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [HuggingFace multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- **Dependency Management**: [uv](https://docs.astral.sh/uv/)
