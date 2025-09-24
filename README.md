# ReID (Person Re-identification) System

人物再識別（Person Re-identification）システムです。YOLOによる人物検出とCLIP-ReIDによる人物識別を組み合わせたリアルタイム人物追跡システムです。

## 機能

- **リアルタイム人物検出**: YOLOモデルによる人物検出
- **人物再識別**: CLIP-ReIDモデルによる人物識別
- **WebSocket通信**: リアルタイムデータ送信
- **REST API**: 人物識別結果の取得

## システム構成

### カメラアプリ (`camra_app/`)
- カメラからの映像取得
- YOLOによる人物検出
- 人物画像のクロップとメタデータ生成

### サーバーアプリ (`server_app/`)
- FastAPIによるREST API
- 人物識別処理
- WebSocket通信

### 共通ライブラリ (`src/`)
- データ処理ライブラリ
- 前処理・後処理モジュール
- ReIDモデル実装

## 依存関係

主要な依存関係は `requirements.txt` を参照してください。

## ライセンス

MIT License

Copyright (c) 2024 komori_lab

詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 使用方法

1. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

2. カメラアプリの実行:
```bash
cd camra_app
python main.py
```

3. サーバーアプリの実行:
```bash
cd server_app
python main.py
```

## 注意事項

- このプロジェクトには外部ライブラリ（clip_reid等）が含まれており、それぞれ独自のライセンスが適用されます
- 商用利用前に各依存関係のライセンス要件を確認してください
