"""複数人物画像直接処理アプリケーション"""
from config import PersonImageAppConfig
from managers.post_processing_manager import PostProcessingManager
from managers.reid_model_manager import ReIDModelManager
from managers.data_set_manager import DataSetManager
from managers.data_manager import DataManager
import logging
import warnings
import numpy as np
import sys
import os
from pathlib import Path

# 警告を完全に抑制（環境変数レベル）
os.environ['PYTHONWARNINGS'] = 'ignore'

# 警告の抑制（より強力な方法）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Cython evaluation.*")
warnings.filterwarnings(
    "ignore", message=".*Importing from timm.models.layers.*")


class PersonImageReIDApp:
    """複数人物画像直接処理アプリケーション"""

    def __init__(
        self,
        data_set_name: str = PersonImageAppConfig.Default.data_set_name,
        reid_backend: str = PersonImageAppConfig.Default.reid_backend
    ) -> None:
        """初期化

        Args:
            data_set_name: データセット名
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
        """
        print("PersonImageReIDApp初期化開始...")

        self.data_set_name = data_set_name
        self.reid_backend = reid_backend

        # ログ設定
        self._setup_logging()

        # ディレクトリの設定
        self._initialize_directories()

        # コンポーネント初期化
        self._initialize_components()

        print("PersonImageReIDApp初期化完了")

    def _setup_logging(self) -> None:
        """ログ設定の初期化"""
        print("ログ設定開始...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('person_image_app.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        print("ログ設定完了")

    def _initialize_directories(self) -> None:
        """ディレクトリの設定"""
        print("ディレクトリ設定開始...")
        self.logger.info("ディレクトリの設定を開始します...")
        self.input_dir_path = Path(
            PersonImageAppConfig.Directories.input_dir_str)
        self.output_dir_path = Path(
            PersonImageAppConfig.Directories.output_dir_str)
        self.data_set_dir_path = self.input_dir_path / self.data_set_name
        self.data_set_query_dir_path = self.data_set_dir_path / "query"
        self.data_set_gallery_dir_path = self.data_set_dir_path / "gallery"
        self.logger.info("ディレクトリの設定が完了しました")
        print("ディレクトリ設定完了")

    def _initialize_components(self) -> None:
        """コンポーネントの初期化"""
        try:
            print("コンポーネント初期化開始...")
            self.logger.info("コンポーネントの初期化を開始します...")

            print("DataManager初期化...")
            self.data_manager = DataManager()
            print("DataManager初期化完了")

            print("DataSetManager初期化...")
            self.data_set_manager = DataSetManager(
                data_set_name=self.data_set_name)
            print("DataSetManager初期化完了")

            print("ReIDModelManager初期化...")
            self.reid_model_manager = ReIDModelManager(
                backend=self.reid_backend)
            print("ReIDModelManager初期化完了")

            print("PostProcessingManager初期化...")
            self.post_processing_manager = PostProcessingManager()
            print("PostProcessingManager初期化完了")

            self.logger.info("全てのコンポーネントの初期化が完了しました")
            print("コンポーネント初期化完了")

        except Exception as e:
            print(f"コンポーネント初期化エラー: {e}")
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise Exception(f"コンポーネントの初期化に失敗しました: {e}")

    def _validate_directories(self) -> None:
        """ディレクトリの存在確認と作成"""
        print("ディレクトリ検証開始...")
        self.logger.info("ディレクトリの存在確認と作成を開始します...")

        # 入力ディレクトリの確認
        if not self.input_dir_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir_path}")

        if not self.input_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.input_dir_path}")

        # データセットディレクトリの確認
        if not self.data_set_dir_path.exists():
            raise FileNotFoundError(
                f"データセットディレクトリが存在しません: {self.data_set_dir_path}")

        if not self.data_set_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_dir_path}")

        # 入力のqueryディレクトリの確認
        if not self.data_set_query_dir_path.exists():
            raise FileNotFoundError(
                f"queryディレクトリが存在しません: {self.data_set_query_dir_path}")

        if not self.data_set_query_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_query_dir_path}")

        # 入力のgalleryディレクトリの確認
        if not self.data_set_gallery_dir_path.exists():
            raise FileNotFoundError(
                f"galleryディレクトリが存在しません: {self.data_set_gallery_dir_path}")

        if not self.data_set_gallery_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_gallery_dir_path}")

        # 出力ディレクトリの作成
        try:
            self.output_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"出力ディレクトリを準備しました: {self.output_dir_path}")
        except Exception as e:
            raise OSError(f"出力ディレクトリの作成に失敗しました: {e}")

        self.logger.info("ディレクトリの存在確認と作成が完了しました")
        print("ディレクトリ検証完了")

    def _main_process(self) -> None:
        """データの読み込み"""
        print("メインプロセス開始...")
        self.logger.info("データセットから特徴量を抽出します...")

        # Gallery画像の処理
        gallery_count = 0
        for file_path in self.data_set_gallery_dir_path.glob("*"):
            if not file_path.is_file():
                continue

            try:
                process_id, camera_id, _, image = self.data_set_manager.load_image(
                    file_path)
                features = self.reid_model_manager.extract_features(
                    image, camera_id)
                self.data_manager.add_gallery(process_id, camera_id, features)
                gallery_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Gallery画像の処理でエラーが発生しました: {file_path} - {e}")
                continue

        self.logger.info(f"Gallery画像の処理が完了しました: {gallery_count}件")

        # Query画像の処理
        query_count = 0
        for file_path in self.data_set_query_dir_path.glob("*"):
            if not file_path.is_file():
                continue

            try:
                process_id, camera_id, _, image = self.data_set_manager.load_image(
                    file_path)
                features = self.reid_model_manager.extract_features(
                    image, camera_id)
                self.data_manager.add_query(process_id, camera_id, features)
                query_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Query画像の処理でエラーが発生しました: {file_path} - {e}")
                continue

        self.logger.info(f"Query画像の処理が完了しました: {query_count}件")
        self.logger.info("データセットから特徴量の抽出が完了しました")
        print("メインプロセス完了")

    def _post_process(self) -> None:
        """後処理"""
        print("後処理開始...")
        self.logger.info("評価を開始します...")

        if not self.data_manager.query_feats or not self.data_manager.gallery_feats:
            self.logger.error("特徴量が抽出されていません。評価をスキップします。")
            return

        try:
            cmc, mAP = self.post_processing_manager.evaluate(
                self.data_manager.query_feats,
                self.data_manager.gallery_feats,
                self.data_manager.query_process_ids,
                self.data_manager.gallery_process_ids,
                self.data_manager.query_camera_ids,
                self.data_manager.gallery_camera_ids
            )

            self.logger.info(f"評価結果 - CMC: {cmc}, mAP: {mAP:.4f}")

            # 結果をファイルに保存
            self._save_evaluation_results(cmc, mAP)

        except Exception as e:
            self.logger.error(f"評価処理でエラーが発生しました: {e}")
            raise

        self.logger.info("評価が完了しました")
        print("後処理完了")

    def _save_evaluation_results(self, cmc: np.ndarray, mAP: float) -> None:
        """評価結果をファイルに保存"""
        try:
            import json
            results = {
                "cmc": cmc.tolist(),
                "mAP": float(mAP),
                "dataset": self.data_set_name,
                "backend": self.reid_backend
            }

            output_file = self.output_dir_path / "evaluation_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"評価結果を保存しました: {output_file}")

        except Exception as e:
            self.logger.error(f"評価結果の保存でエラーが発生しました: {e}")

    def run(self) -> None:
        """アプリケーションの実行"""
        print("アプリケーション実行開始...")
        self.logger.info("アプリケーションの実行を開始します...")

        try:
            self._validate_directories()
            self._main_process()
            self._post_process()
            self.logger.info("アプリケーションの実行が完了しました")
            print("アプリケーション実行完了")

        except Exception as e:
            print(f"アプリケーション実行エラー: {e}")
            self.logger.error(f"アプリケーションの実行でエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    print("メイン関数開始...")
    try:
        app = PersonImageReIDApp()
        app.run()
        print("メイン関数完了")
    except Exception as e:
        print(f"メイン関数エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
