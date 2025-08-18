"""複数人物画像直接処理アプリケーション"""
from dataclasses import dataclass

from managers.post_processing_manager import PostProcessingManager
from processors.reid.clip_reid_processor import ClipReIDProcessor
from managers.data_set_manager import DataSetManager
from managers.data_manager import DataManager
from managers.pre_processing_manager import PreProcessingManager
from pathlib import Path
import logging
import numpy as np
import torch

@dataclass
class PersonImageAppConfig:
    input_dir_str: str = "./resources/person_images/input"
    output_dir_str: str = "./resources/person_images/output"
    k_reciprocal_re_ranking: bool = False
    clahe: bool = False
    retinex: bool = False

    class DATA_SET:
        name: str = "market1501"

    class PostProcessing:
        max_rank: int = 50
        metric: str = "cosine"
        use_metric_cuhk03: bool = False
        use_cython: bool = False

PERSON_IMAGE_APP_CONFIG = PersonImageAppConfig()

class PersonImageReIDApp:
    """複数人物画像直接処理アプリケーション"""

    def __init__(
        self,
        input_dir_str: str = PERSON_IMAGE_APP_CONFIG.input_dir_str,
        output_dir_str: str = PERSON_IMAGE_APP_CONFIG.output_dir_str,
        k_reciprocal_re_ranking: bool = PERSON_IMAGE_APP_CONFIG.k_reciprocal_re_ranking,
        clahe: bool = PERSON_IMAGE_APP_CONFIG.clahe,
        retinex: bool = PERSON_IMAGE_APP_CONFIG.retinex,
        data_set_name: str = PERSON_IMAGE_APP_CONFIG.DATA_SET.name,
        max_rank: int = PERSON_IMAGE_APP_CONFIG.PostProcessing.max_rank,
        metric: str = PERSON_IMAGE_APP_CONFIG.PostProcessing.metric,
        use_metric_cuhk03: bool = PERSON_IMAGE_APP_CONFIG.PostProcessing.use_metric_cuhk03,
        use_cython: bool = PERSON_IMAGE_APP_CONFIG.PostProcessing.use_cython,
    ) -> None:
        """初期化

        Args:
            data_set_name: データセット名
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
        """
        self._setup_logging()
        self.data_set_name = data_set_name
        self.input_dir_str = input_dir_str
        self.output_dir_str = output_dir_str
        self.max_rank = max_rank
        self.metric = metric
        self.use_metric_cuhk03 = use_metric_cuhk03
        self.use_cython = use_cython
        self.clahe = clahe
        self.k_reciprocal_re_ranking = k_reciprocal_re_ranking
        self.retinex = retinex
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _setup_logging(self) -> None:
        """ログ設定の初期化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('person_image_app.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _set_components(self) -> None:
        """コンポーネントの初期化"""
        self.logger.info("コンポーネントの初期化を開始します...")

        self.data_manager = DataManager()

        self.data_set_manager = DataSetManager(
            data_set_name=self.data_set_name)

        self.clip_reid_processor = ClipReIDProcessor()

        self.post_processing_manager = PostProcessingManager(
            max_rank=self.max_rank,
            metric=self.metric,
            use_metric_cuhk03=self.use_metric_cuhk03,
            use_cython=self.use_cython
        )

        self.pre_processing_manager = PreProcessingManager(
            device=self.device
        )
        self.logger.info("全てのコンポーネントの初期化が完了しました")

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
        self.logger.info("アプリケーションの実行を開始します...")

        self.logger.info("処理の初期化を開始します...")
        self._set_directories()
        self._validate_directories()
        self._set_components()
        self.logger.info("処理の初期化が完了しました")

        self.logger.info("データセットから特徴量を抽出します...")
        gallery_count = 0
        for file_path in self.data_set_gallery_dir_path.glob("*"):
            if not file_path.is_file():
                continue

            try:
                person_id, camera_id, view_id, image = self.data_set_manager.load_image(
                    file_path)
                if person_id == -1:
                    continue
                if self.clahe:
                    image = self.pre_processing_manager.clahe(image)
                    file_path = self.output_dir_path / f"{person_id}_{camera_id}_{view_id}.jpg"
                    self.pre_processing_manager.np_image_output(image, str(file_path))
                if self.retinex:
                    image = self.pre_processing_manager.retinex(image)
                    file_path = self.output_dir_path / f"{person_id}_{camera_id}_{view_id}.jpg"
                    self.pre_processing_manager.np_image_output(image, str(file_path))
                features = self.clip_reid_processor.extract_features(
                    image, camera_id, view_id)
                self.data_manager.add_gallery(
                    person_id, camera_id, view_id, features)
                gallery_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Gallery画像の処理でエラーが発生しました: {file_path} - {e}")
                continue
        self.logger.info(f"Gallery画像の処理が完了しました: {gallery_count}件")

        query_count = 0
        for file_path in self.data_set_query_dir_path.glob("*"):
            if not file_path.is_file():
                continue
            try:
                person_id, camera_id, view_id, image = self.data_set_manager.load_image(
                    file_path)
                if person_id == -1:
                    continue
                if self.clahe:
                    image = self.pre_processing_manager.clahe(image)
                    file_path = self.output_dir_path / f"{person_id}_{camera_id}_{view_id}.jpg"
                    self.pre_processing_manager.np_image_output(image, str(file_path))
                if self.retinex:
                    image = self.pre_processing_manager.retinex(image)
                    file_path = self.output_dir_path / f"{person_id}_{camera_id}_{view_id}.jpg"
                    self.pre_processing_manager.np_image_output(image, str(file_path))
                features = self.clip_reid_processor.extract_features(
                    image, camera_id, view_id)
                self.data_manager.add_query(
                    person_id, camera_id, view_id, features)
                query_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Query画像の処理でエラーが発生しました: {file_path} - {e}")
                continue
        self.logger.info(f"Query画像の処理が完了しました: {query_count}件")
        self.logger.info("データセットから特徴量の抽出が完了しました")

        if self.data_manager.query_feats is None or self.data_manager.gallery_feats is None:
            raise Exception("特徴量が抽出されていません。")

        self.logger.info("後処理を開始します...")
        self.logger.info("評価を開始します...")

        dist = None
        if self.k_reciprocal_re_ranking:
            dist = self.post_processing_manager.k_reciprocal_re_ranking(
                self.data_manager.query_feats,
                self.data_manager.gallery_feats
            )

        cmc, mAP = self.post_processing_manager.evaluate(
            self.data_manager.query_feats,
            self.data_manager.gallery_feats,
            self.data_manager.query_person_ids,
            self.data_manager.gallery_person_ids,
            self.data_manager.query_camera_ids,
            self.data_manager.gallery_camera_ids,
            dist=dist
        )

        self.logger.info("評価が完了しました")
        self.logger.info(f"評価結果 - CMC: {cmc}, mAP: {mAP:.4f}")

        self._save_evaluation_results(cmc, mAP)

        self.logger.info("ROC曲線を計算します...")

        fpr, tpr, roc_thresholds, eer, eer_threshold, best_f1, best_f1_threshold = self.post_processing_manager.compute_roc_eer_f1(
            self.data_manager.query_feats,
            self.data_manager.gallery_feats,
            self.data_manager.query_person_ids,
            self.data_manager.gallery_person_ids
        )

        self.logger.info(
            f"ROC曲線 - FPR: {fpr}, TPR: {tpr}, ROC閾値: {roc_thresholds}")
        self.logger.info(f"EER: {eer}, EER閾値: {eer_threshold}")
        self.logger.info(f"F1スコア: {best_f1}, F1閾値: {best_f1_threshold}")
        self.logger.info("後処理が完了しました")

        self.logger.info("アプリケーションの実行が完了しました")


def main():
    """メイン関数"""
    app = PersonImageReIDApp()
    app.run()


if __name__ == "__main__":
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(30)
