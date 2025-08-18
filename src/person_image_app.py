"""複数人物画像直接処理アプリケーション"""
import torch
from dataclasses import dataclass
from processors.logger import LoggerProcessor
from processors.directory.data_set import DataSetDirectoryProcessor
from processors.reid.clip import ClipReIDProcessor
from processors.post.evaluate_post import EvaluatePostProcessor
from processors.data_load.market1501 import Market1501DataLoadProcessor
from processors.post.compute_roc_eer_f1 import ComputeRocEerF1PostProcessor

@dataclass
class Config:
    class Application:
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        use_data_set_name: str = "market1501"

    class PostProcessing:
        class Evaluation:
            max_rank: int = 50
            metric: str = "cosine"
            use_metric_cuhk03: bool = False
            use_cython: bool = False
            k_reciprocal_re_ranking: bool = False

CONFIG = Config()

class PersonImageReIDApp:
    """複数人物画像直接処理アプリケーション"""
    def __init__(self):
        self.logger = LoggerProcessor().setup_logging()
        self.data_set_processor = DataSetDirectoryProcessor(
            use_data_set_name=CONFIG.Application.use_data_set_name)
        self.data_set_processor.validate_directories()
        self.data_set_processor.create_output_directory()
        self.market1501_data_load_processor = Market1501DataLoadProcessor(
            use_data_set_name=CONFIG.Application.use_data_set_name)
        self.clip_reid_processor = ClipReIDProcessor()
        self.evaluate_post_processor = EvaluatePostProcessor(
            max_rank=CONFIG.PostProcessing.Evaluation.max_rank,
            metric=CONFIG.PostProcessing.Evaluation.metric,
            use_metric_cuhk03=CONFIG.PostProcessing.Evaluation.use_metric_cuhk03,
            use_cython=CONFIG.PostProcessing.Evaluation.use_cython,
            k_reciprocal_re_ranking=CONFIG.PostProcessing.Evaluation.k_reciprocal_re_ranking
        )
        self.compute_roc_eer_f1_post_processor = ComputeRocEerF1PostProcessor()

    def run(self) -> None:
        """アプリケーションの実行"""
        self.logger.info("アプリケーションの実行を開始します...")

        self.logger.info("データセットから特徴量を抽出します...")
        self.market1501_data_load_processor.load_gallery(
            extract_feat=self.clip_reid_processor.extract_feat)
        self.market1501_data_load_processor.load_query(
            extract_feat=self.clip_reid_processor.extract_feat)
        gallery_features = self.market1501_data_load_processor.get_gallery_features()
        query_features = self.market1501_data_load_processor.get_query_features()
        self.logger.info("データセットから特徴量の抽出が完了しました")

        self.logger.info("評価を開始します...")
        self.evaluate_post_processor.evaluate(
            query_features=query_features,
            gallery_features=gallery_features,
            query_person_ids=query_features.persons_id,
            gallery_person_ids=gallery_features.persons_id,
            query_camera_ids=query_features.cameras_id,
            gallery_camera_ids=gallery_features.cameras_id
        )
        cmc = self.evaluate_post_processor.get_cmc()
        mAP = self.evaluate_post_processor.get_mAP()
        self.logger.info("評価が完了しました")
        self.logger.info(f"評価結果 - CMC: {cmc}, mAP: {mAP:.4f}")

        self.logger.info("ROC曲線を計算します...")
        self.compute_roc_eer_f1_post_processor.compute(
            query_feats=query_features.features,
            gallery_feats=gallery_features.features,
            query_person_ids=query_features.persons_id,
            gallery_person_ids=gallery_features.persons_id
        )
        best_f1 = self.compute_roc_eer_f1_post_processor.get_best_f1()
        best_f1_threshold = self.compute_roc_eer_f1_post_processor.get_best_f1_threshold()

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
