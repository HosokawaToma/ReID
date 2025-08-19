"""複数人物画像直接処理アプリケーション"""
from dataclasses import dataclass
from processors.logger import LoggerProcessor
from processors.directory.data_set import DataSetDirectoryProcessor
from processors.data_load.market1501 import Market1501DataLoadProcessor
from processors.reid.clip import ClipReIDProcessor
from data_class.person_data_set_features import PersonDataSetFeatures
from processors.post.evaluate_post import EvaluatePostProcessor
from processors.post.compute_roc_eer_f1 import ComputeRocEerF1PostProcessor

@dataclass
class Config:
    use_data_set_name: str = "market1501"


CONFIG = Config()

class PersonImageReIDApp:
    """複数人物画像直接処理アプリケーション"""
    def __init__(self):
        self.logger = LoggerProcessor.setup_logging()
        self.data_set_directory_processor = DataSetDirectoryProcessor(
            use_data_set_name=CONFIG.use_data_set_name)
        self.clip_reid_processor = ClipReIDProcessor()
        self.gallery_features = PersonDataSetFeatures()
        self.query_features = PersonDataSetFeatures()
        self.evaluate_processor = EvaluatePostProcessor()
        self.compute_roc_eer_f1_processor = ComputeRocEerF1PostProcessor()

    def run(self) -> None:
        self.logger.info("アプリケーションの実行を開始します...")

        self._process_directory()

        self._process_images()

        self._evaluate()

        self.logger.info("アプリケーションの実行が完了しました")

    def _process_directory(self) -> None:
        """ディレクトリの処理を行う"""
        self.logger.info("ディレクトリの処理を開始します...")

        self.logger.info("必要なディレクトリの確認を開始します...")
        if not self.data_set_directory_processor.validate_directories():
            self.logger.error("必要なディレクトリが存在しません")
            return
        self.logger.info("必要なディレクトリの確認が完了しました")

        self.logger.info("出力ディレクトリの作成を開始します...")
        self.data_set_directory_processor.create_output_directory()
        self.logger.info("出力ディレクトリの作成が完了しました")

        self.logger.info("ディレクトリの処理が完了しました")

    def _process_images(self) -> None:
        """画像の処理を行う"""
        self.logger.info("ギャラリー画像の処理を開始します...")
        for image_file_path in self.data_set_directory_processor.get_data_set_gallery_image_file_paths():
            person_id, camera_id, view_id, image = Market1501DataLoadProcessor.load_image(image_file_path)
            feat = self.clip_reid_processor.extract_feat(image, camera_id, view_id)
            self.gallery_features.add_feature(feat)
            self.gallery_features.add_person_id(person_id)
            self.gallery_features.add_camera_id(camera_id)
            self.gallery_features.add_view_id(view_id)
        self.logger.info("ギャラリー画像の処理が完了しました")

        self.logger.info("クエリ画像の処理を開始します...")
        for image_file_path in self.data_set_directory_processor.get_data_set_query_image_file_paths():
            person_id, camera_id, view_id, image = Market1501DataLoadProcessor.load_image(image_file_path)
            feat = self.clip_reid_processor.extract_feat(image, camera_id, view_id)
            self.query_features.add_feature(feat)
            self.query_features.add_person_id(person_id)
            self.query_features.add_camera_id(camera_id)
            self.query_features.add_view_id(view_id)
        self.logger.info("クエリ画像の処理が完了しました")

    def _evaluate(self) -> None:
        """評価を行う"""
        self.logger.info("評価を開始します...")
        gallery_features = self.gallery_features.get_features()
        query_features = self.query_features.get_features()
        gallery_person_ids = self.gallery_features.get_person_ids()
        query_person_ids = self.query_features.get_person_ids()
        self.evaluate_processor.evaluate(gallery_features, query_features)
        a1 = self.evaluate_processor.get_A1()
        a5 = self.evaluate_processor.get_A5()
        mAP = self.evaluate_processor.get_mAP()
        self.compute_roc_eer_f1_processor.compute(
            gallery_features,
            query_features,
            gallery_person_ids,
            query_person_ids
        )
        best_f1 = self.compute_roc_eer_f1_processor.get_best_f1()
        best_f1_threshold = self.compute_roc_eer_f1_processor.get_best_f1_threshold()
        self.logger.info("評価が完了しました")
        self.logger.info(f"A1: {a1}")
        self.logger.info(f"A5: {a5}")
        self.logger.info(f"mAP: {mAP}")
        self.logger.info(f"best_f1: {best_f1}")
        self.logger.info(f"best_f1_threshold: {best_f1_threshold}")


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
