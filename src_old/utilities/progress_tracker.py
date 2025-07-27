"""進捗管理の共通ロジック"""
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class ProgressTracker:
    """進捗管理の共通ロジック"""

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self.total_items = 0
        self.current_item = 0
        self.start_time = None
        self.last_update_time = None
        self.processing_start_time = None

    def start_processing(self, total_items: int) -> None:
        """処理開始

        Args:
            total_items: 処理対象の総アイテム数
        """
        self.total_items = total_items
        self.current_item = 0
        self.start_time = time.time()
        self.processing_start_time = datetime.now()
        self.last_update_time = self.start_time

        self.logger.info(f"処理を開始します - 総アイテム数: {total_items}")
        print(f"\n=== 処理開始 ===")
        print(f"総アイテム数: {total_items}")
        print(f"開始時刻: {self.processing_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"進捗: 0/{total_items} (0.0%)")

    def update_progress(self, current: int, message: str = "") -> None:
        """進捗更新

        Args:
            current: 現在の処理済みアイテム数
            message: 追加メッセージ
        """
        if self.start_time is None:
            self.logger.warning("start_processing()が呼ばれていません")
            return

        self.current_item = current
        current_time = time.time()

        # 進捗率計算
        progress_percentage = (current / self.total_items * 100) if self.total_items > 0 else 0

        # 経過時間計算
        elapsed_time = current_time - self.start_time

        # 推定残り時間計算
        if current > 0:
            avg_time_per_item = elapsed_time / current
            remaining_items = self.total_items - current
            estimated_remaining_time = avg_time_per_item * remaining_items
            remaining_str = self._format_time(estimated_remaining_time)
        else:
            remaining_str = "計算中..."

        # 進捗表示
        progress_bar = self._create_progress_bar(progress_percentage)
        elapsed_str = self._format_time(elapsed_time)

        print(f"\r進捗: {current}/{self.total_items} ({progress_percentage:.1f}%) {progress_bar} "
              f"経過: {elapsed_str} 残り: {remaining_str}", end="", flush=True)

        if message:
            print(f" - {message}", end="", flush=True)

        # ログ出力（定期的に）
        if current_time - self.last_update_time >= 10.0:  # 10秒ごと
            self.logger.info(f"進捗更新: {current}/{self.total_items} ({progress_percentage:.1f}%)")
            self.last_update_time = current_time

    def finish_processing(self, summary: Dict[str, Any]) -> None:
        """処理完了

        Args:
            summary: 処理結果のサマリー辞書
        """
        if self.start_time is None:
            self.logger.warning("start_processing()が呼ばれていません")
            return

        end_time = time.time()
        total_time = end_time - self.start_time
        processing_end_time = datetime.now()

        print(f"\n\n=== 処理完了 ===")
        print(f"終了時刻: {processing_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総処理時間: {self._format_time(total_time)}")

        # サマリー情報を表示
        if summary:
            print(f"\n=== 処理結果サマリー ===")

            # 基本統計
            if 'total_files' in summary:
                print(f"総ファイル数: {summary['total_files']}")
            if 'successful_files' in summary:
                print(f"成功: {summary['successful_files']}")
            if 'failed_files' in summary:
                print(f"失敗: {summary['failed_files']}")

            # 人物検出統計
            if 'total_persons_detected' in summary:
                print(f"検出された人物総数: {summary['total_persons_detected']}")
            if 'unique_person_count' in summary:
                print(f"ユニーク人物数: {summary['unique_person_count']}")

            # ReIDバックエンド情報
            if 'reid_backend' in summary:
                print(f"使用ReIDバックエンド: {summary['reid_backend']}")

            # 処理速度
            if self.total_items > 0:
                avg_time_per_item = total_time / self.total_items
                print(f"平均処理時間/ファイル: {self._format_time(avg_time_per_item)}")

        # ログ出力
        self.logger.info(f"処理が完了しました - 総処理時間: {self._format_time(total_time)}")

        # 状態リセット
        self._reset()

    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """プログレスバーを作成

        Args:
            percentage: 進捗率（0-100）
            width: プログレスバーの幅

        Returns:
            プログレスバー文字列
        """
        filled_width = int(width * percentage / 100)
        bar = '█' * filled_width + '░' * (width - filled_width)
        return f"[{bar}]"

    def _format_time(self, seconds: float) -> str:
        """時間を読みやすい形式でフォーマット

        Args:
            seconds: 秒数

        Returns:
            フォーマットされた時間文字列
        """
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}時間{minutes}分{secs}秒"

    def _reset(self) -> None:
        """内部状態をリセット"""
        self.total_items = 0
        self.current_item = 0
        self.start_time = None
        self.last_update_time = None
        self.processing_start_time = None

    def get_current_progress(self) -> Dict[str, Any]:
        """現在の進捗情報を取得

        Returns:
            進捗情報の辞書
        """
        if self.start_time is None:
            return {}

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        progress_percentage = (self.current_item / self.total_items * 100) if self.total_items > 0 else 0

        return {
            'total_items': self.total_items,
            'current_item': self.current_item,
            'progress_percentage': progress_percentage,
            'elapsed_time': elapsed_time,
            'start_time': self.processing_start_time.isoformat() if self.processing_start_time else None
        }
