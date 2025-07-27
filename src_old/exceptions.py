"""カスタム例外クラス"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


class ApplicationError(Exception):
    """アプリケーション基底例外クラス"""

    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details


class CameraError(ApplicationError):
    """カメラ関連のエラー"""
    pass


class CameraNotFoundError(CameraError):
    """カメラが見つからない場合のエラー"""
    pass


class CameraInitializationError(CameraError):
    """カメラの初期化に失敗した場合のエラー"""
    pass


class CameraOperationError(CameraError):
    """カメラ操作中のエラー"""
    pass


class ModelError(ApplicationError):
    """モデル関連のエラー"""
    pass


class ModelLoadError(ModelError):
    """モデルの読み込みに失敗した場合のエラー"""
    pass


class ModelInferenceError(ModelError):
    """モデル推論中のエラー"""
    pass


class ReIDError(ApplicationError):
    """ReID関連のエラー"""
    pass


class FeatureExtractionError(ReIDError):
    """特徴抽出エラー"""
    pass


class PersonIdentificationError(ReIDError):
    """人物識別エラー"""
    pass


class FrameProcessingError(ApplicationError):
    """フレーム処理関連のエラー"""
    pass


class DisplayError(ApplicationError):
    """表示関連のエラー"""
    pass


class ConfigurationError(ApplicationError):
    """設定関連のエラー"""
    pass


@dataclass
class ProcessingResult:
    """処理結果データモデル"""
    file_path: str
    success: bool
    persons_detected: int
    unique_person_ids: List[int]
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class ProcessingSummary:
    """処理サマリーデータモデル"""
    total_files: int
    successful_files: int
    failed_files: int
    total_persons_detected: int
    unique_person_count: int
    total_processing_time: float
    reid_backend: str
    results: List[ProcessingResult]
    evaluation_results: Optional[Dict[str, Any]] = None
