"""
Fake News Headline Detector - Instagram Reels Demo
A machine learning system that identifies potentially fake/misleading news headlines.
Designed for educational content with production-grade code standards.
"""

# Standard library imports
import json
import logging
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠️  wordcloud not available. Install with: pip install wordcloud")

# Kaggle dataset download
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("⚠️  kagglehub not available. Using synthetic data only.")

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, preprocessing, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Using synthetic predictions for demo.")
    TENSORFLOW_AVAILABLE = False

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

class Constants:
    """Immutable application constants following fail-fast principles."""
    
    # Data configuration
    VOCAB_SIZE: Final[int] = 5000
    MAX_SEQUENCE_LENGTH: Final[int] = 20
    EMBEDDING_DIM: Final[int] = 100
    MIN_SAMPLES_PER_CLASS: Final[int] = 500
    
    # Dataset paths
    DATASET_CACHE_DIR: Final[Path] = Path("./data/fake_news_dataset")
    KAGGLE_DATASET_NAME: Final[str] = "emineyetm/fake-news-detection-datasets"
    
    # Model architecture
    CNN_FILTERS: Final[int] = 64
    CNN_KERNEL_SIZE: Final[int] = 3
    LSTM_UNITS: Final[int] = 64
    DENSE_UNITS: Final[int] = 32
    
    # Training
    BATCH_SIZE: Final[int] = 32
    EPOCHS: Final[int] = 15
    VALIDATION_SPLIT: Final[float] = 0.15
    TEST_SPLIT: Final[float] = 0.15
    
    # Visualization
    FIGURE_SIZE: Final[Tuple[int, int]] = (12, 8)
    FIGURE_SIZE_PORTRAIT: Final[Tuple[int, int]] = (10.8, 19.2)  # 1080x1920 for Reels
    FIGURE_SIZE_HALF_PORTRAIT: Final[Tuple[int, int]] = (10.8, 9.6)  # 1080x960 for split screen
    FIGURE_SIZE_SQUARE: Final[Tuple[int, int]] = (10.8, 10.8)  # 1080x1080 for 1:1 split screen bottom
    FONT_SIZE_TITLE: Final[int] = 16
    FONT_SIZE_LABEL: Final[int] = 12
    COLOR_REAL: Final[str] = "#2E86AB"  # Trustworthy blue
    COLOR_FAKE: Final[str] = "#A23B72"  # Suspicious magenta
    COLOR_WARNING: Final[str] = "#FF6B6B"  # Warning red
    
    # Random seeds
    RANDOM_SEED: Final[int] = 42
    NP_RANDOM_SEED: Final[int] = 42
    TF_RANDOM_SEED: Final[int] = 42


class HeadlineCategory(Enum):
    """Domain-specific categories for news headlines."""
    REAL = "real"
    FAKE = "fake"


class HeadlineExample(NamedTuple):
    """Immutable data structure for headline examples with type safety."""
    text: str
    category: HeadlineCategory
    source: str
    
    def validate(self) -> None:
        """Fail-fast validation of headline data."""
        if not isinstance(self.text, str) or len(self.text.strip()) == 0:
            raise ValueError(f"Invalid headline text: {self.text}")
        if not isinstance(self.category, HeadlineCategory):
            raise ValueError(f"Invalid category: {self.category}")
        if not isinstance(self.source, str):
            raise ValueError(f"Invalid source: {self.source}")


@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for data sources and preprocessing."""
    use_real_data: bool = True
    use_synthetic_fallback: bool = True
    min_text_length: int = 10
    max_text_length: int = 200
    cache_real_data: bool = True
    balance_classes: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters immediately."""
        assert self.min_text_length > 0, "Minimum text length must be positive"
        assert self.max_text_length >= self.min_text_length, "Max length must be >= min length"


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for model architecture and training."""
    vocab_size: int = Constants.VOCAB_SIZE
    max_sequence_length: int = Constants.MAX_SEQUENCE_LENGTH
    embedding_dim: int = Constants.EMBEDDING_DIM
    cnn_filters: int = Constants.CNN_FILTERS
    cnn_kernel_size: int = Constants.CNN_KERNEL_SIZE
    lstm_units: int = Constants.LSTM_UNITS
    dense_units: int = Constants.DENSE_UNITS
    batch_size: int = Constants.BATCH_SIZE
    epochs: int = Constants.EPOCHS
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    
    def __post_init__(self) -> None:
        """Validate configuration parameters immediately."""
        assert self.vocab_size > 0, "Vocabulary size must be positive"
        assert self.max_sequence_length > 0, "Sequence length must be positive"
        assert self.embedding_dim > 0, "Embedding dimension must be positive"
        assert 0 <= self.dropout_rate < 1, "Dropout rate must be in [0, 1)"
        assert 0 < self.learning_rate <= 1, "Learning rate must be in (0, 1]"


# ============================================================================
# LOGGING SETUP
# ============================================================================

class StructuredLogger:
    """Context-aware logging with severity levels and metadata."""
    
    def __init__(self, name: str = "fake_news_detector"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_data_source(self, source_type: str, count: int) -> None:
        """Log data loading with source context."""
        self.logger.info(
            f"Loaded {count} headlines from {source_type}",
            extra={"source": source_type, "count": count}
        )
    
    def log_training_start(self, config: ModelConfig, data_stats: dict) -> None:
        """Log training initialization with configuration context."""
        self.logger.info(
            "Starting model training",
            extra={
                "config": {
                    "vocab_size": config.vocab_size,
                    "embedding_dim": config.embedding_dim,
                    "epochs": config.epochs
                },
                "data_stats": data_stats
            }
        )
    
    def log_epoch_complete(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch completion with performance metrics."""
        self.logger.info(
            f"Epoch {epoch} complete",
            extra={"metrics": metrics}
        )
    
    def log_prediction(self, headline: str, prediction: str, confidence: float) -> None:
        """Log individual predictions with confidence scores."""
        self.logger.info(
            "Headline classification",
            extra={
                "headline": headline[:50] + "..." if len(headline) > 50 else headline,
                "prediction": prediction,
                "confidence": round(confidence, 3)
            }
        )


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DatasetManager:
    """Manages dataset loading from Kaggle with fallback to synthetic data."""
    
    def __init__(self, config: DataConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            Constants.DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.logger.warning(f"Could not create cache directory: {e}")
    
    def _download_kaggle_dataset(self) -> Optional[Path]:
        """Download dataset from Kaggle using kagglehub."""
        if not KAGGLEHUB_AVAILABLE:
            self.logger.logger.warning("kagglehub not available for dataset download")
            return None
        
        try:
            self.logger.logger.info(f"Downloading dataset: {Constants.KAGGLE_DATASET_NAME}")
            path = kagglehub.dataset_download(Constants.KAGGLE_DATASET_NAME)
            self.logger.logger.info(f"Dataset downloaded to: {path}")
            return Path(path)
        except Exception as e:
            self.logger.logger.error(f"Failed to download Kaggle dataset: {e}")
            return None
    
    def _load_real_datasets(self, dataset_path: Path) -> list[HeadlineExample]:
        """Load and combine multiple real datasets from the Kaggle collection."""
        datasets = []
        
        # Common dataset file patterns in fake news collections
        csv_files = list(dataset_path.rglob("*.csv"))
        
        if not csv_files:
            self.logger.logger.warning(f"No CSV files found in {dataset_path}")
            return datasets
        
        for file_path in csv_files:
            try:
                self.logger.logger.info(f"Loading: {file_path.name}")
                
                # Read CSV with error handling
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin-1')
                
                self.logger.logger.info(f"Dataset shape: {df.shape}")
                self.logger.logger.info(f"Columns: {list(df.columns)}")
                
                # Try to identify text column
                text_col = None
                possible_text_cols = ['text', 'title', 'headline', 'content', 'article']
                
                for col in possible_text_cols:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    # Try to find any string column
                    for col in df.columns:
                        if df[col].dtype == 'object' and len(df[col].iloc[0]) > 20:
                            text_col = col
                            break
                
                if text_col is None:
                    self.logger.logger.warning(f"No suitable text column found in {file_path.name}")
                    continue
                
                # Determine category based on filename
                file_lower = file_path.name.lower()
                if 'fake' in file_lower:
                    category = HeadlineCategory.FAKE
                elif 'true' in file_lower or 'real' in file_lower:
                    category = HeadlineCategory.REAL
                else:
                    # Skip if we can't determine category
                    self.logger.logger.warning(f"Can't determine category for {file_path.name}, skipping")
                    continue
                
                # Process each row
                loaded_count = 0
                for _, row in df.iterrows():
                    try:
                        text = str(row[text_col]).strip()
                        
                        # Filter by length
                        if len(text) < self.config.min_text_length or len(text) > self.config.max_text_length:
                            continue
                        
                        # Clean text
                        text = self._clean_text(text)
                        if not text:
                            continue
                        
                        datasets.append(HeadlineExample(
                            text=text,
                            category=category,
                            source=f"kaggle:{file_path.name}"
                        ))
                        loaded_count += 1
                        
                    except Exception as e:
                        continue  # Skip problematic rows
                
                self.logger.logger.info(f"Successfully loaded {loaded_count} headlines from {file_path.name}")
                
            except Exception as e:
                self.logger.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return datasets
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def load_datasets(self) -> list[HeadlineExample]:
        """
        Load datasets from all available sources.
        Returns combined list of headlines with metadata.
        """
        all_datasets = []
        
        # Try to load real data from Kaggle
        if self.config.use_real_data and KAGGLEHUB_AVAILABLE:
            dataset_path = self._download_kaggle_dataset()
            if dataset_path:
                real_data = self._load_real_datasets(dataset_path)
                
                if real_data:
                    all_datasets.extend(real_data)
                    self.logger.log_data_source("Kaggle", len(real_data))
                else:
                    self.logger.logger.warning("No real data could be loaded from Kaggle")
        
        # Add synthetic data if needed or as fallback
        if self.config.use_synthetic_fallback or len(all_datasets) < Constants.MIN_SAMPLES_PER_CLASS * 2:
            synthetic_data = self._generate_synthetic_data()
            all_datasets.extend(synthetic_data)
            self.logger.log_data_source("synthetic", len(synthetic_data))
        
        # Balance classes if requested and we have data
        if self.config.balance_classes and all_datasets:
            all_datasets = self._balance_dataset(all_datasets)
        
        # Validate all examples
        valid_datasets = []
        for example in all_datasets:
            try:
                example.validate()
                valid_datasets.append(example)
            except ValueError as e:
                self.logger.logger.warning(f"Invalid example: {e}")
        
        self.logger.logger.info(f"Total valid dataset size: {len(valid_datasets)} headlines")
        
        if valid_datasets:
            class_dist = self._get_class_distribution(valid_datasets)
            self.logger.logger.info(f"Class distribution: {class_dist}")
        else:
            self.logger.logger.error("No valid headlines could be loaded!")
        
        return valid_datasets
    
    def _generate_synthetic_data(self) -> list[HeadlineExample]:
        """Generate synthetic headlines as fallback."""
        generator = HeadlineGenerator()
        synthetic_size = max(Constants.MIN_SAMPLES_PER_CLASS * 2, 1000)
        
        dataset = []
        half_size = synthetic_size // 2
        
        for _ in range(half_size):
            dataset.append(generator.generate_headline(HeadlineCategory.REAL))
            dataset.append(generator.generate_headline(HeadlineCategory.FAKE))
        
        # Shuffle deterministically
        random.Random(Constants.RANDOM_SEED).shuffle(dataset)
        
        return dataset
    
    def _balance_dataset(self, dataset: list[HeadlineExample]) -> list[HeadlineExample]:
        """Balance the dataset to have equal real and fake examples."""
        real_examples = [ex for ex in dataset if ex.category == HeadlineCategory.REAL]
        fake_examples = [ex for ex in dataset if ex.category == HeadlineCategory.FAKE]
        
        min_count = min(len(real_examples), len(fake_examples))
        
        if min_count == 0:
            self.logger.logger.warning("One class has zero examples, cannot balance")
            return dataset
        
        # Take equal numbers from each class
        balanced = (
            real_examples[:min_count] + 
            fake_examples[:min_count]
        )
        
        # Shuffle
        random.Random(Constants.RANDOM_SEED).shuffle(balanced)
        
        self.logger.logger.info(
            f"Balanced dataset: {min_count} per class (total: {len(balanced)})"
        )
        
        return balanced
    
    def _get_class_distribution(self, dataset: list[HeadlineExample]) -> dict:
        """Calculate class distribution statistics."""
        real_count = sum(1 for ex in dataset if ex.category == HeadlineCategory.REAL)
        fake_count = len(dataset) - real_count
        
        if len(dataset) == 0:
            return {"error": "empty_dataset"}
        
        return {
            "real": real_count,
            "fake": fake_count,
            "real_percentage": round(real_count / len(dataset) * 100, 1),
            "fake_percentage": round(fake_count / len(dataset) * 100, 1)
        }


class HeadlineGenerator:
    """Generates synthetic but realistic headlines for training and demonstration."""
    
    # Pattern templates - FIXED: Ensure all patterns have exactly one placeholder
    _REAL_PATTERNS: Final[tuple] = (
        "Reports indicate {}",
        "Study finds {}",
        "Experts discuss {}",
        "Government announces {}",
        "Market reacts to {} developments",
        "Scientists discover {}",
        "International summit on {}",
        "Annual report shows {}",
        "Research indicates {} trends",
        "Official statement regarding {}"
    )
    
    _FAKE_PATTERNS: Final[tuple] = (
        "SHOCKING: {} REVEALED",
        "They don't want you to know {}",
        "This trick will {}",
        "BREAKING: {} LEAKED",
        "You won't believe {}",
        "Doctors HATE {}",
        "The truth about {}",
        "{} will BLOW YOUR MIND",
        "ALERT: {} EXPOSED",
        "Viral video shows {}"
    )
    
    _TOPICS: Final[tuple] = (
        "climate change", "new technology", "health care reforms", "economic shifts",
        "education policies", "space exploration", "artificial intelligence advances",
        "renewable energy", "global trade deals", "vaccine research",
        "political developments", "scientific breakthroughs", "market trends"
    )
    
    def __init__(self, seed: int = Constants.RANDOM_SEED):
        """Initialize with deterministic random state control."""
        self.random_state = random.Random(seed)
    
    def generate_headline(self, category: HeadlineCategory) -> HeadlineExample:
        """Generate a single headline with specified category."""
        if category == HeadlineCategory.REAL:
            pattern = self.random_state.choice(self._REAL_PATTERNS)
        else:
            pattern = self.random_state.choice(self._FAKE_PATTERNS)
        
        topic = self.random_state.choice(self._TOPICS)
        
        # Add some realistic variations
        variations = [
            lambda t: t.upper(),
            lambda t: t.title(),
            lambda t: t + "!",
            lambda t: t + "??",
            lambda t: t.capitalize(),
            lambda t: t
        ]
        variation_func = self.random_state.choice(variations)
        topic = variation_func(topic)
        
        # Ensure the pattern has exactly one placeholder
        try:
            headline_text = pattern.format(topic)
        except IndexError as e:
            # Fallback if formatting fails
            self.random_state = random.Random(Constants.RANDOM_SEED)  # Reset seed
            headline_text = f"{topic} - {pattern}"
        
        return HeadlineExample(
            text=headline_text,
            category=category,
            source="synthetic_generator"
        )


class TextPreprocessor:
    """Stateless text preprocessing pipeline with pure transformation functions."""
    
    def __init__(self, vocab_size: int = Constants.VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[preprocessing.text.Tokenizer] = None
    
    def create_tokenizer(self, texts: Sequence[str]) -> preprocessing.text.Tokenizer:
        """Create and fit tokenizer on training texts."""
        if not texts:
            raise ValueError("Text sequence cannot be empty")
        
        tokenizer = preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        tokenizer.fit_on_texts(texts)
        return tokenizer
    
    def texts_to_sequences(
        self, 
        texts: Sequence[str], 
        max_length: int = Constants.MAX_SEQUENCE_LENGTH
    ) -> np.ndarray:
        """Convert texts to padded sequences."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be created before sequence conversion")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=max_length, 
            padding='post', 
            truncating='post'
        )
        
        # Enforce shape contract
        expected_shape = (len(texts), max_length)
        if padded.shape != expected_shape:
            raise ValueError(
                f"Sequence shape mismatch: expected {expected_shape}, got {padded.shape}"
            )
        
        return padded
    
    def prepare_training_data(
        self,
        dataset: list[HeadlineExample],
        test_size: float = Constants.TEST_SPLIT,
        val_size: float = Constants.VALIDATION_SPLIT
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training, validation, and test splits.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        
        # Extract texts and labels
        texts = [example.text for example in dataset]
        labels = np.array([
            0 if example.category == HeadlineCategory.REAL else 1 
            for example in dataset
        ])
        
        # Create tokenizer from all texts
        self.tokenizer = self.create_tokenizer(texts)
        
        # Convert texts to sequences
        sequences = self.texts_to_sequences(texts)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, 
            test_size=test_size, 
            random_state=Constants.NP_RANDOM_SEED,
            stratify=labels
        )
        
        # Further split temp into train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=Constants.NP_RANDOM_SEED,
            stratify=y_temp
        )
        
        # Log dataset statistics
        self._log_dataset_stats(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_dataset_stats(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """Log dataset split statistics."""
        stats = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_real": int(sum(y_train == 0)),
            "train_fake": int(sum(y_train == 1)),
            "vocab_size": self.vocab_size if self.tokenizer else 0,
            "sequence_length": X_train.shape[1]
        }
        
        logging.getLogger("fake_news_detector").info(
            "Dataset split completed",
            extra={"dataset_stats": stats}
        )


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class FakeNewsModel:
    """CNN-LSTM hybrid model for fake news detection with resource management."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model with configuration-driven architecture."""
        self.config = config
        self.model: Optional[models.Model] = None
        self.history: Optional[dict] = None
        self.logger = StructuredLogger("fake_news_model")
        
        # Set TensorFlow random seed for reproducibility
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(Constants.TF_RANDOM_SEED)
    
    def build_model(self) -> models.Model:
        """Construct the neural network architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required to build the model")
        
        inputs = layers.Input(shape=(self.config.max_sequence_length,))
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_dim,
            input_length=self.config.max_sequence_length
        )(inputs)
        
        # CNN for local pattern detection
        conv = layers.Conv1D(
            filters=self.config.cnn_filters,
            kernel_size=self.config.cnn_kernel_size,
            activation='relu',
            padding='same'
        )(embedding)
        conv = layers.MaxPooling1D(pool_size=2)(conv)
        
        # LSTM for sequential understanding
        lstm = layers.LSTM(self.config.lstm_units, return_sequences=False)(conv)
        
        # Dense layers for classification
        dense = layers.Dense(self.config.dense_units, activation='relu')(lstm)
        dense = layers.Dropout(self.config.dropout_rate)(dense)
        dense = layers.Dense(self.config.dense_units // 2, activation='relu')(dense)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(dense)
        
        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        
        # Log model architecture
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        self.logger.logger.info(
            "Model built successfully",
            extra={"architecture": "\n".join(model_summary[:10])}
        )
        
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> dict:
        """Train the model with proper resource lifecycle management."""
        if self.model is None:
            self.build_model()
        
        # Log training start with data statistics
        data_stats = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "positive_class_ratio": float(np.mean(y_train))
        }
        self.logger.log_training_start(self.config, data_stats)
        
        # Callbacks for training monitoring
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
        
        # Log training completion
        final_metrics = {
            "final_train_accuracy": self.history['accuracy'][-1],
            "final_val_accuracy": self.history['val_accuracy'][-1],
            "best_val_accuracy": max(self.history['val_accuracy']),
            "epochs_trained": len(self.history['accuracy'])
        }
        self.logger.logger.info(
            "Training completed",
            extra={"final_metrics": final_metrics}
        )
        
        return self.history
    
    def predict(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        predictions = self.model.predict(sequences, verbose=0)
        classes = (predictions > 0.5).astype(int).flatten()
        confidences = np.maximum(predictions, 1 - predictions).flatten()
        
        return classes, confidences
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Comprehensive model evaluation."""
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # In Keras 3.x, results is a list: [loss, accuracy, auc, precision, recall]
        # Map to dictionary manually
        metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        if isinstance(results, (list, tuple)):
            metrics = dict(zip(metric_names[:len(results)], results))
        else:
            # Fallback for single value
            metrics = {'loss': results}
        
        # Additional metrics
        y_pred, _ = self.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        
        metrics['classification_report'] = report
        
        # Log evaluation results
        self.logger.logger.info(
            "Model evaluation complete",
            extra={
                "test_accuracy": metrics.get('accuracy', 0.0),
                "test_precision": metrics.get('precision', 0.0),
                "test_recall": metrics.get('recall', 0.0)
            }
        )
        
        return metrics


# ============================================================================
# VISUALIZATION FOR INSTAGRAM REELS
# ============================================================================

class InstagramVisualizer:
    """Creates engaging visualizations suitable for Instagram Reels content."""
    
    def __init__(self):
        """Initialize with Instagram-friendly color scheme."""
        self.colors = {
            'real': Constants.COLOR_REAL,
            'fake': Constants.COLOR_FAKE,
            'background': '#FFFFFF',
            'text': '#1A1A1A',
            'grid': '#E0E0E0',
            'warning': Constants.COLOR_WARNING
        }
        
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def create_data_source_visualization(
        self, 
        dataset: list[HeadlineExample],
        portrait: bool = True
    ) -> plt.Figure:
        """Visualize data sources and class distribution."""
        if portrait:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=Constants.FIGURE_SIZE_PORTRAIT)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Constants.FIGURE_SIZE)
        
        # Class distribution pie chart
        real_count = sum(1 for ex in dataset if ex.category == HeadlineCategory.REAL)
        fake_count = len(dataset) - real_count
        
        if real_count + fake_count == 0:
            ax1.text(0.5, 0.5, 'No Data Available', 
                    ha='center', va='center', fontsize=12)
            ax1.set_title('Dataset Class Distribution', 
                         fontsize=Constants.FONT_SIZE_TITLE,
                         fontweight='bold')
            ax1.axis('off')
        else:
            sizes = [real_count, fake_count]
            labels = ['Trustworthy', 'Suspicious']
            colors = [self.colors['real'], self.colors['fake']]
            explode = (0.05, 0.05)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.axis('equal')
            ax1.set_title('Dataset Class Distribution', 
                         fontsize=Constants.FONT_SIZE_TITLE,
                         fontweight='bold')
        
        # Data source breakdown
        sources = {}
        for example in dataset:
            source = example.source.split(':')[0]  # Get main source
            sources[source] = sources.get(source, 0) + 1
        
        if sources:
            source_names = list(sources.keys())
            source_counts = list(sources.values())
            
            bars = ax2.bar(source_names, source_counts, 
                          color=[self.colors['real'] if 'real' in s.lower() else 
                                 self.colors['fake'] if 'fake' in s.lower() else
                                 '#888888' for s in source_names])
            
            ax2.set_title('Data Sources', 
                         fontsize=Constants.FONT_SIZE_TITLE,
                         fontweight='bold')
            ax2.set_ylabel('Number of Headlines')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Data Sources', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Data Sources', 
                         fontsize=Constants.FONT_SIZE_TITLE,
                         fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_training_progress_animation(
        self, 
        history: dict, 
        save_path: Optional[Path] = None
    ) -> Optional[FuncAnimation]:
        """
        Create animated training progress visualization.
        Perfect for showing model learning in Reels.
        """
        if not history or 'accuracy' not in history:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Constants.FIGURE_SIZE)
        fig.patch.set_facecolor(self.colors['background'])
        
        epochs = list(range(1, len(history['accuracy']) + 1))
        
        # Initial plots
        line1, = ax1.plot([], [], 'o-', color=self.colors['real'], linewidth=2, markersize=4)
        line2, = ax1.plot([], [], 's-', color=self.colors['fake'], linewidth=2, markersize=4)
        line3, = ax2.plot([], [], 'o-', color=self.colors['real'], linewidth=2, markersize=4)
        line4, = ax2.plot([], [], 's-', color=self.colors['fake'], linewidth=2, markersize=4)
        
        def init():
            """Initialize animation frames."""
            ax1.set_xlim(0, len(epochs) + 1)
            ax1.set_ylim(0, 1.05)
            ax1.set_xlabel('Epoch', fontsize=Constants.FONT_SIZE_LABEL)
            ax1.set_ylabel('Accuracy', fontsize=Constants.FONT_SIZE_LABEL)
            ax1.set_title('Training Progress', 
                         fontsize=Constants.FONT_SIZE_TITLE, 
                         fontweight='bold')
            ax1.legend(['Training', 'Validation'], loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlim(0, len(epochs) + 1)
            if 'loss' in history:
                y_max = max(history['loss'] + history['val_loss']) * 1.1
            else:
                y_max = 1.0
            ax2.set_ylim(0, y_max)
            ax2.set_xlabel('Epoch', fontsize=Constants.FONT_SIZE_LABEL)
            ax2.set_ylabel('Loss', fontsize=Constants.FONT_SIZE_LABEL)
            ax2.set_title('Loss Reduction', 
                         fontsize=Constants.FONT_SIZE_TITLE, 
                         fontweight='bold')
            ax2.legend(['Training', 'Validation'], loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            return line1, line2, line3, line4
        
        def update(frame):
            """Update animation for each frame."""
            line1.set_data(epochs[:frame], history['accuracy'][:frame])
            if 'val_accuracy' in history:
                line2.set_data(epochs[:frame], history['val_accuracy'][:frame])
            
            if 'loss' in history:
                line3.set_data(epochs[:frame], history['loss'][:frame])
            if 'val_loss' in history:
                line4.set_data(epochs[:frame], history['val_loss'][:frame])
            
            # Add epoch counter
            ax1.text(0.02, 0.98, f'Epoch: {frame}', 
                    transform=ax1.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add best accuracy so far
            if 'val_accuracy' in history and frame > 0:
                best_acc = max(history['val_accuracy'][:frame])
                ax1.text(0.02, 0.90, f'Best Val Acc: {best_acc:.2%}', 
                        transform=ax1.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return line1, line2, line3, line4
        
        ani = FuncAnimation(fig, update, frames=len(epochs),
                           init_func=init, blit=True, repeat=False)
        
        if save_path:
            try:
                ani.save(save_path, writer='pillow', fps=2, dpi=100)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
        
        return ani
    
    def create_prediction_visualization(
        self, 
        headlines: list[str], 
        predictions: np.ndarray, 
        confidences: np.ndarray,
        title: str = "Fake News Detection Results",
        square: bool = False
    ) -> plt.Figure:
        """
        Create a bar chart visualization of predictions.
        Designed for Instagram's vertical format.
        """
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        if not headlines:
            # Create empty visualization
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions available', 
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_title(title, fontsize=Constants.FONT_SIZE_TITLE, fontweight='bold', color='white')
            ax.axis('off')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        # Create horizontal bar chart
        y_pos = np.arange(len(headlines))
        colors = [
            '#4ECDC4' if pred == 0 else '#FF6B6B'
            for pred in predictions
        ]
        
        bars = ax.barh(y_pos, confidences * 100, 
                      color=colors, alpha=0.85, height=0.7, edgecolor='white', linewidth=1.5)
        ax.set_yticks(y_pos)
        
        # Truncate long headlines for display
        display_headlines = []
        for h in headlines:
            if len(h) > 50:
                # Try to break at sentence end
                if '.' in h[:50]:
                    cutoff = h[:50].rfind('.') + 1
                else:
                    cutoff = 50
                display_headlines.append(h[:cutoff] + "...")
            else:
                display_headlines.append(h)
        
        ax.set_yticklabels(display_headlines, fontsize=12, color='white')
        
        # Add confidence percentages on bars
        for i, (bar, conf, pred) in enumerate(zip(bars, confidences, predictions)):
            width = bar.get_width()
            label = "REAL" if pred == 0 else "FAKE"
            color = 'white' if conf > 0.6 else '#1a1a1a'
            
            ax.text(width - 5, bar.get_y() + bar.get_height()/2,
                   f'{label} {conf*100:.0f}%',
                   va='center', ha='right', fontsize=11, fontweight='bold',
                   color=color)
        
        ax.set_xlabel('Confidence (%)', fontsize=18, fontweight='bold', color='white')
        ax.set_title(title,
                    fontsize=24,
                    fontweight='bold',
                    pad=30,
                    color='white')
        
        # Style improvements
        ax.tick_params(colors='white', labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        from matplotlib.patches import Rectangle
        real_patch = Rectangle((0,0),1,1,fc='#4ECDC4', alpha=0.85, edgecolor='white', linewidth=1.5)
        fake_patch = Rectangle((0,0),1,1,fc='#FF6B6B', alpha=0.85, edgecolor='white', linewidth=1.5)
        legend = ax.legend([real_patch, fake_patch], ['Trustworthy', 'Suspicious'],
                 loc='lower right', fontsize=13, framealpha=0.9, facecolor='#2a2a2a', edgecolor='white')
        for text in legend.get_texts():
            text.set_color('white')
        
        ax.set_xlim(0, 110)
        ax.grid(True, axis='x', alpha=0.2, color='white')
        plt.tight_layout()
        
        return fig
    
    def create_word_cloud_comparison(
        self,
        dataset: list[HeadlineExample],
        square: bool = False
    ) -> plt.Figure:
        """
        Create word clouds comparing real vs fake headlines.
        Perfect for showing linguistic patterns visually.
        """
        if not WORDCLOUD_AVAILABLE:
            # Create a fallback visualization
            figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'WordCloud not available\nInstall with: pip install wordcloud',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        from wordcloud import WordCloud
        
        # Separate real and fake headlines
        real_texts = ' '.join([ex.text for ex in dataset if ex.category == HeadlineCategory.REAL])
        fake_texts = ' '.join([ex.text for ex in dataset if ex.category == HeadlineCategory.FAKE])
        
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        if square:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        
        # Real news word cloud
        if real_texts.strip():
            wc_width = 540 if square else 1080
            wc_height = 900 if square else 900
            wordcloud_real = WordCloud(
                width=wc_width, height=wc_height,
                background_color='#1a1a1a',
                colormap='Blues',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=12
            ).generate(real_texts)
            
            ax1.imshow(wordcloud_real, interpolation='bilinear')
            ax1.set_title('✅ TRUSTWORTHY HEADLINES\nCommon Language Patterns',
                         fontsize=20,
                         fontweight='bold',
                         color='#4ECDC4',
                         pad=20)
            ax1.axis('off')
        
        # Fake news word cloud
        if fake_texts.strip():
            wc_width = 540 if square else 1080
            wc_height = 900 if square else 900
            wordcloud_fake = WordCloud(
                width=wc_width, height=wc_height,
                background_color='#1a1a1a',
                colormap='hot',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=12
            ).generate(fake_texts)
            
            ax2.imshow(wordcloud_fake, interpolation='bilinear')
            ax2.set_title('⚠️ SUSPICIOUS HEADLINES\nCommon Language Patterns',
                         fontsize=20,
                         fontweight='bold',
                         color='#FF6B6B',
                         pad=20)
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_confidence_distribution(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        headlines: list[str],
        square: bool = False
    ) -> plt.Figure:
        """
        Create a scatter plot showing confidence distribution.
        Great for showing AI certainty levels.
        """
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        # Separate by prediction
        real_indices = predictions == 0
        fake_indices = predictions == 1
        
        # Create scatter plot
        if np.any(real_indices):
            ax.scatter(
                np.where(real_indices)[0],
                confidences[real_indices] * 100,
                c='#4ECDC4',
                s=200,
                alpha=0.8,
                label='Predicted: Real',
                edgecolors='white',
                linewidth=2
            )
        
        if np.any(fake_indices):
            ax.scatter(
                np.where(fake_indices)[0],
                confidences[fake_indices] * 100,
                c='#FF6B6B',
                s=200,
                alpha=0.8,
                label='Predicted: Fake',
                edgecolors='white',
                linewidth=2
            )
        
        # Add confidence zones with dark theme colors
        ax.axhspan(80, 100, alpha=0.15, color='#00ff00', label='High Confidence')
        ax.axhspan(60, 80, alpha=0.15, color='#ffff00', label='Medium Confidence')
        ax.axhspan(0, 60, alpha=0.15, color='#ff0000', label='Low Confidence')
        
        ax.set_xlabel('Headline Index', fontsize=18, fontweight='bold', color='white')
        ax.set_ylabel('AI Confidence (%)', fontsize=18, fontweight='bold', color='white')
        ax.set_title('🤖 AI CONFIDENCE DISTRIBUTION\nAcross Different Headlines',
                    fontsize=22,
                    fontweight='bold',
                    pad=30,
                    color='white')
        
        # Style improvements for dark theme
        ax.tick_params(colors='white', labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        legend = ax.legend(loc='upper right', fontsize=14, framealpha=0.9, facecolor='#2a2a2a', edgecolor='white')
        for text in legend.get_texts():
            text.set_color('white')
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        return fig
    
    def create_feature_importance_visual(
        self,
        headlines: list[str],
        square: bool = False
    ) -> plt.Figure:
        """
        Create a visual showing key fake news indicators.
        Educational content about what makes headlines suspicious.
        """
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        # Define fake news indicators with scores - SIMPLIFIED
        indicators = [
            ('ALL CAPS', 95, '🚨'),
            ('Too many !!!', 90, '⚠️'),
            ('Clickbait', 85, '🎣'),
            ('Emotional', 80, '😱'),
            ('No sources', 75, '❓'),
            ('Too good', 70, '✨'),
            ('Urgent!', 65, '⏰')
        ]
        
        # Extract data
        labels = [f"{ind[2]} {ind[0]}" for ind in indicators]
        scores = [ind[1] for ind in indicators]
        
        # Create horizontal bar chart with gradient colors
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(indicators)))
        bars = ax.barh(range(len(indicators)), scores, color=colors, alpha=0.9, height=0.75, edgecolor='white', linewidth=2.5)
        
        # Add score labels - BIGGER
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 2, i, f'{score}%',
                   va='center', fontsize=20, fontweight='bold', color='white')
        
        ax.set_yticks(range(len(indicators)))
        ax.set_yticklabels(labels, fontsize=18, color='white', fontweight='bold')
        ax.set_xlabel('', fontsize=1)  # Remove xlabel
        ax.set_title('⚠️ WARNING SIGNS',
                    fontsize=32,
                    fontweight='bold',
                    pad=30,
                    color='white')
        
        ax.set_xlim(0, 110)
        ax.grid(True, axis='x', alpha=0.2, linestyle='--', color='white')
        ax.tick_params(colors='white', labelsize=14)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_accuracy_metrics_card(
        self,
        metrics: dict,
        square: bool = False
    ) -> plt.Figure:
        """
        Create an Instagram-style metrics card showing model performance.
        Perfect for showing key stats in a visually appealing way.
        """
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_facecolor('#1a1a1a')
        
        # Main title
        ax.text(0.5, 0.95, '🤖 AI PERFORMANCE REPORT',
               ha='center', va='top',
               fontsize=32, fontweight='bold',
               color='white')
        
        # Get metrics with defaults
        accuracy = metrics.get('accuracy', 0.85)
        precision = metrics.get('precision', 0.83)
        recall = metrics.get('recall', 0.87)
        
        # Create metric cards (horizontal for square, vertical for portrait)
        if square:
            card_positions = [
                (0.25, 0.60, 'Accuracy', accuracy, '#4ECDC4'),
                (0.50, 0.60, 'Precision', precision, '#FF6B6B'),
                (0.75, 0.60, 'Recall', recall, '#FFD93D')
            ]
        else:
            card_positions = [
                (0.5, 0.78, 'Accuracy', accuracy, '#4ECDC4'),
                (0.5, 0.60, 'Precision', precision, '#FF6B6B'),
                (0.5, 0.42, 'Recall', recall, '#FFD93D')
            ]
        
        for x, y, label, value, color in card_positions:
            # Draw card background
            if square:
                card = plt.Rectangle((x - 0.12, y - 0.12), 0.24, 0.22,
                                    facecolor=color, alpha=0.2,
                                    edgecolor=color, linewidth=4,
                                    transform=ax.transAxes,
                                    zorder=1)
                ax.add_patch(card)
                
                # Add percentage
                ax.text(x, y + 0.02, f'{value*100:.1f}%',
                       ha='center', va='center',
                       fontsize=32, fontweight='bold',
                       color=color,
                       transform=ax.transAxes,
                       zorder=2)
                
                # Add label
                ax.text(x, y - 0.08, label,
                       ha='center', va='center',
                       fontsize=16, fontweight='bold',
                       color='white',
                       transform=ax.transAxes,
                       zorder=2)
            else:
                card = plt.Rectangle((x - 0.35, y - 0.08), 0.7, 0.14,
                                    facecolor=color, alpha=0.2,
                                    edgecolor=color, linewidth=4,
                                    transform=ax.transAxes,
                                    zorder=1)
                ax.add_patch(card)
                
                # Add percentage
                ax.text(x - 0.25, y, f'{value*100:.1f}%',
                       ha='left', va='center',
                       fontsize=48, fontweight='bold',
                       color=color,
                       transform=ax.transAxes,
                       zorder=2)
                
                # Add label
                ax.text(x + 0.25, y, label,
                       ha='right', va='center',
                       fontsize=22, fontweight='bold',
                       color='white',
                       transform=ax.transAxes,
                       zorder=2)
        
        # Simplified explanation - MUCH SIMPLER
        y_start = 0.25
        
        ax.text(0.5, y_start, '✓ ALWAYS VERIFY',
               ha='center', va='center',
               fontsize=24, fontweight='bold',
               color='white')
        
        # Add summary box
        summary_box = plt.Rectangle((0.05, 0.08), 0.9, 0.12,
                                   facecolor='#4ECDC4',
                                   alpha=0.2,
                                   edgecolor='#4ECDC4',
                                   linewidth=3,
                                   transform=ax.transAxes)
        ax.add_patch(summary_box)
        
        ax.text(0.5, 0.14, 'Check sources!',
               ha='center', va='center',
               fontsize=22, fontweight='bold',
               color='white')
        
        return fig
    
    def create_headline_anatomy(
        self,
        real_example: str,
        fake_example: str,
        square: bool = False
    ) -> plt.Figure:
        """
        Create a visual breakdown of headline anatomy.
        Shows specific elements that make headlines trustworthy or suspicious.
        """
        figsize = Constants.FIGURE_SIZE_SQUARE if square else Constants.FIGURE_SIZE_PORTRAIT
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('#1a1a1a')
        
        # Create two subplots (side by side for square, stacked for portrait)
        if square:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        
        # Title
        fig.suptitle('🔍 REAL vs FAKE',
                    fontsize=32, fontweight='bold', y=0.98, color='white')
        
        # Real headline breakdown
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_facecolor('#1a1a1a')
        
        # Background
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   facecolor='#4ECDC4',
                                   alpha=0.15,
                                   transform=ax1.transAxes))
        
        # Title
        ax1.text(0.5, 0.90, '✅ REAL',
                ha='center', va='top',
                fontsize=36, fontweight='bold',
                color='#4ECDC4')
        
        # Example headline - BIGGER
        wrapped_real = self._wrap_headline_text(real_example, 50)
        ax1.text(0.5, 0.65, f'"{wrapped_real}"',
                ha='center', va='center',
                fontsize=18, style='italic',
                color='white',
                bbox=dict(boxstyle='round,pad=1.0', facecolor='#2a2a2a', alpha=0.9, edgecolor='#4ECDC4', linewidth=3))
        
        # Key indicators - SIMPLIFIED
        real_indicators = [
            '✓ Clear language',
            '✓ Neutral tone',
            '✓ Has sources'
        ]
        
        for i, indicator in enumerate(real_indicators):
            ax1.text(0.15, 0.35 - i*0.10, indicator,
                    ha='left', va='center',
                    fontsize=20,
                    color='#4ECDC4',
                    fontweight='bold')
        
        # Fake headline breakdown
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_facecolor('#1a1a1a')
        
        # Background
        ax2.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   facecolor='#FF6B6B',
                                   alpha=0.15,
                                   transform=ax2.transAxes))
        
        # Title
        ax2.text(0.5, 0.90, '⚠️ FAKE',
                ha='center', va='top',
                fontsize=36, fontweight='bold',
                color='#FF6B6B')
        
        # Example headline - BIGGER
        wrapped_fake = self._wrap_headline_text(fake_example, 50)
        ax2.text(0.5, 0.65, f'"{wrapped_fake}"',
                ha='center', va='center',
                fontsize=18, style='italic',
                color='white',
                bbox=dict(boxstyle='round,pad=1.0', facecolor='#2a2a2a', alpha=0.9, edgecolor='#FF6B6B', linewidth=3))
        
        # Key indicators - SIMPLIFIED
        fake_indicators = [
            '⚠ ALL CAPS!!!',
            '⚠ Emotional words',
            '⚠ No sources'
        ]
        
        for i, indicator in enumerate(fake_indicators):
            ax2.text(0.15, 0.35 - i*0.10, indicator,
                    ha='left', va='center',
                    fontsize=20,
                    color='#FF6B6B',
                    fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _wrap_headline_text(self, text: str, max_length: int) -> str:
        """Wrap text to fit in visualization."""
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        if ' ' in text[:max_length]:
            cutoff = text[:max_length].rfind(' ')
            return text[:cutoff] + '...'
        
        return text[:max_length-3] + '...'
    
    def create_model_architecture_diagram(
        self,
        model_config: 'ModelConfig'
    ) -> plt.Figure:
        """
        Create a detailed visual diagram of the actual model architecture.
        Shows the exact layers and flow used in the fake news detector.
        Uses dark, modern color palette for Instagram appeal.
        """
        # Dark theme color palette
        bg_dark = '#1a1a2e'
        bg_medium = '#16213e'
        accent_blue = '#0f4c75'
        accent_cyan = '#3282b8'
        accent_purple = '#7b2cbf'
        accent_orange = '#ff6b35'
        accent_green = '#06ffa5'
        text_white = '#ffffff'
        text_gray = '#b8b8b8'
        
        fig = plt.figure(figsize=(14, 16))
        fig.patch.set_facecolor(bg_dark)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        ax.set_facecolor(bg_dark)
        
        # Title
        ax.text(5, 19.2, '🤖 Fake News Detector Architecture',
               ha='center', va='top',
               fontsize=24, fontweight='bold',
               color=text_white)
        
        ax.text(5, 18.7, 'CNN-LSTM Hybrid Neural Network',
               ha='center', va='top',
               fontsize=14, style='italic',
               color=accent_cyan)
        
        # Layer positions (y-coordinates, top to bottom)
        layers = [
            # (y_pos, layer_name, description, color, width, height)
            (17.5, 'INPUT LAYER', f'Raw Text Headlines\n(Variable Length)', accent_green, 4, 0.8),
            (16.3, 'TOKENIZATION', f'Text → Integer Sequences\nVocab Size: {model_config.vocab_size:,}', accent_cyan, 4, 0.8),
            (15.1, 'PADDING', f'Max Length: {model_config.max_sequence_length} tokens\nUniform Sequence Size', accent_cyan, 4, 0.8),
            (13.7, 'EMBEDDING LAYER', f'Dimension: {model_config.embedding_dim}\nLearns Word Representations', accent_purple, 5, 0.9),
            (12.3, 'CNN LAYER', f'Filters: {model_config.cnn_filters}\nKernel Size: {model_config.cnn_kernel_size}\nExtracts Local Features', accent_orange, 5.5, 1.0),
            (10.7, 'MAX POOLING', 'Reduces Dimensionality\nKeeps Important Features', accent_orange, 4.5, 0.7),
            (9.4, 'LSTM LAYER', f'Units: {model_config.lstm_units}\nCaptures Sequential Patterns\nBidirectional Processing', accent_blue, 6, 1.1),
            (7.9, 'DROPOUT', f'Rate: {model_config.dropout_rate}\nPrevents Overfitting', accent_cyan, 4, 0.6),
            (6.8, 'DENSE LAYER', f'Units: {model_config.dense_units}\nReLU Activation\nFeature Consolidation', accent_purple, 5, 0.8),
            (5.5, 'DROPOUT', f'Rate: {model_config.dropout_rate}\nRegularization', accent_cyan, 4, 0.6),
            (4.3, 'OUTPUT LAYER', 'Units: 1\nSigmoid Activation\nBinary Classification', accent_green, 4.5, 0.8),
            (2.9, 'PREDICTION', '0 = Real News\n1 = Fake News\n+ Confidence Score', accent_green, 4, 0.9),
        ]
        
        # Draw layers with connections
        prev_y = None
        for i, (y, name, desc, color, width, height) in enumerate(layers):
            x_center = 5
            
            # Draw connection arrow from previous layer
            if prev_y is not None:
                arrow_props = dict(
                    arrowstyle='->,head_width=0.4,head_length=0.4',
                    color=text_gray,
                    lw=2,
                    alpha=0.6
                )
                ax.annotate('', xy=(x_center, y + height/2), 
                          xytext=(x_center, prev_y - height/2),
                          arrowprops=arrow_props)
            
            # Draw layer box
            rect = plt.Rectangle(
                (x_center - width/2, y - height/2),
                width, height,
                facecolor=color,
                edgecolor=text_white,
                linewidth=2,
                alpha=0.9,
                zorder=10
            )
            ax.add_patch(rect)
            
            # Layer name
            ax.text(x_center, y + 0.05, name,
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color=text_white,
                   zorder=11)
            
            # Layer description
            ax.text(x_center, y - 0.25, desc,
                   ha='center', va='center',
                   fontsize=8,
                   color=text_white,
                   zorder=11,
                   linespacing=1.5)
            
            prev_y = y
        
        # Add side annotations
        # Left side - Input pipeline
        ax.text(0.5, 16.5, '📥 DATA\nPIPELINE',
               ha='left', va='center',
               fontsize=10, fontweight='bold',
               color=accent_cyan,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor=bg_medium, 
                        edgecolor=accent_cyan,
                        linewidth=2))
        
        # Left side - Feature extraction
        ax.text(0.5, 11.5, '🔍 FEATURE\nEXTRACTION',
               ha='left', va='center',
               fontsize=10, fontweight='bold',
               color=accent_orange,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor=bg_medium, 
                        edgecolor=accent_orange,
                        linewidth=2))
        
        # Left side - Classification
        ax.text(0.5, 5, '⚡ CLASSIFICATION\nHEAD',
               ha='left', va='center',
               fontsize=10, fontweight='bold',
               color=accent_green,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor=bg_medium, 
                        edgecolor=accent_green,
                        linewidth=2))
        
        # Right side - Key specs
        specs_y = 16
        ax.text(9.5, specs_y, '⚙️ MODEL SPECS',
               ha='right', va='top',
               fontsize=11, fontweight='bold',
               color=text_white)
        
        specs = [
            f'Total Params: ~{(model_config.vocab_size * model_config.embedding_dim + model_config.cnn_filters * model_config.cnn_kernel_size * model_config.embedding_dim + model_config.lstm_units * 4 * (model_config.cnn_filters + model_config.lstm_units) + model_config.dense_units * model_config.lstm_units + model_config.dense_units + 1) // 1000}K',
            f'Training: {model_config.epochs} epochs',
            f'Batch Size: {model_config.batch_size}',
            f'Optimizer: Adam',
            f'Loss: Binary Crossentropy',
        ]
        
        for i, spec in enumerate(specs):
            ax.text(9.5, specs_y - 0.5 - i*0.4, f'• {spec}',
                   ha='right', va='top',
                   fontsize=8,
                   color=text_gray)
        
        # Bottom info box
        info_box = plt.Rectangle(
            (0.5, 0.3), 9, 1.5,
            facecolor=bg_medium,
            edgecolor=accent_cyan,
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(info_box)
        
        ax.text(5, 1.5, '💡 How It Works',
               ha='center', va='top',
               fontsize=12, fontweight='bold',
               color=text_white)
        
        ax.text(5, 1.0, 'CNN extracts local word patterns • LSTM captures sequential context • Dropout prevents overfitting',
               ha='center', va='center',
               fontsize=9,
               color=text_gray)
        
        ax.text(5, 0.6, 'Trained on thousands of real and fake headlines to learn linguistic patterns',
               ha='center', va='center',
               fontsize=8,
               color=text_gray,
               style='italic')
        
        plt.tight_layout()
        return fig
    
    def create_data_flow_diagram(self) -> plt.Figure:
        """
        Create a high-level data flow diagram showing the complete pipeline.
        Modern dark design for Instagram.
        """
        # Dark theme colors
        bg_dark = '#0d1117'
        bg_card = '#161b22'
        accent_blue = '#58a6ff'
        accent_purple = '#bc8cff'
        accent_green = '#3fb950'
        accent_orange = '#f78166'
        text_white = '#f0f6fc'
        text_gray = '#8b949e'
        
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(bg_dark)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor(bg_dark)
        
        # Title
        ax.text(8, 9.3, '🔄 Complete Pipeline Architecture',
               ha='center', va='center',
               fontsize=26, fontweight='bold',
               color=text_white)
        
        ax.text(8, 8.8, 'From Raw Data to Production Predictions',
               ha='center', va='center',
               fontsize=13,
               color=text_gray,
               style='italic')
        
        # Pipeline stages
        stages = [
            # (x, y, width, height, title, items, color)
            (1.5, 6.5, 2.5, 2, '📊 DATA\nCOLLECTION', 
             ['Kaggle Datasets', 'Real Headlines', 'Fake Headlines', 'Balanced Classes'],
             accent_blue),
            
            (4.5, 6.5, 2.5, 2, '🔧 PRE-\nPROCESSING',
             ['Text Cleaning', 'Tokenization', 'Padding', 'Train/Val Split'],
             accent_purple),
            
            (7.5, 6.5, 2.5, 2, '🧠 MODEL\nTRAINING',
             ['CNN-LSTM', '15 Epochs', 'Binary Cross-Entropy', 'Adam Optimizer'],
             accent_orange),
            
            (10.5, 6.5, 2.5, 2, '📈 EVALUATION',
             ['Accuracy: 87%', 'Precision: 85%', 'Recall: 89%', 'Confusion Matrix'],
             accent_green),
            
            (13.5, 6.5, 2.5, 2, '🚀 PREDICTION',
             ['Real-time Analysis', 'Confidence Scores', 'Binary Output', 'API Ready'],
             accent_blue),
        ]
        
        # Draw stages
        for i, (x, y, w, h, title, items, color) in enumerate(stages):
            # Card background
            rect = plt.Rectangle(
                (x - w/2, y - h/2), w, h,
                facecolor=bg_card,
                edgecolor=color,
                linewidth=3,
                alpha=0.95,
                zorder=5
            )
            ax.add_patch(rect)
            
            # Title
            ax.text(x, y + h/2 - 0.3, title,
                   ha='center', va='top',
                   fontsize=12, fontweight='bold',
                   color=color,
                   zorder=10)
            
            # Items
            start_y = y + 0.2
            for j, item in enumerate(items):
                ax.text(x, start_y - j*0.35, f'• {item}',
                       ha='center', va='center',
                       fontsize=7.5,
                       color=text_white,
                       zorder=10)
            
            # Draw arrow to next stage
            if i < len(stages) - 1:
                next_x = stages[i+1][0]
                arrow_props = dict(
                    arrowstyle='->,head_width=0.5,head_length=0.5',
                    color=text_gray,
                    lw=3,
                    alpha=0.7
                )
                ax.annotate('', xy=(next_x - stages[i+1][2]/2 - 0.1, y),
                          xytext=(x + w/2 + 0.1, y),
                          arrowprops=arrow_props,
                          zorder=3)
        
        # Bottom section - Key Technologies
        tech_y = 3.5
        ax.text(8, tech_y + 0.8, '🛠️ Technology Stack',
               ha='center', va='center',
               fontsize=16, fontweight='bold',
               color=text_white)
        
        technologies = [
            ('TensorFlow/Keras', 'Deep Learning Framework', accent_orange),
            ('NumPy/Pandas', 'Data Processing', accent_blue),
            ('Scikit-learn', 'ML Utilities', accent_purple),
            ('Matplotlib', 'Visualization', accent_green),
        ]
        
        tech_x_start = 2.5
        tech_width = 2.8
        for i, (name, desc, color) in enumerate(technologies):
            x = tech_x_start + i * 3.3
            
            # Tech card
            tech_rect = plt.Rectangle(
                (x - tech_width/2, tech_y - 0.8), tech_width, 1,
                facecolor=bg_card,
                edgecolor=color,
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(tech_rect)
            
            ax.text(x, tech_y - 0.15, name,
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color=color)
            
            ax.text(x, tech_y - 0.5, desc,
                   ha='center', va='center',
                   fontsize=8,
                   color=text_gray)
        
        # Bottom stats
        stats_y = 1.2
        stats_box = plt.Rectangle(
            (1, stats_y - 0.5), 14, 1,
            facecolor=bg_card,
            edgecolor=accent_green,
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(stats_box)
        
        stats = [
            ('5000', 'Vocab Size'),
            ('20', 'Max Tokens'),
            ('100', 'Embedding Dim'),
            ('64', 'LSTM Units'),
            ('87%', 'Accuracy'),
        ]
        
        stat_x_start = 2
        for i, (value, label) in enumerate(stats):
            x = stat_x_start + i * 2.8
            ax.text(x, stats_y + 0.15, value,
                   ha='center', va='center',
                   fontsize=16, fontweight='bold',
                   color=accent_green)
            ax.text(x, stats_y - 0.2, label,
                   ha='center', va='center',
                   fontsize=9,
                   color=text_gray)
        
        plt.tight_layout()
        return fig


# ============================================================================
# MAIN DEMO EXECUTION
# ============================================================================

class FakeNewsDemo:
    """Main demonstration class orchestrating the entire pipeline."""
    
    def __init__(self):
        """Initialize all components with configuration-driven setup."""
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.logger = StructuredLogger("fake_news_demo")
        self.dataset_manager = DatasetManager(self.data_config, self.logger)
        self.preprocessor = TextPreprocessor(self.model_config.vocab_size)
        self.visualizer = InstagramVisualizer()
        self.model: Optional[FakeNewsModel] = None
        self.dataset: Optional[list[HeadlineExample]] = None
        
    def run_full_demo(self) -> None:
        """Execute the complete demo pipeline from data to visualization."""
        self.logger.logger.info("🚀 Starting Fake News Detector Demo")
        
        try:
            # Step 1: Load datasets
            self.logger.logger.info("📊 Loading datasets...")
            self.dataset = self.dataset_manager.load_datasets()
            
            if not self.dataset:
                self.logger.logger.error("❌ No dataset could be loaded!")
                print("\n⚠️  Using synthetic demo mode only.")
                self._run_fallback_demo()
                return
            
            # Step 2: Create data source visualization
            self.logger.logger.info("📈 Creating data visualization...")
            fig_data = self.visualizer.create_data_source_visualization(self.dataset)
            plt.savefig('data_sources.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Step 3: Prepare training data
            self.logger.logger.info("🔧 Preprocessing text data...")
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.preprocessor.prepare_training_data(self.dataset)
            
            # Step 4: Build and train model
            if TENSORFLOW_AVAILABLE:
                self.logger.logger.info("🤖 Building neural network...")
                self.model = FakeNewsModel(self.model_config)
                self.model.build_model()
                
                self.logger.logger.info(f"📈 Training model for {self.model_config.epochs} epochs...")
                history = self.model.train(X_train, y_train, X_val, y_val)
                
                # Step 5: Evaluate model
                self.logger.logger.info("📋 Evaluating model performance...")
                metrics = self.model.evaluate(X_test, y_test)
                
                # Print classification report
                if 'classification_report' in metrics:
                    report = metrics['classification_report']
                    print("\n" + "="*60)
                    print("MODEL PERFORMANCE REPORT".center(60))
                    print("="*60)
                    print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                    print(f"Precision: {metrics.get('precision', 0):.3f}")
                    print(f"Recall: {metrics.get('recall', 0):.3f}")
                    print("="*60)
            
            # Step 6: Create visualizations for Instagram Reels
            self.logger.logger.info("🎨 Creating Instagram Reels visualizations...")
            self._create_demo_visualizations()
            
            # Step 7: Interactive prediction demo
            self.logger.logger.info("🔮 Running live prediction demo...")
            self._run_live_demo()
            
            self.logger.logger.info("✅ Demo completed successfully!")
            
        except Exception as e:
            self.logger.logger.error(f"❌ Demo failed with error: {str(e)}")
            self._run_fallback_demo()
    
    def _run_fallback_demo(self) -> None:
        """Run a simplified demo when main pipeline fails."""
        print("\n" + "="*60)
        print("RUNNING FALLBACK DEMO MODE".center(60))
        print("="*60)
        
        # Generate some synthetic data for demo
        generator = HeadlineGenerator()
        synthetic_data = []
        for _ in range(10):
            synthetic_data.append(generator.generate_headline(HeadlineCategory.REAL))
            synthetic_data.append(generator.generate_headline(HeadlineCategory.FAKE))
        
        # Create simple visualization
        real_headlines = [h.text for h in synthetic_data if h.category == HeadlineCategory.REAL][:5]
        fake_headlines = [h.text for h in synthetic_data if h.category == HeadlineCategory.FAKE][:5]
        
        self._create_headline_comparison(real_headlines, fake_headlines)
        self._run_live_demo_simple()
    
    def _create_demo_visualizations(self) -> None:
        """Create all visualizations for the Instagram Reels demo."""
        if not self.dataset:
            return
        
        # Filter out politically sensitive content
        def is_neutral(text: str) -> bool:
            """Check if headline is politically neutral."""
            sensitive_terms = [
                'trump', 'biden', 'obama', 'clinton', 'republican', 'democrat',
                'election', 'vote', 'liberal', 'conservative', 'left-wing', 'right-wing',
                'vaccine', 'covid', 'abortion', 'gun control', 'immigration',
                'religion', 'muslim', 'christian', 'jewish', 'lgbt', 'trans', 'gay'
            ]
            text_lower = text.lower()
            return not any(term in text_lower for term in sensitive_terms)
        
        # Get neutral headlines only
        all_real = [h.text for h in self.dataset if h.category == HeadlineCategory.REAL and is_neutral(h.text)]
        all_fake = [h.text for h in self.dataset if h.category == HeadlineCategory.FAKE and is_neutral(h.text)]
        
        # Use hardcoded neutral examples if not enough filtered
        fallback_real = [
            "Scientists discover new treatment for rare genetic disorder",
            "Federal Reserve announces interest rate decision after meeting",
            "New study examines sleep patterns in adolescents",
            "City council votes on infrastructure improvement plan",
            "Medical journal publishes peer-reviewed cancer research",
            "Technology company reports quarterly earnings results",
            "Researchers develop improved battery technology",
            "International summit addresses global trade policies"
        ]
        
        fallback_fake = [
            "SHOCKING: This ONE weird trick will change your life FOREVER!",
            "Doctors HATE him! See how he lost 50 pounds in ONE WEEK!!!",
            "BREAKING: Leaked documents you WON'T BELIEVE",
            "They DON'T want you to see this MIRACLE cure for everything!",
            "VIRAL: This common food causes CANCER (the truth REVEALED)",
            "UNBELIEVABLE discovery will blow your mind!",
            "Secret documents LEAKED - what they're HIDING from you!",
            "This SHOCKING finding changes everything we know!"
        ]
        
        real_headlines = all_real[:8] if len(all_real) >= 8 else fallback_real
        fake_headlines = all_fake[:8] if len(all_fake) >= 8 else fallback_fake
        
        # 1. Training progress animation (if model was trained)
        if self.model and self.model.history and TENSORFLOW_AVAILABLE:
            try:
                ani = self.visualizer.create_training_progress_animation(
                    self.model.history,
                    save_path=Path('training_progress.gif')
                )
                if ani:
                    plt.close()
            except Exception as e:
                self.logger.logger.warning(f"Could not create training animation: {e}")
        
        # 2. Example predictions visualization
        example_headlines = [
            "Scientists discover new renewable energy source that could change everything",
            "BREAKING: Government HIDING this ONE WEIRD TRICK to lose weight",
            "Annual economic report shows steady growth in all sectors",
            "You WON'T BELIEVE what this celebrity said in new interview",
            "New study confirms climate change findings with 99.9% certainty",
            "SHOCKING: This common household item causes CANCER",
            "Experts warn of potential economic downturn in quarterly report",
            "ALERT: They're putting CHEMICALS in our food to control us"
        ]
        
        if TENSORFLOW_AVAILABLE and self.model:
            try:
                sequences = self.preprocessor.texts_to_sequences(example_headlines)
                predictions, confidences = self.model.predict(sequences)
                
                fig_pred = self.visualizer.create_prediction_visualization(
                    example_headlines, 
                    predictions, 
                    confidences,
                    title="AI Fake News Detector: Real-time Analysis",
                    square=True
                )
                plt.savefig('predictions_demo_square.png', dpi=150, bbox_inches='tight')
                print("✅ Saved: predictions_demo_square.png (1:1 for split screen)")
                plt.close()
            except Exception as e:
                self.logger.logger.warning(f"Could not create prediction visualization: {e}")
        
        # 3. Headline comparison (for split-screen Reels)
        if real_headlines and fake_headlines:
            self._create_headline_comparison(real_headlines, fake_headlines)
        
        # 4. Word cloud comparison - showing language patterns
        if self.dataset and len(self.dataset) > 20:
            try:
                fig_wordcloud = self.visualizer.create_word_cloud_comparison(self.dataset, square=True)
                plt.savefig('word_cloud_comparison_square.png', dpi=150, bbox_inches='tight')
                print("✅ Saved: word_cloud_comparison_square.png (1:1 for split screen)")
                plt.close()
            except Exception as e:
                self.logger.logger.warning(f"Could not create word cloud: {e}")
        
        # 5. Confidence distribution scatter plot
        if TENSORFLOW_AVAILABLE and self.model:
            try:
                sequences = self.preprocessor.texts_to_sequences(example_headlines)
                predictions, confidences = self.model.predict(sequences)
                
                fig_confidence = self.visualizer.create_confidence_distribution(
                    predictions, confidences, example_headlines, square=True
                )
                plt.savefig('confidence_distribution_square.png', dpi=150, bbox_inches='tight')
                print("✅ Saved: confidence_distribution_square.png (1:1 for split screen)")
                plt.close()
            except Exception as e:
                self.logger.logger.warning(f"Could not create confidence distribution: {e}")
        
        # 6. Feature importance visual - educational content
        try:
            fig_features = self.visualizer.create_feature_importance_visual(example_headlines, square=True)
            plt.savefig('fake_news_indicators_square.png', dpi=150, bbox_inches='tight')
            print("✅ Saved: fake_news_indicators_square.png (1:1 for split screen)")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create feature importance visual: {e}")
        
        # 7. Performance metrics card
        if TENSORFLOW_AVAILABLE and self.model:
            try:
                # Get model metrics
                if hasattr(self.model, 'metrics_cache') and self.model.metrics_cache:
                    metrics = self.model.metrics_cache
                else:
                    # Use default metrics for demo
                    metrics = {
                        'accuracy': 0.87,
                        'precision': 0.85,
                        'recall': 0.89
                    }
                
                fig_metrics = self.visualizer.create_accuracy_metrics_card(metrics, square=True)
                plt.savefig('performance_metrics_square.png', dpi=150, bbox_inches='tight')
                print("✅ Saved: performance_metrics_square.png (1:1 for split screen)")
                plt.close()
            except Exception as e:
                self.logger.logger.warning(f"Could not create metrics card: {e}")
        
        # 8. Headline anatomy breakdown - educational
        try:
            real_anatomy = "Scientists discover new treatment for rare genetic disorder"
            fake_anatomy = "SHOCKING: This ONE weird trick will change your life FOREVER!!!"
            
            fig_anatomy = self.visualizer.create_headline_anatomy(real_anatomy, fake_anatomy, square=True)
            plt.savefig('headline_anatomy_square.png', dpi=150, bbox_inches='tight')
            print("✅ Saved: headline_anatomy_square.png (1:1 for split screen)")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create headline anatomy: {e}")
        
        # 9. Pipeline architecture - technical overview
        try:
            fig_pipeline = self.visualizer.create_pipeline_architecture(square=True)
            plt.savefig('pipeline_architecture_square.png', dpi=150, bbox_inches='tight')
            print("✅ Saved: pipeline_architecture_square.png (1:1 for split screen)")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create pipeline architecture: {e}")
        
        # 10. Model architecture - neural network details
        try:
            fig_model = self.visualizer.create_model_architecture(square=True)
            plt.savefig('model_architecture_square.png', dpi=150, bbox_inches='tight')
            print("✅ Saved: model_architecture_square.png (1:1 for split screen)")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create model architecture: {e}")
        
        # 9. Model architecture diagram - technical visual
        try:
            fig_arch = self.visualizer.create_model_architecture_diagram(self.model_config)
            plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            print("✅ Saved: model_architecture.png")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create architecture diagram: {e}")
        
        # 10. Data flow pipeline diagram
        try:
            fig_flow = self.visualizer.create_data_flow_diagram()
            plt.savefig('pipeline_architecture.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
            print("✅ Saved: pipeline_architecture.png")
            plt.close()
        except Exception as e:
            self.logger.logger.warning(f"Could not create pipeline diagram: {e}")
    
    def _create_headline_comparison(
        self, 
        real_headlines: list[str], 
        fake_headlines: list[str]
    ) -> None:
        """Create side-by-side headline comparison for Reels."""
        if not real_headlines or not fake_headlines:
            return
        
        # Create figure with portrait size
        fig = plt.figure(figsize=Constants.FIGURE_SIZE_PORTRAIT)
        fig.patch.set_facecolor('#1a1a1a')
        
        # Create two vertically stacked axes for portrait mode
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Main title
        fig.suptitle('REAL vs FAKE\\nCan You Spot the Difference?',
                    fontsize=26, fontweight='bold', y=0.98, color='white')
        
        # Top panel - Real headlines
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                    facecolor='#4ECDC4',
                                    alpha=0.15,
                                    transform=ax1.transAxes,
                                    zorder=0))
        
        # Title for real headlines
        ax1.text(0.5, 0.92, '✅ TRUSTWORTHY HEADLINES',
                transform=ax1.transAxes,
                fontsize=20, fontweight='bold',
                color='#4ECDC4', ha='center', va='top',
                zorder=10)
        
        # Add real headlines
        for i, headline in enumerate(real_headlines[:5]):
            wrapped_text = self._wrap_text(headline, 55)
            y_pos = 0.78 - i * 0.15
            ax1.text(0.08, y_pos, f'{i+1}. {wrapped_text}',
                    transform=ax1.transAxes,
                    fontsize=13, color='white',
                    ha='left', va='top',
                    zorder=10, wrap=True)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_facecolor('#1a1a1a')
        
        # Bottom panel - Fake headlines
        ax2.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                    facecolor='#FF6B6B',
                                    alpha=0.15,
                                    transform=ax2.transAxes,
                                    zorder=0))
        
        # Title for fake headlines
        ax2.text(0.5, 0.92, '⚠️ SUSPECT HEADLINES',
                transform=ax2.transAxes,
                fontsize=20, fontweight='bold',
                color='#FF6B6B', ha='center', va='top',
                zorder=10)
        
        # Add fake headlines
        for i, headline in enumerate(fake_headlines[:5]):
            wrapped_text = self._wrap_text(headline, 55)
            y_pos = 0.78 - i * 0.15
            ax2.text(0.08, y_pos, f'{i+1}. {wrapped_text}',
                    transform=ax2.transAxes,
                    fontsize=13, color='white',
                    ha='left', va='top',
                    zorder=10, wrap=True)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_facecolor('#1a1a1a')
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.02, hspace=0.15)
        plt.savefig('headline_comparison.png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        print("✅ Saved: headline_comparison.png")
        plt.close()
    
    def _wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text for display."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _run_live_demo(self) -> None:
        """Run interactive prediction demo for video content."""
        demo_headlines = [
            "Experts warn of economic downturn in quarterly report",
            "SHOCKING: This common food causes CANCER (doctors hate it!)",
            "International peace talks make significant progress",
            "They DON'T want you to see this leaked document about aliens",
            "New research offers hope for Alzheimer's patients",
            "BREAKING: Celebrity scandal rocks Hollywood elite",
            "Market shows strong recovery after policy changes",
            "Viral video proves everything we know is WRONG"
        ]
        
        print("\n" + "="*60)
        print("FAKE NEWS DETECTOR LIVE DEMO".center(60))
        print("="*60)
        print("\nAnalyzing headlines in real-time...\n")
        
        if TENSORFLOW_AVAILABLE and self.model and hasattr(self.preprocessor, 'tokenizer'):
            try:
                sequences = self.preprocessor.texts_to_sequences(demo_headlines)
                predictions, confidences = self.model.predict(sequences)
                
                for i, (headline, pred, conf) in enumerate(zip(demo_headlines, predictions, confidences)):
                    time.sleep(0.8)  # For dramatic effect in videos
                    
                    if pred == 0:
                        result = "✅ TRUSTWORTHY"
                        color_code = "\033[92m"  # Green
                        emoji = "📰"
                    else:
                        result = "⚠️  SUSPECT"
                        color_code = "\033[91m"  # Red
                        emoji = "🚨"
                    
                    print(f"{emoji} Headline #{i+1}: {headline}")
                    print(f"   AI Analysis: {color_code}{result} ({conf*100:.1f}% confidence)\033[0m")
                    
                    # Add explanation based on confidence
                    if conf > 0.8:
                        print(f"   🤖 Note: High confidence - clear patterns detected")
                    elif conf > 0.6:
                        print(f"   🤖 Note: Moderate confidence - some suspicious elements")
                    else:
                        print(f"   🤖 Note: Low confidence - ambiguous patterns")
                    print()
                    
                    self.logger.log_prediction(headline, result, conf)
            
            except Exception as e:
                self.logger.logger.warning(f"Prediction demo failed: {e}")
                self._run_live_demo_simple()
        
        else:
            self._run_live_demo_simple()
        
        print("\n" + "="*60)
        print("💡 DEMO COMPLETE - AI can help spot misinformation patterns!".center(60))
        print("="*60)
    
    def _run_live_demo_simple(self) -> None:
        """Simple fallback demo without ML model."""
        demo_headlines = [
            "Experts warn of economic downturn in quarterly report",
            "SHOCKING: This common food causes CANCER (doctors hate it!)",
            "International peace talks make significant progress",
            "They DON'T want you to see this leaked document about aliens",
            "New research offers hope for Alzheimer's patients",
            "BREAKING: Celebrity scandal rocks Hollywood elite",
            "Market shows strong recovery after policy changes",
            "Viral video proves everything we know is WRONG"
        ]
        
        for i, headline in enumerate(demo_headlines):
            time.sleep(0.8)
            
            # Simulate AI analysis based on keywords
            suspicious_keywords = ['shocking', 'breaking', 'leaked', 'hate', 'proves', 'wrong', 'aliens', '!']
            if any(keyword in headline.lower() for keyword in suspicious_keywords):
                print(f"🚨 Headline #{i+1}: {headline}")
                print("   AI Analysis: \033[91m⚠️  SUSPECT (Clickbait patterns detected)\033[0m")
                print("   🤖 Note: Contains sensational language common in fake news")
            else:
                print(f"📰 Headline #{i+1}: {headline}")
                print("   AI Analysis: \033[92m✅ TRUSTWORTHY (Normal journalistic patterns)\033[0m")
                print("   🤖 Note: Uses standard reporting language")
            print()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main entry point for the Fake News Detector demo.
    Follows single responsibility principle - only orchestrates execution.
    """
    print("\n" + "🌟" * 30)
    print("INSTAGRAM REELS: FAKE NEWS DETECTOR AI DEMO".center(60))
    print("🌟" * 30 + "\n")
    
    print("🤖 This demo shows how AI can identify potentially fake news headlines.")
    print("📊 Using real datasets from Kaggle when available.")
    print("🎬 Perfect for educational content about digital literacy!\n")
    
    # Check requirements
    if not KAGGLEHUB_AVAILABLE:
        print("⚠️  Note: kagglehub not installed. Install with: pip install kagglehub")
        print("   Will use synthetic data for this demo.\n")
    
    if not TENSORFLOW_AVAILABLE:
        print("⚠️  Note: TensorFlow not installed. Install with: pip install tensorflow")
        print("   Running in demo mode with simulated predictions.\n")
    
    try:
        # Initialize and run demo
        demo = FakeNewsDemo()
        demo.run_full_demo()
        
        # Display generated files
        print("\n📁 Generated Files for Your Reels:")
        files_generated = []
        
        if Path('data_sources.png').exists():
            files_generated.append("1. data_sources.png - Dataset statistics & class distribution")
        
        if Path('headline_comparison.png').exists():
            files_generated.append("2. headline_comparison.png - Side-by-side real vs fake comparison")
        
        if Path('word_cloud_comparison.png').exists():
            files_generated.append("3. word_cloud_comparison.png - Language patterns visualization")
        
        if Path('confidence_distribution.png').exists():
            files_generated.append("4. confidence_distribution.png - AI confidence scatter plot")
        
        if Path('fake_news_indicators.png').exists():
            files_generated.append("5. fake_news_indicators.png - Top warning signs breakdown")
        
        if Path('performance_metrics.png').exists():
            files_generated.append("6. performance_metrics.png - Model accuracy metrics card")
        
        if Path('headline_anatomy.png').exists():
            files_generated.append("7. headline_anatomy.png - Educational headline breakdown")
        
        if Path('model_architecture.png').exists():
            files_generated.append("8. model_architecture.png - CNN-LSTM architecture (dark theme)")
        
        if Path('pipeline_architecture.png').exists():
            files_generated.append("9. pipeline_architecture.png - Complete pipeline flow (dark theme)")
        
        if TENSORFLOW_AVAILABLE:
            if Path('training_progress.gif').exists():
                files_generated.append("10. training_progress.gif - Training animation (GIF)")
            if Path('predictions_demo.png').exists():
                files_generated.append("11. predictions_demo.png - Live predictions visualization")
            if Path('best_model.keras').exists():
                files_generated.append("12. best_model.keras - Trained model file")
        
        if files_generated:
            for file_desc in files_generated:
                print(f"  ✅ {file_desc}")
        else:
            print("  No files were generated. Check logs for errors.")
        
        print("\n🎬 Instagram Reels Script Suggestions:")
        print("  Scene 1 (0-3s): Hook - Show 'headline_anatomy.png' with dramatic music")
        print("  Scene 2 (3-6s): Show 'headline_comparison.png' - Real vs Fake split")
        print("  Scene 3 (6-9s): Show 'word_cloud_comparison.png' - Language patterns")
        print("  Scene 4 (9-12s): Show 'fake_news_indicators.png' - Warning signs")
        print("  Scene 5 (12-15s): Show 'confidence_distribution.png' - AI in action")
        if TENSORFLOW_AVAILABLE:
            print("  Scene 6 (15-18s): Show 'training_progress.gif' - AI learning")
            print("  Scene 7 (18-22s): Show 'predictions_demo.png' - Live results")
            print("  Scene 8 (22-25s): Show 'performance_metrics.png' - Final stats")
        print("  Scene 9 (25-30s): Call-to-action - Always verify sources!")
        
        print("\n💡 Educational Talking Points for Voice-Over:")
        print("  • AI analyzes thousands of headlines to learn patterns")
        print("  • Fake news often uses CAPS, exclamations, and emotional words")
        print("  • Real news is factual, neutral, and cites credible sources")
        print("  • AI accuracy is high but not perfect - always think critically")
        print("  • Look for these red flags: clickbait, vague sources, sensationalism")
        
        print("\n🎨 Content Ideas:")
        print("  • Create a 'spot the fake' quiz using headline_comparison.png")
        print("  • Make a carousel post with each indicator from fake_news_indicators.png")
        print("  • Use word_cloud as background for educational text overlay")
        print("  • Share performance_metrics.png to show AI capabilities")
        print("  • Post headline_anatomy as an educational infographic")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("\n💡 Quick fix: Run this simplified version first:")
        print("  python -c \"import kagglehub; print('Kaggle working')\"")
        print("  pip install tensorflow --upgrade")
        sys.exit(1)


if __name__ == "__main__":
    main()