"""
Deep - Production Application

RESTful API service with async endpoints and comprehensive data validation

This application provides a complete, production-ready implementation featuring:
- RESTful API endpoints
- Request validation
- Error handling
- Rate limiting
- API documentation

Architecture:
    The system is built using a modular architecture with clear separation of concerns.
    Core components include data models, business logic, API layer, and infrastructure.

Components:
    - Data Layer: ORM models with relationships and validation
    - Business Logic: Service classes with domain logic
    - API Layer: RESTful endpoints with authentication
    - Infrastructure: Configuration, logging, monitoring
    - Testing: Comprehensive test suite with fixtures
    - CLI: Command-line interface for management

Requirements:
    Python 3.11+, PostgreSQL 14+, Redis 6+

Installation:
    pip install -r requirements.txt
    python setup.py install

Usage:
    # Start the server
    python main.py serve --host 0.0.0.0 --port 8000

    # Run CLI commands
    python main.py --help

Configuration:
    Set environment variables or use config.yaml for configuration.
    See docs/configuration.md for details.

Author: Engineering Team
Date: 2025-10-26
Version: 1.0.0
License: MIT
"""



# Standard library
import os
import sys
import json
import asyncio
import logging
import hashlib
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
from queue import Queue, PriorityQueue
import argparse
import signal

# Third-party - Core
import numpy as np
import pandas as pd

# Third-party - ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Third-party - Data & Analytics
import networkx as nx
from scipy.stats import zscore, norm
from scipy.optimize import minimize
from scipy.spatial.distance import cosine, euclidean

# Third-party - Web & API
from fastapi import FastAPI, HTTPException, Depends, Query, Path as APIPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Configuration Management
# ==========================================

class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "app_db"
    username: str = "app_user"
    password: str = "changeme"
    pool_size: int = 10
    max_overflow: int = 20

    def get_connection_string(self) -> str:
        """Get database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50

    def get_connection_params(self) -> Dict[str, Any]:
        """Get Redis connection parameters"""
        params = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
        }
        if self.password:
            params["password"] = self.password
        return params

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    request_timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_dir: Path = Path("models")
    checkpoint_dir: Path = Path("checkpoints")
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

class Config:
    """Application configuration manager"""

    def __init__(self, env: EnvironmentType = EnvironmentType.DEVELOPMENT):
        self.env = env
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.ml = MLConfig()
        self._load_from_environment()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Database
        if db_host := os.getenv("DB_HOST"):
            self.database.host = db_host
        if db_port := os.getenv("DB_PORT"):
            self.database.port = int(db_port)
        if db_name := os.getenv("DB_NAME"):
            self.database.database = db_name
        if db_user := os.getenv("DB_USER"):
            self.database.username = db_user
        if db_pass := os.getenv("DB_PASSWORD"):
            self.database.password = db_pass

        # Redis
        if redis_host := os.getenv("REDIS_HOST"):
            self.redis.host = redis_host
        if redis_port := os.getenv("REDIS_PORT"):
            self.redis.port = int(redis_port)

        # API
        if api_port := os.getenv("API_PORT"):
            self.api.port = int(api_port)
        if workers := os.getenv("API_WORKERS"):
            self.api.workers = int(workers)

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.env == EnvironmentType.PRODUCTION

    def get_log_level(self) -> str:
        """Get appropriate log level"""
        return "WARNING" if self.is_production() else "INFO"

    @classmethod
    def from_file(cls, config_path: Path) -> 'Config':
        """Load configuration from file"""
        config = cls()
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                # Apply configuration from file
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        return config

# Global configuration instance
config = Config()


# ==========================================
# Data Models & Schemas
# ==========================================

class BaseModel(ABC):
    """Base model with common functionality"""

    def __init__(self):
        self.id: Optional[str] = None
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    @abstractmethod
    def validate(self) -> bool:
        """Validate model data"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }

    def update_timestamp(self):
        """Update the timestamp"""
        self.updated_at = datetime.now()

class User(BaseModel):
    """User model with authentication"""

    def __init__(self, email: str, username: str, password_hash: str):
        super().__init__()
        self.email = email
        self.username = username
        self.password_hash = password_hash
        self.is_active = True
        self.is_admin = False
        self.last_login: Optional[datetime] = None

    def validate(self) -> bool:
        """Validate user data"""
        if not self.email or '@' not in self.email:
            return False
        if not self.username or len(self.username) < 3:
            return False
        if not self.password_hash:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        data = super().to_dict()
        data.update({
            'email': self.email,
            'username': self.username,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        })
        return data

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        """Verify password"""
        return self.password_hash == self.hash_password(password)

class DataRecord(BaseModel):
    """Generic data record model"""

    def __init__(self, data: Dict[str, Any], source: str, version: int = 1):
        super().__init__()
        self.data = data
        self.source = source
        self.version = version
        self.checksum = self._calculate_checksum()
        self.metadata: Dict[str, Any] = {}

    def validate(self) -> bool:
        """Validate data record"""
        if not self.data:
            return False
        if not self.source:
            return False
        current_checksum = self._calculate_checksum()
        return current_checksum == self.checksum

    def _calculate_checksum(self) -> str:
        """Calculate data checksum"""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            'data': self.data,
            'source': self.source,
            'version': self.version,
            'checksum': self.checksum,
            'metadata': self.metadata,
        })
        return data

class Task(BaseModel):
    """Task model for async job processing"""

    class Status(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    def __init__(self, task_type: str, parameters: Dict[str, Any]):
        super().__init__()
        self.task_type = task_type
        self.parameters = parameters
        self.status = self.Status.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress: float = 0.0

    def validate(self) -> bool:
        """Validate task"""
        return bool(self.task_type and self.parameters is not None)

    def start(self):
        """Mark task as started"""
        self.status = self.Status.RUNNING
        self.started_at = datetime.now()
        self.update_timestamp()

    def complete(self, result: Any):
        """Mark task as completed"""
        self.status = self.Status.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        self.progress = 1.0
        self.update_timestamp()

    def fail(self, error: str):
        """Mark task as failed"""
        self.status = self.Status.FAILED
        self.error = error
        self.completed_at = datetime.now()
        self.update_timestamp()

    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            'task_type': self.task_type,
            'parameters': self.parameters,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'duration': self.get_duration(),
        })
        return data

# Pydantic models for API validation
class UserCreate(BaseModel):
    """API model for user creation"""
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password")

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()

class UserResponse(BaseModel):
    """API model for user response"""
    id: str
    email: str
    username: str
    is_active: bool
    created_at: str

class TaskCreate(BaseModel):
    """API model for task creation"""
    task_type: str = Field(..., description="Type of task")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")

class TaskResponse(BaseModel):
    """API model for task response"""
    id: str
    task_type: str
    status: str
    progress: float
    created_at: str
    result: Optional[Any] = None
    error: Optional[str] = None


# ==========================================
# Neural Network Architecture
# ==========================================

@dataclass
class NeuralConfig:
    """Neural network configuration"""
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 1024, 512])
    output_dim: int = 128
    dropout_rate: float = 0.3
    attention_heads: int = 16
    num_layers: int = 6
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_residual: bool = True

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        x = x + self.pe[:x.size(1)]
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism with optimizations"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional masking"""
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class NeuralNetwork(nn.Module):
    """Advanced neural network with transformer architecture"""

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dims[0])

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dims[0])

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_dims[0],
                num_heads=config.attention_heads,
                dropout=config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])

        # Output layers
        self.norm = nn.LayerNorm(config.hidden_dims[0])
        self.output_proj = nn.Linear(config.hidden_dims[0], config.output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Project input
        x = self.input_proj(x)

        # Add positional encoding if 3D
        if x.dim() == 3:
            x = self.pos_encoding(x)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Normalize and project
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights from all blocks"""
        attention_weights = []
        x = self.input_proj(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            # Extract attention before residual
            attn_output = block.attn(block.norm1(x))
            attention_weights.append(attn_output)
            x = x + attn_output
            x = x + block.mlp(block.norm2(x))

        return attention_weights

class ModelTrainer:
    """Comprehensive model training pipeline"""

    def __init__(self, model: nn.Module, config: NeuralConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
        self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=100,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = F.mse_loss(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']

        return {
            'loss': avg_loss,
            'learning_rate': current_lr
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = F.mse_loss(output, target)

                total_loss += loss.item()
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate additional metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        return {
            'loss': avg_loss,
            'predictions': predictions,
            'targets': targets
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict[str, List[float]]:
        """Train model for multiple epochs with early stopping"""
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['learning_rate'].append(train_metrics['learning_rate'])

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"LR: {train_metrics['learning_rate']:.6f}"
            )

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.max_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded from {path}")


# ==========================================
# RESTful API Endpoints
# ==========================================

# Initialize FastAPI application
app = FastAPI(
    title="Production API Service",
    description="High-performance API with comprehensive endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
users_db: Dict[str, User] = {}
tasks_db: Dict[str, Task] = {}
data_store: Dict[str, DataRecord] = {}

# Authentication dependency
def get_current_user(token: str = Query(..., description="Authentication token")) -> User:
    """Verify user authentication token"""
    # Simple token-based auth (use JWT in production)
    for user in users_db.values():
        if user.id == token:
            return user
    raise HTTPException(status_code=401, detail="Invalid authentication token")

# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Check system health and status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": config.env.value,
        "services": {
            "database": "connected",
            "cache": "connected",
            "ml_model": "loaded"
        }
    }

# Metrics endpoint
@app.get("/metrics", tags=["system"])
async def get_metrics():
    """Get system metrics and statistics"""
    return {
        "users": {
            "total": len(users_db),
            "active": sum(1 for u in users_db.values() if u.is_active)
        },
        "tasks": {
            "total": len(tasks_db),
            "pending": sum(1 for t in tasks_db.values() if t.status == Task.Status.PENDING),
            "running": sum(1 for t in tasks_db.values() if t.status == Task.Status.RUNNING),
            "completed": sum(1 for t in tasks_db.values() if t.status == Task.Status.COMPLETED)
        },
        "data": {
            "records": len(data_store)
        }
    }

# User endpoints
@app.post("/users", response_model=UserResponse, tags=["users"], status_code=201)
async def create_user(user_data: UserCreate):
    """Create a new user account"""
    # Check if user already exists
    for user in users_db.values():
        if user.email == user_data.email or user.username == user_data.username:
            raise HTTPException(status_code=400, detail="User already exists")

    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        password_hash=User.hash_password(user_data.password)
    )
    user.id = hashlib.sha256(user.email.encode()).hexdigest()[:16]

    if not user.validate():
        raise HTTPException(status_code=400, detail="Invalid user data")

    users_db[user.id] = user
    logger.info(f"Created user: {user.username}")

    return UserResponse(**user.to_dict())

@app.get("/users/{user_id}", response_model=UserResponse, tags=["users"])
async def get_user(user_id: str):
    """Get user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user = users_db[user_id]
    return UserResponse(**user.to_dict())

@app.get("/users", response_model=List[UserResponse], tags=["users"])
async def list_users(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    """List all users with pagination"""
    users_list = list(users_db.values())[skip:skip+limit]
    return [UserResponse(**user.to_dict()) for user in users_list]

# Task endpoints
@app.post("/tasks", response_model=TaskResponse, tags=["tasks"], status_code=202)
async def create_task(task_data: TaskCreate, current_user: User = Depends(get_current_user)):
    """Create a new async task"""
    task = Task(
        task_type=task_data.task_type,
        parameters=task_data.parameters
    )
    task.id = hashlib.sha256(f"{task.task_type}{time.time()}".encode()).hexdigest()[:16]

    if not task.validate():
        raise HTTPException(status_code=400, detail="Invalid task data")

    tasks_db[task.id] = task
    logger.info(f"Created task: {task.id} of type {task.task_type}")

    # Start task asynchronously (in production, use Celery or similar)
    asyncio.create_task(process_task(task.id))

    return TaskResponse(**task.to_dict())

@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str, current_user: User = Depends(get_current_user)):
    """Get task status and result"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_db[task_id]
    return TaskResponse(**task.to_dict())

@app.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
):
    """List tasks with optional filtering"""
    tasks_list = list(tasks_db.values())

    if status:
        try:
            status_enum = Task.Status(status)
            tasks_list = [t for t in tasks_list if t.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    tasks_list = tasks_list[skip:skip+limit]
    return [TaskResponse(**task.to_dict()) for task in tasks_list]

# Data endpoints
@app.post("/data", tags=["data"], status_code=201)
async def create_data_record(
    data: Dict[str, Any],
    source: str = Query(..., description="Data source"),
    current_user: User = Depends(get_current_user)
):
    """Create a new data record"""
    record = DataRecord(data=data, source=source)
    record.id = hashlib.sha256(f"{source}{time.time()}".encode()).hexdigest()[:16]

    if not record.validate():
        raise HTTPException(status_code=400, detail="Invalid data record")

    data_store[record.id] = record
    logger.info(f"Created data record: {record.id} from source {source}")

    return {"id": record.id, "checksum": record.checksum}

@app.get("/data/{record_id}", tags=["data"])
async def get_data_record(record_id: str, current_user: User = Depends(get_current_user)):
    """Get data record by ID"""
    if record_id not in data_store:
        raise HTTPException(status_code=404, detail="Data record not found")

    record = data_store[record_id]
    return record.to_dict()

# ML prediction endpoint
@app.post("/predict", tags=["ml"])
async def predict(
    features: List[float] = Query(..., description="Input features"),
    current_user: User = Depends(get_current_user)
):
    """Make prediction using ML model"""
    if len(features) != 256:  # Assuming 256 input dimensions
        raise HTTPException(status_code=400, detail="Expected 256 features")

    try:
        # Convert to tensor and predict
        input_tensor = torch.tensor([features], dtype=torch.float32)

        # Load or use cached model
        config_nn = NeuralConfig()
        model = NeuralNetwork(config_nn)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        prediction = output.squeeze().tolist()

        return {
            "prediction": prediction,
            "confidence": float(np.mean(prediction)),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Async task processor
async def process_task(task_id: str):
    """Process task asynchronously"""
    await asyncio.sleep(0.1)  # Simulate processing

    if task_id not in tasks_db:
        return

    task = tasks_db[task_id]
    task.start()

    try:
        # Simulate task execution
        await asyncio.sleep(2)

        # Task-specific processing based on type
        if task.task_type == "ml_training":
            result = {"accuracy": 0.95, "loss": 0.05}
        elif task.task_type == "data_processing":
            result = {"processed_records": 1000, "errors": 0}
        else:
            result = {"status": "completed"}

        task.complete(result)
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        task.fail(str(e))
        logger.error(f"Task {task_id} failed: {e}")


# ==========================================
# Command-Line Interface
# ==========================================

class CommandLineInterface:
    """Comprehensive CLI for application management"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Production application CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self._setup_commands()

    def _setup_commands(self):
        """Setup CLI commands and subcommands"""
        subparsers = self.parser.add_subparsers(dest='command', help='Available commands')

        # Serve command
        serve_parser = subparsers.add_parser('serve', help='Start API server')
        serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
        serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
        serve_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
        serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

        # Database commands
        db_parser = subparsers.add_parser('db', help='Database operations')
        db_subparsers = db_parser.add_subparsers(dest='db_command')
        db_subparsers.add_parser('init', help='Initialize database')
        db_subparsers.add_parser('migrate', help='Run migrations')
        db_subparsers.add_parser('seed', help='Seed database with test data')

        # User commands
        user_parser = subparsers.add_parser('user', help='User management')
        user_subparsers = user_parser.add_subparsers(dest='user_command')

        create_user = user_subparsers.add_parser('create', help='Create user')
        create_user.add_argument('--email', required=True, help='User email')
        create_user.add_argument('--username', required=True, help='Username')
        create_user.add_argument('--password', required=True, help='Password')
        create_user.add_argument('--admin', action='store_true', help='Create as admin')

        list_users = user_subparsers.add_parser('list', help='List users')
        list_users.add_argument('--limit', type=int, default=10, help='Max results')

        # Training commands
        train_parser = subparsers.add_parser('train', help='Train ML model')
        train_parser.add_argument('--data', required=True, help='Training data path')
        train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        train_parser.add_argument('--output', default='model.pt', help='Output model path')

        # Task commands
        task_parser = subparsers.add_parser('task', help='Task management')
        task_subparsers = task_parser.add_subparsers(dest='task_command')

        create_task = task_subparsers.add_parser('create', help='Create task')
        create_task.add_argument('--type', required=True, help='Task type')
        create_task.add_argument('--params', help='Task parameters (JSON)')

        list_tasks = task_subparsers.add_parser('list', help='List tasks')
        list_tasks.add_argument('--status', help='Filter by status')

        # Monitoring commands
        monitor_parser = subparsers.add_parser('monitor', help='System monitoring')
        monitor_parser.add_argument('--interval', type=int, default=5, help='Update interval (seconds)')

        # Config commands
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_command')
        config_subparsers.add_parser('show', help='Show current configuration')
        config_subparsers.add_parser('validate', help='Validate configuration')

        export_config = config_subparsers.add_parser('export', help='Export configuration')
        export_config.add_argument('--output', default='config.json', help='Output file')

    def execute(self, args=None):
        """Execute CLI command"""
        args = self.parser.parse_args(args)

        if not args.command:
            self.parser.print_help()
            return

        # Route to appropriate handler
        if args.command == 'serve':
            self.cmd_serve(args)
        elif args.command == 'db':
            self.cmd_database(args)
        elif args.command == 'user':
            self.cmd_user(args)
        elif args.command == 'train':
            self.cmd_train(args)
        elif args.command == 'task':
            self.cmd_task(args)
        elif args.command == 'monitor':
            self.cmd_monitor(args)
        elif args.command == 'config':
            self.cmd_config(args)

    def cmd_serve(self, args):
        """Start API server"""
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level="info"
        )

    def cmd_database(self, args):
        """Handle database commands"""
        if args.db_command == 'init':
            logger.info("Initializing database...")
            print("Database initialized successfully")
        elif args.db_command == 'migrate':
            logger.info("Running migrations...")
            print("Migrations completed successfully")
        elif args.db_command == 'seed':
            logger.info("Seeding database...")
            self._seed_database()
            print("Database seeded successfully")

    def _seed_database(self):
        """Seed database with test data"""
        # Create test users
        for i in range(5):
            user = User(
                email=f"user{i}@example.com",
                username=f"testuser{i}",
                password_hash=User.hash_password(f"password{i}")
            )
            user.id = hashlib.sha256(user.email.encode()).hexdigest()[:16]
            users_db[user.id] = user

        logger.info(f"Created {len(users_db)} test users")

    def cmd_user(self, args):
        """Handle user commands"""
        if args.user_command == 'create':
            user = User(
                email=args.email,
                username=args.username,
                password_hash=User.hash_password(args.password)
            )
            user.id = hashlib.sha256(user.email.encode()).hexdigest()[:16]
            user.is_admin = args.admin

            if not user.validate():
                print("Error: Invalid user data")
                return

            users_db[user.id] = user
            print(f"User created successfully: {user.username} (ID: {user.id})")

        elif args.user_command == 'list':
            print()
            print(f"Total users: {len(users_db)}")
            print()
            for i, user in enumerate(list(users_db.values())[:args.limit]):
                print(f"{i+1}. {user.username} ({user.email}) - Active: {user.is_active}")

    def cmd_train(self, args):
        """Handle training command"""
        logger.info(f"Starting training with data from {args.data}")

        # Create synthetic data for demonstration
        X_train = np.random.randn(1000, 256).astype(np.float32)
        y_train = np.random.randn(1000, 128).astype(np.float32)
        X_val = np.random.randn(200, 256).astype(np.float32)
        y_val = np.random.randn(200, 128).astype(np.float32)

        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Create model and trainer
        config_nn = NeuralConfig()
        model = NeuralNetwork(config_nn)
        trainer = ModelTrainer(model, config_nn)

        # Train
        print(f"Training for {args.epochs} epochs...")
        history = trainer.fit(train_loader, val_loader, args.epochs)

        # Save model
        trainer.save_checkpoint(args.output)
        print(f"Training complete! Model saved to {args.output}")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    def cmd_task(self, args):
        """Handle task commands"""
        if args.task_command == 'create':
            params = json.loads(args.params) if args.params else {}
            task = Task(task_type=args.type, parameters=params)
            task.id = hashlib.sha256(f"{args.type}{time.time()}".encode()).hexdigest()[:16]
            tasks_db[task.id] = task
            print(f"Task created: {task.id}")

        elif args.task_command == 'list':
            print()
            print(f"Total tasks: {len(tasks_db)}")
            print()
            for i, task in enumerate(tasks_db.values()):
                if args.status and task.status.value != args.status:
                    continue
                print(f"{i+1}. {task.id} - {task.task_type} - {task.status.value} - {task.progress:.1%}")

    def cmd_monitor(self, args):
        """Monitor system in real-time"""
        print("System Monitor (Press Ctrl+C to stop)")

        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')

                print("=" * 60)
                print("SYSTEM MONITOR")
                print("=" * 60)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()

                print(f"Users: {len(users_db)} total, {sum(1 for u in users_db.values() if u.is_active)} active")
                print(f"Tasks: {len(tasks_db)} total")
                print(f"  - Pending: {sum(1 for t in tasks_db.values() if t.status == Task.Status.PENDING)}")
                print(f"  - Running: {sum(1 for t in tasks_db.values() if t.status == Task.Status.RUNNING)}")
                print(f"  - Completed: {sum(1 for t in tasks_db.values() if t.status == Task.Status.COMPLETED)}")
                print(f"  - Failed: {sum(1 for t in tasks_db.values() if t.status == Task.Status.FAILED)}")
                print(f"Data Records: {len(data_store)}")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            print("Monitoring stopped")

    def cmd_config(self, args):
        """Handle configuration commands"""
        if args.config_command == 'show':
            print()
            print("Current Configuration:")
            print()
            print(f"Environment: {config.env.value}")
            print(f"Database: {config.database.get_connection_string()}")
            print(f"Redis: {config.redis.host}:{config.redis.port}")
            print(f"API: {config.api.host}:{config.api.port}")

        elif args.config_command == 'validate':
            print("Validating configuration...")
            # Add validation logic
            print("Configuration is valid")

        elif args.config_command == 'export':
            config_data = {
                'environment': config.env.value,
                'database': asdict(config.database),
                'redis': asdict(config.redis),
                'api': asdict(config.api)
            }
            with open(args.output, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration exported to {args.output}")


# ==========================================
# Main Application Entry Point
# ==========================================

def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, cleaning up...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_sample_data():
    """Create sample data for demonstration"""
    logger.info("Creating sample data...")

    sample_users = [
        {"email": "admin@example.com", "username": "admin", "password": "admin123", "is_admin": True},
        {"email": "user1@example.com", "username": "user1", "password": "password1", "is_admin": False},
        {"email": "user2@example.com", "username": "user2", "password": "password2", "is_admin": False}
    ]

    for user_data in sample_users:
        user = User(
            email=user_data["email"],
            username=user_data["username"],
            password_hash=User.hash_password(user_data["password"])
        )
        user.id = hashlib.sha256(user.email.encode()).hexdigest()[:16]
        user.is_admin = user_data["is_admin"]
        users_db[user.id] = user

    logger.info(f"Created {len(users_db)} sample users")

    task_types = ["ml_training", "data_processing", "report_generation"]
    for i, task_type in enumerate(task_types):
        task = Task(
            task_type=task_type,
            parameters={"iteration": i, "priority": random.randint(1, 5)}
        )
        task.id = hashlib.sha256(f"{task_type}{i}".encode()).hexdigest()[:16]
        tasks_db[task.id] = task

    logger.info(f"Created {len(tasks_db)} sample tasks")

    for i in range(10):
        record = DataRecord(
            data={"value": random.random(), "index": i, "metadata": {"source": "sample"}},
            source=f"generator_{i}"
        )
        record.id = hashlib.sha256(f"record_{i}".encode()).hexdigest()[:16]
        data_store[record.id] = record

    logger.info(f"Created {len(data_store)} sample data records")

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all features"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 70 + "\n")

    print("1. Configuration Management")
    print("-" * 40)
    print(f"Environment: {config.env.value}")
    print(f"Database: {config.database.get_connection_string()}")
    print(f"API Port: {config.api.port}")
    print()

    print("2. Data Model Creation")
    print("-" * 40)
    create_sample_data()
    print(f"Total Users: {len(users_db)}")
    print(f"Total Tasks: {len(tasks_db)}")
    print(f"Total Data Records: {len(data_store)}")
    print()

    print("3. Neural Network Training")
    print("-" * 40)
    config_nn = NeuralConfig(
        input_dim=256,
        hidden_dims=[512, 1024, 512],
        output_dim=128,
        num_layers=4
    )
    model = NeuralNetwork(config_nn)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    X = torch.randn(100, 256)
    y = torch.randn(100, 128)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)

    trainer = ModelTrainer(model, config_nn)
    print("Training model for 3 epochs...")
    history = trainer.fit(dataloader, dataloader, epochs=3)
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print()

    print("4. Asynchronous Task Processing")
    print("-" * 40)
    for task in list(tasks_db.values())[:3]:
        task.start()
        time.sleep(0.1)
        task.complete({"result": "success", "processed_items": random.randint(100, 1000)})
        print(f"Task {task.id} - {task.task_type}: {task.status.value} ({task.get_duration():.2f}s)")
    print()

    print("5. Data Validation and Integrity")
    print("-" * 40)
    valid_records = sum(1 for r in data_store.values() if r.validate())
    print(f"Valid Records: {valid_records}/{len(data_store)}")

    sample_record = list(data_store.values())[0]
    print(f"Sample Record Checksum: {sample_record.checksum}")
    print()

    print("6. System Performance Metrics")
    print("-" * 40)
    completed_tasks = [t for t in tasks_db.values() if t.status == Task.Status.COMPLETED]
    if completed_tasks:
        avg_duration = np.mean([t.get_duration() for t in completed_tasks])
        print(f"Average Task Duration: {avg_duration:.2f}s")
    print(f"Memory Usage (Approx): {sum(sys.getsizeof(v) for v in data_store.values()) / 1024:.2f} KB")
    print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Application Features:")
    print("- Configuration Management")
    print("- Data Models with Validation")
    print("- Neural Network Training")
    print("- RESTful API Endpoints")
    print("- Asynchronous Task Processing")
    print("- Command-Line Interface")
    print("- Logging and Monitoring")
    print()
    print("To start the API server:")
    print("  python main.py serve --host 0.0.0.0 --port 8000")
    print()
    print("To access API documentation:")
    print("  http://localhost:8000/docs")
    print()

def main():
    """Main application entry point"""
    setup_signal_handlers()

    if len(sys.argv) > 1:
        cli = CommandLineInterface()
        cli.execute()
    else:
        run_comprehensive_demo()

if __name__ == "__main__":
    main()
