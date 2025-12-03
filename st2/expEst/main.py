
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class EMOCAConfig:
    """Configuration for EMOCA model"""
    img_size: int = 224
    latent_dim: int = 128
    num_emotions: int = 7
    num_expression_coeffs: int = 50  # 3DMM expression coefficients
    backbone: str = "resnet50"
    use_pretrained: bool = True
    dropout_rate: float = 0.3
    # Regularization weights
    lambda_emo: float = 1.0
    lambda_expr: float = 0.5
    lambda_photo: float = 5.0
    lambda_percep: float = 0.1
    lambda_temp: float = 0.2

class HybridBackbone(nn.Module):
    """ResNet backbone with hybrid features for emotion capture"""
    def __init__(self, config: EMOCAConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet
        if config.backbone == "resnet50":
            resnet = models.resnet50(pretrained=config.use_pretrained)
        elif config.backbone == "resnet18":
            resnet = models.resnet18(pretrained=config.use_pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # Remove fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Additional emotion-specific convolutional layers
        self.emotion_conv = nn.Sequential(
            nn.Conv2d(2048 if config.backbone == "resnet50" else 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout_rate),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        emotion_features = self.emotion_conv(features)
        pooled = self.pool(emotion_features)
        return pooled.flatten(1)

class LatentEmotionEmbedding(nn.Module):
    """128D latent emotion embedding capturing arousal, valence, and discrete emotions"""
    def __init__(self, config: EMOCAConfig):
        super().__init__()
        self.config = config
        
        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(256, config.latent_dim),
            nn.BatchNorm1d(config.latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.BatchNorm1d(config.latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Arousal and valence regressors
        self.arousal_head = nn.Linear(config.latent_dim, 1)
        self.valence_head = nn.Linear(config.latent_dim, 1)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Generate latent embedding
        embedding = self.embedding_net(features)
        
        # Predict arousal and valence (continuous emotion dimensions)
        arousal = torch.sigmoid(self.arousal_head(embedding)) * 2 - 1  # [-1, 1]
        valence = torch.sigmoid(self.valence_head(embedding)) * 2 - 1  # [-1, 1]
        
        return {
            "embedding": embedding,
            "arousal": arousal,
            "valence": valence
        }

class EMOCA(nn.Module):
    """Main EMOCA model combining hybrid backbone, emotion embedding, and expression decoder"""
    def __init__(self, config: EMOCAConfig):
        super().__init__()
        self.config = config
        
        # Backbone network
        self.backbone = HybridBackbone(config)
        
        # Bridge layer
        backbone_out_dim = 256  # Output from emotion_conv
        self.bridge = nn.Linear(
            2048 if config.backbone == "resnet50" else 512,
            backbone_out_dim
        )
        
        # Latent emotion embedding
        self.emotion_embedding = LatentEmotionEmbedding(config)
        
        # Emotion classification head (discrete emotions)
        self.emotion_head = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, config.num_emotions)
        )
        
        # Expression decoder (3DMM coefficients)
        self.expression_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.num_expression_coeffs),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        backbone_features = self.backbone(x)
        
        # Bridge to emotion features
        emotion_features = F.relu(self.bridge(backbone_features))
        
        # Generate latent emotion embedding
        embedding_output = self.emotion_embedding(emotion_features)
        latent_embedding = embedding_output["embedding"]
        
        # Emotion classification
        emotion_logits = self.emotion_head(latent_embedding)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        # Expression coefficients
        expression_coeffs = self.expression_decoder(latent_embedding)
        
        return {
            "latent_embedding": latent_embedding,
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "expression_coeffs": expression_coeffs,
            "arousal": embedding_output["arousal"],
            "valence": embedding_output["valence"]
        }

# 2. Loss Functions
class EMOCALoss(nn.Module):
    """Total loss function for EMOCA training"""
    def __init__(self, config: EMOCAConfig):
        super().__init__()
        self.config = config
        
        # Individual loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # For perceptual loss (using VGG)
        self.vgg = None
        self._init_perceptual_loss()
        
    def _init_perceptual_loss(self):
        """Initialize VGG network for perceptual loss"""
        try:
            vgg = models.vgg16(pretrained=True).features[:16].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
        except:
            print("Warning: Could not load VGG for perceptual loss")
            self.vgg = None
    
    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        return self.l1_loss(pred_features, target_features)
    
    def temporal_consistency_loss(self, 
                                 current: torch.Tensor, 
                                 previous: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Temporal consistency loss for video sequences"""
        if mask is None:
            mask = torch.ones_like(current)
        
        diff = torch.abs(current - previous) * mask
        return torch.mean(diff)
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                previous_frame: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # Emotion classification loss
        if "emotion_labels" in targets:
            losses["emo"] = self.ce_loss(
                predictions["emotion_logits"], 
                targets["emotion_labels"]
            )
        
        # Expression regression loss
        if "expression_coeffs" in targets:
            losses["expr"] = self.mse_loss(
                predictions["expression_coeffs"],
                targets["expression_coeffs"]
            )
        
        # Photometric loss (if reconstruction available)
        if "reconstruction" in predictions and "image" in targets:
            losses["photo"] = self.l1_loss(
                predictions["reconstruction"],
                targets["image"]
            )
        
        # Perceptual loss
        if "reconstruction" in predictions and "image" in targets and self.vgg is not None:
            losses["percep"] = self.perceptual_loss(
                predictions["reconstruction"],
                targets["image"]
            )
        
        # Temporal consistency loss (for video)
        if previous_frame is not None:
            # Apply temporal loss to embeddings and expression coefficients
            if "latent_embedding" in predictions and "latent_embedding" in previous_frame:
                losses["temp_embed"] = self.temporal_consistency_loss(
                    predictions["latent_embedding"],
                    previous_frame["latent_embedding"]
                )
            
            if "expression_coeffs" in predictions and "expression_coeffs" in previous_frame:
                losses["temp_expr"] = self.temporal_consistency_loss(
                    predictions["expression_coeffs"],
                    previous_frame["expression_coeffs"]
                )
            
            losses["temp"] = losses.get("temp_embed", 0) + losses.get("temp_expr", 0)
        
        # Total weighted loss
        total_loss = (
            self.config.lambda_emo * losses.get("emo", 0) +
            self.config.lambda_expr * losses.get("expr", 0) +
            self.config.lambda_photo * losses.get("photo", 0) +
            self.config.lambda_percep * losses.get("percep", 0) +
            self.config.lambda_temp * losses.get("temp", 0)
        )
        
        losses["total"] = total_loss
        
        return losses

# 3. Data Preparation and Augmentation
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mediapipe as mp
from retinaface import RetinaFace
from PIL import Image

class FacePreprocessor:
    """Face detection, alignment, and preprocessing"""
    def __init__(self, method: str = "mediapipe"):
        self.method = method
        
        if method == "mediapipe":
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        elif method == "retinaface":
            pass  # RetinaFace is called directly
        else:
            raise ValueError(f"Unknown face detection method: {method}")
    
    def detect_and_align(self, image: np.ndarray, target_size: int = 224) -> np.ndarray:
        """Detect face and align to target size"""
        h, w = image.shape[:2]
        
        if self.method == "mediapipe":
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = self.face_detection.process(rgb_image)
            if not results.detections:
                raise ValueError("No face detected")
            
            # Get bounding box
            bbox = results.detections[0].location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Get face landmarks for alignment
            mesh_results = self.face_mesh.process(rgb_image)
            if not mesh_results.multi_face_landmarks:
                raise ValueError("No face landmarks detected")
            
            # Simple alignment using eye positions
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            
            # Get eye landmarks (simplified)
            left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
            right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
            
            # Calculate angle for rotation
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Rotate image
            center = ((left_eye[0] + right_eye[0]) // 2, 
                     (left_eye[1] + right_eye[1]) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            
            # Crop face
            face_crop = rotated[y:y+height, x:x+width]
            
        else:  # retinaface
            faces = RetinaFace.detect_faces(image)
            if not faces:
                raise ValueError("No face detected")
            
            # Get first face
            face_data = list(faces.values())[0]
            facial_area = face_data["facial_area"]
            
            # Extract coordinates
            x1, y1, x2, y2 = facial_area
            face_crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_crop = cv2.resize(face_crop, (target_size, target_size))
        
        return face_crop

class EmotionDataset(Dataset):
    """Dataset for emotion recognition with augmentation"""
    def __init__(self, 
                 image_paths: list,
                 labels: Optional[list] = None,
                 config: EMOCAConfig = None,
                 is_train: bool = True):
        
        self.image_paths = image_paths
        self.labels = labels
        self.config = config or EMOCAConfig()
        self.is_train = is_train
        self.preprocessor = FacePreprocessor()
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),  # ±15° rotation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess face
        try:
            face = self.preprocessor.detect_and_align(image, self.config.img_size)
        except:
            # If face detection fails, use center crop
            face = cv2.resize(image, (self.config.img_size, self.config.img_size))
        
        # Apply transforms
        transform = self.train_transform if self.is_train else self.val_transform
        face_tensor = transform(face)
        
        # Prepare output
        sample = {"image": face_tensor}
        
        if self.labels is not None:
            if isinstance(self.labels[idx], dict):
                # Multi-task labels
                sample.update(self.labels[idx])
            else:
                # Single emotion label
                sample["emotion_labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return sample

# 4. Training Loop
class EMOCATrainer:
    """Trainer for EMOCA model"""
    def __init__(self, model: nn.Module, config: EMOCAConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lambda_emo * 1e-4,  # Learning rate scaled by lambda_emo
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        self.criterion = EMOCALoss(config)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            losses["total"].backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += losses["total"].item()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {losses['total'].item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}
                
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                
                total_loss += losses["total"].item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch + 1)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"emoca_best.pth")
                print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"emoca_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Loaded checkpoint from {path}")

# 5. Real-time Inference with WebSocket Server
import asyncio
import websockets
import json
import base64
from collections import deque
import threading

class RealTimeEmotionAnalyzer:
    """Real-time emotion analysis with temporal smoothing"""
    def __init__(self, model_path: str = None, config: EMOCAConfig = None):
        self.config = config or EMOCAConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = EMOCA(self.config).to(self.device)
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Face preprocessor
        self.preprocessor = FacePreprocessor()
        
        # Temporal smoothing
        self.buffer_size = 5
        self.expression_buffer = deque(maxlen=self.buffer_size)
        self.emotion_buffer = deque(maxlen=self.buffer_size)
        
        # Emotion labels (AffectNet 7 basic emotions)
        self.emotion_labels = [
            "Neutral", "Happy", "Sad", "Surprise", 
            "Fear", "Disgust", "Anger"
        ]
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for inference"""
        try:
            # Detect and align face
            face = self.preprocessor.detect_and_align(frame, self.config.img_size)
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            face_tensor = transform(face).unsqueeze(0).to(self.device)
            return face_tensor
        except Exception as e:
            print(f"Face preprocessing failed: {e}")
            return None
    
    def analyze_frame(self, frame: np.ndarray) -> dict:
        """Analyze single frame for emotions"""
        # Preprocess
        face_tensor = self.preprocess_frame(frame)
        if face_tensor is None:
            return None
        
        # Inference
        with torch.no_grad():
            output = self.model(face_tensor)
        
        # Get predictions
        emotion_probs = output["emotion_probs"][0].cpu().numpy()
        expression_coeffs = output["expression_coeffs"][0].cpu().numpy()
        arousal = output["arousal"][0].cpu().numpy()
        valence = output["valence"][0].cpu().numpy()
        
        # Apply temporal smoothing
        self.expression_buffer.append(expression_coeffs)
        self.emotion_buffer.append(emotion_probs)
        
        smoothed_expression = np.mean(self.expression_buffer, axis=0)
        smoothed_emotion = np.mean(self.emotion_buffer, axis=0)
        
        # Get dominant emotion
        dominant_idx = np.argmax(smoothed_emotion)
        dominant_emotion = self.emotion_labels[dominant_idx]
        
        # Prepare results
        results = {
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else time.time(),
            "emotions": {
                label: float(prob) for label, prob in zip(self.emotion_labels, smoothed_emotion)
            },
            "dominant_emotion": dominant_emotion,
            "expression_coefficients": smoothed_expression.tolist(),
            "arousal": float(arousal),
            "valence": float(valence),
            "latent_embedding": output["latent_embedding"][0].cpu().numpy().tolist()
        }
        
        return results

class WebSocketEmotionServer:
    """WebSocket server for streaming emotion parameters"""
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.analyzer = RealTimeEmotionAnalyzer()
        self.clients = set()
        self.is_running = False
        
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connection"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                # Parse message
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "frame")
                    
                    if message_type == "frame":
                        # Decode base64 image
                        image_data = base64.b64decode(data["image"])
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            await websocket.send(json.dumps({"error": "Invalid image"}))
                            continue
                        
                        # Analyze frame
                        results = self.analyzer.analyze_frame(frame)
                        
                        if results:
                            # Send results back
                            await websocket.send(json.dumps({
                                "type": "emotion_data",
                                "data": results
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": "Face not detected"
                            }))
                    
                    elif message_type == "command":
                        command = data.get("command", "")
                        if command == "start_stream":
                            # Start continuous streaming
                            pass
                        elif command == "stop_stream":
                            # Stop streaming
                            pass
                        
                except Exception as e:
                    print(f"Error processing message: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)
    
    async def broadcast_emotions(self, frame: np.ndarray):
        """Broadcast emotion data to all connected clients"""
        if not self.clients:
            return
        
        # Analyze frame
        results = self.analyzer.analyze_frame(frame)
        if not results:
            return
        
        # Prepare message
        message = json.dumps({
            "type": "emotion_update",
            "data": results
        })
        
        # Broadcast to all clients
        tasks = [client.send(message) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start_server(self):
        """Start WebSocket server"""
        self.is_running = True
        print(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    def stop_server(self):
        """Stop WebSocket server"""
        self.is_running = False
        print("Stopping WebSocket server")



import argparse
import time
from pathlib import Path

def train_model(args):
    """Train EMOCA model from scratch"""
    print("Initializing EMOCA training...")
    
    # Configuration
    config = EMOCAConfig(
        img_size=224,
        latent_dim=128,
        lambda_emo=args.lambda_emo,
        lambda_expr=args.lambda_expr,
        lambda_photo=args.lambda_photo,
        lambda_percep=args.lambda_percep,
        lambda_temp=args.lambda_temp
    )
    
    # Create model
    model = EMOCA(config)
    
    # Create trainer
    trainer = EMOCATrainer(model, config)
    
    # Load datasets
    # Note: You need to implement dataset loading based on your data structure
    print("Loading datasets...")
    # train_dataset = EmotionDataset(...)
    # val_dataset = EmotionDataset(...)
    
    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("Starting training...")
    # trainer.fit(train_loader, val_loader, epochs=args.epochs)

def start_websocket_server(args):
    """Start real-time emotion analysis server"""
    print("Starting EMOCA WebSocket server...")
    
    # Initialize server
    server = WebSocketEmotionServer(host=args.host, port=args.port)
    
    # Load model if provided
    if args.model_path:
        server.analyzer = RealTimeEmotionAnalyzer(model_path=args.model_path)
    
    # Start server
    asyncio.run(server.start_server())

def process_video(args):
    """Process video file and save emotion data"""
    print(f"Processing video: {args.video_path}")
    
    # Initialize analyzer
    analyzer = RealTimeEmotionAnalyzer(model_path=args.model_path)
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Prepare output
    output_data = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze every Nth frame based on fps
        if frame_count % int(fps / args.analysis_fps) == 0:
            results = analyzer.analyze_frame(frame)
            if results:
                output_data.append(results)
        
        frame_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    
    # Save results
    output_path = Path(args.video_path).stem + "_emotions.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="EMOCA: Emotion Capture System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train EMOCA model')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--lambda-emo', type=float, default=1.0)
    train_parser.add_argument('--lambda-expr', type=float, default=0.5)
    train_parser.add_argument('--lambda-photo', type=float, default=5.0)
    train_parser.add_argument('--lambda-percep', type=float, default=0.1)
    train_parser.add_argument('--lambda-temp', type=float, default=0.2)
    
    # Serve command (WebSocket server)
    serve_parser = subparsers.add_parser('serve', help='Start WebSocket server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0')
    serve_parser.add_argument('--port', type=int, default=8765)
    serve_parser.add_argument('--model-path', type=str, default=None)
    
    # Process video command
    video_parser = subparsers.add_parser('process-video', help='Process video file')
    video_parser.add_argument('video_path', type=str)
    video_parser.add_argument('--model-path', type=str, default=None)
    video_parser.add_argument('--analysis-fps', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'serve':
        start_websocket_server(args)
    elif args.command == 'process-video':
        process_video(args)
    else:
        print("Please specify a command: train, serve, or process-video")

if __name__ == "__main__":
    main()