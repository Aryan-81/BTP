"""
Comprehensive 3D Face Reconstruction from Single Image
A complete implementation with multiple methods and visualization
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL.Image
import os
import sys
import json
from typing import Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
import time
from scipy import ndimage
from scipy.spatial import Delaunay
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FaceReconstructionConfig:
    """Configuration for face reconstruction parameters"""
    image_size: Tuple[int, int] = (512, 512)
    mesh_resolution: int = 256
    depth_scale: float = 0.2
    smoothing_iterations: int = 5
    landmark_confidence: float = 0.5
    output_format: str = "obj"  # obj, gltf, ply
    enable_texturing: bool = True
    quality: str = "high"  # low, medium, high 

class FaceDetector:
    """Enhanced face detection with multiple methods"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces.tolist()
    
    def detect_mediapipe(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces and landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y, landmark.z))
                
                # Calculate bounding box
                xs = [l[0] for l in landmarks]
                ys = [l[1] for l in landmarks]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                detections.append({
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'landmarks': landmarks,
                    'confidence': 0.9  
                })
        
        return detections
    
    def detect_faces(self, image: np.ndarray, method: str = "mediapipe") -> List[Dict[str, Any]]:
        """Detect faces using specified method"""
        if method == "mediapipe":
            return self.detect_mediapipe(image)
        elif method == "haar":
            faces = self.detect_haar(image)
            return [{'bbox': face, 'landmarks': [], 'confidence': 0.7} for face in faces]
        else:
            raise ValueError(f"Unknown detection method: {method}")

class DepthEstimator:
    """Advanced depth estimation for facial geometry"""
    
    def __init__(self):
        self.face_landmark_model = self._load_landmark_model()
    
    def _load_landmark_model(self):
        """Placeholder for landmark-based depth model"""
        return None
    
    def estimate_depth_from_landmarks(self, landmarks: List[Tuple[int, int, float]], 
                                    image_shape: Tuple[int, int]) -> np.ndarray:
        """Estimate depth map from facial landmarks"""
        height, width = image_shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        if not landmarks:
            return self._estimate_depth_from_intensity(depth_map)
        
        # Convert landmarks to numpy array
        landmarks_array = np.array(landmarks)
        
        # Create depth basis from landmark z-coordinates
        for i, (x, y, z) in enumerate(landmarks_array):
            if 0 <= x < width and 0 <= y < height:
                # Create Gaussian blob at landmark position
                sigma = max(width, height) // 50
                depth_map = self._add_gaussian_blob(depth_map, x, y, z, sigma)
        
        # Smooth the depth map
        depth_map = ndimage.gaussian_filter(depth_map, sigma=3)
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_map
    
    def _add_gaussian_blob(self, depth_map: np.ndarray, x: int, y: int, 
                          z: float, sigma: int) -> np.ndarray:
        """Add Gaussian blob to depth map at specified position"""
        height, width = depth_map.shape
        y_grid, x_grid = np.ogrid[:height, :width]
        
        # Calculate Gaussian
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        gaussian = gaussian * z
        
        # Add to depth map
        depth_map += gaussian
        return depth_map
    
    def _estimate_depth_from_intensity(self, image: np.ndarray) -> np.ndarray:
        """Fallback depth estimation from image intensity"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        depth_map = cv2.Laplacian(gray, cv2.CV_64F)
        depth_map = np.absolute(depth_map)
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        
        # Normalize
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        return depth_map.astype(np.float32)

class MeshGenerator:
    """3D mesh generation from 2D image and depth map"""
    
    def __init__(self, config: FaceReconstructionConfig):
        self.config = config
    
    def create_textured_mesh(self, image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """Create textured 3D mesh from image and depth map"""
        height, width = image.shape[:2]
        
        # Generate vertices with colors
        vertices = []
        vertex_colors = []
        
        for y in range(height):
            for x in range(width):
                # Normalize coordinates
                nx = x / width - 0.5
                ny = 0.5 - y / height
                nz = depth_map[y, x] * self.config.depth_scale
                
                vertices.append([nx, ny, nz])
                
                # Store color (BGR to RGB)
                if len(image.shape) == 3:
                    r, g, b = image[y, x][2] / 255.0, image[y, x][1] / 255.0, image[y, x][0] / 255.0
                else:
                    r = g = b = image[y, x] / 255.0
                vertex_colors.append([r, g, b])
        
        # Generate faces
        faces = self._generate_faces(height, width)
        
        # Smooth mesh
        vertices = self._smooth_mesh(vertices, faces, height, width)
        
        return {
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'colors': np.array(vertex_colors)
        }
    
    def _generate_faces(self, height: int, width: int) -> List[List[int]]:
        """Generate triangular faces for the mesh"""
        faces = []
        
        for y in range(height - 1):
            for x in range(width - 1):
                # Calculate vertex indices
                v1 = y * width + x
                v2 = y * width + x + 1
                v3 = (y + 1) * width + x
                v4 = (y + 1) * width + x + 1
                
                # Create two triangles for each quad
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        return faces
    
    def _smooth_mesh(self, vertices: List[List[float]], faces: List[List[int]], 
                    height: int, width: int, iterations: int = 3) -> List[List[float]]:
        """Apply Laplacian smoothing to the mesh"""
        if iterations == 0:
            return vertices
        
        vertices_array = np.array(vertices)
        smoothed_vertices = vertices_array.copy()
        
        # Build vertex adjacency
        adjacency = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[face[i]].append(face[j])
        
        # Apply Laplacian smoothing
        for _ in range(iterations):
            for i in range(len(vertices)):
                if len(adjacency[i]) > 0:
                    # Average neighboring vertices (excluding boundary vertices)
                    neighbors = vertices_array[adjacency[i]]
                    smoothed_vertices[i] = 0.5 * vertices_array[i] + 0.5 * neighbors.mean(axis=0)
        
        return smoothed_vertices.tolist()

class NeuralFaceReconstructor(nn.Module):
    """Neural network based 3D face reconstruction"""
    
    def __init__(self, input_channels=3, output_channels=1):
        super(NeuralFaceReconstructor, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FileExporter:
    """Export 3D models in various formats"""
    
    @staticmethod
    def export_obj(vertices: np.ndarray, faces: np.ndarray, 
                  colors: np.ndarray, filename: str, texture_image: np.ndarray = None):
        """Export mesh as OBJ file with optional texture"""
        with open(filename, 'w') as f:
            f.write("# 3D Face Model Generated from Single Image\n")
            f.write("# Vertices: {}\n".format(len(vertices)))
            f.write("# Faces: {}\n".format(len(faces)))
            
            # Write vertices
            for i, vertex in enumerate(vertices):
                if colors is not None and i < len(colors):
                    r, g, b = colors[i]
                    f.write("v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        vertex[0], vertex[1], vertex[2], r, g, b))
                else:
                    f.write("v {:.6f} {:.6f} {:.6f}\n".format(
                        vertex[0], vertex[1], vertex[2]))
            
            # Write faces
            for face in faces:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
        
        # Save texture if provided
        if texture_image is not None:
            texture_filename = filename.replace('.obj', '_texture.png')
            cv2.imwrite(texture_filename, texture_image)
            logger.info(f"Texture saved as: {texture_filename}")
    
    @staticmethod
    def export_gltf(vertices: np.ndarray, faces: np.ndarray, 
                   colors: np.ndarray, filename: str):
        """Export mesh as GLTF file (simplified)"""
        gltf_data = {
            "asset": {
                "version": "2.0",
                "generator": "Face3DReconstructor"
            },
            "scenes": [{
                "nodes": [0]
            }],
            "nodes": [{
                "mesh": 0
            }],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "COLOR_0": 1
                    },
                    "indices": 2,
                    "mode": 4
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(vertices),
                    "type": "VEC3",
                    "min": vertices.min(axis=0).tolist(),
                    "max": vertices.max(axis=0).tolist()
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": len(colors),
                    "type": "VEC3"
                },
                {
                    "bufferView": 2,
                    "componentType": 5123,
                    "count": len(faces) * 3,
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(vertices) * 4 * 3
                },
                {
                    "buffer": 0,
                    "byteOffset": len(vertices) * 4 * 3,
                    "byteLength": len(colors) * 4 * 3
                },
                {
                    "buffer": 0,
                    "byteOffset": len(vertices) * 4 * 3 + len(colors) * 4 * 3,
                    "byteLength": len(faces) * 3 * 2
                }
            ],
            "buffers": [{
                "byteLength": len(vertices) * 4 * 3 + len(colors) * 4 * 3 + len(faces) * 3 * 2,
                "uri": "data:application/octet-stream;base64," + 
                       FileExporter._encode_binary_data(vertices, colors, faces)
            }]
        }
        
        with open(filename, 'w') as f:
            json.dump(gltf_data, f, indent=2)
    
    @staticmethod
    def _encode_binary_data(vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> str:
        """Encode binary data for GLTF (simplified)"""
        # This is a simplified implementation
        vertex_data = vertices.astype(np.float32).tobytes()
        color_data = colors.astype(np.float32).tobytes()
        face_data = faces.astype(np.uint16).tobytes()
        
        combined_data = vertex_data + color_data + face_data
        return combined_data.hex()

class VisualizationEngine:
    """Advanced visualization for 3D face models"""
    
    @staticmethod
    def create_comparison_visualization(original_image: np.ndarray, 
                                      depth_map: np.ndarray, 
                                      mesh_data: Dict[str, Any],
                                      output_path: str = None):
        """Create comprehensive visualization of results"""
        fig = plt.figure(figsize=(20, 10))
        
        # Original image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Face Image')
        ax1.axis('off')
        
        # Depth map
        ax2 = fig.add_subplot(2, 3, 2)
        depth_display = depth_map.copy()
        im = ax2.imshow(depth_display, cmap='viridis')
        ax2.set_title('Estimated Depth Map')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3D mesh (multiple views)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        colors = mesh_data['colors']
        
        # Front view
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        VisualizationEngine._plot_mesh_3d(ax3, vertices, faces, colors, elevation=0, azimuth=0)
        ax3.set_title('3D Model - Front View')
        
        # Side view
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        VisualizationEngine._plot_mesh_3d(ax4, vertices, faces, colors, elevation=0, azimuth=90)
        ax4.set_title('3D Model - Side View')
        
        # Top view
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        VisualizationEngine._plot_mesh_3d(ax5, vertices, faces, colors, elevation=90, azimuth=0)
        ax5.set_title('3D Model - Top View')
        
        # Wireframe
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        VisualizationEngine._plot_wireframe(ax6, vertices, faces)
        ax6.set_title('3D Model - Wireframe')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved as: {output_path}")
        
        plt.show()
    
    @staticmethod
    def _plot_mesh_3d(ax, vertices: np.ndarray, faces: np.ndarray, 
                     colors: np.ndarray, elevation: float = 0, azimuth: float = 0):
        """Plot 3D mesh with colors"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create collection of polygons
        polygons = []
        face_colors = []
        
        for face in faces:
            polygon = vertices[face]
            polygons.append(polygon)
            
            # Average vertex colors for face color
            if colors is not None:
                face_color = np.mean(colors[face], axis=0)
                face_colors.append(face_color)
        
        collection = Poly3DCollection(polygons, facecolors=face_colors, 
                                    edgecolor='none', alpha=0.8)
        ax.add_collection3d(collection)
        
        # Set axis limits
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(), 
                             vertices[:,1].max()-vertices[:,1].min(), 
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    @staticmethod
    def _plot_wireframe(ax, vertices: np.ndarray, faces: np.ndarray):
        """Plot wireframe of the mesh"""
        for face in faces:
            for i in range(3):
                start = vertices[face[i]]
                end = vertices[face[(i + 1) % 3]]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                       'b-', linewidth=0.5, alpha=0.6)
        
        # Set equal aspect ratio
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(), 
                             vertices[:,1].max()-vertices[:,1].min(), 
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

class Face3DReconstructor:
    """Main class for 3D face reconstruction from single image"""
    
    def __init__(self, config: FaceReconstructionConfig = None):
        self.config = config or FaceReconstructionConfig()
        self.face_detector = FaceDetector()
        self.depth_estimator = DepthEstimator()
        self.mesh_generator = MeshGenerator(self.config)
        self.file_exporter = FileExporter()
        self.visualization_engine = VisualizationEngine()
        
        # Initialize neural model (if GPU available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_model = NeuralFaceReconstructor().to(self.device)
        logger.info(f"Using device: {self.device}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and validate input image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        logger.info(f"Loaded image with shape: {image.shape}")
        return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for face reconstruction"""
        # Resize to configured size
        processed = cv2.resize(image, self.config.image_size)
        
        # Enhance image quality
        processed = self._enhance_image(processed)
        
        return processed
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better reconstruction"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def reconstruct_3d_face(self, image_path: str, output_base_path: str = None,
                           enable_visualization: bool = True) -> Dict[str, Any]:
        """
        Main method to reconstruct 3D face from single image
        
        Args:
            image_path: Path to input face image
            output_base_path: Base path for output files (without extension)
            enable_visualization: Whether to generate visualization
        
        Returns:
            Dictionary containing reconstruction results
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            logger.info("Loading and preprocessing image...")
            original_image = self.load_image(image_path)
            processed_image = self.preprocess_image(original_image)
            
            # Detect face
            logger.info("Detecting face...")
            face_detections = self.face_detector.detect_faces(processed_image, method="mediapipe")
            
            if not face_detections:
                raise ValueError("No face detected in the image")
            
            # Use first detection
            face_data = face_detections[0]
            logger.info(f"Face detected with confidence: {face_data.get('confidence', 'N/A')}")
            
            # Extract face region
            x, y, w, h = face_data['bbox']
            face_region = processed_image[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, self.config.image_size)
            
            # Estimate depth map
            logger.info("Estimating depth map...")
            landmarks = face_data.get('landmarks', [])
            depth_map = self.depth_estimator.estimate_depth_from_landmarks(
                landmarks, face_region.shape
            )
            
            # Generate 3D mesh
            logger.info("Generating 3D mesh...")
            mesh_data = self.mesh_generator.create_textured_mesh(face_region, depth_map)
            
            # Prepare output
            results = {
                'success': True,
                'mesh_data': mesh_data,
                'depth_map': depth_map,
                'face_region': face_region,
                'original_image': original_image,
                'face_detection': face_data,
                'processing_time': time.time() - start_time
            }
            
            # Export results if output path provided
            if output_base_path:
                self._export_results(results, output_base_path)
            
            # Generate visualization
            if enable_visualization:
                viz_path = f"{output_base_path}_visualization.png" if output_base_path else None
                self.visualization_engine.create_comparison_visualization(
                    face_region, depth_map, mesh_data, viz_path
                )
            
            logger.info(f"3D face reconstruction completed in {results['processing_time']:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"3D face reconstruction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _export_results(self, results: Dict[str, Any], output_base_path: str):
        """Export reconstruction results to files"""
        mesh_data = results['mesh_data']
        
        # Export OBJ file
        obj_path = f"{output_base_path}.obj"
        self.file_exporter.export_obj(
            mesh_data['vertices'],
            mesh_data['faces'],
            mesh_data['colors'],
            obj_path,
            results['face_region']
        )
        logger.info(f"3D model exported as: {obj_path}")
        
        # Export depth map
        depth_path = f"{output_base_path}_depth.png"
        depth_display = (results['depth_map'] * 255).astype(np.uint8)
        cv2.imwrite(depth_path, depth_display)
        logger.info(f"Depth map exported as: {depth_path}")
        
        # Export face region
        face_path = f"{output_base_path}_face.png"
        cv2.imwrite(face_path, results['face_region'])
        logger.info(f"Face region exported as: {face_path}")
        
        # Export metadata
        meta_path = f"{output_base_path}_metadata.json"
        metadata = {
            'vertices_count': len(mesh_data['vertices']),
            'faces_count': len(mesh_data['faces']),
            'processing_time': results['processing_time'],
            'image_size': results['face_region'].shape,
            'face_detection_confidence': results['face_detection'].get('confidence'),
            'landmarks_count': len(results['face_detection'].get('landmarks', []))
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata exported as: {meta_path}")

def main():
    """Main function with example usage"""
    # Configuration
    config = FaceReconstructionConfig(
        image_size=(512, 512),
        mesh_resolution=256,
        depth_scale=0.3,
        quality="high",
        enable_texturing=True
    )
    
    # Initialize reconstructor
    reconstructor = Face3DReconstructor(config)
    
    # Example usage
    image_path = "sample.png"  
    output_path = "output"
    
    if os.path.exists(image_path):
        # Perform 3D reconstruction
        results = reconstructor.reconstruct_3d_face(
            image_path=image_path,
            output_base_path=output_path,
            enable_visualization=True
        )
        
        if results['success']:
            print(f"\n3D Face Reconstruction Successful!")
            print(f"Statistics:")
            print(f"   - Vertices: {len(results['mesh_data']['vertices'])}")
            print(f"   - Faces: {len(results['mesh_data']['faces'])}")
            print(f"   - Processing time: {results['processing_time']:.2f} seconds")
            print(f"   - Output files:")
            print(f"       • {output_path}.obj (3D model)")
            print(f"       • {output_path}_depth.png (Depth map)")
            print(f"       • {output_path}_face.png (Processed face)")
            print(f"       • {output_path}_visualization.png (Results overview)")
        else:
            print(f"Reconstruction failed: {results['error']}")
    else:
        print(f"Example image not found: {image_path}")
        print("Please provide a valid image path and run again.")
        
        # Create a sample demonstration with a synthetic face
        print("\n Creating synthetic demonstration...")
        demo_image = create_demo_face()
        cv2.imwrite("demo_face.jpg", demo_image)
        print("Demo face created: demo_face.jpg")
        
        # Run reconstruction on demo face
        results = reconstructor.reconstruct_3d_face(
            image_path="demo_face.jpg",
            output_base_path="demo_output",
            enable_visualization=True
        )

def create_demo_face() -> np.ndarray:
    """Create a synthetic face image for demonstration"""
    size = 512
    image = np.ones((size, size, 3), dtype=np.uint8) * 255  # White background
    
    # Face oval
    center = (size // 2, size // 2)
    axes = (size // 3, size // 4)
    cv2.ellipse(image, center, axes, 0, 0, 360, (255, 220, 200), -1)
    
    # Eyes
    eye_y = center[1] - size // 10
    left_eye_center = (center[0] - size // 6, eye_y)
    right_eye_center = (center[0] + size // 6, eye_y)
    cv2.ellipse(image, left_eye_center, (size // 15, size // 20), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(image, right_eye_center, (size // 15, size // 20), 0, 0, 360, (0, 0, 0), -1)
    
    # Nose
    nose_top = (center[0], eye_y + size // 10)
    nose_bottom = (center[0], center[1] + size // 10)
    cv2.line(image, nose_top, nose_bottom, (0, 0, 0), 3)
    
    # Mouth
    mouth_center = (center[0], center[1] + size // 5)
    mouth_axes = (size // 8, size // 20)
    cv2.ellipse(image, mouth_center, mouth_axes, 0, 0, 180, (0, 0, 0), 3)
    
    return image

if __name__ == "__main__":
    main()