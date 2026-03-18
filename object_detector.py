"""
Object Detection Module - Uses YOLO for real-time player and ball detection
Detects: Players, ball, referee, intended action
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
from typing import Dict, List, Tuple, Optional
import logging
from event_classifier import EventClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballObjectDetector:
    """
    Real-time object detection for football/soccer video streams.
    Uses YOLOv8 for multi-object detection (players, ball, etc.)
    """
    
    def __init__(self, model_size: str = "medium", custom_weights: Optional[str] = None):
        """
        Initialize YOLO detector
        
        Args:
            model_size: "nano", "small", "medium", "large" or "xlarge"
                       Larger = more accurate, slower
            custom_weights: Path to a custom-trained .pt weights file.
                           If provided, this takes priority over model_size.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model: custom weights if available, else generic pretrained
        if custom_weights and os.path.exists(custom_weights):
            logger.info(f"Loading CUSTOM football weights: {custom_weights}")
            self.model = YOLO(custom_weights)
            self.using_custom_model = True
        else:
            if custom_weights:
                logger.warning(f"Custom weights not found at '{custom_weights}', falling back to generic YOLOv8{model_size}")
            self.model = YOLO(f"yolov8{model_size}.pt")
            self.using_custom_model = False
        self.model.to(self.device)
        
        # Class mappings
        # Generic COCO classes (used when no custom model)
        self.coco_class_names = {
            0: "person",
            32: "sports ball"
        }
        # Custom football classes (set by Roboflow training)
        # These map to the classes defined in the training data.yaml
        self.football_class_names = {
            0: "ball",
            1: "goalkeeper",
            2: "player",
            3: "referee",
        }
        
        # Tracking for players across frames
        self.player_tracks = {}
        self.frame_count = 0
        
        # Tier 2: Event classifier
        self.event_classifier = EventClassifier()
        
    def detect_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        Detect objects in a single frame
        
        Returns:
            {
                "players": [{"id": int, "bbox": (x, y, w, h), "confidence": float, "team": str}],
                "ball": {"bbox": (x, y, w, h), "confidence": float},
                "actions": [{"player_id": int, "action": str, "confidence": float}]
            }
        """
        self.frame_count += 1
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = {
            "players": [],
            "ball": None,
            "actions": [],
            "frame_count": self.frame_count
        }
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Determine what this detection is
                is_ball = False
                is_person = False
                
                if self.using_custom_model:
                    # Custom model: use football-specific classes
                    cls_name = self.football_class_names.get(cls_id, "unknown")
                    is_ball = (cls_name == "ball")
                    is_person = (cls_name in ["player", "goalkeeper", "referee"])
                else:
                    # Generic COCO model
                    is_ball = (cls_id == 32)
                    is_person = (cls_id == 0)
                
                if is_ball and confidence > conf_threshold:
                    detections["ball"] = {
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "confidence": confidence
                    }
                
                elif is_person:
                    player = {
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "confidence": confidence,
                        "height": y2 - y1,
                        "width": x2 - x1
                    }
                    
                    # Assign to team based on jersey color or position
                    team = self._classify_team(frame, x1, y1, x2, y2)
                    player["team"] = team
                    
                    # Assign tracking ID
                    player_id = self._track_player(player)
                    player["id"] = player_id
                    
                    detections["players"].append(player)
        
        # Infer actions from player/ball positions (Tier 1 heuristics)
        detections["actions"] = self._infer_actions(detections)
        
        # Tier 2: Classify events from the detection sequence
        event = self.event_classifier.update(detections)
        if event:
            detections["event"] = event
            logger.info(f"⚽ EVENT DETECTED: {event['event']} (conf: {event['confidence']:.2f}) — {event['description']}")
        
        return detections
    
    def _classify_team(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
        """
        Classify player into team based on jersey color
        
        Returns: "home", "away", or "unknown"
        """
        try:
            # Extract jersey region (upper part of bounding box)
            jersey_region = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
            
            if jersey_region.size == 0:
                return "unknown"
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Count dominant colors
            # Red team (home): Red jerseys
            # Blue team (away): Blue jerseys
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            red_count = cv2.countNonZero(red_mask)
            blue_count = cv2.countNonZero(blue_mask)
            white_count = cv2.countNonZero(white_mask)
            
            # Determine team by dominant color
            if red_count > blue_count and red_count > white_count:
                return "home"
            elif blue_count > red_count and blue_count > white_count:
                return "away"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Team classification error: {e}")
            return "unknown"
    
    def _track_player(self, player: Dict) -> int:
        """
        Simple player tracking - assign IDs across frames
        Uses centroid distance for association
        """
        center = player["center"]
        max_distance = 50  # pixels
        
        best_match_id = None
        best_distance = max_distance
        
        # Find closest existing player track
        for player_id, track_history in self.player_tracks.items():
            if track_history:
                last_center = track_history[-1]["center"]
                distance = np.sqrt(
                    (center[0] - last_center[0])**2 + 
                    (center[1] - last_center[1])**2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = player_id
        
        # Assign ID
        if best_match_id is not None:
            player_id = best_match_id
            self.player_tracks[player_id].append(player)
        else:
            player_id = max(self.player_tracks.keys()) + 1 if self.player_tracks else 0
            self.player_tracks[player_id] = [player]
        
        # Keep only last 30 frames of history per player
        if len(self.player_tracks[player_id]) > 30:
            self.player_tracks[player_id] = self.player_tracks[player_id][-30:]
        
        return player_id
    
    def _infer_actions(self, detections: Dict) -> List[Dict]:
        """
        Infer player actions based on spatial relationships
        
        Actions: "shooting", "passing", "dribbling", "defending",
                "receiving", "heading", "sliding", "celebrating"
        """
        actions = []
        
        if not detections["players"] or detections["ball"] is None:
            return actions
        
        ball_pos = detections["ball"]["center"]
        
        for player in detections["players"]:
            player_pos = player["center"]
            distance_to_ball = np.sqrt(
                (player_pos[0] - ball_pos[0])**2 + 
                (player_pos[1] - ball_pos[1])**2
            )
            
            # Player within 50px of ball = interacting
            if distance_to_ball < 50:
                
                # Check if player has ball (very close, ~30px)
                if distance_to_ball < 30:
                    action = self._detect_ball_interaction(player, detections)
                    if action:
                        actions.append({
                            "player_id": player["id"],
                            "action": action,
                            "confidence": 0.8
                        })
        
        return actions
    
    def _detect_ball_interaction(self, player: Dict, detections: Dict) -> Optional[str]:
        """
        More detailed action detection when player has ball
        Uses player movement history and trajectory
        """
        player_id = player["id"]
        
        if player_id not in self.player_tracks or len(self.player_tracks[player_id]) < 3:
            return "receiving"
        
        # Get last 3 positions
        history = self.player_tracks[player_id][-3:]
        positions = [h["center"] for h in history]
        
        # Calculate velocity
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # High velocity toward goal = shooting
        if velocity > 15:
            ball = detections["ball"]["center"]
            frame_width = 1920  # Adjust to your frame size
            
            # Moving toward bottom of frame (goal direction) = shooting
            if dy > abs(dx):
                return "shooting"
            return "dribbling"
        
        return "receiving"
    
    def get_player_by_name(self, frame: np.ndarray, detections: Dict) -> Dict[int, str]:
        """
        ADVANCED: Use face/uniform recognition to nameplate players
        This is where you'd integrate with a face detection + recognition model
        
        Returns mapping of player_id -> player_name
        """
        # This requires a separate model trained on your team's rosters
        # For now, return placeholder
        player_names = {}
        
        for player in detections["players"]:
            # TODO: Integrate with face recognition model
            player_names[player["id"]] = f"{player['team'].upper()}_Player_{player['id']}"
        
        return player_names
    
    def visualize(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Overlay detections on frame for debugging"""
        frame_vis = frame.copy()
        
        # Draw players
        for player in detections["players"]:
            x, y, w, h = player["bbox"]
            team_color = (0, 255, 0) if player["team"] == "home" else (255, 0, 0)
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), team_color, 2)
            cv2.putText(
                frame_vis, 
                f"ID:{player['id']} {player['team'][:1]}", 
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                team_color, 
                2
            )
        
        # Draw ball
        if detections["ball"]:
            cx, cy = detections["ball"]["center"]
            cv2.circle(frame_vis, (cx, cy), 8, (0, 255, 255), 2)
        
        # Draw actions
        for action in detections["actions"]:
            logger.info(f"Action: Player {action['player_id']} - {action['action']}")
        
        return frame_vis


# Example usage
if __name__ == "__main__":
    detector = FootballObjectDetector(
        model_size="small",
        custom_weights="football_yolov8s_best.pt"
    )
    
    # Process video
    cap = cv2.VideoCapture("test_match.mp4")
    
    # Speed optimization parameters
    FRAME_SKIP = 2  # Process every Nth frame
    PROCESS_WIDTH = 854  # 480p resolution for faster YOLO
    PROCESS_HEIGHT = 480
    
    frame_count = 0
    last_detections = {"players": [], "ball": None, "actions": []}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # Process every Nth frame, but display every frame
        if frame_count % FRAME_SKIP == 0:
            detections = detector.detect_frame(frame, conf_threshold=0.5)
            last_detections = detections
        else:
             # Just increment frame counts or time on the visualizer without detection overhead
             detections = last_detections
             
        frame_vis = detector.visualize(frame, detections)
        cv2.imshow("Detection", frame_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} frames")
    
    cap.release()
    cv2.destroyAllWindows()
