"""
Real-Time Football Commentary Pipeline
Integrates object detection + ML commentary + voice synthesis
"""

import cv2
import numpy as np
import threading
import queue
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from object_detector import FootballObjectDetector
from commentary_generator import (
    ContextAwareCommentator,
    PlayerAction,
    OllamaCommentary,
    OpenAICommentary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineFrame:
    """Frame with all analysis results"""
    frame_id: int
    timestamp: datetime
    frame: np.ndarray
    detections: Dict
    actions: list
    commentary: Optional[str] = None
    processing_time: float = 0.0


class CommentaryPipeline:
    """
    Real-time pipeline for live football commentary
    
    Flow:
    1. Video frame input
    2. Object detection (players, ball, actions)
    3. Game state update
    4. ML commentary generation
    5. Text-to-speech synthesis
    6. Audio output
    """
    
    def __init__(self,
                 video_source: str,  # File path or 0 for webcam
                 generator_backend: str = "ollama",  # "ollama", "openai", "huggingface"
                 home_team: str = "Home",
                 away_team: str = "Away",
                 enable_visualization: bool = True):
        
        self.video_source = video_source
        self.home_team = home_team
        self.away_team = away_team
        self.enable_visualization = enable_visualization
        
        # Initialize detector
        logger.info("Initializing object detector...")
        self.detector = FootballObjectDetector(model_size="small")
        
        # Initialize commentary generator
        logger.info(f"Initializing {generator_backend} commentary generator...")
        if generator_backend == "openai":
            commentary_gen = OpenAICommentary()
        elif generator_backend == "huggingface":
            from commentary_generator import HuggingFaceCommentary
            commentary_gen = HuggingFaceCommentary()
        else:  # ollama (default)
            commentary_gen = OllamaCommentary()
        
        self.commentator = ContextAwareCommentator(
            generator=commentary_gen,
            home_team=home_team,
            away_team=away_team
        )
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=30)
        self.commentary_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        # Stats
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.last_commentary_time = 0
        self.min_commentary_interval = 2.0  # Min seconds between commentary
        
    def run(self):
        """Main pipeline execution"""
        
        # Start threads
        video_thread = threading.Thread(target=self._capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self._detect_and_analyze, daemon=True)
        commentary_thread = threading.Thread(target=self._generate_commentary, daemon=True)
        
        video_thread.start()
        detection_thread.start()
        commentary_thread.start()
        
        # Display results
        self._display_loop()
        
        # Cleanup
        self.stop_event.set()
        video_thread.join(timeout=5)
        detection_thread.join(timeout=5)
        commentary_thread.join(timeout=5)
    
    def _capture_frames(self):
        """Capture frames from video source"""
        logger.info(f"Opening video source: {self.video_source}")
        
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.video_source}")
            return
        
        frame_id = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (1280, 720))
            
            try:
                self.frame_queue.put(
                    PipelineFrame(
                        frame_id=frame_id,
                        timestamp=datetime.now(),
                        frame=frame,
                        detections=None,
                        actions=[]
                    ),
                    timeout=1
                )
                frame_id += 1
            except queue.Full:
                logger.warning("Frame queue full, dropping frame")
        
        cap.release()
        logger.info("Video capture thread stopped")
    
    def _detect_and_analyze(self):
        """Detect objects and analyze actions"""
        logger.info("Detection thread started")
        
        while not self.stop_event.is_set():
            try:
                pipeline_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            start_time = time.time()
            
            # Object detection
            detections = self.detector.detect_frame(
                pipeline_frame.frame,
                conf_threshold=0.5
            )
            
            pipeline_frame.detections = detections
            pipeline_frame.actions = self._extract_meaningful_actions(detections)
            pipeline_frame.processing_time = time.time() - start_time
            
            # Update game state
            self._update_game_state(detections)
            
            try:
                self.commentary_queue.put(pipeline_frame, timeout=1)
            except queue.Full:
                logger.warning("Commentary queue full")
            
            self.frame_count += 1
            self.total_processing_time += pipeline_frame.processing_time
            
            if self.frame_count % 30 == 0:
                avg_time = self.total_processing_time / self.frame_count
                logger.info(f"Processed {self.frame_count} frames (avg {avg_time*1000:.1f}ms)")
    
    def _extract_meaningful_actions(self, detections: Dict) -> list:
        """
        Filter actions to only meaningful events
        (avoid generating commentary for every frame)
        """
        meaningful_actions = []
        
        for action in detections.get("actions", []):
            # Only action-types worth commenting on
            if action["action"] in ["shooting", "dribbling", "defending"]:
                if action["confidence"] > 0.7:
                    meaningful_actions.append(action)
        
        return meaningful_actions
    
    def _update_game_state(self, detections: Dict):
        """Update game state based on detections"""
        
        # Update possession based on who has the ball
        if detections["ball"]:
            closest_player = self._find_closest_player(
                detections["ball"]["center"],
                detections["players"]
            )
            
            if closest_player:
                self.commentator.update_state({
                    "possession": closest_player["team"]
                })
    
    def _find_closest_player(self, ball_pos: Tuple, players: list) -> Optional[Dict]:
        """Find player closest to ball"""
        if not players:
            return None
        
        min_distance = float('inf')
        closest = None
        
        for player in players:
            distance = np.sqrt(
                (player["center"][0] - ball_pos[0])**2 +
                (player["center"][1] - ball_pos[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest = player
        
        return closest if min_distance < 100 else None
    
    def _generate_commentary(self):
        """Generate commentary for actions"""
        logger.info("Commentary generation thread started")
        
        while not self.stop_event.is_set():
            try:
                pipeline_frame = self.commentary_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Only generate if meaningful actions and enough time has passed
            if pipeline_frame.actions:
                current_time = time.time()
                
                if current_time - self.last_commentary_time >= self.min_commentary_interval:
                    
                    # Generate commentary for the most confident action
                    best_action = max(
                        pipeline_frame.actions,
                        key=lambda a: a["confidence"]
                    )
                    
                    # Find player
                    player = self._get_player_by_id(
                        best_action["player_id"],
                        pipeline_frame.detections
                    )
                    
                    if player:
                        # Create action object
                        player_action = PlayerAction(
                            player_name=f"{player['team']}_Player_{player['id']}",
                            player_id=player['id'],
                            team=player['team'],
                            action=best_action["action"],
                            action_confidence=best_action["confidence"],
                            nearby_players=self._get_nearby_players(
                                player,
                                pipeline_frame.detections["players"]
                            ),
                            ball_position={
                                "x": pipeline_frame.detections["ball"]["center"][0],
                                "y": pipeline_frame.detections["ball"]["center"][1],
                                "z": 0  # 2D video
                            },
                            field_zone=self._get_field_zone(player)
                        )
                        
                        # Generate commentary
                        try:
                            commentary = self.commentator.generate_commentary(player_action)
                            
                            # Add to pipeline frame
                            pipeline_frame.commentary = commentary
                            
                            logger.info(f"✓ {commentary}")
                            
                            self.last_commentary_time = current_time
                            
                        except Exception as e:
                            logger.error(f"Commentary generation error: {e}")
    
    def _get_player_by_id(self, player_id: int, detections: Dict) -> Optional[Dict]:
        """Find player by ID in detections"""
        for player in detections.get("players", []):
            if player.get("id") == player_id:
                return player
        return None
    
    def _get_nearby_players(self, player: Dict, all_players: list) -> list:
        """Get players near a specific player"""
        nearby = []
        threshold = 150  # pixels
        
        for other in all_players:
            if other["id"] != player["id"]:
                distance = np.sqrt(
                    (player["center"][0] - other["center"][0])**2 +
                    (player["center"][1] - other["center"][1])**2
                )
                
                if distance < threshold:
                    nearby.append({
                        "player_id": other["id"],
                        "distance": int(distance),
                        "team": other["team"]
                    })
        
        return nearby
    
    def _get_field_zone(self, player: Dict, frame_width: int = 1280) -> str:
        """Determine which third of field the player is in"""
        x = player["center"][0]
        
        if x < frame_width / 3:
            return "defensive_third"
        elif x < 2 * frame_width / 3:
            return "midfield"
        else:
            return "attacking_third"
    
    def _display_loop(self):
        """Display visualization and results"""
        logger.info("Starting display loop")
        
        frame_count = 0
        while not self.stop_event.is_set():
            
            # Try to get latest frame
            frames_to_skip = max(0, self.frame_queue.qsize() - 1)
            for _ in range(frames_to_skip):
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            try:
                pipeline_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if pipeline_frame.detections:
                display_frame = self.detector.visualize(
                    pipeline_frame.frame,
                    pipeline_frame.detections
                )
                
                # Add commentary to frame
                if pipeline_frame.commentary:
                    cv2.putText(
                        display_frame,
                        pipeline_frame.commentary[:80],
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Add stats
                fps = 1.0 / max(pipeline_frame.processing_time, 0.001)
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                if self.enable_visualization:
                    cv2.imshow("Football Commentary Pipeline", display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested exit")
                        self.stop_event.set()
                        break
                
                frame_count += 1
        
        cv2.destroyAllWindows()
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "frames_processed": self.frame_count,
            "avg_detection_time_ms": (self.total_processing_time / max(self.frame_count, 1)) * 1000,
            "commentary_stats": self.commentator.get_commentary_stats(),
            "frame_queue_size": self.frame_queue.qsize(),
            "commentary_queue_size": self.commentary_queue.qsize()
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Use video file or webcam
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 for webcam
    
    pipeline = CommentaryPipeline(
        video_source=video_source,
        generator_backend="ollama",  # Change to "openai" or "huggingface"
        home_team="Manchester United",
        away_team="Liverpool",
        enable_visualization=True
    )
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    finally:
        stats = pipeline.get_statistics()
        logger.info(f"Pipeline statistics: {stats}")
