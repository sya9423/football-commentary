"""
Complete Football Commentary System - Integrated
Combines real-time detection + ML commentary + voice synthesis
"""

import cv2
import numpy as np
import threading
import queue
import logging
import asyncio
from typing import Optional, Union
from datetime import datetime
import pygame

from object_detector import FootballObjectDetector
from commentary_generator import (
    ContextAwareCommentator,
    PlayerAction,
    OllamaCommentary,
    OpenAICommentary
)
from commentator import VoiceEngine, generate_tts_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullCommentarySystem:
    """
    Complete end-to-end system:
    1. Real-time video analysis
    2. Player/action detection
    3. ML commentary generation
    4. Voice synthesis and playback
    """
    
    def __init__(self,
                 video_source: Union[str, int] = 0,
                 generator_backend: str = "ollama",
                 home_team: str = "Home",
                 away_team: str = "Away",
                 enable_voice: bool = True,
                 voice_model: str = "en-GB-ThomasNeural",
                 custom_weights: Optional[str] = None):
        
        logger.info("=" * 60)
        logger.info("FOOTBALL COMMENTARY SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        self.video_source = video_source
        self.home_team = home_team
        self.away_team = away_team
        self.enable_voice = enable_voice
        
        # Initialize detector (with custom weights if available)
        logger.info("▸ Loading object detector (YOLOv8)...")
        self.detector = FootballObjectDetector(
            model_size="small",
            custom_weights=custom_weights
        )
        
        # Initialize commentary generator
        logger.info(f"▸ Loading {generator_backend} commentary generator...")
        if generator_backend == "openai":
            commentary_gen = OpenAICommentary()
        elif generator_backend == "huggingface":
            from commentary_generator import HuggingFaceCommentary
            commentary_gen = HuggingFaceCommentary()
        else:
            commentary_gen = OllamaCommentary()
        
        self.commentator = ContextAwareCommentator(
            generator=commentary_gen,
            home_team=home_team,
            away_team=away_team
        )
        
        # Initialize voice
        if self.enable_voice:
            logger.info("▸ Initializing voice synthesis...")
            self.voice_engine = VoiceEngine()
            self.voice_engine.voice = voice_model
        
        # Queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.commentary_queue = queue.Queue(maxsize=5)
        self.voice_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        
        # Stats
        self.stats = {
            "frames_processed": 0,
            "commentaries_generated": 0,
            "voice_outputs": 0,
            "start_time": datetime.now()
        }
        
        logger.info("✓ System initialized successfully\n")
    
    def run(self):
        """Execute the complete pipeline"""
        
        logger.info("▸ Starting pipeline threads...")
        
        # Create worker threads
        threads = [
            threading.Thread(target=self._capture_and_detect, daemon=True, name="Detector"),
            threading.Thread(target=self._generate_commentary_worker, daemon=True, name="Commentary"),
        ]
        
        if self.enable_voice:
            threads.append(
                threading.Thread(target=self._voice_worker, daemon=True, name="Voice")
            )
        
        for t in threads:
            t.start()
        
        # Display loop (main thread)
        try:
            self._display_loop()
        except KeyboardInterrupt:
            logger.info("\n✓ User stopped pipeline")
        finally:
            self.stop_event.set()
            for t in threads:
                t.join(timeout=2)
            self._print_statistics()
    
    def _capture_and_detect(self):
        """Capture frames and detect objects"""
        logger.info("▸ Video capture thread started")
        
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"✗ Failed to open: {self.video_source}")
            return
        
        frame_id = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.info("✓ End of video stream")
                break
            
            # Resize
            frame = cv2.resize(frame, (1280, 720))
            
            # Detect
            detections = self.detector.detect_frame(frame, conf_threshold=0.5)
            
            try:
                self.frame_queue.put({
                    "frame_id": frame_id,
                    "frame": frame,
                    "detections": detections,
                    "timestamp": datetime.now()
                }, timeout=0.5)
                
                frame_id += 1
                self.stats["frames_processed"] += 1
                
            except queue.Full:
                pass
        
        cap.release()
        logger.info("✓ Video capture thread stopped")
    
    def _generate_commentary_worker(self):
        """Generate commentary for detected actions"""
        logger.info("▸ Commentary generation thread started")
        
        min_interval = 1.5  # Min seconds between commentary
        last_time = 0
        
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            detections = frame_data["detections"]
            
            # Check for Tier 2 events first (higher quality)
            event = detections.get("event")
            if event:
                current_time = datetime.now().timestamp()
                
                if (current_time - last_time) >= min_interval:
                    try:
                        # Use the event description as the action context
                        player_action = PlayerAction(
                            player_name=f"{event.get('team', 'TEAM').upper()}_P{event.get('player_id', 0)}",
                            player_id=event.get('player_id', 0),
                            team=event.get('team', 'unknown'),
                            action=event.get('event', 'unknown'),
                            action_confidence=event.get('confidence', 0.5),
                            nearby_players=[],
                            ball_position={
                                "x": detections["ball"]["center"][0] if detections.get("ball") else 0,
                                "y": detections["ball"]["center"][1] if detections.get("ball") else 0,
                                "z": 0
                            },
                            field_zone=event.get('zone', 'midfield')
                        )
                        
                        commentary = self.commentator.generate_commentary(player_action)
                        self.stats["commentaries_generated"] += 1
                        last_time = current_time
                        
                        try:
                            self.commentary_queue.put({
                                "text": commentary,
                                "timestamp": datetime.now()
                            }, timeout=0.5)
                        except queue.Full:
                            pass
                    
                    except Exception as e:
                        logger.error(f"Commentary error: {e}")
        
        logger.info("✓ Commentary thread stopped")
    
    def _voice_worker(self):
        """Synthesize and play voice commentary"""
        logger.info("▸ Voice synthesis thread started")
        
        while not self.stop_event.is_set():
            try:
                commentary_data = self.commentary_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            text = commentary_data["text"]
            
            try:
                logger.info(f"🔊 Synthesizing: '{text[:50]}...'")
                self.voice_engine.speak(text)
                self.stats["voice_outputs"] += 1
                
            except Exception as e:
                logger.error(f"Voice error: {e}")
        
        logger.info("✓ Voice synthesis thread stopped")
    
    def _get_field_zone(self, player: dict) -> str:
        """Determine field zone (attacking/midfield/defensive)"""
        x = player["center"][0]
        
        if x < 426:  # 1280 / 3
            return "defensive_third"
        elif x < 853:  # 2 * 1280 / 3
            return "midfield"
        else:
            return "attacking_third"
    
    def _display_loop(self):
        """Main display/UI loop"""
        logger.info("▸ Display loop started\n")
        
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            detections = frame_data["detections"]
            frame = frame_data["frame"]
            
            # Visualize
            display_frame = self.detector.visualize(frame, detections)
            
            # Get latest commentary
            latest_commentary = None
            while True:
                try:
                    latest_commentary = self.commentary_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Draw info
            if latest_commentary:
                text = latest_commentary["text"][:70]
                cv2.putText(
                    display_frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )
            
            # Draw players count
            cv2.putText(
                display_frame,
                f"Players: {len(detections['players'])}",
                (20, 670),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Draw stats
            elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
            fps = self.stats["frames_processed"] / max(elapsed, 1)
            
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Commentaries: {self.stats['commentaries_generated']}",
                (20, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            cv2.imshow("Football Commentary System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                self.stop_event.set()
                break
            
            frame_count += 1
        
        cv2.destroyAllWindows()
    
    def _print_statistics(self):
        """Print performance statistics"""
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Average FPS: {self.stats['frames_processed']/max(elapsed, 1):.1f}")
        logger.info(f"Commentaries generated: {self.stats['commentaries_generated']}")
        logger.info(f"Voice outputs: {self.stats['voice_outputs']}")
        logger.info("=" * 60)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    backend = sys.argv[2] if len(sys.argv) > 2 else "ollama"
    
    system = FullCommentarySystem(
        video_source=video_source,
        generator_backend=backend,
        home_team="Manchester United",
        away_team="Liverpool",
        enable_voice=True,
        voice_model="en-GB-ThomasNeural",
        custom_weights="football_yolov8s_best.pt"
    )
    
    system.run()
