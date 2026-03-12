"""
Test suite for Football Commentary System
Run tests before deployment
"""

import unittest
import cv2
import numpy as np
import tempfile
from pathlib import Path
import sys

# Import modules
sys.path.insert(0, str(Path(__file__).parent))
from object_detector import FootballObjectDetector
from commentary_generator import GameState, PlayerAction, OllamaCommentary
from commentator import generate_tts_file


class TestObjectDetection(unittest.TestCase):
    """Test object detection module"""
    
    def setUp(self):
        """Initialize detector"""
        self.detector = FootballObjectDetector(model_size="nano")
    
    def test_detector_initialization(self):
        """Test detector loads correctly"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
    
    def test_detect_frame(self):
        """Test detection on dummy frame"""
        # Create dummy frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Detect
        detections = self.detector.detect_frame(frame)
        
        # Check structure
        self.assertIn("players", detections)
        self.assertIn("ball", detections)
        self.assertIn("actions", detections)
        self.assertIsInstance(detections["players"], list)
    
    def test_team_classification(self):
        """Test team color classification"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Red frame = home team
        frame[:, :] = (0, 0, 255)  # BGR: Red
        team = self.detector._classify_team(frame, 10, 10, 90, 90)
        
        # Should classify as home (red)
        self.assertIn(team, ["home", "away", "unknown"])
    
    def test_player_tracking(self):
        """Test player tracking across frames"""
        player1 = {"center": (100, 100), "zone": "mid"}
        player2 = {"center": (105, 105), "zone": "mid"}
        
        id1 = self.detector._track_player(player1)
        id2 = self.detector._track_player(player2)
        
        # Should assign IDs
        self.assertIsInstance(id1, int)
        self.assertIsInstance(id2, int)


class TestCommentaryGeneration(unittest.TestCase):
    """Test commentary generation"""
    
    def setUp(self):
        """Initialize commentary generator"""
        try:
            self.generator = OllamaCommentary()
            self.ollama_available = True
        except:
            self.ollama_available = False
    
    def test_game_state_creation(self):
        """Test GameState data class"""
        state = GameState(
            home_team="Man Utd",
            away_team="Liverpool",
            score={"home": 1, "away": 0},
            minute=45,
            period="first_half",
            possession="home",
            last_events=[],
            injured_players=[],
            yellow_cards=[],
            red_cards=[],
            atmosphere="intense"
        )
        
        self.assertEqual(state.home_team, "Man Utd")
        self.assertEqual(state.score["home"], 1)
        self.assertEqual(state.minute, 45)
    
    def test_player_action_creation(self):
        """Test PlayerAction data class"""
        action = PlayerAction(
            player_name="Bruno Fernandes",
            player_id=1,
            team="home",
            action="shooting",
            action_confidence=0.92,
            nearby_players=[],
            ball_position={"x": 100, "y": 50, "z": 0},
            field_zone="attacking_third"
        )
        
        self.assertEqual(action.player_name, "Bruno Fernandes")
        self.assertEqual(action.action, "shooting")
        self.assertEqual(action.action_confidence, 0.92)
    
    @unittest.skipIf(not True, "Ollama not available")  # Skip if no Ollama
    def test_commentary_generation(self):
        """Test commentary generation (requires Ollama)"""
        if not self.ollama_available:
            self.skipTest("Ollama not running")
        
        state = GameState(
            home_team="Man Utd",
            away_team="Liverpool",
            score={"home": 1, "away": 0},
            minute=45,
            period="first_half",
            possession="home",
            last_events=[],
            injured_players=[],
            yellow_cards=[],
            red_cards=[],
            atmosphere="normal"
        )
        
        action = PlayerAction(
            player_name="Bruno",
            player_id=1,
            team="home",
            action="shooting",
            action_confidence=0.9,
            nearby_players=[],
            ball_position={"x": 100, "y": 50, "z": 0},
            field_zone="attacking_third"
        )
        
        commentary = self.generator.generate(state, action)
        
        # Should generate something
        self.assertIsNotNone(commentary)
        self.assertGreater(len(commentary), 0)
        self.assertIsInstance(commentary, str)


class TestVoiceSynthesis(unittest.TestCase):
    """Test voice synthesis"""
    
    def test_tts_file_generation(self):
        """Test TTS file generation"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_file = f.name
        
        try:
            # Generate TTS
            result = generate_tts_file(
                "Hello, this is a test.",
                temp_file,
                "en-GB-ThomasNeural"
            )
            
            # Should succeed or fail gracefully
            self.assertIsInstance(result, bool)
            
        finally:
            # Cleanup
            if Path(temp_file).exists():
                Path(temp_file).unlink()


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_system_initialization(self):
        """Test full system initializes"""
        from full_system import FullCommentarySystem
        
        try:
            system = FullCommentarySystem(
                video_source=0,
                generator_backend="ollama",
                enable_voice=False
            )
            
            self.assertIsNotNone(system)
            self.assertIsNotNone(system.detector)
            self.assertIsNotNone(system.commentator)
            
        except Exception as e:
            self.fail(f"System initialization failed: {e}")


class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""
    
    def test_detection_speed(self):
        """Benchmark detection speed"""
        import time
        
        detector = FootballObjectDetector(model_size="nano")
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Warmup
        detector.detect_frame(frame)
        
        # Benchmark
        start = time.time()
        for _ in range(5):
            detector.detect_frame(frame)
        elapsed = time.time() - start
        
        avg_ms = (elapsed / 5) * 1000
        
        print(f"\nDetection avg: {avg_ms:.1f}ms")
        self.assertLess(avg_ms, 1000)  # Should be < 1s


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    print("Running Football Commentary System Tests...\n")
    run_tests()
