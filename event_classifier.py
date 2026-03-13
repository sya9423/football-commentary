"""
Tier 2: Football Event Classifier
Analyzes a sliding window of Tier 1 (YOLO) detections to classify game events.
Events: shot, pass, tackle, dribble, goal_kick, corner, free_kick, throw_in, goal
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventClassifier:
    """
    Rule-based event classifier that watches sequences of YOLO detections
    and infers football events from spatial/temporal patterns.
    
    This serves as a strong baseline. It can later be replaced with a trained
    neural network (LSTM/Transformer) once we have labeled event data.
    """

    # Event types this classifier can detect
    EVENTS = [
        "shot", "pass", "tackle", "dribble", "goal_kick",
        "corner", "free_kick", "throw_in", "goal", "save",
        "clearance", "cross", "header",
    ]

    def __init__(self, window_size: int = 20, frame_width: int = 1280, frame_height: int = 720):
        """
        Args:
            window_size: Number of past frames to consider for event detection.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
        """
        self.window_size = window_size
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Sliding window of recent detections
        self.detection_history: deque = deque(maxlen=window_size)

        # Ball trajectory
        self.ball_positions: deque = deque(maxlen=window_size)

        # Cooldown to prevent spamming the same event
        self.last_event_frame = -30
        self.cooldown_frames = 15  # Min frames between events

        # Goal zone boundaries (rough pixel regions)
        # Left goal: x < 10% of frame
        # Right goal: x > 90% of frame
        self.goal_zone_left = frame_width * 0.08
        self.goal_zone_right = frame_width * 0.92

        # Define pitch zones (in pixels)
        self.zones = {
            "defensive_left": (0, frame_width * 0.33),
            "midfield": (frame_width * 0.33, frame_width * 0.66),
            "attacking_right": (frame_width * 0.66, frame_width),
        }

        self.frame_count = 0
        logger.info("EventClassifier initialized (rule-based mode)")

    def update(self, detections: Dict) -> Optional[Dict]:
        """
        Feed a new frame's detections and check for events.
        
        Args:
            detections: Output from FootballObjectDetector.detect_frame()
                {
                    "players": [...],
                    "ball": {"center": (x, y), ...} or None,
                    ...
                }
        
        Returns:
            Event dict if an event was detected, else None.
            {
                "event": str,           # e.g. "shot", "pass"
                "confidence": float,    # 0.0 - 1.0
                "player_id": int,       # ID of the player involved
                "team": str,            # "home", "away", "unknown"
                "zone": str,            # pitch zone
                "description": str,     # Human-readable description
            }
        """
        self.frame_count += 1
        self.detection_history.append(detections)

        # Track ball position
        if detections.get("ball") and detections["ball"].get("center"):
            self.ball_positions.append(detections["ball"]["center"])
        else:
            self.ball_positions.append(None)

        # Don't classify until we have enough history
        if len(self.detection_history) < 5:
            return None

        # Cooldown check
        if (self.frame_count - self.last_event_frame) < self.cooldown_frames:
            return None

        # Run event detection rules (ordered by priority)
        event = (
            self._detect_goal()
            or self._detect_shot()
            or self._detect_save()
            or self._detect_tackle()
            or self._detect_pass()
            or self._detect_dribble()
            or self._detect_clearance()
            or self._detect_cross()
        )

        if event:
            self.last_event_frame = self.frame_count
            return event

        return None

    # =========================================================================
    # Event Detection Rules
    # =========================================================================

    def _detect_goal(self) -> Optional[Dict]:
        """
        Detect a goal: ball was moving fast toward goal zone and then disappears
        or enters the goal area.
        """
        if len(self.ball_positions) < 5:
            return None

        recent_positions = [p for p in list(self.ball_positions)[-8:] if p is not None]
        if len(recent_positions) < 3:
            return None

        # Ball velocity toward goal
        velocity = self._get_ball_velocity(recent_positions)
        if velocity is None:
            return None

        vx, vy = velocity
        last_pos = recent_positions[-1]

        # Ball near goal zone + moving fast horizontally
        near_goal = (last_pos[0] < self.goal_zone_left or last_pos[0] > self.goal_zone_right)
        fast_horizontal = abs(vx) > 15

        # Ball disappeared after being near goal (went in the net)
        recent_ball_present = [p is not None for p in list(self.ball_positions)[-5:]]
        ball_disappeared = recent_ball_present.count(False) >= 3

        if near_goal and (fast_horizontal or ball_disappeared):
            closest = self._get_closest_player_to_ball()
            return {
                "event": "goal",
                "confidence": 0.7,
                "player_id": closest["id"] if closest else -1,
                "team": closest.get("team", "unknown") if closest else "unknown",
                "zone": "attacking_third",
                "description": f"GOAL! {closest.get('team', 'A player').upper()} scores!",
            }

        return None

    def _detect_shot(self) -> Optional[Dict]:
        """
        Detect a shot: ball accelerates rapidly toward the goal from the
        attacking third.
        """
        if len(self.ball_positions) < 5:
            return None

        recent = [p for p in list(self.ball_positions)[-6:] if p is not None]
        if len(recent) < 3:
            return None

        velocity = self._get_ball_velocity(recent)
        if velocity is None:
            return None

        vx, vy = velocity
        speed = np.sqrt(vx**2 + vy**2)
        last_pos = recent[-1]

        # High speed + ball in attacking zone + moving toward goal line
        in_attack_zone = (
            last_pos[0] > self.frame_width * 0.6
            or last_pos[0] < self.frame_width * 0.4
        )

        if speed > 20 and in_attack_zone:
            closest = self._get_closest_player_to_ball()
            return {
                "event": "shot",
                "confidence": min(0.5 + speed / 80, 0.95),
                "player_id": closest["id"] if closest else -1,
                "team": closest.get("team", "unknown") if closest else "unknown",
                "zone": self._get_zone(last_pos[0]),
                "description": f"Shot by {closest.get('team', 'unknown').upper()} player!",
            }

        return None

    def _detect_save(self) -> Optional[Dict]:
        """
        Detect a save: ball was heading toward goal at speed, then stops or
        reverses direction.
        """
        if len(self.ball_positions) < 8:
            return None

        recent = [p for p in list(self.ball_positions)[-8:] if p is not None]
        if len(recent) < 5:
            return None

        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        v1 = self._get_ball_velocity(first_half)
        v2 = self._get_ball_velocity(second_half)

        if v1 is None or v2 is None:
            return None

        # Direction reversal near goal
        direction_reversed = (v1[0] * v2[0] < 0)  # x-velocity changed sign
        near_goal = (
            recent[-1][0] < self.goal_zone_left + 50
            or recent[-1][0] > self.goal_zone_right - 50
        )

        if direction_reversed and near_goal and abs(v1[0]) > 10:
            # Find goalkeeper
            gk = self._find_goalkeeper()
            return {
                "event": "save",
                "confidence": 0.65,
                "player_id": gk["id"] if gk else -1,
                "team": gk.get("team", "unknown") if gk else "unknown",
                "zone": "goal_area",
                "description": "Save by the goalkeeper!",
            }

        return None

    def _detect_tackle(self) -> Optional[Dict]:
        """
        Detect a tackle: two players from different teams very close together,
        and ball changes possession direction.
        """
        if len(self.detection_history) < 3:
            return None

        current = self.detection_history[-1]
        players = current.get("players", [])
        ball = current.get("ball")

        if not ball or len(players) < 2:
            return None

        ball_pos = ball["center"]

        # Find players very close to each other AND close to ball
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                if p1.get("team") == p2.get("team"):
                    continue  # Same team, skip

                dist_players = np.sqrt(
                    (p1["center"][0] - p2["center"][0])**2
                    + (p1["center"][1] - p2["center"][1])**2
                )
                dist_to_ball_1 = np.sqrt(
                    (p1["center"][0] - ball_pos[0])**2
                    + (p1["center"][1] - ball_pos[1])**2
                )
                dist_to_ball_2 = np.sqrt(
                    (p2["center"][0] - ball_pos[0])**2
                    + (p2["center"][1] - ball_pos[1])**2
                )

                # Two opponents within 40px of each other AND both near ball
                if dist_players < 40 and min(dist_to_ball_1, dist_to_ball_2) < 60:
                    tackler = p1 if dist_to_ball_1 > dist_to_ball_2 else p2
                    return {
                        "event": "tackle",
                        "confidence": 0.6,
                        "player_id": tackler.get("id", -1),
                        "team": tackler.get("team", "unknown"),
                        "zone": self._get_zone(ball_pos[0]),
                        "description": f"Tackle by {tackler.get('team', 'unknown').upper()} player!",
                    }

        return None

    def _detect_pass(self) -> Optional[Dict]:
        """
        Detect a pass: ball moves steadily between two players on the same team,
        moderate speed.
        """
        if len(self.ball_positions) < 6:
            return None

        recent = [p for p in list(self.ball_positions)[-6:] if p is not None]
        if len(recent) < 4:
            return None

        velocity = self._get_ball_velocity(recent)
        if velocity is None:
            return None

        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # Moderate speed (not a shot, not stationary)
        if 5 < speed < 18:
            closest = self._get_closest_player_to_ball()
            if closest:
                return {
                    "event": "pass",
                    "confidence": 0.55,
                    "player_id": closest.get("id", -1),
                    "team": closest.get("team", "unknown"),
                    "zone": self._get_zone(recent[-1][0]),
                    "description": f"Pass by {closest.get('team', 'unknown').upper()} player.",
                }

        return None

    def _detect_dribble(self) -> Optional[Dict]:
        """
        Detect a dribble: one player stays near the ball across multiple frames
        while moving forward.
        """
        if len(self.detection_history) < 8:
            return None

        # Check if the same player has been closest to ball for 5+ frames
        closest_ids = []
        for det in list(self.detection_history)[-8:]:
            ball = det.get("ball")
            if not ball:
                continue
            players = det.get("players", [])
            if not players:
                continue

            ball_pos = ball["center"]
            dists = []
            for p in players:
                d = np.sqrt(
                    (p["center"][0] - ball_pos[0])**2
                    + (p["center"][1] - ball_pos[1])**2
                )
                dists.append((d, p))

            if dists:
                closest = min(dists, key=lambda x: x[0])
                if closest[0] < 50:
                    closest_ids.append(closest[1].get("id", -1))

        if len(closest_ids) >= 5:
            # Same player holding ball?
            most_common_id = max(set(closest_ids), key=closest_ids.count)
            if closest_ids.count(most_common_id) >= 5:
                # Find this player in current frame
                current = self.detection_history[-1]
                for p in current.get("players", []):
                    if p.get("id") == most_common_id:
                        return {
                            "event": "dribble",
                            "confidence": 0.6,
                            "player_id": most_common_id,
                            "team": p.get("team", "unknown"),
                            "zone": self._get_zone(p["center"][0]),
                            "description": f"{p.get('team', 'unknown').upper()} player on the dribble!",
                        }

        return None

    def _detect_clearance(self) -> Optional[Dict]:
        """
        Detect a clearance: ball launched from defensive zone at high speed upfield.
        """
        if len(self.ball_positions) < 5:
            return None

        recent = [p for p in list(self.ball_positions)[-5:] if p is not None]
        if len(recent) < 3:
            return None

        first_pos = recent[0]
        velocity = self._get_ball_velocity(recent)
        if velocity is None:
            return None

        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # Fast ball launched from defensive zone
        in_defense = (
            first_pos[0] < self.frame_width * 0.25
            or first_pos[0] > self.frame_width * 0.75
        )

        if speed > 18 and in_defense:
            closest = self._get_closest_player_to_ball()
            return {
                "event": "clearance",
                "confidence": 0.55,
                "player_id": closest.get("id", -1) if closest else -1,
                "team": closest.get("team", "unknown") if closest else "unknown",
                "zone": "defensive_third",
                "description": "Clearance! Ball booted upfield!",
            }

        return None

    def _detect_cross(self) -> Optional[Dict]:
        """
        Detect a cross: ball moves from wide position into the box (from the side
        of the frame toward the center-goal area).
        """
        if len(self.ball_positions) < 5:
            return None

        recent = [p for p in list(self.ball_positions)[-6:] if p is not None]
        if len(recent) < 3:
            return None

        first_pos = recent[0]
        last_pos = recent[-1]

        # Ball started wide (near top/bottom of frame) and moved centrally
        started_wide = (
            first_pos[1] < self.frame_height * 0.2
            or first_pos[1] > self.frame_height * 0.8
        )
        moved_central = (
            self.frame_height * 0.3 < last_pos[1] < self.frame_height * 0.7
        )

        velocity = self._get_ball_velocity(recent)
        if velocity and started_wide and moved_central:
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            if speed > 8:
                closest = self._get_closest_player_to_ball()
                return {
                    "event": "cross",
                    "confidence": 0.55,
                    "player_id": closest.get("id", -1) if closest else -1,
                    "team": closest.get("team", "unknown") if closest else "unknown",
                    "zone": "attacking_third",
                    "description": "Cross delivered into the box!",
                }

        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_ball_velocity(self, positions: List[Tuple]) -> Optional[Tuple[float, float]]:
        """Calculate average ball velocity from a list of (x, y) positions."""
        if len(positions) < 2:
            return None
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        n = len(positions) - 1
        return (dx / n, dy / n)

    def _get_closest_player_to_ball(self) -> Optional[Dict]:
        """Find the player closest to the ball in the most recent frame."""
        if not self.detection_history:
            return None

        current = self.detection_history[-1]
        ball = current.get("ball")
        if not ball:
            return None

        ball_pos = ball["center"]
        players = current.get("players", [])

        best = None
        best_dist = float("inf")
        for p in players:
            d = np.sqrt(
                (p["center"][0] - ball_pos[0])**2
                + (p["center"][1] - ball_pos[1])**2
            )
            if d < best_dist:
                best_dist = d
                best = p

        return best

    def _find_goalkeeper(self) -> Optional[Dict]:
        """Find a player positioned near a goal line (likely the goalkeeper)."""
        if not self.detection_history:
            return None

        current = self.detection_history[-1]
        for p in current.get("players", []):
            x = p["center"][0]
            if x < self.goal_zone_left + 30 or x > self.goal_zone_right - 30:
                return p
        return None

    def _get_zone(self, x: float) -> str:
        """Get pitch zone name from x-coordinate."""
        if x < self.frame_width * 0.33:
            return "defensive_third"
        elif x < self.frame_width * 0.66:
            return "midfield"
        else:
            return "attacking_third"
