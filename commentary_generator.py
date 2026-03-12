"""
ML-based Commentary Generator
Generates fresh, contextual commentary in real-time using LLMs
Supports multiple backends: OpenAI, Ollama (local), Hugging Face
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Current match context"""
    home_team: str
    away_team: str
    score: Dict[str, int]
    minute: int
    period: str  # "first_half", "second_half", "extra_time", "penalties"
    possession: str  # which team
    last_events: List[Dict]  # recent events
    injured_players: List[str]
    yellow_cards: List[str]
    red_cards: List[str]
    atmosphere: str  # "intense", "calm", "chaotic", etc.


@dataclass
class PlayerAction:
    """Current player action context"""
    player_name: str
    player_id: int
    team: str
    action: str  # "shooting", "passing", "dribbling", "defending", etc.
    action_confidence: float
    nearby_players: List[Dict]  # {"player_name", "distance", "team"}
    ball_position: Dict  # {"x", "y", "z"}
    field_zone: str  # "defensive_third", "midfield", "attacking_third"


class CommentaryGenerator(ABC):
    """Abstract base class for commentary generators"""
    
    @abstractmethod
    def generate(
        self, 
        game_state: GameState, 
        player_action: PlayerAction
    ) -> str:
        """Generate commentary based on game and player state"""
        pass


class OpenAICommentary(CommentaryGenerator):
    """
    Uses OpenAI's GPT models for commentary generation
    Requires: pip install openai
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4",
                 style: str = "Peter Drury"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.style = style
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def generate(
        self, 
        game_state: GameState, 
        player_action: PlayerAction
    ) -> str:
        """Generate commentary using GPT"""
        
        prompt = self._build_prompt(game_state, player_action)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a world-class football (soccer) commentator 
                        with the style of {self.style}. Generate live match commentary.
                        - Keep commentary to 1-2 sentences (natural speaking length)
                        - Use vivid, descriptive language
                        - React to the action authentically
                        - Vary your commentary - no repetition
                        - Use the player's actual name when possible"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,  # Higher for more varied output
                max_tokens=100,
                top_p=0.95
            )
            
            commentary = response.choices[0].message.content.strip()
            logger.info(f"Generated: {commentary[:80]}...")
            return commentary
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._fallback_commentary(player_action)
    
    def _build_prompt(self, game_state: GameState, player_action: PlayerAction) -> str:
        """Build detailed prompt for GPT"""
        
        score_str = f"{game_state.score['home']}-{game_state.score['away']}"
        
        prompt = f"""Match Context:
- Match: {game_state.home_team} vs {game_state.away_team}
- Score: {score_str}
- Time: {game_state.minute}'
- Period: {game_state.period}
- Possession: {game_state.possession}
- Recent action: {game_state.last_events[-1] if game_state.last_events else 'Match start'}

Current Action:
- Player: {player_action.player_name} ({player_action.team})
- Action: {player_action.action}
- Confidence: {player_action.action_confidence:.1%}
- Zone: {player_action.field_zone}
- Nearby opponents: {len([p for p in player_action.nearby_players if p['team'] != player_action.team])}

Generate authentic, exciting commentary for this moment. Be specific and natural."""
        
        return prompt
    
    def _fallback_commentary(self, player_action: PlayerAction) -> str:
        """Fallback if API fails"""
        actions = {
            "shooting": f"{player_action.player_name} strikes for goal!",
            "passing": f"{player_action.player_name} looks for a teammate.",
            "dribbling": f"{player_action.player_name} drives forward!",
            "defending": f"{player_action.player_name} slides in to block!",
            "receiving": f"{player_action.player_name} takes possession.",
            "heading": f"{player_action.player_name} rises for the header!"
        }
        return actions.get(player_action.action, "The action continues...")


class OllamaCommentary(CommentaryGenerator):
    """
    Uses Ollama for local LLM inference (privacy, no API costs)
    Requires: pip install requests
    Ensure Ollama is running: ollama serve
    """
    
    def __init__(self, 
                 model: str = "mistral",
                 host: str = "http://localhost:11434",
                 style: str = "Peter Drury"):
        self.model = model
        self.host = host
        self.style = style
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")
    
    def generate(
        self, 
        game_state: GameState, 
        player_action: PlayerAction
    ) -> str:
        """Generate commentary using local Ollama model"""
        
        prompt = self._build_prompt(game_state, player_action)
        
        try:
            response = self.requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.9,
                    "top_p": 0.95,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                commentary = result.get("response", "").strip()
                logger.info(f"Generated (Ollama): {commentary[:80]}...")
                return commentary
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._fallback_commentary(player_action)
                
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            return self._fallback_commentary(player_action)
    
    def _build_prompt(self, game_state: GameState, player_action: PlayerAction) -> str:
        score_str = f"{game_state.score['home']}-{game_state.score['away']}"
        
        prompt = f"""[INST] You are a world-class {self.style}-style football commentator.

Match: {game_state.home_team} vs {game_state.away_team}
Score: {score_str} | Time: {game_state.minute}' | Period: {game_state.period}

Current action: {player_action.player_name} performing {player_action.action} in the {player_action.field_zone}.

Generate 1-2 sentences of vivid, authentic match commentary. Be specific and exciting. [/INST]"""
        
        return prompt
    
    def _fallback_commentary(self, player_action: PlayerAction) -> str:
        actions = {
            "shooting": f"{player_action.player_name} goes for goal!",
            "passing": f"{player_action.player_name} and a pass...",
            "dribbling": f"{player_action.player_name} advances down the field!",
            "defending": f"{player_action.player_name} makes a clearance.",
            "receiving": f"{player_action.player_name} controls it.",
            "heading": f"{player_action.player_name} heads it away!"
        }
        return actions.get(player_action.action, "Play continues...")


class HuggingFaceCommentary(CommentaryGenerator):
    """
    Uses Hugging Face transformers for local inference
    Requires: pip install transformers torch
    """
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 device: str = "cuda",
                 style: str = "Peter Drury"):
        self.model_name = model_name
        self.device = device
        self.style = style
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            logger.info("Model loaded successfully")
            
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def generate(
        self, 
        game_state: GameState, 
        player_action: PlayerAction
    ) -> str:
        """Generate commentary using Hugging Face model"""
        
        prompt = self._build_prompt(game_state, player_action)
        
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True
                )
            
            commentary = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the generated part (remove prompt)
            commentary = commentary[len(prompt):].strip()
            
            logger.info(f"Generated (HF): {commentary[:80]}...")
            return commentary
            
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return self._fallback_commentary(player_action)
    
    def _build_prompt(self, game_state: GameState, player_action: PlayerAction) -> str:
        score_str = f"{game_state.score['home']}-{game_state.score['away']}"
        
        prompt = f"""You are a world-class {self.style}-style football commentator.

Match: {game_state.home_team} vs {game_state.away_team}
Score: {score_str} | Time: {game_state.minute}'

Current action: {player_action.player_name} ({player_action.team}) {player_action.action}

Commentary: """
        
        return prompt
    
    def _fallback_commentary(self, player_action: PlayerAction) -> str:
        return f"{player_action.player_name} with the {player_action.action}!"


class ContextAwareCommentator:
    """
    High-level interface combining game state tracking + commentary generation
    """
    
    def __init__(self, 
                 generator: CommentaryGenerator,
                 home_team: str,
                 away_team: str):
        self.generator = generator
        self.game_state = GameState(
            home_team=home_team,
            away_team=away_team,
            score={"home": 0, "away": 0},
            minute=0,
            period="first_half",
            possession="home",
            last_events=[],
            injured_players=[],
            yellow_cards=[],
            red_cards=[],
            atmosphere="normal"
        )
        self.commentary_history = []
    
    def update_state(self, updates: Dict):
        """Update game state"""
        for key, value in updates.items():
            if hasattr(self.game_state, key):
                setattr(self.game_state, key, value)
    
    def generate_commentary(self, 
                          player_action: PlayerAction) -> str:
        """Generate commentary for current action"""
        
        # Track recent events
        event = {
            "player": player_action.player_name,
            "action": player_action.action,
            "zone": player_action.field_zone,
            "timestamp": datetime.now()
        }
        self.game_state.last_events.append(event)
        
        # Keep only last 5 events
        if len(self.game_state.last_events) > 5:
            self.game_state.last_events = self.game_state.last_events[-5:]
        
        # Generate
        commentary = self.generator.generate(self.game_state, player_action)
        
        # Track
        self.commentary_history.append({
            "minute": self.game_state.minute,
            "action": player_action.action,
            "commentary": commentary
        })
        
        return commentary
    
    def get_commentary_stats(self) -> Dict:
        """Get commentary generation statistics"""
        return {
            "total_generated": len(self.commentary_history),
            "by_action": self._count_by_action(),
            "generation_time": self._avg_generation_time()
        }
    
    def _count_by_action(self) -> Dict[str, int]:
        """Count commentary by action type"""
        from collections import Counter
        actions = [c["action"] for c in self.commentary_history]
        return dict(Counter(actions))
    
    def _avg_generation_time(self) -> float:
        """Average generation time"""
        if len(self.commentary_history) < 2:
            return 0.0
        times = []
        for i in range(1, len(self.commentary_history)):
            prev_time = self.commentary_history[i-1].get("timestamp", datetime.now())
            curr_time = self.commentary_history[i].get("timestamp", datetime.now())
            times.append((curr_time - prev_time).total_seconds())
        return sum(times) / len(times) if times else 0.0


# Example usage
if __name__ == "__main__":
    # Choose your backend
    # generator = OpenAICommentary()  # Requires OPENAI_API_KEY
    generator = OllamaCommentary()  # Requires Ollama running locally
    # generator = HuggingFaceCommentary()  # Requires transformers
    
    commentator = ContextAwareCommentator(
        generator=generator,
        home_team="Manchester United",
        away_team="Liverpool"
    )
    
    # Simulate game state
    commentator.update_state({
        "minute": 25,
        "possession": "home",
        "score": {"home": 1, "away": 0}
    })
    
    # Simulate player action
    action = PlayerAction(
        player_name="Bruno Fernandes",
        player_id=1,
        team="home",
        action="shooting",
        action_confidence=0.92,
        nearby_players=[
            {"player_name": "Haaland", "distance": 5, "team": "away"}
        ],
        ball_position={"x": 90, "y": 50, "z": 0.5},
        field_zone="attacking_third"
    )
    
    commentary = commentator.generate_commentary(action)
    print(f"\nGenerated Commentary:\n{commentary}")
