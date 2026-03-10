import random
import time
import pygame
import os
import sys

# Use synchronous TTS generation via subprocess
def generate_tts_file(text, filename, voice):
    """Generate TTS file using edge-tts CLI"""
    import subprocess
    try:
        cmd = [
            sys.executable, "-m", "edge_tts",
            "--text", text,
            "--voice", voice,
            "--write-media", filename
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return os.path.exists(filename)
    except Exception as e:
        print(f"TTS Generation Error: {e}")
        return False

class ContextualCommentator:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.recent_events = []
        self.goal_scorers = {}
        
    def generate_goal_commentary(self, player, team, minute, score):
        """Context-aware goal commentary in Peter Drury's style"""
        is_equalizer = score["home"] == score["away"]
        is_go_ahead = abs(score["home"] - score["away"]) == 1
        is_late_game = minute > 75
        is_first_goal = score["home"] + score["away"] == 1
        scorer_has_brace = self.goal_scorers.get(player, 0) >= 1
        
        if is_late_game and is_go_ahead:
            templates = [
                f"And it's there! {player}! In the ninety-th minute!",
                f"Can you believe it? {player} has done it!",
                f"Late drama. {player} strikes. This is extraordinary!",
                f"The moment. The stage. {player}!"
            ]
        elif scorer_has_brace:
            templates = [
                f"{player} again! His second! And what a performance this is!",
                f"It's {player}! Twice now! He's the man of the match!",
                f"Two for {player}. That's a brace. Complete dominance.",
                f"{player} with his second. He's been magnificent!"
            ]
        elif is_equalizer:
            templates = [
                f"And it's level! {player} does it! All square now!",
                f"{player} levels it! That's the equalizer!",
                f"Back they come. {player}. And it's equal.",
                f"We're level. {player} makes it so!"
            ]
        elif is_first_goal:
            templates = [
                f"It's in! {player} opens the scoring!",
                f"The deadlock is broken! {player}!",
                f"{player} with the first goal of the match.",
                f"And it's there. {player}. First blood."
            ]
        else:
            templates = [
                f"{player}! And that's a goal!",
                f"It's in. {player} finds the net.",
                f"{player}! Into the back of the net!",
                f"Goal! {player}!"
            ]
        
        self.goal_scorers[player] = self.goal_scorers.get(player, 0) + 1
        self.recent_events.append({"type": "goal", "player": player, "minute": minute})
        
        commentary = random.choice(templates)
        score_line = f"The score is now {self.home_team} {score['home']}, {self.away_team} {score['away']}."
        
        return f"{commentary} {score_line}"
    
    def generate_shot_commentary(self, player, team, minute, on_target):
        """Commentary for shots - Peter Drury style"""
        if on_target:
            templates = [
                f"{player} shoots! The keeper saves!",
                f"There's an effort! {player}! Saved!",
                f"{player} goes for goal. Good save.",
                f"A chance there for {player}! But the keeper is alert!"
            ]
        else:
            templates = [
                f"{player} tries from distance. Just wide.",
                f"There's an attempt. {player}. Off target.",
                f"{player} shoots. That's over the bar.",
                f"A wild effort from {player}. Well over."
            ]
        
        self.recent_events.append({"type": "shot", "player": player, "minute": minute})
        return random.choice(templates)


class VoiceEngine:
    def __init__(self):
        pygame.mixer.init()
        self.temp_file = "commentary.mp3"
        # Use ThomasNeural - deeper voice closer to Peter Drury's tone
        self.voice = "en-GB-ThomasNeural"
        
    def speak(self, text):
        """Convert text to speech using Microsoft Edge TTS"""
        try:
            print(f"Speaking: {text[:60]}...")
            
            # Generate TTS file
            if not generate_tts_file(text, self.temp_file, self.voice):
                print(f"Failed to generate audio for: {text}")
                return
            
            # Load and play audio
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            pygame.mixer.music.unload()
            time.sleep(0.1)
            
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                
        except Exception as e:
            print(f"Voice error: {e}")
            print(f"[Commentary]: {text}")


class FootballMatch:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.minute = 0
        self.score = {"home": 0, "away": 0}
        
        self.home_players = self.load_squad(home_team)
        self.away_players = self.load_squad(away_team)
        
        self.commentator = ContextualCommentator(home_team, away_team)
        self.voice = VoiceEngine()
        
    def load_squad(self, team):
        """Load player names"""
        squads = {
            "Arsenal": ["Saka", "Martinelli", "Jesus", "Odegaard", "Rice", "Havertz"],
            "Chelsea": ["Sterling", "Jackson", "Palmer", "Enzo", "Caicedo", "Mudryk"],
            "Liverpool": ["Salah", "Nunez", "Diaz", "Jota", "Gakpo", "Jones"],
            "Man City": ["Haaland", "Foden", "Grealish", "Alvarez", "Silva", "De Bruyne"]
        }
        return squads.get(team, ["Player 1", "Player 2", "Player 3"])
    
    def simulate_minute(self):
        """Simulate one minute of play"""
        self.minute += 1
        roll = random.random()
        
        if roll < 0.02:  # 2% goal chance
            team = random.choice([self.home_team, self.away_team])
            players = self.home_players if team == self.home_team else self.away_players
            player = random.choice(players)
            
            if team == self.home_team:
                self.score["home"] += 1
            else:
                self.score["away"] += 1
            
            commentary = self.commentator.generate_goal_commentary(
                player, team, self.minute, self.score
            )
            self.voice.speak(commentary)
            return commentary
            
        elif roll < 0.08:  # 6% shot chance
            team = random.choice([self.home_team, self.away_team])
            players = self.home_players if team == self.home_team else self.away_players
            player = random.choice(players)
            on_target = random.random() < 0.4
            
            commentary = self.commentator.generate_shot_commentary(
                player, team, self.minute, on_target
            )
            self.voice.speak(commentary)
            return commentary
        
        return None
    
    def run_match(self, speed=0.3):
        """Run full 90 minute match"""
        print(f"\n{'='*60}")
        print(f"  {self.home_team.upper()} vs {self.away_team.upper()}")
        print(f"{'='*60}\n")
        
        self.voice.speak(f"Welcome to today's match. {self.home_team} against {self.away_team}. Let's get underway!")
        time.sleep(1)
        
        for _ in range(90):
            event = self.simulate_minute()
            
            if event:
                print(f"[Min {self.minute}'] {event}")
            
            # Score update every 15 minutes
            if self.minute % 15 == 0 and self.minute < 90:
                score_update = f"After {self.minute} minutes, {self.home_team} {self.score['home']}, {self.away_team} {self.score['away']}."
                print(f"\n--- {score_update} ---\n")
            
            time.sleep(speed)
        
        # Full time
        final = f"Full time whistle! Final score: {self.home_team} {self.score['home']}, {self.away_team} {self.score['away']}."
        print(f"\n{'='*60}")
        print(f"  {final}")
        print(f"{'='*60}\n")
        self.voice.speak(final)


if __name__ == "__main__":
    # Run a match!
    match = FootballMatch("Arsenal", "Chelsea")
    match.run_match(speed=0.5)  # 0.5 seconds per minute (45 second match)