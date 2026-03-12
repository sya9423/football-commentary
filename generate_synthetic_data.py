import os
import json
import random
from statsbombpy import sb

def generate_synthetic_statsbomb_commentary(output_file='training_data/commentary.jsonl', num_samples=500):
    """
    Downloads real match data from StatsBomb Open Data and uses templates 
    to generate realistic (Game State, Action, Commentary) triplets.
    """
    os.makedirs('training_data', exist_ok=True)
    print("🏟️  Connecting to StatsBomb Open Data...")
    
    # Using the 2018 Men's World Cup (Competition 43, Season 3) as a reliable free open data source
    matches = sb.matches(competition_id=43, season_id=3)
    match_ids = matches['match_id'].tolist()
    
    print(f"Found {len(match_ids)} matches. Generating {num_samples} training samples...\n")
    
    generated_data = []
    
    # Dictionaries for synthetic commentary templates based on event types
    templates = {
        'Pass': [
            "A crisp pass from {player}.",
            "{player} sprays the ball wide.",
            "Great vision by {player} to pick out that pass.",
            "{player} keeps the possession moving.",
            "Simple ball played by {player}."
        ],
        'Shot': [
            "{player} takes the shot! Oh, it was close!",
            "It's an effort on goal from {player}!",
            "{player} strikes it cleanly, but it's not quite there.",
            "Brilliant strike from {player}!",
            "{player} goes for goal!"
        ],
        'Foul Committed': [
            "Clumsy challenge there by {player}.",
            "The referee blows the whistle. Clear foul by {player}.",
            "{player} goes into the book for that tackle.",
            "That's a free kick. {player} was too aggressive there.",
            "{player} brings his man down."
        ],
        'Ball Recovery': [
            "Important interception by {player}.",
            "{player} reads the game beautifully and wins it back.",
            "Great defensive work from {player}.",
            "{player} recovers the loose ball.",
            "Possession turns over as {player} steps in."
        ],
        'Dribble': [
            "{player} takes on his man!",
            "Winding run by {player}...",
            "{player} shows fantastic footwork.",
            "He drops the shoulder... beautiful skill by {player}.",
            "{player} is driving forward with purpose."
        ]
    }

    # Loop through matches and extract events
    for match_id in match_ids:
        if len(generated_data) >= num_samples:
            break
            
        try:
            events = sb.events(match_id=match_id)
            
            # Filter to events we have templates for
            valid_events = events[events['type'].isin(templates.keys())].dropna(subset=['player'])
            
            # Subsample to get a mix of different games
            sampled_events = valid_events.sample(n=min(50, len(valid_events)))
            
            for _, row in sampled_events.iterrows():
                if len(generated_data) >= num_samples:
                    break
                    
                event_type = row['type']
                player_name = row['player']
                minute = row['minute']
                team = row['team']
                
                # Mock a scoreline (since calculating running score per minute is complex, we mock it for the prompt)
                mock_score = f"{random.randint(0,2)}-{random.randint(0,2)}"
                
                game_state = f"{mock_score}, {minute}th minute"
                
                # Format Action
                if event_type == 'Pass':
                    action = f"{player_name} ({team}) completes a pass"
                else:
                    action = f"{player_name} ({team}) performs a {event_type.lower()}"
                
                # Format Commentary
                commentary_template = random.choice(templates[event_type])
                commentary = commentary_template.format(player=player_name)
                
                # Add flavor based on the minute
                if minute > 85:
                    commentary += " They're running out of time here!"
                elif minute < 10:
                    commentary += " Just trying to settle into the game early on."
                
                generated_data.append({
                    "game_state": game_state,
                    "action": action,
                    "commentary": commentary
                })
        except Exception as e:
            # Skip matches that fail to load
            continue

    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in generated_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"✅ Successfully generated {len(generated_data)} pieces of realistic commentary!")
    print(f"✅ Data saved to: {output_file}")
    
    # Show a sample
    print("\n--- Example Sample ---")
    sample = generated_data[0]
    print(f"Prompt Input: Game: {sample['game_state']} | Action: {sample['action']}")
    print(f"Target Commentary: {sample['commentary']}")

if __name__ == "__main__":
    generate_synthetic_statsbomb_commentary(num_samples=1000)
