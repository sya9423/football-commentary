import os
import json
import random
from statsbombpy import sb

# =============================================================================
# MASSIVE TEMPLATE LIBRARY - Sounds like real TV commentary
# =============================================================================

TEMPLATES = {
    'Shot': {
        'on_target': [
            "WHAT A STRIKE from {player}! The keeper just about gets a hand to it!",
            "{player} FIRES towards goal! Saved brilliantly!",
            "{player} lets fly from distance! The goalkeeper pushes it wide!",
            "OH! {player} rattles the crossbar! So close!",
            "{player} unleashes a thunderbolt! Corner!",
            "A fierce drive from {player}! Tipped over!",
            "{player} goes low and hard! The keeper gets down well!",
            "Brilliant effort from {player}! Just wide of the post!",
            "{player} cuts inside and curls one towards the far corner! Saved!",
            "{player} tries his luck from 25 yards! Not far away!",
        ],
        'off_target': [
            "{player} blazes it over the bar! He'll be disappointed with that.",
            "Wild effort from {player}! That's gone into Row Z!",
            "{player} snatches at it and drags it wide. Wasted opportunity.",
            "Skied! {player} got under that one. Should have done better.",
            "{player} pulls the trigger but it's well off target.",
            "That's gone sailing over! {player} will want that one back.",
            "{player} loses his composure in front of goal. Blazed over.",
            "Dragged wide by {player}! He had options there.",
        ],
        'goal': [
            "GOAAAAAL! {player} smashes it into the back of the net! INCREDIBLE!",
            "HE'S DONE IT! {player} scores! What a moment!",
            "IT'S IN! {player} buries it! The crowd goes absolutely wild!",
            "GOAL! GOAL! GOAL! {player} with a clinical finish!",
            "{player} slots it home cool as you like! That is ice cold!",
            "GET IN! {player} with a stunning strike! Unstoppable!",
            "The net BULGES! {player} has found the bottom corner! Beautiful!",
            "{player} heads it in! WHAT A HEADER! The defence was all at sea!",
            "A tap-in for {player}! Great team goal, worked it beautifully!",
            "OH MY WORD! {player} from outside the box! That is SENSATIONAL!",
            "{player} rounds the keeper and rolls it into an empty net! Cool as ice!",
            "SCREAMER from {player}! Top bins! The goalkeeper had absolutely no chance!",
        ],
    },
    'Pass': {
        'normal': [
            "{player} keeps it ticking over nicely.",
            "Neat and tidy from {player}. Keeps possession.",
            "{player} recycles the ball back across the pitch.",
            "Patient build-up play here. {player} finds a teammate.",
            "{player} with a short pass. They're probing for an opening.",
            "Sideways from {player}. Looking for the gap.",
        ],
        'key_pass': [
            "OH LOVELY BALL from {player}! That splits the defence wide open!",
            "Inch-perfect pass from {player}! What vision!",
            "{player} threads the needle! Brilliant through ball!",
            "Magnificent delivery from {player}! That's world class!",
            "{player} picks out the run with a gorgeous pass!",
            "WHAT A BALL by {player}! He's seen the run early!",
            "Delightful pass from {player}! Right into the danger zone!",
            "{player} whips in a beauty! Someone needs to get on the end of that!",
        ],
    },
    'Foul Committed': {
        'normal': [
            "That's a foul by {player}. Free kick.",
            "{player} goes through the back of his man. Referee blows immediately.",
            "Cynical foul from {player}. He knew exactly what he was doing.",
            "{player} clips the ankle. That's got to be a free kick.",
            "Late challenge from {player}! The referee has words with him.",
            "{player} catches him on the follow through. Unfortunate.",
            "No need for that from {player}. Unnecessary challenge.",
            "Tactical foul by {player}. Stops a dangerous counter-attack.",
        ],
        'yellow': [
            "YELLOW CARD! {player} is booked! He's walking a tightrope now!",
            "That's a booking for {player}! He can't be doing that!",
            "The card comes out! {player} was far too reckless there!",
            "{player} picks up a caution. He needs to be careful for the rest of this match.",
            "Into the book goes {player}! Deserved yellow card.",
        ],
        'red': [
            "RED CARD! {player} is OFF! That's a straight red!",
            "HE'S BEEN SENT OFF! {player} sees red! What was he thinking?!",
            "That is a HORROR tackle from {player}! Red card, no question about it!",
        ],
    },
    'Dribble': {
        'success': [
            "{player} glides past his marker! Silky smooth!",
            "OH THE SKILL! {player} leaves the defender for dead!",
            "{player} drops the shoulder and he's away! Magnificent!",
            "Incredible footwork from {player}! You cannot stop this man!",
            "{player} dances through the challenge! Poetry in motion!",
            "He's gone past one... past two... {player} is on fire tonight!",
            "{player} nutmegs the defender! The crowd love that!",
            "Breathtaking from {player}! He makes it look so, so easy!",
        ],
        'fail': [
            "{player} tries to beat his man but runs into trouble.",
            "Good defending! {player} is dispossessed.",
            "{player} overcomplicates it. Should have passed earlier.",
            "The defender reads {player}'s intentions perfectly. Ball won.",
        ],
    },
    'Ball Recovery': {
        'normal': [
            "{player} with the interception! Great reading of the game!",
            "Superb defensive awareness from {player}! Mopped that up nicely.",
            "{player} picks the pocket of the attacker! Brilliant!",
            "Vital intervention from {player}! That was heading towards danger!",
            "{player} wins it back with a perfectly timed challenge!",
            "Recovery tackle from {player}! The crowd appreciate that!",
            "{player} nips in and steals possession. Very alert.",
            "Outstanding from {player}! He covers so much ground!",
        ],
    },
    'Clearance': {
        'normal': [
            "{player} hacks it clear! Job done!",
            "Headed away by {player}! No nonsense defending!",
            "{player} boots it into Row Z! Safety first!",
            "Big clearance from {player}! They were under serious pressure there!",
            "{player} gets a vital block in! Brave defending!",
        ],
    },
}

# Extra context injectors based on game state
MINUTE_FLAVORS = {
    'early': [
        " Both teams are still feeling each other out.",
        " It's early days yet.",
        " The match is just getting started.",
        " Settling into the rhythm of the game.",
    ],
    'first_half': [
        "",  # Often no extra context needed mid-half
        " The tempo is picking up now.",
        " This has been an entertaining first half.",
    ],
    'halftime_approaching': [
        " We're approaching the break!",
        " Not long until the half-time whistle.",
        " Can they get one more before the break?",
    ],
    'second_half': [
        "",
        " The second half has been lively!",
        " Both managers will be making adjustments.",
    ],
    'late': [
        " Time is running out!",
        " We're into the final minutes here!",
        " It's now or never!",
        " Everything on the line in these closing stages!",
        " The clock is ticking!",
    ],
    'stoppage': [
        " We're deep into stoppage time!",
        " Is there one last twist in this tale?!",
        " Surely the final whistle is coming any moment!",
    ],
}

SCORE_FLAVORS = {
    'winning': [
        " They're protecting their lead.",
        " Comfortable cushion for them.",
    ],
    'losing': [
        " They desperately need something here.",
        " Backs against the wall now.",
    ],
    'drawing': [
        " It's all level.",
        " Nothing separating the two sides.",
    ],
}


def get_minute_context(minute):
    if minute <= 10:
        return random.choice(MINUTE_FLAVORS['early'])
    elif minute <= 40:
        return random.choice(MINUTE_FLAVORS['first_half'])
    elif minute <= 45:
        return random.choice(MINUTE_FLAVORS['halftime_approaching'])
    elif minute <= 75:
        return random.choice(MINUTE_FLAVORS['second_half'])
    elif minute <= 90:
        return random.choice(MINUTE_FLAVORS['late'])
    else:
        return random.choice(MINUTE_FLAVORS['stoppage'])


def generate_commentary(output_file='training_data/commentary.jsonl', num_samples=1500):
    os.makedirs('training_data', exist_ok=True)
    print("🏟️  Connecting to StatsBomb Open Data...")

    matches = sb.matches(competition_id=43, season_id=3)
    match_ids = matches['match_id'].tolist()
    print(f"Found {len(match_ids)} matches. Generating {num_samples} training samples...\n")

    generated_data = []
    event_type_map = {
        'Pass': 'Pass',
        'Shot': 'Shot',
        'Foul Committed': 'Foul Committed',
        'Dribble': 'Dribble',
        'Ball Recovery': 'Ball Recovery',
        'Clearance': 'Clearance',
    }

    for match_id in match_ids:
        if len(generated_data) >= num_samples:
            break

        try:
            events = sb.events(match_id=match_id)
            valid = events[events['type'].isin(event_type_map.keys())].dropna(subset=['player'])
            sampled = valid.sample(n=min(80, len(valid)))

            for _, row in sampled.iterrows():
                if len(generated_data) >= num_samples:
                    break

                event_type = row['type']
                player = row['player']
                minute = int(row['minute'])
                team = row['team']

                # Build game state
                home_score = random.randint(0, 3)
                away_score = random.randint(0, 3)
                game_state = f"{home_score}-{away_score}, {minute}th minute"

                # Build action description
                if event_type == 'Shot':
                    # Determine shot outcome for template selection
                    outcome = row.get('shot_outcome', 'Off Target')
                    if outcome == 'Goal':
                        sub = 'goal'
                        action = f"{player} ({team}) shoots and SCORES"
                    elif outcome in ['Saved', 'Saved To Post']:
                        sub = 'on_target'
                        action = f"{player} ({team}) shoots on target"
                    else:
                        sub = 'off_target'
                        action = f"{player} ({team}) shoots off target"
                elif event_type == 'Pass':
                    is_key = random.random() < 0.25
                    sub = 'key_pass' if is_key else 'normal'
                    action = f"{player} ({team}) plays a {'key ' if is_key else ''}pass"
                elif event_type == 'Foul Committed':
                    card = row.get('foul_committed_card', None)
                    if card == 'Yellow Card':
                        sub = 'yellow'
                    elif card == 'Red Card':
                        sub = 'red'
                    else:
                        sub = 'normal'
                    action = f"{player} ({team}) commits a foul"
                elif event_type == 'Dribble':
                    outcome = row.get('dribble_outcome', 'Complete')
                    sub = 'success' if outcome == 'Complete' else 'fail'
                    action = f"{player} ({team}) attempts a dribble"
                elif event_type == 'Ball Recovery':
                    sub = 'normal'
                    action = f"{player} ({team}) recovers the ball"
                elif event_type == 'Clearance':
                    sub = 'normal'
                    action = f"{player} ({team}) makes a clearance"
                else:
                    continue

                # Pick template
                template_list = TEMPLATES.get(event_type, {}).get(sub)
                if not template_list:
                    continue

                commentary = random.choice(template_list).format(player=player)

                # Add game context flavor (50% of the time for variety)
                if random.random() < 0.5:
                    commentary += get_minute_context(minute)

                generated_data.append({
                    "game_state": game_state,
                    "action": action,
                    "commentary": commentary,
                })

        except Exception:
            continue

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in generated_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Successfully generated {len(generated_data)} commentary examples!")
    print(f"✅ Data saved to: {output_file}")

    # Show samples
    print("\n--- Example Samples ---")
    for s in random.sample(generated_data, min(5, len(generated_data))):
        print(f"  Game: {s['game_state']}")
        print(f"  Action: {s['action']}")
        print(f"  Commentary: {s['commentary']}\n")


if __name__ == "__main__":
    generate_commentary(num_samples=1500)
