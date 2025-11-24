import os
import torch
import random
import uuid
from flask import Flask, render_template, request, jsonify
from game import GameV2, Card, Suit, PassDirection
from train import AIPlayer, HeartsTransformer, HIDDEN_DIM, DROPOUT
from strategies import ExpertPolicy
import gpu_selector

app = Flask(__name__)

# --- Model Loading ---
device = gpu_selector.select_device()
print(f"Running on {device}")

model = HeartsTransformer(d_model=HIDDEN_DIM, dropout=DROPOUT).to(device)
model_path = 'hearts_model.pth'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval() # Set to eval mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print("Warning: No model found. AI will be random/untrained.")

# --- Helper Functions ---
def card_to_dict(card):
    return {'suit': card.suit.name[0], 'rank': card.rank} # S, H, D, C

def dict_to_card(d):
    suit_map = {'S': Suit.SPADES, 'H': Suit.HEARTS, 'D': Suit.DIAMONDS, 'C': Suit.CLUBS}
    return Card(suit_map[d['suit']], d['rank'])

# --- Game Manager ---
class WebGame:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.game = GameV2()
        self.game.reset()
        
        # Player 0 is Human. Players 1, 2, 3 are AI.
        # We use the same model for all AI agents for now.
        self.ai_agents = [AIPlayer(model, device) for _ in range(3)]
        
        self.phase = 'passing' # passing, playing, finished
        self.pass_direction = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        self.msg = f"New Game! Pass Direction: {self.pass_direction.name}"
        
        if self.pass_direction == PassDirection.KEEP:
            self.phase = 'playing'
            self.start_playing_phase()
            
        self.trick_turn_count = 0 # 0 to 3
        self.current_player_idx = -1

    def start_playing_phase(self):
        self.game.gamestate.players[0].hand.sort() # Sort human hand
        self.current_player_idx = self.game.gamestate.get_first_player()
        self.trick_turn_count = 0
        self.game.gamestate.current_table = []
        self.game.gamestate.current_suit = None
        
        self.msg = "Game Started! "
        if self.current_player_idx == 0:
            self.msg += "Your turn (Must lead 2♣)."
        else:
            self.msg += f"Player {self.current_player_idx}'s turn."
            self.advance_ai()

    def advance_ai(self):
        """Run AI turns until it's Human's turn or trick ends."""
        while self.current_player_idx != 0 and self.phase == 'playing':
            ai_idx = self.current_player_idx
            player = self.game.gamestate.players[ai_idx]
            
            # 1. Determine Legal Moves
            # We need to replicate available_actions logic or import it
            from game import available_actions
            scored = any(p.points > 0 for p in self.game.gamestate.players)
            is_first_trick = (self.game.rounds == 1)
            
            legal_moves = available_actions(player, self.game.gamestate.current_suit, is_first_trick, scored)
            
            # 2. AI Decision
            # We use the AIPlayer.play_policy
            # We need to construct info dict
            info = self.game.get_player_info(ai_idx)
            
            # Use Expert Policy mixed with Model? Or just Model?
            # Let's use the Model (AIPlayer)
            # But AIPlayer needs to be reset? No, it's stateless per move mostly, but has history.
            # We should reset AI agents at start of game. (Done in init)
            
            # We use the trained model
            # Note: AIPlayer.play_policy expects 'order' (0-3)
            order = self.trick_turn_count
            
            selected_card = self.ai_agents[ai_idx-1].play_policy(player, info, legal_moves, order)
            
            # 3. Apply Move
            self.apply_move(ai_idx, selected_card)
            
            # 4. Check if trick ended
            if self.trick_turn_count == 0: # Reset to 0 means trick finished
                # Trick finished in apply_move
                # Check if human is next leader
                if self.current_player_idx == 0:
                    self.msg = f"Trick finished. You won the trick! Your lead."
                    return
                else:
                    # AI won, continue loop
                    continue
            
            # If not finished, loop continues to next AI
            # If next is Human (0), loop breaks

    def apply_move(self, player_idx, card):
        player = self.game.gamestate.players[player_idx]
        player.hand.remove(card)
        
        # Update Game State
        self.game.gamestate.current_table.append((card, player_idx))
        self.game.gamestate.table.append((card, player_idx))
        player.table.append(card)
        
        if self.trick_turn_count == 0:
            self.game.gamestate.current_suit = card.suit
            
        # Check Hearts Broken / Piggy
        if not self.game.gamestate.piggy_pulled and (card.suit == Suit.SPADES and card.rank == 12):
            self.game.gamestate.piggy_pulled = True
            self.game.gamestate.heart_broken = True
        
        if card.suit == Suit.HEARTS and not self.game.gamestate.heart_broken:
            self.game.gamestate.heart_broken = True
            
        self.trick_turn_count += 1
        self.current_player_idx = (self.current_player_idx + 1) % 4
        
        # Check Trick End
        if self.trick_turn_count == 4:
            self.finish_trick()

    def finish_trick(self):
        # Determine Winner
        trick = self.game.gamestate.current_table
        lead_suit = self.game.gamestate.current_suit
        
        lead_suit_cards = [(c, pid) for c, pid in trick if c.suit == lead_suit]
        winner_card, winner_idx = max(lead_suit_cards, key=lambda x: (x[0].rank - 2) % 13)
        
        # Calculate Points
        points = sum(c.value() for c, _ in trick)
        self.game.gamestate.players[winner_idx].points += points
        
        # Record History
        from data_structure import TrickRecord
        record = TrickRecord(winner=winner_idx, score=points, lead_suit=lead_suit)
        self.game.trick_history.append(record)
        
        # Reset for next trick
        self.game.gamestate.current_table = []
        self.game.gamestate.current_suit = None
        self.trick_turn_count = 0
        self.current_player_idx = winner_idx
        self.game.rounds += 1
        
        if self.game.rounds > 13:
            self.phase = 'finished'
            self.msg = "Game Over!"
            # Handle STM scoring
            scores = [p.points for p in self.game.gamestate.players]
            if 26 in scores:
                for i in range(4):
                    if self.game.gamestate.players[i].points == 26:
                        self.game.gamestate.players[i].points = 0
                    else:
                        self.game.gamestate.players[i].points = 26
        else:
            self.msg = f"Trick finished. Player {winner_idx} won."

    def human_pass(self, cards):
        # Human passes 'cards'
        # AI agents pass using Expert Policy (for now, to be safe/smart)
        # Or use Model Pass Policy? Let's use Expert for stability in demo.
        
        pass_plans = {0: cards}
        
        for i in range(1, 4):
            # AI Pass
            # Use Expert Policy for passing in demo
            ai_cards = ExpertPolicy.pass_policy(self.game.gamestate.players[i], {})
            pass_plans[i] = ai_cards
            
        # Execute Pass
        # Logic from GameV2.perform_pass
        passed_cards_map = pass_plans
        
        # Distribute
        for sender_idx in range(4):
            target_idx = -1
            if self.pass_direction == PassDirection.LEFT:
                target_idx = (sender_idx + 1) % 4
            elif self.pass_direction == PassDirection.RIGHT:
                target_idx = (sender_idx - 1) % 4
            elif self.pass_direction == PassDirection.ACROSS:
                target_idx = (sender_idx + 2) % 4
            
            received = passed_cards_map[sender_idx]
            # Remove from sender
            for c in received:
                if c in self.game.gamestate.players[sender_idx].hand:
                    self.game.gamestate.players[sender_idx].hand.remove(c)
            
            # Add to target
            self.game.gamestate.players[target_idx].hand.extend(received)
            
        self.phase = 'playing'
        self.start_playing_phase()

    def get_state(self):
        # Return JSON state for UI
        p0 = self.game.gamestate.players[0]
        return {
            'game_id': self.id,
            'phase': self.phase,
            'hand': [card_to_dict(c) for c in sorted(p0.hand)],
            'current_table': [{'card': card_to_dict(c), 'player_id': pid} for c, pid in self.game.gamestate.current_table],
            'scores': [p.points for p in self.game.gamestate.players],
            'message': self.msg,
            'pass_direction': self.pass_direction.name
        }

# Store active games
active_games = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    game = WebGame()
    active_games[game.id] = game
    return jsonify(game.get_state())

@app.route('/pass_cards', methods=['POST'])
def pass_cards():
    data = request.json
    game_id = data.get('game_id')
    cards_data = data.get('cards')
    
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
        
    game = active_games[game_id]
    cards = [dict_to_card(c) for c in cards_data]
    
    game.human_pass(cards)
    return jsonify(game.get_state())

@app.route('/play_card', methods=['POST'])
def play_card():
    data = request.json
    game_id = data.get('game_id')
    card_data = data.get('card')
    
    if game_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
        
    game = active_games[game_id]
    card = dict_to_card(card_data)
    
    # Validate Move
    from game import available_actions
    scored = any(p.points > 0 for p in game.game.gamestate.players)
    is_first_trick = (game.game.rounds == 1)
    legal_moves = available_actions(game.game.gamestate.players[0], game.game.gamestate.current_suit, is_first_trick, scored)
    
    if card not in legal_moves:
        return jsonify({'error': f'Illegal Move! You must follow suit or lead 2♣. Legal: {[str(c) for c in legal_moves]}'}), 400
        
    game.apply_move(0, card)
    game.advance_ai()
    
    return jsonify(game.get_state())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
