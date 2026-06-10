"""Rule-based agent with hand-coded heuristics."""

from shengji import GameState, Action, ActionType, Suit, Rank
from shengji.card import Card

from shengji_ai.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """Plays using hand-coded heuristics."""

    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose action based on game phase and heuristics."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Handle non-trick phases
        if state.phase.name == "DEALING":
            # Just pass or take action based on what's available
            return legal_actions[0]

        if state.phase.name == "TRUMP_DECLARATION":
            # Simple: bid if we have strong cards, else pass
            return self._trump_action(state, legal_actions)

        if state.phase.name == "KITTY":
            # Bury weak cards
            return self._kitty_action(state, legal_actions)

        if state.phase.name == "CALL_HELPER":
            # Call a helper strategically
            return self._call_helper_action(state, legal_actions)

        if state.phase.name == "TRICK_PLAYING":
            # Main game logic
            return self._trick_action(state, legal_actions)

        if state.phase.name == "SCORING":
            # Game is over
            return legal_actions[0]

        return legal_actions[0]

    def _trump_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Decide whether to bid trump or pass."""
        # Simple: pass for now. A smarter agent would count trump cards in hand
        pass_actions = [a for a in legal_actions if a.action_type == ActionType.PASS_TRUMP]
        if pass_actions:
            return pass_actions[0]
        return legal_actions[0]

    def _kitty_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose which 6 cards to bury from the kitty."""
        # For now, just take the first legal kitty action
        # A smarter agent would bury weak cards
        kitty_actions = [
            a for a in legal_actions if a.action_type == ActionType.TAKE_KITTY
        ]
        if kitty_actions:
            return kitty_actions[0]
        return legal_actions[0]

    def _call_helper_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose which card to call as helper."""
        # For now, just call the first legal action
        # A smarter agent would call a low card to keep hands balanced
        return legal_actions[0]

    def _trick_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Play a card during trick playing."""
        # Get our hand (stored in state)
        our_hand = self._get_our_hand(state)

        # Check if we're leading (trick is empty)
        is_leading = len(state.current_trick) == 0

        if is_leading:
            return self._lead_action(state, legal_actions, our_hand)
        else:
            return self._follow_action(state, legal_actions, our_hand)

    def _is_red_five(self, card: Card) -> bool:
        """Check if card is ♥5 or ♦5."""
        return (card.suit in (Suit.HEARTS, Suit.DIAMONDS)) and card.rank == Rank.FIVE

    def _lead_action(
        self, state: GameState, legal_actions: list[Action], our_hand: list[Card]
    ) -> Action:
        """Play when leading (first in trick).

        Priority:
        1. Play highest tractor if available
        2. Else highest pair
        3. Else highest single
        4. Avoid red fives if possible
        """
        play_actions = [a for a in legal_actions if a.action_type == ActionType.PLAY_CARDS]
        if not play_actions:
            return legal_actions[0]

        # For simplicity, just play the highest card that isn't a red 5
        cards_in_actions = set()
        for action in play_actions:
            for card in action.cards:
                cards_in_actions.add(card)

        # Sort by rank (descending), but avoid red 5s
        safe_cards = [c for c in cards_in_actions if not self._is_red_five(c)]
        if safe_cards:
            highest_safe = max(safe_cards, key=lambda c: self._card_value(c))
            # Find action with this card
            for action in play_actions:
                if highest_safe in action.cards:
                    return action

        # If no safe card, play anything
        return play_actions[0]

    def _follow_action(
        self, state: GameState, legal_actions: list[Action], our_hand: list[Card]
    ) -> Action:
        """Play when following (not first in trick).

        Priority:
        1. If partner winning: play lowest card
        2. If opponent winning:
           a. Play highest trump if can win
           b. Else play lowest card
        3. Protect red fives
        """
        play_actions = [a for a in legal_actions if a.action_type == ActionType.PLAY_CARDS]
        if not play_actions:
            return legal_actions[0]

        # Check who's winning the trick so far
        # Placeholder: just play lowest card that's not a red 5
        cards_in_actions = set()
        for action in play_actions:
            for card in action.cards:
                cards_in_actions.add(card)

        safe_cards = [c for c in cards_in_actions if not self._is_red_five(c)]
        if safe_cards:
            lowest_safe = min(safe_cards, key=lambda c: self._card_value(c))
            for action in play_actions:
                if lowest_safe in action.cards:
                    return action

        # If no safe card, play lowest
        lowest = min(cards_in_actions, key=lambda c: self._card_value(c))
        for action in play_actions:
            if lowest in action.cards:
                return action

        return play_actions[0]

    def _card_value(self, card: Card) -> int:
        """Return a numeric value for card comparison (higher = better)."""
        rank_order = {
            Rank.TWO: 2,
            Rank.THREE: 3,
            Rank.FOUR: 4,
            Rank.FIVE: 5,
            Rank.SIX: 6,
            Rank.SEVEN: 7,
            Rank.EIGHT: 8,
            Rank.NINE: 9,
            Rank.TEN: 10,
            Rank.JACK: 11,
            Rank.QUEEN: 12,
            Rank.KING: 13,
            Rank.ACE: 14,
            Rank.SMALL_JOKER: 15,
            Rank.LARGE_JOKER: 16,
        }
        return rank_order.get(card.rank, 0)

    def _get_our_hand(self, state: GameState) -> list[Card]:
        """Extract our hand from game state.

        Our hand is the only non-empty hand (due to privacy filtering from server).
        """
        for hand in state.hands:
            if hand:
                return list(hand)
        return []
