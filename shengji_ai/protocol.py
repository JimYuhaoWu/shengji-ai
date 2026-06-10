"""Protocol for serializing/deserializing game state and actions."""

from shengji import GameState, Action, ActionType, Card, Suit, Rank, GamePhase, TrumpBid


def deserialize_card(card_dict: dict) -> Card:
    """Convert {"suit": "H", "rank": "7", "deck_id": 0} to Card object."""
    return Card(
        suit=Suit(card_dict["suit"]),
        rank=Rank(card_dict["rank"]),
        deck_id=card_dict["deck_id"],
    )


def card_to_dict(card: Card) -> dict:
    """Convert Card to {"suit": "H", "rank": "7", "deck_id": 0}."""
    return {
        "suit": card.suit.value,
        "rank": card.rank.value,
        "deck_id": card.deck_id,
    }


def deserialize_action(action_dict: dict) -> Action:
    """Convert action dict from server to Action object.

    The server may send either:
    - {"type": "play_cards", "cards": [...]}
    - {"type": "bid_trump", "count": 1, "suit": "H"}
    - {"type": "call_helper", "suit": "D", "rank": "K"}
    - etc.
    """
    action_type_str = action_dict.get("type")

    if action_type_str == "play_cards":
        cards = tuple(deserialize_card(c) for c in action_dict.get("cards", []))
        return Action(action_type=ActionType.PLAY_CARDS, cards=cards)

    elif action_type_str == "bid_trump":
        count = action_dict["count"]
        suit = Suit(action_dict["suit"])
        # We don't have bidder_id in the message, so we set it to 0 (placeholder)
        trump_bid = TrumpBid(count=count, suit=suit, bidder_id=0)
        return Action(action_type=ActionType.BID_TRUMP, trump_bid=trump_bid)

    elif action_type_str == "pass_trump":
        return Action(action_type=ActionType.PASS_TRUMP)

    elif action_type_str == "take_kitty":
        cards = tuple(deserialize_card(c) for c in action_dict.get("cards", []))
        return Action(action_type=ActionType.TAKE_KITTY, cards=cards)

    elif action_type_str == "call_helper":
        suit = Suit(action_dict["suit"])
        rank = Rank(action_dict["rank"])
        card = Card(suit=suit, rank=rank, deck_id=0)
        return Action(action_type=ActionType.CALL_HELPER, cards=(card,))

    else:
        raise ValueError(f"Unknown action type: {action_type_str}")


def action_to_dict(action: Action) -> dict:
    """Convert Action object to JSON dict for server."""
    if action.action_type == ActionType.PLAY_CARDS:
        return {
            "type": "play_cards",
            "cards": [card_to_dict(c) for c in action.cards],
        }

    elif action.action_type == ActionType.BID_TRUMP:
        return {
            "type": "bid_trump",
            "count": action.trump_bid.count,
            "suit": action.trump_bid.suit.value,
        }

    elif action.action_type == ActionType.PASS_TRUMP:
        return {"type": "pass_trump"}

    elif action.action_type == ActionType.TAKE_KITTY:
        return {
            "type": "take_kitty",
            "cards": [card_to_dict(c) for c in action.cards],
        }

    elif action.action_type == ActionType.CALL_HELPER:
        card = action.cards[0]
        return {
            "type": "call_helper",
            "suit": card.suit.value,
            "rank": card.rank.value,
        }

    else:
        raise ValueError(f"Unknown action type: {action.action_type}")


def deserialize_state(state_dict: dict) -> GameState:
    """Convert state dict from server to GameState object.

    The server sends the full state for the player:
    {
        "phase": "TRICK_PLAYING",
        "current_player": 2,
        "your_hand": [{"suit": "H", "rank": "7", ...}],
        "hands_size": [10, 12, 8, 9, 11, 10],
        "legal_actions": [...],
        "dealer_id": 1,
        ...
    }
    """
    # Deserialize hands - we have our hand, but only sizes for others
    your_hand = [deserialize_card(c) for c in state_dict.get("your_hand", [])]
    hands_size = state_dict.get("hands_size", [])
    hands = []
    for i, size in enumerate(hands_size):
        if len(hands) == state_dict.get("your_player_id", 0):
            hands.append(tuple(your_hand))
        else:
            # For other players, we don't have the actual cards, so we create empty
            # This is a limitation - the GameState will have placeholder hands
            hands.append(tuple())

    # Deserialize kitty
    kitty_cards = state_dict.get("kitty", [])
    kitty = tuple(deserialize_card(c) for c in kitty_cards) if kitty_cards else ()

    # Deserialize trump info
    phase = GamePhase[state_dict.get("phase", "DEALING")]

    hands_tuple = tuple(hands)

    # Create GameState - we need to reconstruct from available data
    # This is tricky because GameState may have validation/invariants
    # For now, we'll assume the server provides enough info to reconstruct
    state = GameState(
        phase=phase,
        current_player=state_dict.get("current_player", 0),
        dealer_id=state_dict.get("dealer_id", 0),
        hands=hands_tuple,
        kitty=kitty,
        cards_dealt=state_dict.get("cards_dealt", 0),
        trump_suit=Suit(state_dict["trump_suit"]) if "trump_suit" in state_dict else None,
        trump_level=state_dict.get("trump_level"),
        trump_locked=state_dict.get("trump_locked", False),
        current_trump_bid=state_dict.get("current_trump_bid"),
        current_trick=state_dict.get("current_trick", []),
        tricks_won=state_dict.get("tricks_won", []),
        scores=state_dict.get("scores"),
        player_levels=state_dict.get("player_levels"),
        called_rank=state_dict.get("called_rank"),
        called_suit=state_dict.get("called_suit"),
        helper_players=state_dict.get("helper_players"),
        buried_cards=state_dict.get("buried_cards"),
        legal_actions=tuple(
            deserialize_action(a) for a in state_dict.get("legal_actions", [])
        ),
    )

    return state
