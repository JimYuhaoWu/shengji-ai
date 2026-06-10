"""Protocol for serializing/deserializing game state and actions."""

from shengji import Action, ActionType, Card, Suit, Rank, GamePhase, GameState, TrumpBid

# Reverse lookups from wire values to enums
SUIT_BY_VALUE = {s.value: s for s in Suit}
RANK_BY_VALUE = {r.value: r for r in Rank}


def deserialize_card(card_dict: dict) -> Card:
    """Convert {"suit": "H", "rank": "7", "deck_id": 0} to Card object."""
    suit = SUIT_BY_VALUE[card_dict["suit"]]
    rank = RANK_BY_VALUE[card_dict["rank"]]
    deck_id = int(card_dict.get("deck_id", 0))
    return Card(suit=suit, rank=rank, deck_id=deck_id)


def deserialize_action(action_dict: dict) -> Action:
    """Deserialize action dict from server's legal_actions list.

    The server sends action dicts with:
    {
        "index": 0,
        "action_type": "PLAY_CARDS",
        "cards": [...],
        "trump_bid": {...},
        ...
    }
    """
    action_type_str = action_dict.get("action_type")
    action_type = ActionType[action_type_str]

    cards = tuple(
        deserialize_card(c) for c in action_dict.get("cards", [])
    )

    trump_bid = None
    if action_dict.get("trump_bid"):
        bid_dict = action_dict["trump_bid"]
        trump_bid = TrumpBid(
            count=bid_dict["count"],
            suit=SUIT_BY_VALUE[bid_dict["suit"]],
            bidder_id=bid_dict["bidder_id"],
        )

    target_suit = None
    if action_dict.get("target_suit"):
        target_suit = SUIT_BY_VALUE[action_dict["target_suit"]]

    target_card = None
    if action_dict.get("target_card"):
        target_card = deserialize_card(action_dict["target_card"])

    return Action(
        action_type=action_type,
        cards=cards,
        trump_bid=trump_bid,
        target_suit=target_suit,
        target_card=target_card,
    )


def action_to_dict(action: Action) -> dict:
    """Convert Action object to JSON dict for server.

    For simple actions, send semantic message format.
    For complex actions, reconstruct from the Action object.
    """
    if action.action_type == ActionType.PLAY_CARDS:
        return {
            "type": "play_cards",
            "cards": [
                {"suit": c.suit.value, "rank": c.rank.value, "deck_id": c.deck_id}
                for c in action.cards
            ],
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
            "cards": [
                {"suit": c.suit.value, "rank": c.rank.value, "deck_id": c.deck_id}
                for c in action.cards
            ],
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
    """Reconstruct GameState from server's state_update dict.

    Note: The server sends a privacy-filtered view, so opponent hands are unknown.
    We reconstruct what we can with placeholders for missing data.
    """
    our_player_id = state_dict["your_player_id"]
    phase = GamePhase[state_dict["phase"]]

    # Reconstruct hands tuple: we have our hand, sizes for others
    your_hand_dicts = state_dict.get("your_hand", [])
    your_hand = tuple(deserialize_card(c) for c in your_hand_dicts)
    hands_size = state_dict.get("hands_size", [0] * 6)

    # Build hands tuple with our hand at the right index, empties for others
    hands_list = []
    for i in range(6):
        if i == our_player_id:
            hands_list.append(your_hand)
        else:
            hands_list.append(tuple())  # Opponent hands unknown
    hands = tuple(hands_list)

    # Kitty (dealer only, visible during KITTY phase)
    kitty_dicts = state_dict.get("kitty", [])
    kitty = tuple(deserialize_card(c) for c in kitty_dicts) if kitty_dicts else ()

    # Buried cards (revealed at SCORING)
    buried_dicts = state_dict.get("buried_cards", [])
    buried_cards = tuple(deserialize_card(c) for c in buried_dicts) if buried_dicts else ()

    # Trump
    trump_suit = None
    if state_dict.get("trump_suit"):
        trump_suit = SUIT_BY_VALUE[state_dict["trump_suit"]]

    # Current trump bid
    current_trump_bid = None
    if state_dict.get("current_trump_bid"):
        bid_dict = state_dict["current_trump_bid"]
        current_trump_bid = TrumpBid(
            count=bid_dict["count"],
            suit=SUIT_BY_VALUE[bid_dict["suit"]],
            bidder_id=bid_dict["bidder_id"],
        )

    # Current trick: [[player_id, [cards]], ...]
    current_trick_list = state_dict.get("current_trick", [])
    current_trick = tuple(
        (int(pid), tuple(deserialize_card(c) for c in cards))
        for pid, cards in current_trick_list
    )

    # Tricks won: [[winner_id, [cards]], ...]
    tricks_won_list = state_dict.get("tricks_won", [])
    tricks_won = tuple(
        (int(winner_id), tuple(deserialize_card(c) for c in cards))
        for winner_id, cards in tricks_won_list
    )

    # Called suit
    called_suit = None
    if state_dict.get("called_suit"):
        called_suit = SUIT_BY_VALUE[state_dict["called_suit"]]

    # Legal actions (may be None if truncated)
    legal_actions_dicts = state_dict.get("legal_actions", [])
    legal_actions = tuple(
        deserialize_action(a) for a in legal_actions_dicts
    ) if legal_actions_dicts else ()

    # Create GameState with all available data
    state = GameState(
        phase=phase,
        current_player=state_dict["current_player"],
        dealer_id=state_dict["dealer_id"],
        hands=hands,
        kitty=kitty,
        cards_dealt=state_dict.get("cards_dealt", 0),
        trump_suit=trump_suit,
        trump_level=state_dict.get("trump_level", "2"),
        trump_locked=state_dict.get("trump_locked", False),
        current_trump_bid=current_trump_bid,
        buried_cards=buried_cards,
        called_rank=state_dict.get("called_rank"),
        called_suit=called_suit,
        helper_players=tuple(state_dict.get("helper_players", [])),
        current_trick=current_trick,
        tricks_won=tricks_won,
        player_levels=tuple(state_dict.get("player_levels", [])),
        scores=tuple(state_dict.get("scores", [])),
        legal_actions=legal_actions,
    )

    return state
