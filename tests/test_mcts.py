import torch
from muzero.mcts import Node, Player

def test_create_a_node():
    prior_prob = 0.22
    player = Player.BLACK
    state = torch.tensor([1, 2, 3])

    node = Node(prior_prob, player, state)
    assert node.value == 0
    assert node.children == {}
    assert node.visits == 0

def test_expand_node():
    pass