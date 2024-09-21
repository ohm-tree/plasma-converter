Design principles:

-   The MCTS algorithm should be agnostic to the game.
-   Each agent should be agnostic to the game.
-   The LLM stuff should be completely wrapped inside of prover_llm;
    this means that everything should be passing around GameState
    instances, and we only "realize" that this is a LeanGameState
    inside of prover_llm, which _knows_ that its a LeanGameState.

-   The things that we store in worker outputs should be agnostic
    to the implementation of prover_llm; they should be essentially
    readable to everyone. So, we shouldn't store an exact prompt
    format or anything. This stuff should all get handled by the
    prover_LLM.
