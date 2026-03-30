import graphviz
from collections import deque

class Acceptor:

    def __init__(self):
        self.Q = {0}                     # set of states
        self.Sigma = set()               # alphabet
        self.delta = {}                  # transition function (state, char) -> state
        self.q0 = 0                      # start state
        self.F = set()                   # accepting states
        self.next_state_id = 1           # helper for new state IDs

    def learn_word(self, word):
        """
        Inserts a word into the prefix tree.
        """
        current = self.q0
        for ch in word:
            self.Sigma.add(ch)
            if (current, ch) not in self.delta:
                new_state = self.next_state_id
                self.Q.add(new_state)
                self.delta[(current, ch)] = new_state
                self.next_state_id += 1
            current = self.delta[(current, ch)]
        self.F.add(current)

    def accepts(self, word):
        """
        Checks if the automaton accepts the word.
        After minimization, the automaton is complete (all transitions defined).
        """
        state = self.q0
        for ch in word:
            if (state, ch) not in self.delta:
                return False
            state = self.delta[(state, ch)]
        return state in self.F

    def minimize(self):
        """
        Minimizes the automaton to a minimal DFA.
        The automaton is first made total (dead state added),
        then minimized using Moore's partitioning algorithm.
        """
        # 1. Make the automaton total (add dead state if needed)
        self._make_total()

        # 2. Partitioning (Moore's algorithm)
        # Initial partition: accepting / non-accepting
        partitions = []
        accepting = {s for s in self.Q if s in self.F}
        non_accepting = self.Q - accepting
        if accepting:
            partitions.append(accepting)
        if non_accepting:
            partitions.append(non_accepting)

        while True:
            new_partitions = []
            for group in partitions:
                # state -> tuple of partition indices of next states (for all symbols)
                profiles = {}
                for s in group:
                    profile = tuple(self._partition_index(self.delta[(s, ch)], partitions)
                                    for ch in sorted(self.Sigma))
                    profiles.setdefault(profile, set()).add(s)
                new_partitions.extend(profiles.values())
            if len(new_partitions) == len(partitions):
                break
            partitions = new_partitions

        # 3. Build new automaton from partitions
        self._build_from_partitions(partitions)

        # 4. Remove unreachable states (optional but clean)
        self._remove_unreachable()

    def _make_total(self):
        """
        Adds a dead state and defines all missing transitions.
        The dead state is non-accepting and loops to itself for all symbols.
        """
        dead = None
        for s in self.Q:
            if s not in self.F:
                all_self = True
                for ch in self.Sigma:
                    if (s, ch) not in self.delta or self.delta[(s, ch)] != s:
                        all_self = False
                        break
                if all_self:
                    dead = s
                    break
        if dead is None:
            dead = self.next_state_id
            self.Q.add(dead)
            self.next_state_id += 1

        for s in list(self.Q):
            for ch in self.Sigma:
                if (s, ch) not in self.delta:
                    self.delta[(s, ch)] = dead
        for ch in self.Sigma:
            self.delta[(dead, ch)] = dead

    def _partition_index(self, state, partitions):
        """
        Returns the index of the partition containing the state.
        """
        for i, group in enumerate(partitions):
            if state in group:
                return i
        raise ValueError(f"State {state} not found in partitions")

    def _build_from_partitions(self, partitions):
        """
        Builds a new automaton where each partition is a state.
        """
        mapping = {}
        for idx, group in enumerate(partitions):
            for s in group:
                mapping[s] = idx

        new_Q = set(range(len(partitions)))
        new_q0 = mapping[self.q0]
        new_F = {mapping[s] for s in self.F if s in mapping}
        new_delta = {}
        for idx, group in enumerate(partitions):
            rep = next(iter(group))
            for ch in sorted(self.Sigma):
                target = self.delta[(rep, ch)]
                new_delta[(idx, ch)] = mapping[target]

        self.Q = new_Q
        self.delta = new_delta
        self.q0 = new_q0
        self.F = new_F

    def _remove_unreachable(self):
        """
        Removes all states not reachable from the start state.
        """
        reachable = set()
        queue = deque([self.q0])
        while queue:
            s = queue.popleft()
            if s in reachable:
                continue
            reachable.add(s)
            for ch in self.Sigma:
                if (s, ch) in self.delta:
                    t = self.delta[(s, ch)]
                    if t not in reachable:
                        queue.append(t)

        unreachable = self.Q - reachable
        if unreachable:
            self.Q = reachable
            self.F &= reachable
            new_delta = {}
            for (s, ch), t in self.delta.items():
                if s in reachable and t in reachable:
                    new_delta[(s, ch)] = t
            self.delta = new_delta

    def visualize(self, filename="acceptor"):
        """
        Generates a graphical representation of the automaton (only reachable states).
        """
        dot = graphviz.Digraph(comment='Automaton')
        dot.attr(rankdir='LR')

        # Start arrow
        dot.node('start', '', shape='none')
        dot.edge('start', f"s{self.q0}")

        for state in self.Q:
            name = f"s{state}"
            if state in self.F:
                dot.node(name, name, shape='doublecircle')
            else:
                dot.node(name, name, shape='circle')

        for (state, char), next_state in self.delta.items():
            if state in self.Q and next_state in self.Q:
                dot.edge(f"s{state}", f"s{next_state}", label=char)

        dot.render(filename, format='png', view=True, cleanup=True)
        print(f"Graph saved and opened as {filename}.png")


if __name__ == "__main__":
    learner = Acceptor()

    #Example
    training_words = ["ab", "aba", "bb", "bba", "abb", "ba"]

    print("--- Learning phase ---")
    for word in training_words:
        learner.learn_word(word)
        learner.minimize()
        print(f"Word '{word}' learned.")

    print("\n--- Formal Representation ---")
    print(f"State set Q: {[f's{s}' for s in sorted(learner.Q)]}")
    print(f"Start state q0: s{learner.q0}")
    print(f"Accepting states F: {[f's{s}' for s in sorted(learner.F)]}")

    print("Transition function delta:")
    for (state, char), target in sorted(learner.delta.items()):
        print(f"  delta(s{state}, '{char}') = s{target}")

    print("\n--- Test phase ---")
    test_words = ["ab", "aba", "a", "bb", "b", "bbaa", "xyz"]

    for word in test_words:
        if learner.accepts(word):
            print(f"The word '{word}' BELONGS to the language.")
        else:
            print(f"The word '{word}' does NOT belong to the language.")

    learner.visualize("learned_acceptor")