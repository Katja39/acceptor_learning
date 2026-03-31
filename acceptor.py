from collections import deque

class Acceptor:

    def __init__(self):
        self.Q = {0}                     # set of states
        self.Sigma = set()               # alphabet
        self.delta = {}                  # transition function (state, char) -> state
        self.q0 = 0                      # start state
        self.F = set()                   # accepting states
        self.next_state_id = 1           # helper for new state IDs
        self.dead_state = None           # tracks the dead state once created

    def learn_word(self, word):
        """
        Inserts a word into the prefix tree.
        """
        current = self.q0
        for ch in word:
            self.Sigma.add(ch)
            needs_new_edge = (current, ch) not in self.delta
            # allow overriding a previous dead-state edge when learning new positive samples
            if not needs_new_edge and self.dead_state is not None:
                needs_new_edge = self.delta[(current, ch)] == self.dead_state

            if needs_new_edge:
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

    def minimize(self, make_total=True):
        """
        Minimizes the automaton to a minimal DFA.
        Optionally make the automaton total (dead state added)
        before Hopcroft's algorithm. Skipping totalization keeps
        partial DFAs minimal without forcing a dead state.
        """
        # 1. Optionally make the automaton total (add dead state if needed)
        if make_total:
            self._make_total()

        # 2. Hopcroft partition refinement
        partitions = self._hopcroft_partitions()

        # 3. Build new automaton from partitions
        self._build_from_partitions(partitions)

        # 4. Remove unreachable states (optional but clean)
        self._remove_unreachable()

        # 5. Keep state-id generator in sync with renumbered states
        self.next_state_id = (max(self.Q) + 1) if self.Q else 0

    def _make_total(self):
        """
        Adds a dead state and defines all missing transitions.
        The dead state is non-accepting and loops to itself for all symbols.
        """
        # Prefer reusing an explicitly tracked dead state if it is still valid,
        # otherwise create a fresh one to avoid misclassifying real states.
        dead = None
        if (
            self.dead_state is not None
            and self.dead_state in self.Q
            and self.dead_state not in self.F
            and all(
                (self.dead_state, ch) in self.delta
                and self.delta[(self.dead_state, ch)] == self.dead_state
                for ch in self.Sigma
            )
        ):
            dead = self.dead_state
        else:
            dead = self.next_state_id
            self.Q.add(dead)
            self.next_state_id += 1
        self.dead_state = dead

        for s in list(self.Q):
            for ch in self.Sigma:
                if (s, ch) not in self.delta:
                    self.delta[(s, ch)] = dead
        for ch in self.Sigma:
            self.delta[(dead, ch)] = dead

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
                if (rep, ch) in self.delta:
                    target = self.delta[(rep, ch)]
                    new_delta[(idx, ch)] = mapping[target]

        self.Q = new_Q
        self.delta = new_delta
        self.q0 = new_q0
        self.F = new_F
        # the dead state (if any) may have merged; recompute handle
        self.dead_state = None

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
            # if dead state was removed, clear handle
            if self.dead_state not in self.Q:
                self.dead_state = None

    def _hopcroft_partitions(self):
        """
        Hopcroft's DFA minimization partition refinement.
        Assumes DFA is total.
        """
        alphabet = sorted(self.Sigma)
        F = set(self.F)
        NF = self.Q - F
        partitions = [F, NF] if NF else [F]
        # remove empties
        partitions = [p for p in partitions if p]

        waiting = partitions.copy()
        while waiting:
            A = waiting.pop()
            for c in alphabet:
                # states that transition via c into A
                X = {q for q in self.Q if (q, c) in self.delta and self.delta[(q, c)] in A}
                new_partitions = []
                for Y in partitions:
                    inter = Y & X
                    diff = Y - X
                    if inter and diff:
                        new_partitions.extend([inter, diff])
                        if Y in waiting:
                            waiting.remove(Y)
                            waiting.extend([inter, diff])
                        else:
                            waiting.append(inter if len(inter) <= len(diff) else diff)
                    else:
                        new_partitions.append(Y)
                partitions = new_partitions
        return partitions
