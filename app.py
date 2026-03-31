import graphviz
import streamlit as st
import pandas as pd
import copy
from collections import deque

from acceptor import Acceptor

def parse_words(raw: str) -> list[str]:
    """Split user input (newlines or commas) into a clean word list."""
    if not raw:
        return []
    separators = raw.replace(",", "\n").replace(";", "\n").splitlines()
    return [word.strip() for word in separators if word.strip()]


def to_graph(automaton: Acceptor) -> graphviz.Digraph:
    """Create a Graphviz graph for the current automaton state."""
    dot = graphviz.Digraph(comment="Automaton")
    dot.attr(rankdir="LR")

    label_map = build_label_map(automaton)

    dot.node("start", "", shape="none")
    dot.edge("start", label_map[automaton.q0])

    for state in sorted(automaton.Q):
        name = label_map[state]
        shape = "doublecircle" if state in automaton.F else "circle"
        dot.node(name, name, shape=shape)

    for (state, target), chars in merge_transitions(automaton).items():
        if state in automaton.Q and target in automaton.Q:
            label = ",".join(sorted(chars))
            dot.edge(label_map[state], label_map[target], label=label)

    return dot


def transitions_to_lines(automaton: Acceptor, label_map: dict[int, str]) -> list[str]:
    """
    Return transitions
    """
    lines = []
    for (state, target), chars in merge_transitions(automaton).items():
        if state in automaton.Q and target in automaton.Q:
            label = ",".join(sorted(chars))
            lines.append(f"{label_map[state]} --{label}--> {label_map[target]}")
    return lines


def merge_transitions(automaton: Acceptor) -> dict[tuple[int, int], set[str]]:
    """
    Group transitions by (source, target) and collect all labels.
    """
    grouped: dict[tuple[int, int], set[str]] = {}
    for (state, char), target in automaton.delta.items():
        grouped.setdefault((state, target), set()).add(char)
    return grouped


def build_trace(automaton: Acceptor, label_map: dict[int, str], word: str):
    """
    Follow the word through the automaton and return trace lines and acceptance.
    """
    lines = []
    state = automaton.q0
    for ch in word:
        next_state = automaton.delta.get((state, ch))
        if next_state is None:
            lines.append(f"{label_map[state]} --{ch}--> (kein Übergang)")
            return lines, False
        lines.append(f"{label_map[state]} --{ch}--> {label_map[next_state]}")
        state = next_state
    return lines, state in automaton.F


def longest_common_prefix(words: list[str]) -> str:
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix) and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def longest_common_suffix(words: list[str]) -> str:
    if not words:
        return ""
    suffix = words[0]
    for w in words[1:]:
        while not w.endswith(suffix) and suffix:
            suffix = suffix[1:]
        if not suffix:
            break
    return suffix


def longest_common_substring(words: list[str]) -> str:
    """
    Longest contiguous substring shared by all words.
    Brute-force over the shortest word; fine for small classroom examples.
    """
    if not words:
        return ""
    shortest = min(words, key=len)
    n = len(shortest)
    for length in range(n, 0, -1):
        for start in range(n - length + 1):
            cand = shortest[start : start + length]
            if all(cand in w for w in words):
                return cand
    return ""


def choose_generalization(words: list[str]) -> tuple[str, str] | None:
    """
    Decide which simple heuristic to use (suffix / prefix / contains substring).
    Returns (mode, token) or None for 'exact'.
    Tie‑break priority: suffix > prefix > contains, with longer token preferred.
    """
    if not words or any(len(w) == 0 for w in words):
        return None

    suffix = longest_common_suffix(words)
    prefix = longest_common_prefix(words)
    substring = longest_common_substring(words)

    candidates = []
    if suffix:
        candidates.append(("suffix", suffix))
    if prefix:
        candidates.append(("prefix", prefix))
    if substring and len(substring) >= 2:
        candidates.append(("contains", substring))

    if not candidates:
        return None

    priority = {"suffix": 2, "prefix": 1, "contains": 0}
    return max(candidates, key=lambda c: (len(c[1]), priority[c[0]]))


def describe_language(words: list[str], generalized: bool) -> tuple[str, str, str]:
    """
    Provide a short language description plus a compact example list.
    Mirrors the generalization heuristic: prefer suffix, else prefix,
    else contains-substring, otherwise stick to the listed samples.
    """
    if not words:
        return "∅", "", "empty"
    examples = "; ".join(words[:6]) + ("; …" if len(words) > 6 else "")
    if generalized and not any(len(w) == 0 for w in words):
        choice = choose_generalization(words)
        if choice:
            mode, token = choice
            if mode == "suffix":
                return f"Σ*·{token}", examples, "suffix"
            if mode == "prefix":
                return f"{token}·Σ*", examples, "prefix"
            if mode == "contains":
                return f"Σ*·{token}·Σ*", examples, "contains"
    return "{" + "; ".join(words) + "}", examples, "exact"


def build_prefix_dfa(sigma: set[str], prefix: str) -> Acceptor:
    """Accepts all words that start with the given prefix."""
    m = len(prefix)
    dead = m + 1
    gen = Acceptor()
    gen.Q = set(range(m + 2))
    gen.q0 = 0
    gen.F = {m}
    gen.Sigma = set(sigma) | set(prefix)
    gen.delta = {}

    for state in range(m):
        expected = prefix[state]
        for ch in gen.Sigma:
            if ch == expected:
                gen.delta[(state, ch)] = state + 1
            else:
                gen.delta[(state, ch)] = dead
    # accepting state: loop on everything
    for ch in gen.Sigma:
        gen.delta[(m, ch)] = m
    # dead state: self-loop
    for ch in gen.Sigma:
        gen.delta[(dead, ch)] = dead

    gen.next_state_id = dead + 1
    return ensure_total(gen)


def build_suffix_dfa(sigma: set[str], suffix: str) -> Acceptor:
    """Accepts all words that end with the given suffix (KMP-style automaton)."""
    m = len(suffix)
    gen = Acceptor()
    gen.Q = set(range(m + 1))
    gen.q0 = 0
    gen.F = {m}
    gen.Sigma = set(sigma) | set(suffix)
    gen.delta = {}

    # prefix function for KMP
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and suffix[k] != suffix[i]:
            k = pi[k - 1]
        if suffix[k] == suffix[i]:
            k += 1
        pi[i] = k

    def next_len(current: int, ch: str) -> int:
        while current > 0 and suffix[current] != ch:
            current = pi[current - 1]
        if current < m and suffix[current] == ch:
            current += 1
        return current

    for state in range(m + 1):
        for ch in gen.Sigma:
            gen.delta[(state, ch)] = next_len(state if state < m else pi[m - 1], ch)

    gen.next_state_id = m + 1
    return ensure_total(gen)


def build_contains_dfa(sigma: set[str], pattern: str) -> Acceptor:
    """Accepts all words that contain the given pattern (Σ*·pattern·Σ*)."""
    m = len(pattern)
    gen = Acceptor()
    gen.Q = set(range(m + 1))
    gen.q0 = 0
    gen.F = {m}
    gen.Sigma = set(sigma) | set(pattern)
    gen.delta = {}

    # prefix function (KMP)
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = pi[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        pi[i] = k

    def next_len(current: int, ch: str) -> int:
        while current > 0 and pattern[current] != ch:
            current = pi[current - 1]
        if current < m and pattern[current] == ch:
            current += 1
        return current

    for state in range(m + 1):
        for ch in gen.Sigma:
            if state == m:
                gen.delta[(state, ch)] = m  # once found, stay accepting
            else:
                gen.delta[(state, ch)] = next_len(state, ch)

    gen.next_state_id = m + 1
    return ensure_total(gen)


def build_generalized_acceptor(words: list[str], generalize: bool) -> Acceptor:
    """
    Build either the exact DEA (prefix tree + minimization) or a generalized DEA.
    Heuristics:
    - längster gemeinsamer Suffix (>=1): akzeptiere Σ*·Suffix
    - längster gemeinsamer Präfix (>=1): akzeptiere Präfix·Σ*
    - längster gemeinsamer Teilstring (>=2): akzeptiere Σ*·Teilstring·Σ*
    - sonst: exakte Minimierung
    """
    sigma = {ch for w in words for ch in w}

    # exact model
    exact = Acceptor()
    for w in words:
        exact.learn_word(w)
    exact.Sigma = sigma
    exact.minimize()
    ensure_total(exact)

    if not generalize or not words:
        return exact

    # allow generalization even mit nur einem Wort (mehr Akzeptanz)
    if any(len(w) == 0 for w in words):
        return exact

    choice = choose_generalization(words)
    if not choice:
        return exact

    mode, token = choice
    if mode == "suffix":
        return build_suffix_dfa(sigma, token)
    if mode == "prefix":
        return build_prefix_dfa(sigma, token)
    if mode == "contains":
        return build_contains_dfa(sigma, token)

    return exact


def models_equal(a: Acceptor, b: Acceptor) -> bool:
    """
    Language equivalence check for two DFAs (state ids may differ).
    """
    sigma = a.Sigma | b.Sigma
    # work on copies to avoid mutating originals
    a_c = copy.deepcopy(a)
    b_c = copy.deepcopy(b)
    a_c.Sigma = set(sigma)
    b_c.Sigma = set(sigma)
    ensure_total(a_c)
    ensure_total(b_c)

    def force_dead_state(automaton: Acceptor):
        """
        Ensure an explicit dead state exists so delta lookups never yield None.
        """
        if automaton.dead_state is None:
            dead = max(automaton.Q) + 1 if automaton.Q else 0
            automaton.Q.add(dead)
            automaton.dead_state = dead
            automaton.next_state_id = max(automaton.next_state_id, dead + 1)
            for ch in automaton.Sigma:
                automaton.delta[(dead, ch)] = dead
        # also fill any remaining gaps to the dead state
        for s in list(automaton.Q):
            for ch in automaton.Sigma:
                automaton.delta.setdefault((s, ch), automaton.dead_state)

    force_dead_state(a_c)
    force_dead_state(b_c)

    seen = set()
    queue = deque([(a_c.q0, b_c.q0)])
    while queue:
        s1, s2 = queue.popleft()
        if (s1, s2) in seen:
            continue
        seen.add((s1, s2))
        if (s1 in a_c.F) != (s2 in b_c.F):
            return False
        for ch in sigma:
            t1 = a_c.delta.get((s1, ch), a_c.dead_state)
            t2 = b_c.delta.get((s2, ch), b_c.dead_state)
            queue.append((t1, t2))
    return True


def ensure_total(automaton: Acceptor) -> Acceptor:
    """
    Ensure DFA completeness: every state has transitions for every symbol.
    Adds a dead state only if necessary.
    """
    missing = False
    for s in automaton.Q:
        for ch in automaton.Sigma:
            if (s, ch) not in automaton.delta:
                missing = True
                break
        if missing:
            break
    if missing:
        automaton._make_total()
    return automaton


def build_label_map(automaton: Acceptor) -> dict[int, str]:
    """
    Map internal state ids to ascending labels
    Start state is always z0, remaining states follow numeric order.
    """
    label_map = {automaton.q0: "z0"}
    counter = 1
    for s in sorted(automaton.Q):
        if s == automaton.q0:
            continue
        label_map[s] = f"z{counter}"
        counter += 1
    return label_map


st.set_page_config(page_title="Lernen von Akzeptoren", layout="wide")
st.title("Lernen von Akzeptoren")

# --- Training-Input -------------------------------------------------------
st.subheader("Trainingsdaten")
training_raw = st.text_area(
    "Wörter (durch Zeilenumbruch, Komma oder Semikolon getrennt)",
    value="ab\naba\nbb\nbba\nabb\nba",
    height=160,
)
col_btn, col_chk = st.columns([1, 1])
with col_btn:
    run_training = st.button("Lernen & minimieren")
with col_chk:
    generalize = st.checkbox(
        "Heuristische Verallgemeinerung",
        value=st.session_state.get("generalize_toggle", True),
        key="generalize_toggle",
        on_change=lambda: st.session_state.update({"generalize_toggle_changed": True}),
    )
    st.caption("Heuristik: längster gemeinsamer Suffix → Σ*·Suffix, "
           "sonst längster gemeinsamer Präfix → Präfix·Σ*, "
           "sonst längster gemeinsamer Teilstring → Σ*·Teilstring·Σ*, "
           "sonst exakte Minimierung.")

# --- Session State ------------------------------------------------------------
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "acceptor" not in st.session_state:
    st.session_state.acceptor = None

# --- Training -----------------------------------------------------------------
recompute = run_training or st.session_state.get("generalize_toggle_changed", False)
if recompute:
    words = parse_words(training_raw)
    if not words:
        st.warning("Bitte mindestens ein Trainingswort eingeben.")
    elif len(words) < 2:
        st.session_state.acceptor = None
        st.session_state.snapshots = []
        st.error("Bitte mindestens zwei Wörter eingeben, damit der Akzeptor gelernt werden kann.")
    else:
        try:
            st.session_state.training_words = words
            snapshots = []
            prev_model = None
            for idx in range(1, len(words) + 1):
                learner = build_generalized_acceptor(words[:idx], generalize)
                labels = build_label_map(learner)
                transitions = transitions_to_lines(learner, labels)
                word = words[idx - 1]
                trace_lines, accepted = build_trace(learner, labels, word)
                prev_accepts = prev_model.accepts(word) if prev_model else None
                changed = True if prev_model is None else not models_equal(prev_model, learner)
                prev_states = len(prev_model.Q) if prev_model else None
                prev_edges = len(prev_model.delta) if prev_model else None
                curr_states = len(learner.Q)
                curr_edges = len(learner.delta)
                dead = learner.dead_state
                dead_incoming = False
                if dead is not None:
                    dead_incoming = any(
                        s != dead and t == dead for (s, ch), t in learner.delta.items()
                    )
                lang_desc, lang_examples, lang_mode = describe_language(words[:idx], generalize)
                snapshots.append(
                    {
                        "step": idx,
                        "word": word,
                        "graph": to_graph(learner),
                        "states": [labels[learner.q0]]
                        + [labels[s] for s in sorted(learner.Q) if s != learner.q0],
                        "accepting": [labels[s] for s in sorted(learner.F)],
                        "alphabet": sorted(learner.Sigma),
                        "start": labels[learner.q0],
                        "transitions": transitions,
                        "trace": trace_lines,
                        "accepted": accepted,
                        "language": lang_desc,
                        "lang_mode": lang_mode,
                        "examples": lang_examples,
                        "generalize": generalize,
                        "prev_accepts": prev_accepts,
                        "changed": changed,
                        "prev_states": prev_states,
                        "prev_edges": prev_edges,
                        "curr_states": curr_states,
                        "curr_edges": curr_edges,
                        "dead_incoming": dead_incoming,
                    }
                )
                prev_model = learner
            final_model = build_generalized_acceptor(words, generalize)

            st.session_state.acceptor = final_model
            st.session_state.snapshots = snapshots
            st.success(f"{len(words)} Wörter gelernt und DEA minimiert.")
        except Exception as exc:
            st.session_state.acceptor = None
            st.session_state.snapshots = []
            st.error(f"Fehler beim Lernen: {exc}")
st.session_state["generalize_toggle_changed"] = False

# --- Visualize -------------------------------------------
snapshots = st.session_state.snapshots
if snapshots:
    st.subheader("Lernverlauf")
    step = st.slider(
        "Schritt auswählen",
        min_value=1,
        max_value=len(snapshots),
        value=len(snapshots),
        format="Schritt %d",
    )
    snap = snapshots[step - 1]
    with st.expander("Was passiert in diesem Schritt?"):
        steps_descr = []
        steps_descr.append(f"1. Neues Wort `{snap['word']}` wird gegen Hypothese H{step-1 if step>1 else 0} getestet.")
        if step == 1:
            steps_descr.append("2. Start mit leerem Akzeptor: Pfad des Wortes wird als Präfix-Baum angelegt.")
        else:
            prev_accepts = snap.get("prev_accepts")
            if prev_accepts:
                steps_descr.append("2. Ergebnis: wird bereits akzeptiert.")
                if snap.get("changed") is False:
                    steps_descr.append("Keine weiteren Aktionen nötig (Hypothese bleibt unverändert).")
                else:
                    steps_descr.append("3. Minimierung fasst Äquivalenzklassen zusammen; Sprache bleibt gleich.")
            else:
                steps_descr.append("2. Ergebnis: wird NICHT akzeptiert.")
                steps_descr.append("3. Fehlende Pfade werden als Präfix-Baum ergänzt (neue Zustände/Kanten).")
        prev_states = snap.get("prev_states")
        prev_edges = snap.get("prev_edges")
        curr_states = snap.get("curr_states")
        curr_edges = snap.get("curr_edges")
        if not snap.get("prev_accepts") or snap.get("changed"):
            if prev_states is not None and (curr_states < prev_states or curr_edges < prev_edges):
                steps_descr.append(f"4. Überprüfung Minimierung: verschmolz Zustände ({prev_states} → {curr_states}), Kanten ({prev_edges} → {curr_edges}).")
            else:
                steps_descr.append("4. Überprüfung Minimierung: keine weiteren Zusammenlegungen nötig.")
            if snap.get("dead_incoming"):
                steps_descr.append("5. Überprüfung Komplettierung: fehlende Übergänge ergänzt → Dead-State genutzt.")
            else:
                steps_descr.append("5. Überprüfung Komplettierung: keine fehlenden Übergänge gefunden (kein neuer Dead-State-Eingang).")

        st.markdown("\n".join(steps_descr))

    col_graph, col_last, col_sig = st.columns([2, 1, 1])
    with col_graph:
        st.markdown(f"**Hypothese {snap['step']}**")
        st.graphviz_chart(snap["graph"])
        with st.expander("Diagramm in Textform"):
            st.code("\n".join(snap["transitions"]), language="text")
    with col_last:
        st.write(f"Letztes Wort: `{snap['word']}`")
        st.markdown("**Pfad**")
        st.code("\n".join(snap["trace"]), language="text")
    with col_sig:
        observed_words = st.session_state.get("training_words", [])[:step]
        lang_desc = snap.get("language", "∅")
        example_str = snap.get("examples", "")
        st.markdown("**Formale Signatur**")
        if observed_words:
            st.write(f"V_{step} = {{{'; '.join(observed_words)}}}")
        else:
            st.write("V = ∅")
        if example_str:
            st.caption(f"Beispiele: {example_str}")
        st.write(f"𝔄 = (Σ, Z, δ, F, z₀)")
        st.write(f"Σ = {{{', '.join(snap['alphabet'])}}}")
        st.write(f"Z = {{{', '.join(snap['states'])}}}")
        st.write(f"F = {{{', '.join(snap['accepting'])}}}")
        st.write(f"z₀ = {snap['start']}")
        mode = snap.get("lang_mode")
        if mode == "suffix":
            st.caption("Hinweis: Generalisierung über gemeinsamen Suffix ⇒ akzeptiert ggf. mehr als V.")
        elif mode == "prefix":
            st.caption("Hinweis: Generalisierung über gemeinsamen Präfix ⇒ akzeptiert ggf. mehr als V.")
        elif mode == "contains":
            st.caption("Hinweis: Generalisierung über gemeinsamen Teilstring ⇒ akzeptiert ggf. mehr als V.")
        elif mode == "exact":
            st.caption("Exakte Hypothese über die bisherigen Wörter.")

# --- Tests --------------------------------------------------------------------
st.subheader("Testen")
test_raw = st.text_area(
    "Testwörter",
    value="",
    height=160,
)
run_tests = st.button("Akzeptanz prüfen")

if run_tests:
    tester = st.session_state.acceptor
    if tester is None:
        st.error("Bitte zuerst trainieren (oben auf 'Lernen & minimieren').")
    else:
        words = parse_words(test_raw)
        if not words:
            st.error("Bitte mindestens ein Testwort eingeben.")
        else:
            results = []
            for w in words:
                accepted = tester.accepts(w)
                results.append(
                    {"Wort": w, "Status": "akzeptiert" if accepted else "abgelehnt"}
                )
            df = pd.DataFrame(results)
            def color_status(val):
                return "color: green;" if val == "akzeptiert" else "color: red;"
            styled = df.style.applymap(color_status, subset=["Status"])
            st.subheader("Testergebnisse")
            st.dataframe(styled, use_container_width=True)

# --- Help ------------------------------------------------------------------
st.caption(
    "Tipp: Starte das Frontend mit `streamlit run app.py`. "
    "Die Graphviz-Ausgabe benoetigt eine funktionierende Graphviz-Installation."
)
