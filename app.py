import graphviz
import streamlit as st
import pandas as pd

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


def build_generalized_acceptor(words: list[str], generalize: bool) -> Acceptor:
    """
    Build either the exact DEA (prefix tree + minimization) or a small generalized DEA
    that accepts all words ending with the common last symbol (if consistent).
    """
    seen = set(ch for w in words for ch in w)
    sigma = seen

    # exact model
    exact = Acceptor()
    for w in words:
        exact.learn_word(w)
    exact.Sigma = sigma
    exact.minimize()
    ensure_total(exact)

    if not generalize or len(words) < 2:
        return exact

    # simple heuristic: if all words end with the same symbol -> accept .*symbol
    if any(len(w) == 0 for w in words):
        return exact
    last_chars = {w[-1] for w in words}
    if len(last_chars) != 1:
        return exact
    end_ch = next(iter(last_chars))
    seen = set(ch for w in words for ch in w)
    sigma = seen

    gen = Acceptor()
    gen.Q = {0, 1}
    gen.q0 = 0
    gen.F = {1}
    gen.Sigma = sigma
    gen.delta = {}
    for s in gen.Q:
        for ch in sigma:
            gen.delta[(s, ch)] = 1 if ch == end_ch else 0
    gen.next_state_id = 2 
    ensure_total(gen)
    return gen


def models_equal(a: Acceptor, b: Acceptor) -> bool:
    """
    Lightweight structural equality check for two DFAs.
    """
    return (
        a.q0 == b.q0
        and a.F == b.F
        and a.Q == b.Q
        and a.Sigma == b.Sigma
        and a.delta == b.delta
    )


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
        
        "Heuristische Verallgemeinerung", #(wenn alle Wörter auf dasselbe Zeichen enden)
        value=st.session_state.get("generalize_toggle", True),
        key="generalize_toggle",
        on_change=lambda: st.session_state.update({"generalize_toggle_changed": True}),
    )

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
        lang_str = "; ".join(observed_words) if observed_words else "∅"
        st.markdown("**Formale Signatur**")
        st.write(f"L(V) = {{{lang_str}}}")
        st.write(f"𝔄 = (Σ, Z, δ, F, z₀)")
        st.write(f"Σ = {{{', '.join(snap['alphabet'])}}}")
        st.write(f"Z = {{{', '.join(snap['states'])}}}")
        st.write(f"F = {{{', '.join(snap['accepting'])}}}")
        st.write(f"z₀ = {snap['start']}")

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
