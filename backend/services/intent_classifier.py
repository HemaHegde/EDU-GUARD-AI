"""
intent_classifier.py

PHASE 2.9 UPDATE (UNIFIED MESSAGE ANALYSIS MODULE):
This module now analyses the student's raw question text along TWO
independent dimensions instead of one:

  - intent  (unchanged -- see "SPRINT 10 ADDITION" docstring below)
  - emotion (new -- see the "PHASE 2.9: EMOTION DETECTION" section
    further down this file)

Rather than adding a second, separate `emotion_classifier.py`, emotion
detection is added to this existing module and exposed through one new
public entry point, `analyze_message(question)`, which returns both
values from a single conceptual pass over the same input string. This
is a deliberate architectural choice:

  - Both intent and emotion are the same KIND of thing: a
    deterministic, rule-based read of the SAME raw question text, with
    no dependency on StudentContext, retrieval, or the LLM. Splitting
    them into two files would mean two modules with an identical
    contract (`str -> str`, pure function, keyword/regex based) and a
    duplicated "why rule-based, not a model" rationale -- exactly the
    duplication this module's existing design discipline argues
    against (see the unchanged rationale below).
  - One deterministic parser is easier to audit than two: this is now
    the single place that documents "given this question text, here is
    everything we deterministically inferred about it," rather than
    two separately-maintained keyword banks that could silently drift
    out of sync over time.
  - `classify_intent(question)` and the new `classify_emotion(question)`
    remain independently callable, independently unit-testable, pure
    functions -- this is additive, not a merge of logic.
    `analyze_message()` is a thin wrapper that calls both and returns
    them together; it introduces no new classification rule of its own.

PERSONA / INTENT / EMOTION INDEPENDENCE:
Persona (defined upstream, in context_builder.py / prompt_builder.py /
recommendation_reasoning.py), intent (this module), and emotion (this
module) are three independent dimensions describing a single
conversational turn:
  - Persona  = WHO Aura is speaking to (a behavioural archetype derived
               upstream from risk/engagement data -- untouched by this
               update).
  - Intent   = WHAT kind of conversation this is (academic / career /
               motivation / emotional / general) -- decides retrieval
               routing, unchanged by this update.
  - Emotion  = HOW Aura should emotionally frame its reply right now
               (stress / frustration / confusion / anxiety /
               hopelessness / burnout / success / confidence /
               neutral) -- brand new, and used ONLY for prompt framing
               (see prompt_builder.py's Identity section). It never
               feeds retrieval routing, confidence scoring, or
               recommendation selection, and none of these three
               dimensions is derived from, or overrides, either of the
               other two. For example, a student can ask an "academic"
               intent question ("I don't get recursion, I keep failing
               every attempt and I want to give up") while ALSO
               carrying "hopelessness" emotion -- both are computed
               independently below and both are handed onward
               unmodified, without either one changing the other.

SPRINT 10 ADDITION (INTENT-AWARE RETRIEVAL ROUTING).

Single responsibility: look at the student's raw question text and
decide which EXECUTION PATH the rest of the pipeline should take,
BEFORE retrieval ever runs. This module does not call retrieval, does
not build prompts, does not call the LLM, and does not read
StudentContext — it is a pure function of the question string, by
design, so it can be unit tested exhaustively with a fixed set of
question -> intent examples and never has a hidden dependency on
upstream state.

WHY RULE-BASED, NOT A MODEL:
- Deterministic and auditable, matching the same design discipline
  already used throughout this pipeline (retrieval.py's similarity
  gate, confidence.py's evidence-only rubric): a human can read every
  branch below and know exactly why a question was routed where it
  was. A misclassification here has real consequences (e.g. an
  emotional-distress message wrongly routed into the academic RAG
  path, surfacing DBMS notes instead of support) — an opaque model
  making that call silently is a materially worse failure mode than a
  keyword rule that occasionally needs a new keyword added to it.
- No added latency or inference cost: this project already runs one
  local LLM call per turn; a second model call just to classify
  intent would add cost for a decision that, empirically, does not
  need semantic subtlety to get right for this system's five buckets.
- No labeled training data is required to stand this up, and its
  behaviour is a complete regression suite: any question -> expected
  intent pair is a valid, permanent test case.
- Swappable by construction: `classify_intent(question: str) -> str`
  is a single, stable, one-argument function boundary. Nothing else
  in the codebase inspects HOW the decision is made — only the return
  value — so this can be replaced later (e.g. an embedding-similarity
  classifier against a few labeled exemplars per intent, or a small
  fine-tuned classifier) without touching mentor_service.py's call
  site or any downstream consumer.

INTENTS:
  "emotional"  - wellbeing/distress. Retrieval must be skipped
                 entirely; academic material must never appear in an
                 emotional-support conversation.
  "motivation" - motivation/procrastination struggles. Retrieval is
                 not helpful here either; academic material should not
                 be forced into a motivation conversation.
  "career"     - placements, projects, career guidance. Retrieval is
                 attempted best-effort (career-guidance material if
                 the corpus has any) but is NOT allowed to block the
                 mentor from answering directly when no strong match
                 exists.
  "academic"   - course-content questions. Retrieval runs exactly as
                 before, including the hard similarity-threshold gate.
  "general"    - anything that matches none of the above. Treated
                 identically to "academic" (same retrieval + gate)
                 rather than skipping retrieval, precisely so that any
                 question this classifier doesn't recognize falls back
                 to the ORIGINAL, unmodified pre-Sprint-10 behaviour.
                 This is a deliberate backward-compatibility choice:
                 the only questions whose execution path actually
                 changes are ones this module confidently recognizes
                 as emotional/motivation/career.

PRECEDENCE:
Checked in this fixed order: emotional -> motivation -> career ->
(default) academic/general. Emotional is checked first because a
question that mixes distress language with an academic-sounding word
("I'm so stressed about this DBMS exam, I feel like giving up") should
still be treated as an emotional-support conversation, not routed into
content retrieval, since the priority here is never surfacing unrelated
material during distress -- the LLM prompt itself may still discuss the
exam in a supportive, non-content way.

SPRINT 1.1 ADDITION (SOCIAL / EXPRESSIVE INTENT COVERAGE).

Many ordinary student messages ("Hello.", "Thank you.", "Bye.", "I
passed my exam!", "I don't think I can continue studying anymore.")
were previously falling into the "general" catch-all simply because
they don't happen to contain an academic keyword. This sprint adds
four new deterministic buckets -- the exact same rule-based
substring/word-boundary style as every bank above, no scoring, no ML:

  "celebration" - the student is sharing a win (passed an exam,
                  finally understood a concept, cleared an interview).
                  Retrieval is not useful here; the mentor should
                  simply acknowledge and encourage.
  "greeting"    - a bare opening ("Hi", "Hello", "Hey"). Detected only
                  when the ENTIRE (trimmed) message is just a greeting
                  plus optional punctuation -- so "Hi, can you explain
                  recursion?" is correctly left for the academic bank
                  below, not hijacked here.
  "gratitude"   - a bare "Thank you" / "Thanks", detected the same
                  whole-message way as greeting, for the same reason.
  "farewell"    - a bare "Bye" / "Goodbye" / "See you", detected the
                  same whole-message way.

PRECEDENCE EXTENSION: the four new buckets are inserted AFTER career
and BEFORE academic/general:

    emotional -> motivation -> career -> celebration -> greeting ->
    gratitude -> farewell -> academic -> general (default)

This ordering is the deliberate backward-compatibility anchor for this
sprint: emotional, motivation, and career keep the EXACT same
patterns and the EXACT same relative precedence as before, so any
message that used to resolve to one of those three labels resolves to
that same label, unchanged, today. The four new buckets can only ever
"claim" a message that would previously have fallen all the way
through to the academic/general default -- and, per the SPRINT 10
docstring above, academic and general were already functionally
identical (same retrieval path). So the only observable change for any
pre-existing message is that some inputs that used to be labeled
"general" now get a more specific, more accurate label; no message's
classification can regress or become less specific than it was before
this sprint.

The one exception is the motivation bank itself, which gains two new
patterns in this sprint (see `_MOTIVATION_PATTERNS` below) to catch
"I don't think I can continue studying anymore" -- phrased as
motivation/burnout-to-study language rather than the existing
"give up" emotional phrasing already handled by the emotional bank.
This is an additive keyword change, not a reordering or a new
mechanism, following the same "extend the bank, don't touch the
mechanism" pattern every prior sprint in this file has used.

SPRINT 1.3 ADDITION (MIXED-INTENT SUPPORT).
Adds a THIRD independent dimension alongside intent and emotion:
`secondary_academic_signal`, a boolean answering "does this message
ALSO reference academic content/evaluation" regardless of which
primary intent bucket it falls into. See `has_academic_reference()`
and the surrounding comment block further down this file for the
full mechanism and rationale. Like emotion, it is computed
independently and never changes, and is never changed by,
`classify_intent()`'s output.
"""

import re
from typing import List

VALID_INTENTS = (
    "emotional",
    "motivation",
    "career",
    "celebration",
    "greeting",
    "gratitude",
    "farewell",
    "academic",
    "general",
)

# ---------------------------------------------------------------
# Keyword/phrase banks. Plain substrings/word-boundary patterns only
# -- no scoring, no weighting, no fuzzy matching -- so every match is
# individually inspectable and the whole bank is easy to extend.
# ---------------------------------------------------------------

_EMOTIONAL_PATTERNS: List[str] = [
    r"\bstress(ed|ful)?\b",
    r"\banxious\b",
    r"\banxiety\b",
    r"\boverwhelm(ed|ing)?\b",
    r"\bdepress(ed|ion)?\b",
    r"\bpanic(k?ing|ked)?\b",
    r"\bburn(ed|t)?\s*out\b",
    r"\bburnout\b",
    r"\bfeel(ing)?\s+like\s+giving\s+up\b",
    r"\bgive\s+up\b",
    r"\bgiving\s+up\b",
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\blonely\b",
    r"\bisolat(ed|ion)\b",
    r"\bcan'?t\s+cope\b",
    r"\bcannot\s+cope\b",
    r"\bcrying\b",
    r"\bscared\b",
    r"\bafraid\b",
    r"\bexhausted\b",
    r"\bmental(ly)?\s+(health|tired|drained)\b",
    r"\bfeel(ing)?\s+down\b",
    r"\bsad\b",
]

_MOTIVATION_PATTERNS: List[str] = [
    r"\bmotivat(ed|ion|e)\b",
    r"\bprocrastinat\w*\b",
    r"\bcan'?t\s+start\b",
    r"\bcannot\s+start\b",
    r"\bcan'?t\s+focus\b",
    r"\bkeep\s+putting\s+(it|things)\s+off\b",
    r"\bno\s+energy\s+to\s+study\b",
    r"\bdon'?t\s+feel\s+like\s+studying\b",
    r"\blazy\b",
    r"\bwant\s+to\s+study\s+but\b",
    # SPRINT 1.1 ADDITION: "I don't think I can continue studying
    # anymore" -- distinct from the emotional bank's "give up" phrasing
    # above; this is language about losing the drive to keep studying,
    # not a general expression of hopelessness.
    r"\b(can'?t|cannot)\s+(continue|keep)\s+(studying|with\s+my\s+studies)\b",
    r"\bdon'?t\s+think\s+i\s+can\s+(continue|keep\s+going|keep\s+studying|do\s+this\s+anymore)\b",
]

_CAREER_PATTERNS: List[str] = [
    r"\bplacement(s)?\b",
    r"\binterview(s)?\b",
    r"\bresume\b",
    r"\bcv\b",
    r"\bcareer(s)?\b",
    r"\bjob(s)?\b",
    r"\binternship(s)?\b",
    r"\bportfolio\b",
    r"\bwhich\s+projects?\s+should\s+i\s+build\b",
    r"\bprojects?\s+to\s+build\b",
    r"\bhiring\b",
    r"\brecruiter(s)?\b",
    r"\bhow\s+do\s+i\s+prepare\s+for\s+placements\b",
]

# NOTE: "academic" and "general" take the IDENTICAL execution path
# (retrieval + hard similarity gate, both unchanged from pre-Sprint-10
# behaviour) -- this bank exists purely so clearly content-shaped
# questions are LABELED "academic" rather than the "general" catch-all,
# for accurate observability/logging (e.g. the `detected_intent` field
# returned by ask_mentor). It does not change routing behaviour at all;
# it is safe to extend or leave incomplete without any functional risk.
_ACADEMIC_PATTERNS: List[str] = [
    r"\bexplain\b",
    r"\bwhat\s+is\b",
    r"\bwhat\s+are\b",
    r"\bdefine\b",
    r"\bdefinition\s+of\b",
    r"\bdifference\s+between\b",
    r"\bhow\s+does\s+.+\s+work\b",
    r"\bhow\s+do(es)?\s+.+\s+work\b",
    r"\bexample(s)?\s+of\b",
]

_EMOTIONAL_RE = re.compile("|".join(_EMOTIONAL_PATTERNS), re.IGNORECASE)
_MOTIVATION_RE = re.compile("|".join(_MOTIVATION_PATTERNS), re.IGNORECASE)
_CAREER_RE = re.compile("|".join(_CAREER_PATTERNS), re.IGNORECASE)
_ACADEMIC_RE = re.compile("|".join(_ACADEMIC_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------
# SPRINT 1.3 ADDITION (MIXED-INTENT SUPPORT: SECONDARY ACADEMIC
# SIGNAL).
#
# PROBLEM: "I'm stressed because I don't understand DBMS." matches
# _EMOTIONAL_RE (on "stress") and, per precedence, `classify_intent`
# correctly returns "emotional" -- distress language must always win
# the PRIMARY label, so persona/prompt framing stay emotion-first.
# But mentor_service.py's retrieval routing keys off that single
# label, and "emotional" is deliberately excluded from
# `_RETRIEVAL_INTENTS` -- so the student's academic question (DBMS)
# never gets a chance to be retrieved at all, even though it was
# clearly also asked.
#
# FIX: this is NOT a new intent bucket, and it does not touch
# `classify_intent()` or its precedence in any way. It is a second,
# INDEPENDENT, deterministic read of the same question text -- the
# same relationship emotion already has to intent (see the module
# docstring's "PERSONA / INTENT / EMOTION INDEPENDENCE" section
# above): a boolean flag answering "does this message ALSO reference
# academic content/evaluation, regardless of which primary bucket it
# was classified into." `has_academic_reference()` is a plain,
# additive, pure function, using the exact same substring/word-
# boundary regex-bank style as every bank above -- no scoring, no
# fuzzy matching, no dependency on the primary intent decision.
#
# This bank is intentionally broader/looser than `_ACADEMIC_PATTERNS`
# above (which exists only to LABEL an unambiguous "academic"-primary
# question). Here we're only looking for a low-risk secondary
# opportunity to attempt retrieval on an otherwise-emotional message
# -- a false positive costs nothing: mentor_service.py never applies
# the strict similarity gate to this case, so weak/irrelevant
# retrieval results are simply discarded (existing non-strict-gate
# behaviour, unchanged), never surfaced and never used to block a
# response.
_ACADEMIC_REFERENCE_PATTERNS: List[str] = [
    r"\bdon'?t\s+understand\b",
    r"\bdo\s+not\s+understand\b",
    r"\bcan'?t\s+understand\b",
    r"\bconfused\s+about\b",
    r"\bstruggling\s+with\b",
    r"\bdifficult\s+to\s+understand\b",
    r"\b(exam|test|quiz|assignment|lab|viva|practical)\b",
    r"\b(subject|course|topic|concept|chapter|module)\b",
]

_ACADEMIC_REFERENCE_RE = re.compile(
    "|".join(_ACADEMIC_REFERENCE_PATTERNS), re.IGNORECASE
)


def has_academic_reference(question: str) -> bool:
    """
    Returns True if `question` references academic content/evaluation
    (e.g. "don't understand DBMS", "operating systems exam"),
    independent of whatever primary intent bucket `classify_intent()`
    assigns it. Pure function of `question` only -- no context, no
    I/O, no randomness.

    This is the SECONDARY signal described in the SPRINT 1.3 ADDITION
    above: it never overrides, is never overridden by, and has no
    effect whatsoever on `classify_intent()`'s return value or
    precedence. It exists solely so mentor_service.py can opt an
    otherwise-"emotional"-labelled question into a best-effort
    (non-strict-gate) retrieval attempt, without changing that
    question's primary label or emotional framing.
    """
    if not question or not question.strip():
        return False
    return bool(_ACADEMIC_REFERENCE_RE.search(question.strip()))

# ---------------------------------------------------------------
# SPRINT 1.1 ADDITION: celebration / greeting / gratitude / farewell.
# Same design discipline as every bank above -- plain substrings/
# word-boundary patterns, no scoring, no weighting, no fuzzy matching.
# ---------------------------------------------------------------

# "celebration" is a positive-event bank, matched the same way as the
# content banks above (substring/word-boundary, anywhere in the
# message) -- these are full sentences ("I finally understood
# recursion!"), not bare exchanges, so no anchoring is needed here.
_CELEBRATION_PATTERNS: List[str] = [
    r"\bi\s+passed\b",
    r"\bpassed\s+my\s+(exam|test|course|semester|subject)\b",
    r"\bi\s+finally\s+understood\b",
    r"\bfinally\s+understood\b",
    r"\bi\s+finally\s+get\s+it\b",
    r"\bi\s+(did\s+it|got\s+it|nailed\s+it|aced\s+it)\b",
    r"\bi\s+cleared\s+(the\s+)?(exam|test|interview)\b",
    r"\bi\s+cracked\s+(the\s+)?(interview|exam)\b",
    r"\bi\s+got\s+(selected|placed|shortlisted)\b",
    r"\bproud\s+of\s+myself\b",
    r"\bscored\s+well\b",
    r"\bgood\s+(grade|marks|score)\b",
    # SPRINT 1.5 ADDITION (Positive Learning Intent): the patterns
    # above only caught the PAST-TENSE "finally understood" phrasing.
    # Present-tense "finally understand" / "finally learned" / a
    # solved-problem statement are equally clear celebration signals
    # that were previously falling through to the "general" default.
    # Purely additive to the existing bank -- same word-boundary/
    # substring mechanism, no new mechanism, no change to precedence.
    r"\bi\s+finally\s+understand\b",
    r"\bfinally\s+understand\b",
    r"\bi\s+finally\s+learn(ed)?\b",
    r"\bfinally\s+learn(ed)?\b",
    r"\bi\s+solved\s+(my\s+)?(first\s+)?(\w+\s+){0,2}problem\b",
]

# "greeting" / "gratitude" / "farewell" are bare-exchange banks. Each
# pattern is anchored with ^...$ over the WHOLE (already-stripped)
# message, tolerating only trailing punctuation/whitespace after the
# phrase -- this is what keeps them from ever hijacking a longer
# message that happens to start with "Hi" or end with "thanks" (e.g.
# "Hi, can you explain recursion?" or "Thanks, that was helpful, but
# what about arrays?" correctly fall through to later banks instead).
_GREETING_PATTERNS: List[str] = [
    r"^(hi+|hello+|hey+|yo|greetings)[\s!.,]*$",
    r"^good\s+(morning|afternoon|evening)[\s!.,]*$",
]

_GRATITUDE_PATTERNS: List[str] = [
    r"^(thanks?|thank\s+you|thank\s+u|thx|ty)(\s+(so\s+much|a\s+lot|very\s+much))?[\s!.,]*$",
]

_FAREWELL_PATTERNS: List[str] = [
    r"^(bye+|goodbye|good\s*bye)[\s!.,]*$",
    r"^see\s+you(\s+later)?[\s!.,]*$",
    r"^(good\s*night|take\s+care)[\s!.,]*$",
]

_CELEBRATION_RE = re.compile("|".join(_CELEBRATION_PATTERNS), re.IGNORECASE)
_GREETING_RE = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)
_GRATITUDE_RE = re.compile("|".join(_GRATITUDE_PATTERNS), re.IGNORECASE)
_FAREWELL_RE = re.compile("|".join(_FAREWELL_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------
# SPRINT 1.5 ADDITION (POSITIVE LEARNING INTENT: ACADEMIC TOPIC
# EXTRACTION).
#
# PROBLEM: "I finally understood recursion!" correctly resolves to
# `intent == "celebration"` (see `_CELEBRATION_RE` above), but the
# mentor's reply had no deterministic way to know WHAT was understood,
# so it could only produce a generic "well done!" instead of
# celebrating the actual concept by name.
#
# FIX: this is NOT a new intent bucket and does not touch
# `classify_intent()`, its precedence, or any existing bank in any
# way. It is a FOURTH independent, deterministic read of the same
# question text -- the exact same relationship `emotion` and
# `secondary_academic_signal` already have to `intent` (see the module
# docstring's "PERSONA / INTENT / EMOTION INDEPENDENCE" and the SPRINT
# 1.3 ADDITION comment above): a plain lookup answering "does this
# message name a recognizable academic concept, and if so, which
# one?" `extract_academic_topic()` uses the exact same substring/
# word-boundary regex-bank style as every bank above -- no scoring, no
# fuzzy matching, no NLP noun-phrase extraction, no dependency on the
# primary intent decision. Patterns are checked in the fixed order
# below and the FIRST match wins (same first-match discipline as
# `classify_intent()`), so behaviour is fully deterministic and
# auditable.
#
# This bank is intentionally a curated list of common CS/engineering
# coursework concepts (easy to extend with more keywords later, just
# like every other bank in this file) rather than an attempt to
# extract an arbitrary topic from arbitrary phrasing -- a missed topic
# simply returns `None` and the caller falls back to a generic
# celebration, exactly the same fail-safe shape as
# `has_academic_reference()` returning `False`.
_ACADEMIC_TOPIC_PATTERNS: List[tuple] = [
    (r"\bdbms\b", "DBMS"),
    (r"\brecursion\b", "recursion"),
    (r"\bdynamic\s+programming\b", "dynamic programming"),
    (r"\bgraph(s)?\b", "graphs"),
    (r"\blinked\s+list(s)?\b", "linked lists"),
    (r"\bbinary\s+search\b", "binary search"),
    (r"\barray(s)?\b", "arrays"),
    (r"\bpointer(s)?\b", "pointers"),
    (r"\bstack(s)?\b", "stacks"),
    (r"\bqueue(s)?\b", "queues"),
    (r"\btree(s)?\b", "trees"),
    (r"\bsorting\b", "sorting"),
    (r"\boperating\s+systems?\b", "operating systems"),
    (r"\bsql\b", "SQL"),
    (r"\bnormalization\b", "normalization"),
    (r"\b(threading|concurrency)\b", "concurrency"),
    (r"\b(oop|object[\s-]oriented)\b", "OOP"),
    (r"\bcompiler(s)?\b", "compilers"),
    (r"\b(networking|tcp\s*/\s*ip|osi\s+model)\b", "networking"),
    (r"\balgorithm(s)?\b", "algorithms"),
    (r"\bdata\s+structures?\b", "data structures"),
]

_ACADEMIC_TOPIC_BANK = [
    (re.compile(pattern, re.IGNORECASE), name)
    for pattern, name in _ACADEMIC_TOPIC_PATTERNS
]


def extract_academic_topic(question: str):
    """
    Returns the first recognized academic concept named in `question`
    (e.g. "recursion", "DBMS", "dynamic programming"), or `None` if no
    known concept is found. Pure function of `question` only -- no
    context, no I/O, no randomness, no ML/embeddings/LLM.

    This is a SECONDARY signal, exactly like `has_academic_reference()`
    above: it never overrides, is never overridden by, and has no
    effect whatsoever on `classify_intent()`'s return value or
    precedence. It exists solely so mentor_service.py can enrich a
    "celebration" reply with the specific concept being celebrated,
    without changing that message's primary label, retrieval routing,
    or the similarity gate (both of which stay OFF for celebration
    regardless of this function's result).
    """
    if not question or not question.strip():
        return None
    text = question.strip()
    for pattern_re, canonical_name in _ACADEMIC_TOPIC_BANK:
        if pattern_re.search(text):
            return canonical_name
    return None


# =========================================================
# PHASE 2.9: EMOTION DETECTION
# =========================================================
# Same design discipline as the intent banks above: plain
# substring/word-boundary regex patterns, no scoring, no weighting, no
# fuzzy matching, no ML/LLM/embeddings, no randomness -- every match is
# individually inspectable and the whole bank is easy to extend.
#
# EMOTIONS: stress, frustration, confusion, anxiety, hopelessness,
# burnout, success, confidence, neutral (default).
#
# PRECEDENCE: checked in this fixed order: hopelessness -> burnout ->
# anxiety -> stress -> frustration -> confusion -> success ->
# confidence -> (default) neutral. The more urgent/severe emotional
# states are checked first, mirroring the intent bank's own precedence
# rationale above -- e.g. a message that mentions both "give up" and
# "stressed" should be read as hopelessness first, since that is the
# more urgent signal for how Aura should open its reply.
VALID_EMOTIONS = (
    "stress",
    "frustration",
    "confusion",
    "anxiety",
    "hopelessness",
    "burnout",
    "success",
    "confidence",
    "neutral",
)

_HOPELESSNESS_PATTERNS: List[str] = [
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\bgive\s+up\b",
    r"\bgiving\s+up\b",
    r"\bno\s+point\b",
    r"\bwhat'?s\s+the\s+point\b",
    r"\bcan'?t\s+do\s+this\s+anymore\b",
    r"\bi'?m\s+done\s+trying\b",
]

_BURNOUT_PATTERNS: List[str] = [
    r"\bburn(ed|t)?\s*out\b",
    r"\bburnout\b",
    r"\bexhausted\b",
    r"\bdrained\b",
    r"\btired\s+of\s+everything\b",
    r"\brunning\s+on\s+empty\b",
]

_ANXIETY_PATTERNS: List[str] = [
    r"\banxious\b",
    r"\banxiety\b",
    r"\bpanic(k?ing|ked)?\b",
    r"\bnervous\b",
    r"\bscared\b",
    r"\bafraid\b",
    r"\bworried\b",
    r"\bworrying\b",
]

_STRESS_PATTERNS: List[str] = [
    r"\bstress(ed|ful)?\b",
    r"\boverwhelm(ed|ing)?\b",
    r"\bunder\s+pressure\b",
    r"\btoo\s+much\s+(going\s+on|work|pressure)\b",
]

_FRUSTRATION_PATTERNS: List[str] = [
    r"\bfrustrat(ed|ing)\b",
    r"\bannoyed\b",
    r"\birritat(ed|ing)\b",
    r"\bfed\s+up\b",
    r"\bsick\s+of\b",
]

_CONFUSION_PATTERNS: List[str] = [
    r"\bconfus(ed|ing)\b",
    r"\bdon'?t\s+understand\b",
    r"\bdo\s+not\s+understand\b",
    r"\bi'?m\s+lost\b",
    r"\bunclear\b",
    r"\bdoesn'?t\s+make\s+sense\b",
    r"\bdo\s+not\s+make\s+sense\b",
]

_SUCCESS_PATTERNS: List[str] = [
    r"\bi\s+(passed|did\s+it|got\s+it|nailed\s+it|aced\s+it)\b",
    r"\bscored\s+well\b",
    r"\bgood\s+(grade|marks|score)\b",
    r"\bproud\s+of\s+myself\b",
    r"\bfinally\s+understood\b",
    r"\bi\s+finally\s+get\s+it\b",
]

_CONFIDENCE_PATTERNS: List[str] = [
    r"\bi\s+feel\s+confident\b",
    r"\bi\s+can\s+do\s+this\b",
    r"\bi\s+got\s+this\b",
    r"\bi'?m\s+ready\s+for\b",
    r"\bi'?m\s+prepared\b",
]

_HOPELESSNESS_RE = re.compile("|".join(_HOPELESSNESS_PATTERNS), re.IGNORECASE)
_BURNOUT_RE = re.compile("|".join(_BURNOUT_PATTERNS), re.IGNORECASE)
_ANXIETY_RE = re.compile("|".join(_ANXIETY_PATTERNS), re.IGNORECASE)
_STRESS_RE = re.compile("|".join(_STRESS_PATTERNS), re.IGNORECASE)
_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_PATTERNS), re.IGNORECASE)
_CONFUSION_RE = re.compile("|".join(_CONFUSION_PATTERNS), re.IGNORECASE)
_SUCCESS_RE = re.compile("|".join(_SUCCESS_PATTERNS), re.IGNORECASE)
_CONFIDENCE_RE = re.compile("|".join(_CONFIDENCE_PATTERNS), re.IGNORECASE)


def classify_emotion(question: str) -> str:
    """
    Returns one of VALID_EMOTIONS. Pure function of `question` only --
    no context, no I/O, no randomness, so identical input always
    yields identical output. Mirrors `classify_intent`'s contract
    exactly (same signature shape, same purity guarantee), so the two
    can be called independently or together via `analyze_message()`.

    Precedence (see module section docstring above): hopelessness >
    burnout > anxiety > stress > frustration > confusion > success >
    confidence > neutral (default).
    """
    if not question or not question.strip():
        return "neutral"

    text = question.strip()

    if _HOPELESSNESS_RE.search(text):
        return "hopelessness"

    if _BURNOUT_RE.search(text):
        return "burnout"

    if _ANXIETY_RE.search(text):
        return "anxiety"

    if _STRESS_RE.search(text):
        return "stress"

    if _FRUSTRATION_RE.search(text):
        return "frustration"

    if _CONFUSION_RE.search(text):
        return "confusion"

    if _SUCCESS_RE.search(text):
        return "success"

    if _CONFIDENCE_RE.search(text):
        return "confidence"

    return "neutral"


# =========================================================
# PHASE 3.1: EMOTION INTENSITY DETECTION
# =========================================================
# Same design discipline as intent/emotion above: plain
# substring/word-boundary regex patterns only, no scoring model, no
# weighting beyond a fixed precedence order, no fuzzy matching, no
# ML/LLM/embeddings, no randomness. Intensity is a THIRD, independent
# dimension layered on top of `emotion` -- it answers "how strongly is
# this emotion being expressed in THIS message," never "which emotion
# is it" (that question is answered exclusively by `classify_emotion`
# above, and is not re-derived or second-guessed here).
#
# INTENSITY LEVELS: "Low" | "Medium" | "High".
#
# SIGNALS (each a plain, inspectable regex bank):
#   - Low intensifiers    -- hedging/minimizing language ("a little",
#     "a bit", "slightly", "somewhat", "kind of", "sort of").
#   - Medium intensifiers -- ordinary amplifying adverbs ("really",
#     "very", "so", "quite", "pretty") and moderate-impact phrases
#     ("can't focus", "can't concentrate", "barely").
#   - High markers        -- absolute/extreme language describing the
#     emotion as overwhelming or all-encompassing ("can't do this
#     anymore", "can't take it", "can't cope", "completely",
#     "totally", "extremely", "unbearable", "nothing is working",
#     repeated exclamation marks), i.e. language that goes beyond
#     amplifying the feeling into describing it as no longer
#     manageable.
#
# PRECEDENCE: High markers are checked first (an extreme/absolute
# phrase dominates even if a hedge word also appears elsewhere in the
# same message), then Medium, then Low. A message carrying NO
# intensity modifier at all defaults to "Medium" -- a plain,
# unqualified statement of an emotion ("I am stressed") is treated as
# an ordinary-strength statement, neither minimized nor escalated. A
# message with no detected emotion (`emotion == "neutral"`) always
# gets "Low" intensity, since there is no emotional signal for
# "Medium"/"High" pacing and validation guidance to attach to.
#
# INDEPENDENCE: this module NEVER reads StudentContext, retrieval
# results, confidence, SHAP, or persona, and its output is consumed
# ONLY by the Identity section of the prompt (see prompt_builder.py's
# Phase 3.1 update) for emotional framing/pacing guidance. It has no
# path into retrieval routing, the similarity gate, confidence
# scoring, evidence prioritization, or recommendation selection --
# those all continue to read exclusively from `detected_intent` and
# StudentContext/retrieval, exactly as before this phase.
VALID_EMOTION_INTENSITIES = ("Low", "Medium", "High")

_LOW_INTENSITY_PATTERNS: List[str] = [
    r"\ba\s+little\b",
    r"\ba\s+bit\b",
    r"\bslightly\b",
    r"\bsomewhat\b",
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\bmildly\b",
    r"\bnot\s+too\b",
]

_MEDIUM_INTENSITY_PATTERNS: List[str] = [
    r"\breally\b",
    r"\bvery\b",
    r"\bso\b",
    r"\bquite\b",
    r"\bpretty\b",
    r"\bcan'?t\s+focus\b",
    r"\bcan'?t\s+concentrate\b",
    r"\bbarely\b",
]

_HIGH_INTENSITY_PATTERNS: List[str] = [
    r"\bcan'?t\s+(do\s+this|take\s+it|take\s+this|cope|go\s+on)\s*(anymore)?\b",
    r"\bcannot\s+(do\s+this|take\s+it|take\s+this|cope|go\s+on)\s*(anymore)?\b",
    r"\bcompletely\b",
    r"\btotally\b",
    r"\bextremely\b",
    r"\bunbearable\b",
    r"\bnever\s+going\s+to\b",
    r"\bnothing\s+is\s+working\b",
    r"\beverything\s+is\s+falling\s+apart\b",
    r"\bcan'?t\s+handle\s+(this|it)\b",
    r"!{2,}",
]

_LOW_INTENSITY_RE = re.compile("|".join(_LOW_INTENSITY_PATTERNS), re.IGNORECASE)
_MEDIUM_INTENSITY_RE = re.compile("|".join(_MEDIUM_INTENSITY_PATTERNS), re.IGNORECASE)
_HIGH_INTENSITY_RE = re.compile("|".join(_HIGH_INTENSITY_PATTERNS), re.IGNORECASE)


def classify_emotion_intensity(question: str, emotion: str = "neutral") -> str:
    """
    Returns one of VALID_EMOTION_INTENSITIES. Pure function of
    `question` and the already-computed `emotion` (passed in, never
    re-derived here) -- no context, no I/O, no randomness, so identical
    input always yields identical output. Mirrors `classify_intent` /
    `classify_emotion`'s contract: independently callable, independently
    unit-testable, and safe to call on its own outside `analyze_message`.

    Precedence (see module section docstring above): High markers >
    Medium intensifiers > Low intensifiers > (default) Medium, except
    that `emotion == "neutral"` always short-circuits to "Low".
    """
    if emotion == "neutral":
        return "Low"

    if not question or not question.strip():
        return "Medium"

    text = question.strip()

    if _HIGH_INTENSITY_RE.search(text):
        return "High"

    if _MEDIUM_INTENSITY_RE.search(text):
        return "Medium"

    if _LOW_INTENSITY_RE.search(text):
        return "Low"

    # No explicit intensity modifier found -- a plain, unqualified
    # statement of an emotion is treated as ordinary strength: neither
    # minimized (Low) nor escalated (High).
    return "Medium"


# =========================================================
# PHASE 3.2: EMOTIONAL TRAJECTORY MEMORY
# =========================================================
# This is NOT a new/second emotion classifier -- it reuses
# `classify_emotion()` (the single existing emotion rule set) against
# each of the student's recent messages, already available read-only
# on `context.chat_history` (built by context_builder.py from the
# existing mentor_history table; no new table, column, query, or write
# is introduced here). What's new is purely a deterministic pattern
# read over an already-computed SEQUENCE of emotion labels -- it asks
# "is the trend across recent turns improving, stable, or worsening,"
# never "what is the emotion" (that question stays exclusively
# `classify_emotion`'s).
#
# WHY DETERMINISTIC, NOT ML: exactly the same rationale as
# `classify_intent` / `classify_emotion` / `classify_emotion_intensity`
# above -- a fixed, auditable numeric ordering over a small, fixed
# emotion vocabulary, with no scoring model, no embeddings, no LLM
# call, and no randomness. Every trajectory label can be traced back to
# a simple sequence of numbers a human can recompute by hand.
#
# MECHANISM:
#   1. Extract the student's raw question text from the last few turns
#      of `chat_history` (same "Student: " / "\nAura: " turn-boundary
#      convention `_section_conversation_history` in prompt_builder.py
#      already relies on -- read-only string parsing, no new storage).
#   2. Re-run the existing `classify_emotion()` on each of those past
#      questions, plus the CURRENT turn's already-computed `emotion`,
#      to get a short, ordered sequence of emotion labels (oldest ->
#      newest).
#   3. Map each label to a fixed integer valence (`_EMOTION_VALENCE`)
#      and read the trend of that numeric sequence:
#        - all valences equal, and negative -> "Stable Emotional Distress"
#        - all valences equal, and positive -> "Positive Stability"
#        - all valences equal, and zero (neutral) -> "Neutral"
#        - monotonically non-decreasing (trending toward positive) ->
#          "Improving Emotional State"
#        - monotonically non-increasing (trending toward negative) ->
#          "Escalating Distress"
#        - anything else (no consistent direction) -> "Mixed Emotional
#          Pattern"
#      This is a plain sequence comparison (`<=`/`>=` over a Python
#      list) -- no statistics, no smoothing, no thresholds beyond the
#      fixed valence table below.
#
# INDEPENDENCE: exactly like emotion and emotion intensity, trajectory
# is consumed ONLY by the Identity section of the prompt (see
# prompt_builder.py's Phase 3.2 update) for continuity-aware emotional
# framing. It has no path into retrieval, the similarity gate,
# confidence, evidence prioritization, SHAP reasoning, persona, or
# recommendation selection -- those continue to read exclusively from
# `detected_intent` and StudentContext/retrieval, untouched by this
# phase. It is never persisted (chat_history is read, never written,
# by this module) and never returned in the API response.
VALID_EMOTION_TRAJECTORIES = (
    "Stable Emotional Distress",
    "Escalating Distress",
    "Improving Emotional State",
    "Positive Stability",
    "Neutral",
    "Mixed Emotional Pattern",
)

# Fixed valence per emotion label -- negative = distress, positive =
# uplifting, zero = neutral. This is the ONLY numeric ordering used by
# `classify_emotional_trajectory`; nothing about `classify_emotion`
# itself is changed or re-derived differently by this table.
_EMOTION_VALENCE = {
    "hopelessness": -4,
    "burnout": -3,
    "anxiety": -3,
    "stress": -2,
    "frustration": -2,
    "confusion": -1,
    "neutral": 0,
    "confidence": 2,
    "success": 2,
}

# Default number of emotion data points (including the current turn)
# considered when computing a trajectory. Small and fixed on purpose:
# this is meant to read the SHAPE of the last few turns, not build a
# long-range emotional profile.
_TRAJECTORY_WINDOW = 3


def _extract_recent_student_messages(chat_history: str, max_turns: int) -> List[str]:
    """
    Read-only parser over `context.chat_history` -- the same string
    context_builder.py already builds and prompt_builder.py's
    `_section_conversation_history` already parses via the
    "Student: " turn-boundary convention. No new persistence, no new
    field: this only extracts the raw student-side question text from
    the most recent `max_turns` turns so `classify_emotion()` can be
    re-run against each one.

    Returns an empty list for empty/missing history or `max_turns <= 0`
    -- e.g. the very first turn of a conversation, where there is no
    prior history to read a trend from.
    """
    if not chat_history or max_turns <= 0:
        return []

    raw_turns = chat_history.split("Student: ")
    turns = [turn for turn in raw_turns if turn.strip()]

    student_messages: List[str] = []
    for turn in turns:
        # Each turn is formatted "Student: <q>\nAura: <a>\n..." by
        # context_builder.py; the question text is everything before
        # the first "\nAura:" marker.
        question_text = turn.split("\nAura:")[0].strip()
        if question_text:
            student_messages.append(question_text)

    return student_messages[-max_turns:] if max_turns else student_messages


def classify_emotional_trajectory(
    chat_history: str = "",
    current_emotion: str = "neutral",
    window: int = _TRAJECTORY_WINDOW,
) -> str:
    """
    Returns one of VALID_EMOTION_TRAJECTORIES. Deterministic function of
    `chat_history` (read-only) and the already-computed
    `current_emotion` for this turn -- no ML, no embeddings, no LLM
    call, no randomness: identical inputs always yield identical
    output. Independently callable/unit-testable outside
    `analyze_message`, exactly like `classify_emotion_intensity`.

    See the module section docstring above for the full mechanism and
    rationale. `window` defaults to `_TRAJECTORY_WINDOW` (3: the
    current turn plus the two most recent prior turns) and is only
    exposed as a parameter for testability -- callers should not
    normally need to override it.
    """
    if window < 1:
        window = 1

    recent_texts = _extract_recent_student_messages(chat_history, window - 1)
    sequence = [classify_emotion(text) for text in recent_texts]
    sequence.append(current_emotion)
    sequence = sequence[-window:]

    valences = [_EMOTION_VALENCE.get(label, 0) for label in sequence]

    if all(v == valences[0] for v in valences):
        v = valences[0]
        if v == 0:
            return "Neutral"
        return "Stable Emotional Distress" if v < 0 else "Positive Stability"

    non_decreasing = all(valences[i] <= valences[i + 1] for i in range(len(valences) - 1))
    non_increasing = all(valences[i] >= valences[i + 1] for i in range(len(valences) - 1))

    if non_decreasing:
        return "Improving Emotional State"

    if non_increasing:
        return "Escalating Distress"

    return "Mixed Emotional Pattern"


def analyze_message(question: str, chat_history: str = "") -> dict:
    """
    PHASE 2.9 ADDITION -- the unified Message Analysis Module's single
    public entry point. Calls `classify_intent()` and
    `classify_emotion()` against the SAME raw question text and returns
    both results together.

    PHASE 3.1 UPDATE: also calls `classify_emotion_intensity()`, passing
    it the SAME question text plus the `emotion` value just computed
    (so intensity is graded relative to the correct emotion without a
    second, independent emotion re-detection).

    PHASE 3.2 UPDATE: new, OPTIONAL `chat_history` parameter (defaults
    to `""`, so any existing caller passing only `question` keeps
    working exactly as before -- with trajectory falling back to
    whatever a single current-turn emotion resolves to). When provided
    (mentor_service.py passes `context.chat_history`, already available
    read-only), `analyze_message` also calls
    `classify_emotional_trajectory()` and the returned dict gains one
    more key:

        {
            "intent": <one of VALID_INTENTS>,
            "emotion": <one of VALID_EMOTIONS>,
            "emotion_intensity": <one of VALID_EMOTION_INTENSITIES>,
            "emotion_trajectory": <one of VALID_EMOTION_TRAJECTORIES>,
        }

    SPRINT 1.3 UPDATE: adds one more additive key,
    `secondary_academic_signal` (bool), computed by the new, fully
    independent `has_academic_reference()` (see the SPRINT 1.3
    ADDITION section above `_ACADEMIC_REFERENCE_PATTERNS`). This
    supports mixed-intent messages such as "I'm stressed because I
    don't understand DBMS" -- `intent` still correctly resolves to
    "emotional" (unchanged precedence), but mentor_service.py can now
    read this new flag to allow a best-effort academic retrieval
    attempt alongside the emotional framing, instead of retrieval
    being skipped outright. It does not change `intent`'s value or
    meaning, and is never used to relabel or override the primary
    intent.

    SPRINT 1.5 UPDATE (POSITIVE LEARNING INTENT): adds one more
    additive key, `academic_topic` (str or None), computed by the new,
    fully independent `extract_academic_topic()` (see the SPRINT 1.5
    ADDITION section above `_ACADEMIC_TOPIC_PATTERNS`). This supports
    celebration messages that name a specific concept, e.g. "I finally
    understood recursion!" -> `academic_topic == "recursion"` --
    `intent` still correctly resolves to "celebration" (unchanged
    precedence, retrieval and the similarity gate stay OFF exactly as
    Sprint 1.4 established), but mentor_service.py can now read this
    new flag to name the specific concept in its celebratory reply
    instead of a generic acknowledgement. It does not change `intent`
    or any other key's value or meaning, and is never used to relabel
    or override the primary intent.

    This still introduces no new classification RULE beyond what
    `classify_intent()`, `classify_emotion()`,
    `classify_emotion_intensity()`, `classify_emotional_trajectory()`,
    `has_academic_reference()`, and `extract_academic_topic()` already
    define individually -- it remains a thin, additive wrapper. All
    six underlying functions stay independently callable for any
    caller that only needs one dimension. Existing callers of
    `classify_intent()` / `classify_emotion()` / the Phase 2.9/3.1/3.2/
    Sprint 1.3 `analyze_message()` shapes are unaffected: `intent`,
    `emotion`, `emotion_intensity`, `emotion_trajectory`, and
    `secondary_academic_signal` keep their exact same values and
    meaning; `academic_topic` is a strictly additive sixth key, not a
    replacement of anything.
    """
    intent = classify_intent(question)
    emotion = classify_emotion(question)
    emotion_intensity = classify_emotion_intensity(question, emotion)
    emotion_trajectory = classify_emotional_trajectory(chat_history, emotion)
    secondary_academic_signal = has_academic_reference(question)
    academic_topic = extract_academic_topic(question)
    return {
        "intent": intent,
        "emotion": emotion,
        "emotion_intensity": emotion_intensity,
        "emotion_trajectory": emotion_trajectory,
        "secondary_academic_signal": secondary_academic_signal,
        "academic_topic": academic_topic,
    }



def classify_intent(question: str) -> str:
    """
    Returns one of VALID_INTENTS. Pure function of `question` only --
    no context, no I/O, no randomness, so identical input always
    yields identical output.

    Precedence (see module docstring, SPRINT 1.1 ADDITION): emotional
    > motivation > career > celebration > greeting > gratitude >
    farewell > academic/general (default). The original
    emotional/motivation/career/academic/general ordering and patterns
    are unchanged from before this sprint; the four new buckets are
    only reachable by messages that would otherwise have fallen
    through to the academic/general default.
    """
    if not question or not question.strip():
        return "general"

    text = question.strip()

    if _EMOTIONAL_RE.search(text):
        return "emotional"

    if _MOTIVATION_RE.search(text):
        return "motivation"

    if _CAREER_RE.search(text):
        return "career"

    if _CELEBRATION_RE.search(text):
        return "celebration"

    if _GREETING_RE.search(text):
        return "greeting"

    if _GRATITUDE_RE.search(text):
        return "gratitude"

    if _FAREWELL_RE.search(text):
        return "farewell"

    if _ACADEMIC_RE.search(text):
        return "academic"

    # Default: unrecognized questions keep the ORIGINAL pre-Sprint-10
    # behaviour (retrieval + hard similarity gate) under the "general"
    # label -- functionally identical to "academic" (see mentor_service
    # .py's _STRICT_GATE_INTENTS / _RETRIEVAL_INTENTS), so an unrecognized
    # question's *behaviour* never regresses, only its label may read
    # "general" instead of "academic". This is the backward-
    # compatibility anchor for this module.
    return "general"
