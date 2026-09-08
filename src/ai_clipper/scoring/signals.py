"""
The language-specific half of the scorer, in one place and editable as JSON.

Every word list in this project was Indonesian, written into the middle of
hook_scorer.py, and adapting the tool to a cooking podcast or a football one
meant editing Python. That is the wrong shape for the one part of the system
that is guaranteed to need changing.

The split is along a real seam, not an arbitrary one. Measured across two
one-hour episodes from different domains:

| layer | what it is | how well it travelled |
|---|---|---|
| structure | a clip must not stop mid-question, must not cross a subject boundary, must open on a line that stands alone | carried over intact |
| vocabulary | which words mean stakes, curiosity, a question, a promise | `stakes` fired on 80% of candidates in one episode and 53% in the other, `intensity` on 42% of both |

The structural half stays in code because it is about how conversation works.
The vocabulary half lives here, and `signals.json` next to the project overrides
or extends any part of it, the same way `styles.json` works for captions.

Adding a language is adding a file. Nothing here is imported by name from
anywhere that would break if the file named a bank this build has never heard of.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Optional, Sequence, Set


@dataclass(frozen=True)
class SignalPack:
    """One language's vocabulary, plus the thresholds measured against it."""

    name: str = "indonesian"

    # --- what makes a hook ---------------------------------------------------
    question_words: Set[str] = field(default_factory=set)
    curiosity_markers: Set[str] = field(default_factory=set)
    contrarian_markers: Set[str] = field(default_factory=set)
    intensity_markers: Set[str] = field(default_factory=set)
    stakes_markers: Set[str] = field(default_factory=set)
    first_person: Set[str] = field(default_factory=set)
    past_markers: Set[str] = field(default_factory=set)
    quantity_words: Set[str] = field(default_factory=set)

    # --- what makes an ending unfinished -------------------------------------
    lead_in_endings: Set[str] = field(default_factory=set)
    speech_act_verbs: Set[str] = field(default_factory=set)
    empty_after_verb: Set[str] = field(default_factory=set)
    question_openers: Set[str] = field(default_factory=set)
    promise_markers: Set[str] = field(default_factory=set)

    # --- what makes an opening work on its own -------------------------------
    discourse_particles: Set[str] = field(default_factory=set)
    back_references: Set[str] = field(default_factory=set)
    not_names: Set[str] = field(default_factory=set)

    # --- general -------------------------------------------------------------
    stopwords: Set[str] = field(default_factory=set)
    filler: Set[str] = field(default_factory=set)

    # --- numbers that were measured, not guessed -----------------------------
    weights: Dict[str, float] = field(default_factory=dict)
    question_min_content: int = 4
    opening_min_content: int = 4
    opening_min_content_question: int = 3
    closing_window: float = 12.0
    payload_words: int = 10
    spillover_window: float = 8.0


INDONESIAN = SignalPack(
    name="indonesian",
    question_words={
        "apa", "kenapa", "mengapa", "gimana", "bagaimana", "siapa",
        "kapan", "kok", "berapa",
    },
    curiosity_markers={
        "ternyata", "rahasia", "gak nyangka", "nggak nyangka", "sebenarnya",
        "yang bikin", "sampai akhirnya", "baru tau", "baru sadar",
    },
    contrarian_markers={
        "padahal", "salah besar", "kebanyakan orang", "banyak yang salah",
        "bukan berarti", "justru", "malah", "salah kaprah",
    },
    intensity_markers={"gila", "parah", "banget", "gokil", "kaget", "shock"},
    # Something was at risk and it nearly went wrong. This bank exists because
    # the clip a reviewer knew had gone viral - a near margin call, negotiating
    # with the broker to avoid being wiped out - fired no curiosity or
    # contrarian marker at all and scored last of fifteen.
    stakes_markers={
        "margin call", "rugi", "kerugian", "bangkrut", "hancur", "wipe out",
        "hampir", "nyaris", "gagal", "utang", "hutang", "pinjam", "anjlok",
        "jatuh", "kesalahan", "fatal", "nyesel", "menyesal", "kepaksa",
        "terpaksa", "kejebak", "nombok", "abis", "habis", "selamat", "kepepet",
        "arb",
    },
    first_person={"gue", "gua", "saya", "aku", "gw"},
    past_markers={
        "waktu itu", "dulu", "akhirnya", "pas itu", "pernah", "ternyata",
        "sempat", "sempet", "kemarin", "dulunya", "awalnya", "jadinya",
    },
    quantity_words={"persen", "juta", "miliar", "ribu", "kali lipat"},
    # Phrases that end a sentence while promising the next one.
    lead_in_endings={
        "gini", "begini", "gitu", "begitu", "yaitu", "adalah", "jadi",
        "maksudnya", "misalnya", "contohnya", "soalnya", "karena", "terus",
        "kayak", "tapi", "cuma", "nah",
    },
    # Verbs that announce speech without carrying it out: "Gue bilang sama
    # Maybank." says he spoke to Maybank and never what he said.
    speech_act_verbs={
        "bilang", "bilangin", "ngomong", "omong", "ngomongin", "tanya", "nanya",
        "tanyain", "jawab", "jawabin", "cerita", "ceritain", "jelasin",
        "jelaskan", "sebut", "sebutin", "nego", "minta", "suruh", "kasih",
        "bahas", "bahasin",
    },
    empty_after_verb={
        "sama", "ke", "kepada", "dia", "gue", "gua", "saya", "lu", "lo", "kamu",
        "itu", "ini", "nya", "lah", "dong", "sih", "aja", "ya", "deh", "kan",
        "tuh", "nih", "juga", "pun", "doang", "terus", "sm",
        "bahwa", "kalau", "kalo", "gini", "begini", "gitu", "begitu",
    },
    # Whisper punctuates a fast exchange badly, so a question has to be
    # recognisable without a question mark.
    question_openers={
        "apa sih", "apa aja", "apa yang", "kenapa", "mengapa", "gimana",
        "bagaimana", "seberapa", "kapan", "siapa", "berapa", "kok bisa",
        "gue penasaran", "gua penasaran", "saya penasaran", "menurut lu",
        "menurut lo", "menurut kamu",
    },
    promise_markers={
        "gue kasih tau", "gua kasih tau", "gue kasih ilmu", "gue buka",
        "gue ceritain", "gue jelasin", "contohnya gini", "misalnya gini",
        "gini deh", "gue kasih contoh", "jadi contoh",
    },
    # Nearly every line in this register opens with one of these and means
    # nothing by it, so they are stripped before an opening is judged.
    discourse_particles={
        "nah", "jadi", "terus", "trus", "oke", "ok", "iya", "ya", "cuman",
        "cuma", "tapi", "dan", "kan", "soalnya", "makanya", "eh", "oh", "wah",
        "kalo", "kalau", "karena", "emang", "memang", "yang", "ini", "itu",
    },
    back_references={"ini", "itu", "dia", "nya", "mereka", "gitu", "situ", "sana"},
    # Capitalised words that are not names. Whisper capitalises weekdays and
    # months like any proper noun, and losing "Senin" costs a clip nothing.
    not_names={
        "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
        "januari", "februari", "maret", "april", "mei", "juni", "juli",
        "agustus", "september", "oktober", "november", "desember",
        "indonesia", "iya", "oke", "nah", "jadi", "gue", "lu",
    },
    stopwords={
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "juga", "ya", "kan",
        "aja", "saja", "untuk", "dengan", "karena", "jadi", "kalau", "kalo",
        "adalah", "atau", "tapi", "tidak", "gak", "nggak", "sudah", "udah",
        "akan", "bisa", "saya", "kamu", "kita", "mereka", "dia", "nya", "loh",
        "sih", "deh", "gua", "gue", "lo", "lu",
    },
    # carries no topic even though it is not a grammatical stopword
    filler={
        "iya", "gitu", "banget", "emang", "terus", "kayak", "udah", "biar",
        "sama", "abis", "bilang", "orang", "maksudnya", "berarti", "soalnya",
        "beneran", "mungkin", "pokoknya", "makanya",
    },
    weights={
        "opening_question": 2.5,
        "curiosity_marker": 2.0,
        "contrarian_marker": 1.5,
        "intensity_marker": 1.0,
        "has_number": 1.0,
        "stakes": 1.2,
        "personal_story": 2.0,
        "topic_distinctiveness": 2.0,
        "self_contained_opening": 1.5,
        "ends_incomplete": -3.0,
        "chatter": -2.5,
        "answered_question": 2.0,
    },
)

# Short names for the banks, so a signals.json reads like a config file rather
# than a mirror of the dataclass. Both forms work.
ALIASES = {
    "curiosity": "curiosity_markers",
    "contrarian": "contrarian_markers",
    "intensity": "intensity_markers",
    "stakes": "stakes_markers",
    "past": "past_markers",
    "quantity": "quantity_words",
    "lead_in": "lead_in_endings",
    "speech_verbs": "speech_act_verbs",
    "promise": "promise_markers",
    "particles": "discourse_particles",
    "back_refs": "back_references",
    "questions": "question_words",
    "openers": "question_openers",
    "not_a_name": "not_names",
}

PACKS = {"indonesian": INDONESIAN}
FILENAME = "signals.json"

_active = INDONESIAN


def active() -> SignalPack:
    """The pack the scorer is currently using."""
    return _active


def use(pack: SignalPack) -> SignalPack:
    """Install a pack. Returns the one it replaced, for tests to put back."""
    global _active
    previous, _active = _active, pack
    return previous


def merge(base: SignalPack, data: Dict) -> SignalPack:
    """
    Apply a JSON document over a pack.

    A list replaces a bank outright. To add without retyping the whole thing,
    prefix the bank name with "+", the way `styles.json` patches a preset rather
    than redefining it:

        {"markers": {"+stakes": ["kena batunya"], "intensity": ["gokil"]}}

    Unknown keys are ignored rather than fatal. A pack written for a later
    version of this project should still load in an older one, minus whatever it
    does not understand.
    """
    changes = {}
    known = {f.name for f in fields(SignalPack)}

    for key, value in (data.get("markers") or {}).items():
        additive = key.startswith("+")
        name = key.lstrip("+")
        name = ALIASES.get(name, name)
        if name not in known:
            continue
        current = getattr(base, name)
        incoming = {str(v).lower() for v in value}
        changes[name] = (set(current) | incoming) if additive else incoming

    weights = data.get("weights") or {}
    if weights:
        changes["weights"] = {**base.weights, **{k: float(v) for k, v in weights.items()}}

    for key, value in (data.get("thresholds") or {}).items():
        if key in known:
            changes[key] = type(getattr(base, key))(value)

    if data.get("name"):
        changes["name"] = str(data["name"])

    return replace(base, **changes)


def load(path: Optional[str] = None, base: Optional[SignalPack] = None) -> SignalPack:
    """
    Read a signals file, falling back to the built-in pack.

    With no path and no file next to the project, the built-in Indonesian pack
    is returned unchanged, so nothing has to exist for the tool to work.
    """
    pack = base or PACKS.get((base or INDONESIAN).name, INDONESIAN)
    if path is None:
        path = FILENAME if os.path.exists(FILENAME) else None
    if not path or not os.path.exists(path):
        return pack

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("extends") in PACKS:
        pack = PACKS[data["extends"]]
    return merge(pack, data)
