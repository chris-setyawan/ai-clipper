import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring.hook_scorer import (
    HookScorer,
    TranscriptSegment,
    ends_dangling,
    is_clean_ending,
)
from ai_clipper.scoring.topics import (
    TopicModel,
    continuation,
    topic_boundaries,
)


# --- the dangling speech act ----------------------------------------------

def test_announcing_speech_without_delivering_it_is_dangling():
    assert ends_dangling("Gue bilang sama Maybank.")
    assert ends_dangling("Nah gue hari Senin gue tanya.")
    assert ends_dangling("Gue nego lah.")


def test_a_delivered_speech_act_is_not_dangling():
    assert not ends_dangling("Bokap gue bilang gausah lah.")
    assert not ends_dangling("Gue nego akhirnya dia kasih reduce ke 5.000.")
    assert not ends_dangling("Gue selamat jadinya.")


def test_a_sentence_without_a_speech_verb_is_never_dangling():
    assert not ends_dangling("Harganya turun sampai 7.500 waktu itu.")


def test_clean_ending_rejects_every_kind_of_unfinished_line():
    assert is_clean_ending("Akhirnya gue jual semuanya.")
    assert not is_clean_ending("Terus gue mikir")          # no full stop
    assert not is_clean_ending("Jadi ceritanya gini.")     # lead-in
    assert not is_clean_ending("Iya.")                     # reaction
    assert not is_clean_ending("Gue bilang sama Maybank.")  # dangling


def test_the_scorer_penalises_a_dangling_ending():
    """
    The clip that prompted this: it ended on a full stop, passed every other
    completeness check, and still stopped in the middle of the story.
    """
    scorer = HookScorer()
    freq, total = {}, 1
    good = [
        TranscriptSegment(0, 10, "Waktu itu harganya turun sampai 7.500 bro."),
        TranscriptSegment(10, 20, "Akhirnya gue nego sama sekuritasnya sampai dapat keringanan."),
    ]
    bad = good[:1] + [TranscriptSegment(10, 20, "Terus gue bilang sama Maybank.")]

    assert scorer.score_window(good, freq, total).signals["ends_incomplete"] == 0.0
    assert scorer.score_window(bad, freq, total).signals["ends_incomplete"] < 0.0


# --- stakes and first-hand experience --------------------------------------

def test_a_near_disaster_story_outscores_the_same_facts_told_flatly():
    scorer = HookScorer()
    freq, total = {}, 1
    story = [
        TranscriptSegment(0, 10, "Waktu itu gue hampir kena margin call di harga 7.500."),
        TranscriptSegment(10, 20, "Akhirnya gue nego sama sekuritasnya biar gak kena likuidasi."),
    ]
    flat = [
        TranscriptSegment(0, 10, "Harga saham itu bergerak di kisaran 7.500."),
        TranscriptSegment(10, 20, "Sekuritas punya aturan sendiri soal batas pembiayaan."),
    ]
    a = scorer.score_window(story, freq, total)
    b = scorer.score_window(flat, freq, total)
    assert a.signals["stakes"] > 0 and a.signals["personal_story"] > 0
    assert b.signals["stakes"] == 0 and b.signals["personal_story"] == 0
    assert a.raw_score > b.raw_score


def test_a_third_person_summary_is_not_a_personal_story():
    scorer = HookScorer()
    seg = [TranscriptSegment(0, 10, "Dulu banyak investor yang akhirnya kena margin call.")]
    assert scorer.score_window(seg, {}, 1).signals["personal_story"] == 0.0


# --- topic structure --------------------------------------------------------

def _episode():
    """Two clearly separate subjects, ten segments each."""
    a = ("kapal tongkang batubara muatan pelabuhan dermaga",
         "tongkang kapal batubara pelabuhan muatan dermaga")
    b = ("startup pendanaan investor valuasi produk pengguna",
         "investor startup valuasi pendanaan pengguna produk")
    segs, t = [], 0.0
    for text in [a[i % 2] for i in range(12)] + [b[i % 2] for i in range(12)]:
        segs.append(TranscriptSegment(t, t + 5, text + "."))
        t += 5
    return segs


def test_a_boundary_is_found_where_the_subject_changes():
    segs = _episode()
    bounds = topic_boundaries(segs, block_words=6, context=3, cutoff=0.0)
    assert bounds, "a hard subject change must produce a boundary"
    assert min(abs(b - 60.0) for b in bounds) < 20.0


def test_too_short_an_episode_gets_no_boundaries():
    segs = [TranscriptSegment(0, 5, "kapal tongkang batubara muatan.")]
    assert topic_boundaries(segs) == []


def test_continuation_is_high_within_a_subject_and_low_across_one():
    segs = _episode()
    first = segs[:8]
    same = segs[8:12]
    other = segs[12:16]
    assert continuation(first, same) > continuation(first, other)


@dataclass
class FakeClip:
    start: float
    end: float


def test_a_clip_cut_mid_subject_is_extended_to_where_it_ends():
    segs = _episode()
    model = TopicModel(segs, sample_every=5.0)
    out = model.extend(FakeClip(0.0, 30.0), hard_max=150.0)
    assert out is not None, "the speaker is still on the same subject at 30s"
    _, new_end, added = out
    assert added > 0
    assert new_end <= 60.0 + 10.0, "must not run far past the subject change"


def test_a_clip_that_already_ends_at_a_subject_change_is_left_alone():
    segs = _episode()
    model = TopicModel(segs, sample_every=5.0)
    assert model.extend(FakeClip(0.0, 60.0), hard_max=150.0) is None


def test_extension_respects_the_hard_maximum():
    segs = _episode()
    model = TopicModel(segs, sample_every=5.0)
    out = model.extend(FakeClip(0.0, 30.0), hard_max=32.0)
    assert out is None, "a cap of 32s leaves no room to extend a 30s clip"


def test_thresholds_are_fitted_to_the_episode():
    model = TopicModel(_episode(), sample_every=5.0)
    assert model.low <= model.high
    assert model.levels, "the episode's own continuation levels are what calibrate it"


# --- the closing exchange ---------------------------------------------------

from ai_clipper.scoring.hook_scorer import closing_unfinished


def _segs(*lines):
    out, t = [], 0.0
    for text in lines:
        out.append(TranscriptSegment(t, t + 3, text))
        t += 3
    return out


def test_a_question_asked_across_several_segments_is_unfinished():
    """
    Whisper does not always punctuate a question with "?". This clip ended with
    the host still asking, and the answer starting one second after the cut.
    """
    body = _segs(
        "Bokap gue juga gak ngerti saham bro.",
        "Nah, tapi gue pengen penasaran nih.",
        "Apa sih yang waktu itu mindset-mindset yang bokap lu ajarin dari kecil.",
        "Karena kan pasti ajarannya beda nih sama orang yang gak terlalu berada.",
    )
    after = _segs("Gue penasaran apa sih ajaran-ajarannya waktu itu.")
    assert closing_unfinished(body, after)


def test_a_clip_cut_before_the_question_it_announced_is_unfinished():
    body = _segs(
        "Nah dia jual 100%.",
        "Terus gue tanya juga.",
        "Kan papa udah jual nih.",
    )
    assert closing_unfinished(body, [])


def test_a_promise_with_nothing_delivered_is_unfinished():
    body = _segs(
        "Jadi akan selalu ada gapnya antara utang dia dan asetnya.",
        "Nah ini mungkin gue kasih ilmu yang cuman mungkin gue buka di.",
        "Di Timothy Ronald doang ya.",
    )
    assert closing_unfinished(body, [])


def test_a_confirmation_question_does_not_make_a_clip_unfinished():
    """
    "Oke, hilirisasi ya waktu itu ya?" needs no answer. Treating every "?" as an
    open question marked four clips the reviewer had passed as broken.
    """
    body = _segs(
        "Nah, jadi alat berat gue gak kesewa bro.",
        "Oke, hilirisasi ya waktu itu ya?",
    )
    assert not closing_unfinished(body, [])


def test_question_words_are_matched_whole():
    """"berapa" inside "beberapa" turned a statement into a question."""
    body = _segs(
        "Ini duitnya dibawa lari ke crypto.",
        "Jadi kemarin gue ada lihat beberapa kasus yang menarik sekali.",
        "Tapi kayaknya dari banknya mereka cek dan mereka tutup celahnya.",
    )
    assert not closing_unfinished(body, [])


def test_a_trailing_off_sentence_is_not_a_clean_ending():
    assert not is_clean_ending("Gue pinjam duit, sebagian gue pinjam duit sama...")


def test_a_complementizer_after_a_speech_verb_is_still_dangling():
    assert ends_dangling("Dia bilang bahwa.")


def test_a_clip_already_past_a_boundary_is_not_extended():
    """
    Extension can only look forward. When the clip's end already sits in the
    next subject, growing it adds more of that subject - which is how one clip
    ended up carrying two.
    """
    segs = _episode()
    model = TopicModel(segs, sample_every=5.0)
    boundary = min(model.boundaries, key=lambda b: abs(b - 60.0))
    assert model.extend(FakeClip(0.0, boundary + 10.0), hard_max=200.0) is None


# --- the opening ------------------------------------------------------------

from ai_clipper.scoring.hook_scorer import opening_stands_alone
from ai_clipper.scoring.topics import shift_to_a_real_opening


def test_an_opening_that_answers_an_unheard_question_does_not_stand_alone():
    assert not opening_stands_alone("Kawan lama, temen dia juga.")
    assert not opening_stands_alone("Ini apa nih?")
    assert not opening_stands_alone("Nah gue hari Senin gue tanya.")


def test_a_real_question_or_a_full_statement_stands_alone():
    assert opening_stands_alone("Gimana caranya lo bisa nilai kualitas manajemen dalam saham?")
    assert opening_stands_alone("Technical, apa fundamental semua gue pelajarin lah.")
    assert opening_stands_alone("Nah menurut lu itu capital marketnya akan jadi seperti apa?")


def test_leading_particles_are_stripped_before_judging():
    """Nearly every line in this register starts with one; they mean nothing."""
    assert opening_stands_alone("Nah, jadi bokap gue itu di bidang impor otomotif.")


class _Clip:
    def __init__(self, start, end, segments):
        self.start, self.end, self.segments = start, end, segments


def test_a_weak_opening_is_moved_forward():
    segs = _segs(
        "Nah gue hari Senin gue tanya.",
        "Nah gue tanya sekuritas tapi dananya udah langsung dibalikin lagi.",
        "Ternyata si hacker ini masuk ke sistem sekuritas lalu withdraw ke nama beda.",
        "Jadi salahnya sebenarnya ada di sisi banknya waktu itu.",
        "Mereka akhirnya tutup celahnya setelah kejadian itu semua.",
        "Dan sekarang sistemnya sudah jauh lebih ketat dari sebelumnya.",
        "Itu pelajaran mahal buat semua investor pasar modal.",
        "Sekarang gue selalu cek dulu sebelum transfer apapun.",
    )
    clip = _Clip(segs[0].start, segs[-1].end, segs)
    moved = shift_to_a_real_opening(clip, segs)
    assert moved is not None
    assert moved[0] == segs[1].start


def test_an_opening_that_already_works_is_left_alone():
    segs = _segs(
        "Gimana caranya lo bisa nilai kualitas manajemen dalam saham?",
        "Ownernya harus ngerti saham dulu baru gue mau masuk ke situ.",
        "Kalau dia gak ngerti, sahamnya naik ujungnya dia jual semua.",
        "Dia balik lagi ke bisnis realnya dan kita yang jadi korban.",
        "Itu yang terjadi di hampir 90% pebisnis real yang gue temui.",
        "Jadi gue selalu lihat psikologis ownernya dulu sebelum beli.",
    )
    clip = _Clip(segs[0].start, segs[-1].end, segs)
    assert shift_to_a_real_opening(clip, segs) is None


def test_the_shift_never_skips_the_only_place_the_subject_is_named():
    """
    "Perusahaan kapal kan itu? Kapal Hilong. Kapal dari China." is a weak
    opening by every test here, and skipping it leaves the clip starting on
    "waktu itu market cap-nya masih ratusan M" with no way to know of what.
    """
    segs = _segs(
        "Perusahaan kapal kan itu?",
        "Kapal Hilong dari China waktu itu.",
        "Waktu itu market capnya masih ratusan miliar doang.",
        "Tapi kapalnya sendiri itu harganya sudah 1,8 triliun bro.",
        "Gue merasa ini kayak mau ada backdoor listing disitu.",
        "Jadi gue mulai beli sedikit dulu buat testing posisi.",
        "Terus gue tanya-tanya ke sekuritas soal akuisisinya.",
    )
    clip = _Clip(segs[0].start, segs[-1].end, segs)
    assert shift_to_a_real_opening(clip, segs) is None


def test_the_name_guard_reads_each_line_on_its_own():
    """
    Whisper capitalises the first word of every segment. Joining the whole
    transcript and splitting on full stops made almost every segment-initial
    word look like a proper noun, and on a transcript with barely any
    punctuation that turned the name guard into a blanket refusal: six weak
    openings had replacements available and not one could be used.
    """
    body = [
        "Terus dia mulai cerita panjang lebar soal kejadian itu.",
        "Sampai akhirnya semua orang di ruangan itu diam saja.",
        "Dan itu jadi titik balik buat gue pribadi waktu itu.",
    ]
    segs = _segs(
        "Kenapa?",
        "Karena waktu itu semuanya berubah dengan sangat cepat.",
        "Gue pastikan lo akan jauh lebih sibuk lagi setelah ini.",
        *(body * 4),
    )
    clip = _Clip(segs[0].start, segs[-1].end, segs)
    moved = shift_to_a_real_opening(clip, segs)
    assert moved is not None, "no real name is lost here, so the shift must happen"
