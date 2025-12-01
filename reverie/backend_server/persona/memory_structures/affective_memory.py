"""
File: affective_memory.py
Description: Defines an affect-rich event memory that tracks event embeddings,
emotions, and occurrence counts across short-term, long-term, and permanent
stores. The memory containers follow fixed sizes for short- and long-term
queues and keep permanent entries indefinitely.
"""
import json
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy

from global_methods import check_if_file_exists, create_folder_if_not_there


class AffectiveMemory:
  """
  Memory system for events enriched with emotions and occurrence counts.
  Each entry follows the structure:
    [event_embedding, event_text, {emotion: score}, times_occurred]
  """

  EMOTION_VOCAB = [
      "admiration", "adoration", "aesthetic_appreciation", "amusement",
      "anger", "anxiety", "awe", "awkwardness", "boredom", "calmness",
      "confusion", "craving", "disgust", "empathic_pain", "entrancement",
      "excitement", "fear", "horror", "interest", "joy", "nostalgia",
      "relief", "romance", "sadness", "satisfaction", "sexual_desire",
      "surprise",
  ]

  MEMORY_STATE_RANK = {
      "short_term_memory": 0,
      "long_term_memory": 1,
      "permanent_memory": 2,
  }

  def __init__(
      self,
      f_saved: Optional[str],
      short_maxlen: int = 100,
      long_maxlen: int = 50,
      similarity_threshold: float = 0.85,
  ):
    self.short_maxlen = short_maxlen
    self.long_maxlen = long_maxlen
    self.similarity_threshold = similarity_threshold

    self.short_term_memory: deque = deque(maxlen=self.short_maxlen)
    self.long_term_memory: deque = deque(maxlen=self.long_maxlen)
    self.permanent_memory: List = []

    if f_saved and check_if_file_exists(f_saved):
      self.load(f_saved)

  def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
    a_arr = numpy.array(a)
    b_arr = numpy.array(b)
    denom = numpy.linalg.norm(a_arr) * numpy.linalg.norm(b_arr)
    if denom == 0:
      return 0.0
    return float(numpy.dot(a_arr, b_arr) / denom)

  def _clean_emotions(self, emotions: Optional[Dict[str, int]]) -> Dict[str, int]:
    if not emotions:
      return {}
    cleaned = {}
    for emo, score in emotions.items():
      if emo not in self.EMOTION_VOCAB:
        continue
      try:
        val = int(score)
      except Exception:
        continue
      val = max(0, min(val, 100))
      cleaned[emo] = val
    top_five = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {k: v for k, v in top_five}

  def _weight_emotions(self, emotions: Dict[str, int], weight: float) -> Dict[str, int]:
    """
    Weight and re-round emotion scores to integers while clamping to [0, 100].
    This lets us bias toward newer observations when merging.
    """
    weighted = {}
    for emo, score in emotions.items():
      weighted_score = int(round(score * weight))
      weighted[emo] = max(0, min(weighted_score, 100))
    return weighted

  def _identify_opposite_pairs(self, emotion_scores: Dict[str, int]) -> List[Tuple[str, str]]:
    """
    Ask an LLM to propose absolute opposite emotion pairs present in this
    event, and fall back to a deterministic map if the LLM call fails.
    """
    if not emotion_scores:
      return []

    # Lazy import to avoid circular dependencies or import-time OpenAI issues.
    try:
      from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_identify_opposite_emotions
    except Exception:
      run_gpt_prompt_identify_opposite_emotions = None

    # Fallback hand-authored opposite map (keeps deterministic behavior when LLM unavailable).
    fallback_pairs = {
        "joy": "anger",
        "sadness": "joy",
        "admiration": "disgust",
        "adoration": "disgust",
        "awe": "boredom",
        "amusement": "anxiety",
        "anger": "calmness",
        "fear": "calmness",
        "horror": "relief",
        "relief": "anxiety",
        "nostalgia": "surprise",
        "interest": "boredom",
        "surprise": "boredom",
        "romance": "awkwardness",
        "satisfaction": "craving",
        "sexual_desire": "disgust",
    }

    if run_gpt_prompt_identify_opposite_emotions:
      try:
        llm_pairs = run_gpt_prompt_identify_opposite_emotions(
            list(emotion_scores.keys()))[0]
        if isinstance(llm_pairs, list):
          parsed_pairs = []
          for pair in llm_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
              parsed_pairs.append((pair[0], pair[1]))
            elif isinstance(pair, str) and "|" in pair:
              left, right = pair.split("|", 1)
              parsed_pairs.append((left.strip(), right.strip()))
          if parsed_pairs:
            return parsed_pairs
      except Exception:
        # We intentionally swallow errors here to preserve a best-effort merge.
        pass

    # Build fallback pairs that exist in the current emotion set only.
    pairs = []
    for emo, opp in fallback_pairs.items():
      if emo in emotion_scores and opp in emotion_scores:
        pairs.append((emo, opp))
    return pairs

  def _merge_emotions(self, existing: Dict[str, int],
                      incoming: Dict[str, int]) -> Dict[str, int]:
    """
    Merge old and new emotions using a weighted blend (old 30%, new 70%),
    resolve absolute opposites by dropping the lower-scored side, and keep the
    top five emotions.
    """
    old_weighted = self._weight_emotions(existing, 0.3)
    new_weighted = self._weight_emotions(incoming, 0.7)

    combined = dict()
    for emo, score in old_weighted.items():
      combined[emo] = combined.get(emo, 0) + score
    for emo, score in new_weighted.items():
      combined[emo] = combined.get(emo, 0) + score

    # Resolve opposites by removing the lower score in each pair.
    for emo_a, emo_b in self._identify_opposite_pairs(combined):
      score_a = combined.get(emo_a)
      score_b = combined.get(emo_b)
      if score_a is None or score_b is None:
        continue
      if score_a > score_b:
        combined.pop(emo_b, None)
      elif score_b > score_a:
        combined.pop(emo_a, None)
      else:
        # If tied, drop the older (favor new by removing the first one we saw).
        combined.pop(emo_a, None)

    top_five = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {k: v for k, v in top_five}

  def _merge_embeddings(self, existing: List[float],
                        incoming: List[float]) -> List[float]:
    try:
      if len(existing) == len(incoming):
        return [(i + j) / 2 for i, j in zip(existing, incoming)]
    except Exception:
      pass
    return incoming

  def _get_container(self, memory_type: str):
    if memory_type == "short_term_memory":
      return self.short_term_memory
    if memory_type == "long_term_memory":
      return self.long_term_memory
    if memory_type == "permanent_memory":
      return self.permanent_memory
    raise ValueError(f"Unknown memory type: {memory_type}")

  def _remove_from_container(self, memory_type: str, idx: int):
    if memory_type == "permanent_memory":
      self.permanent_memory.pop(idx)
      return
    container = list(self._get_container(memory_type))
    container.pop(idx)
    if memory_type == "short_term_memory":
      self.short_term_memory = deque(container, maxlen=self.short_maxlen)
    elif memory_type == "long_term_memory":
      self.long_term_memory = deque(container, maxlen=self.long_maxlen)

  def _add_to_container(self, memory_type: str, entry: List):
    if memory_type == "permanent_memory":
      self.permanent_memory.append(entry)
    elif memory_type == "short_term_memory":
      self.short_term_memory.append(entry)
    elif memory_type == "long_term_memory":
      self.long_term_memory.append(entry)
    else:
      raise ValueError(f"Unknown memory type: {memory_type}")

  def _find_similar(self, event_embedding: List[float]) -> Optional[Tuple[str, int, float]]:
    best_match = None
    for mem_type in ("short_term_memory", "long_term_memory", "permanent_memory"):
      container = self._get_container(mem_type)
      for idx, entry in enumerate(container):
        sim = self._cosine_similarity(entry[0], event_embedding)
        if sim >= self.similarity_threshold:
          if not best_match or sim > best_match[2]:
            best_match = (mem_type, idx, sim)
    return best_match

  def _target_state_new(self, emotions: Dict[str, int]) -> Optional[str]:
    scores = emotions.values()
    if not scores:
      return None
    if max(scores) > 84:
      return "permanent_memory"
    if max(scores) > 50:
      return "long_term_memory"
    else:
      return "short_term_memory"

  def _target_state_existing(self, current_memory_type: str,
                             new_emotions: Dict[str, int],
                             times_occurred: int) -> str:
    state_rank = self.MEMORY_STATE_RANK[current_memory_type]
    if any(score > 81 for score in new_emotions.values()):
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["permanent_memory"])
    elif any(score > 51 for score in new_emotions.values()):
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["long_term_memory"])
    else:
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["short_term_memory"])

    if times_occurred > 99:
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["permanent_memory"])
    elif times_occurred > 49:
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["long_term_memory"])
    else:
      state_rank = max(state_rank, self.MEMORY_STATE_RANK["short_term_memory"])

    for mem_type, rank in self.MEMORY_STATE_RANK.items():
      if rank == state_rank:
        return mem_type

  def store_event(
      self,
      event_embedding: List[float],
      event_text: str,
      emotions: Optional[Dict[str, int]],
      occurrences: int = 1,
  ) -> Optional[str]:
    """
    Store or update an event with affective context.

    Returns the memory tier the event resides in after processing.
    """
    cleaned_emotions = self._clean_emotions(emotions)
    occurrences = max(1, int(occurrences))

    similar = self._find_similar(event_embedding)
    if not similar:
      target_memory = self._target_state_new(cleaned_emotions)
      if not target_memory:
        return None
      entry = [event_embedding, event_text, cleaned_emotions, occurrences]
      self._add_to_container(target_memory, entry)
      return target_memory

    mem_type, idx, _ = similar
    container = self._get_container(mem_type)
    entry = container[idx]
    entry[0] = self._merge_embeddings(entry[0], event_embedding)
    entry[2] = self._merge_emotions(entry[2], cleaned_emotions)
    entry[3] += occurrences

    target_memory = self._target_state_existing(
        mem_type, cleaned_emotions, entry[3])
    if target_memory != mem_type:
      self._remove_from_container(mem_type, idx)
      self._add_to_container(target_memory, entry)
    return target_memory

  def _serialize_memory(self, memory_obj) -> List:
    return [entry for entry in memory_obj]

  def save(self, f_saved: str):
    """
    Persist only the three memory stores, keeping the on-disk shape simple and
    directly aligned with the required queue/list structure.
    """
    create_folder_if_not_there(f_saved)
    payload = {
        "short_term_memory": self._serialize_memory(self.short_term_memory),
        "long_term_memory": self._serialize_memory(self.long_term_memory),
        "permanent_memory": self.permanent_memory,
    }
    with open(f_saved, "w") as outfile:
      json.dump(payload, outfile)

  def load(self, f_saved: str):
    """
    Load the three memory stores from disk, defaulting to empty structures if
    any field is missing.
    """
    saved = json.load(open(f_saved))
    self.short_term_memory = deque(
        saved.get("short_term_memory", []), maxlen=self.short_maxlen)
    self.long_term_memory = deque(
        saved.get("long_term_memory", []), maxlen=self.long_maxlen)
    self.permanent_memory = saved.get("permanent_memory", [])

  def format_for_prompt(self, max_items: int = 5) -> str:
    """
    Create a human-readable snippet for prompt-conditioning that surfaces the
    most recent and most salient affective memories.
    """
    lines: List[str] = []

    def _render_entry(entry, label: str):
      _embedding, text, emotions, times = entry
      emo_str = ", ".join([f"{k}: {v}" for k, v in emotions.items()]) or "no tagged emotions"
      lines.append(f"{label}: {text} | emotions [{emo_str}] | seen {times} time(s)")

    # Recent short-term items (recency bias).
    for entry in list(self.short_term_memory)[-max_items:]:
      _render_entry(entry, "recent")

    # High-salience long-term items.
    long_sorted = sorted(list(self.long_term_memory), key=lambda e: max(e[2].values()) if e[2] else 0, reverse=True)
    for entry in long_sorted[:max_items]:
      _render_entry(entry, "long-term")

    # Permanent anchors.
    for entry in self.permanent_memory[:max_items]:
      _render_entry(entry, "permanent")

    if not lines:
      return "No affective memories recorded yet."
    return "\n".join(lines)
