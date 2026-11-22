# MIT License
# (see repository LICENSE)
"""Offline replay evaluation for contextual-bandit policies."""

import argparse
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Replay evaluation result."""

    processed: int
    matched: int
    clicks: int
    replay_ctr: float

    @property
    def match_rate(self) -> float:
        return self.matched / self.processed if self.processed else 0.0


def load_policy(policy_file: str | Path):
    """Load a policy module without executing policy source text as Python code."""
    policy_path = Path(policy_file).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    spec = importlib.util.spec_from_file_location("yahoo_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load policy module from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_articles(article_file: str | Path) -> dict[int, list[float]]:
    article_path = Path(article_file)
    articles_array = np.loadtxt(article_path, ndmin=2)
    if articles_array.shape[1] != 7:
        raise ValueError(
            f"Expected article rows of article_id plus 6 features, got shape {articles_array.shape}"
        )

    articles: dict[int, list[float]] = {}
    for row in articles_array:
        article_id = int(row[0])
        if article_id in articles:
            raise ValueError(f"Duplicate article ID {article_id} in {article_path}")
        articles[article_id] = [float(value) for value in row[1:]]
    return articles


def parse_log_line(line: str, line_number: int) -> tuple[int, list[float], int, int, list[int]]:
    """Parse timestamp, user context, logged action, reward, and candidate IDs."""
    fields = line.split()
    if len(fields) < 10:
        raise ValueError(f"Line {line_number}: expected at least 10 fields, got {len(fields)}")

    try:
        timestamp = int(fields[0])
        user_features = [float(value) for value in fields[1:7]]
        chosen_article = int(fields[7])
        reward = int(fields[8])
        candidate_ids = [int(value) for value in fields[9:]]
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid numeric value") from exc

    if reward not in (0, 1):
        raise ValueError(f"Line {line_number}: reward must be 0 or 1, got {reward}")
    if not candidate_ids:
        raise ValueError(f"Line {line_number}: candidate list is empty")
    if chosen_article not in candidate_ids:
        raise ValueError(f"Line {line_number}: displayed article {chosen_article} is absent from candidates")

    return timestamp, user_features, chosen_article, reward, candidate_ids


def iter_log_lines(log_file: str | Path) -> Iterator[tuple[int, int, list[float], int, int, list[int]]]:
    with Path(log_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield (line_number, *parse_log_line(line, line_number))


def process_line(policy, logline: Iterable[str], line_number: int = 1):
    # Kept as a convenient compatibility function for the legacy policy interface.
    fields = list(logline)
    timestamp, user_features, chosen, reward, articles = parse_log_line(" ".join(fields), line_number)
    recommendation = policy.recommend(timestamp, user_features, articles)
    return reward, chosen, recommendation


def evaluate_detailed(policy, input_generator, article_count: int | None = None) -> ReplayResult:
    """
    Args:
        policy: A contextual-bandit policy object with recommend() and update() methods.
        input_generator: An iterable of tuples (line_number, timestamp, user_features, chosen, reward, articles)
            where user_features is a list of floats, chosen is the logged article ID,
            reward is 0 or 1, and articles is a list of candidate article IDs.
        article_count: Optional number of known articles. If provided, will check that all candidates
            are known articles.
    Returns:
        A ReplayResult object containing the number of processed lines, matched lines,
        一个包含evaluation 结果的对象
        对象：
        - processed 是处理过的行。 
        - matched 是指这个文章在历史中被展示过（所以能观测到是否被点击）。
        - clicks 如果被点击就加1, 否则不加。
        - replay_ctr 是点击率，等于 clicks / matched。

    """
    clicks = 0
    matched = 0
    processed = 0

    for line_number, timestamp, user_features, chosen, reward, articles in input_generator:
        processed += 1
        if article_count is not None and any(article not in policy._known_article_ids for article in articles):
            raise ValueError(f"Line {line_number}: candidate contains an unknown article")

        recommendation = policy.recommend(timestamp, user_features, articles)
        if recommendation not in articles:
            raise ValueError(f"Line {line_number}: policy recommended article outside candidates: {recommendation}")

        if recommendation == chosen:
            if reward < 0:
                logger.debug("Line %d: matched recommendation %d with reward %d", line_number, recommendation, reward)
            policy.update(reward)
            clicks += reward
            matched += 1
        else:
            if reward < 0:
                logger.debug("Line %d: matched recommendation %d with reward %d", line_number, recommendation, reward)

            # logger.debug("Line %d: recommended %d, logged %d, reward %d", line_number, recommendation, chosen, reward)  
            policy.update(-1)

    replay_ctr = clicks / matched if matched else 0.0
    logger.info("Processed %d; matched %d; matched clicks %d; replay CTR %.5f", processed, matched, clicks, replay_ctr)
    return ReplayResult(processed, matched, clicks, replay_ctr)


def evaluate(policy, input_generator, article_count: int | None = None) -> float:
    return evaluate_detailed(policy, input_generator, article_count).replay_ctr


def run_detailed(policy_file: str | Path, log_file: str | Path, articles_file: str | Path) -> ReplayResult:
    policy = load_policy(policy_file)
    articles = load_articles(articles_file)
    policy.set_articles(articles)
    policy._known_article_ids = set(articles)
    return evaluate_detailed(policy, iter_log_lines(log_file), len(articles))


def run(policy_file: str | Path, log_file: str | Path, articles_file: str | Path) -> float:
    return run_detailed(policy_file, log_file, articles_file).replay_ctr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file")
    parser.add_argument("articles_file")
    parser.add_argument("source_file", help="Python policy module to load")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    run(args.source_file, args.log_file, args.articles_file)
