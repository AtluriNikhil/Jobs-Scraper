#!/usr/bin/env python3
"""
job_scraper_scheduler.py

Continuously scrapes job listings, filters by relevance with a Cross-Encoder,
deduplicates by source priority, and saves both unfiltered and filtered results.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import schedule
from jobspy import scrape_jobs
from sentence_transformers import CrossEncoder

# ------------------ Configuration Flags ------------------
INTERN_MODE = True       # include “Intern” variants
COOP_MODE = True         # include “co-op” variants

# ------------------ Job Titles ------------------
BASE_TITLES = [
    "Data Engineer", "Software Engineer"
]

def generate_titles(base_titles, intern=INTERN_MODE, coop=COOP_MODE):
    """
    Build the full list of job queries based on flags.
    """
    titles = list(base_titles)
    if intern:
        titles += [f"{t} Intern" for t in base_titles]
    if coop:
        titles += [f"{t} co-op" for t in base_titles]
    return titles

# ------------------ Other Configuration ------------------
VALID_LOCATIONS = {"USA"}
MAX_LISTINGS = 25
HOURS_OLD = 1
OUTPUT_DIR = Path("job_reports")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE_UNFILTERED = OUTPUT_DIR / "Jobs_Unfiltered.xlsx"
OUTPUT_FILE_FILTERED   = OUTPUT_DIR / "Jobs.xlsx"

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------ Relevance Model ------------------
logger.info("Loading the Cross Encoder Model....")
relevance_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# ------------------ Helper Functions ------------------

def fetch_job_listings(job_title: str, location: str, max_listings: int) -> pd.DataFrame:
    """
    Scrape up to `max_listings` from multiple sites for a given title/location.
    """
    logger.info(f"Scraping up to {max_listings} '{job_title}' roles in {location} from Linkedin")
    df = scrape_jobs(
        site_name=["linkedin"],
        search_term=job_title,
        location=location,
        results_wanted=max_listings,
        hours_old=HOURS_OLD,
        linkedin_fetch_description=True
    )
    if df.empty:
        logger.warning(f"No listings for '{job_title}' in {location}")
    else:
        logger.info(f"Retrieved {len(df)} listings from LinkedIn")
    return df


def filter_by_relevance(df: pd.DataFrame, query: str, threshold: float = 1.0) -> pd.DataFrame:
    """
    Score each (query, title+description) pair and keep rows with score >= threshold.
    """
    texts = (df['title'].fillna('') + '. ' + df['description'].fillna('')).tolist()
    pairs = [(query, txt) for txt in texts]
    scores = relevance_model.predict(pairs)
    df = df.copy()
    df['relevance_score'] = scores
    return df[df['relevance_score'] >= threshold].drop(columns=['relevance_score'])


def process_and_save(frames: list, existing_path: Path, output_path: Path, description: str) -> int:
    """
    Merge new frames with existing file at existing_path, filter out rows older than 2 days,
    dedupe by site priority, sort within each query by company size, and save to output_path.
    Returns number of rows saved.
    """
    if not frames:
        logger.warning(f"No {description} listings to process.")
        return 0

    df_all = pd.concat(frames, ignore_index=True)
    if existing_path.exists():
        df_exist = pd.read_excel(existing_path)
        df_all = pd.concat([df_exist, df_all], ignore_index=True)
    
    df_all = df_all.drop_duplicates(
        subset=['id'],
        keep='first'
    )
    df_all.to_excel(output_path, index=False)
    logger.info(f"Saved {len(df_all)} {description} listings to {output_path}")
    return len(df_all)

# ------------------ Main Scrape Cycle ------------------

def run_scrape_cycle():
    titles = generate_titles(BASE_TITLES)
    raw_frames = []
    filtered_frames = []

    # scrape and collect
    for title in titles:
        for loc in VALID_LOCATIONS:
            df_raw = fetch_job_listings(title, loc, MAX_LISTINGS)
            if df_raw.empty:
                continue

            df_raw = df_raw.assign(search_query=title, search_location=loc)
            raw_frames.append(df_raw)

            df_filt = filter_by_relevance(df_raw, query=title, threshold=1.0)
            if df_filt.empty:
                logger.info(f"All '{title}' listings fell below relevance threshold—skipping.")
                continue
            filtered_frames.append(df_filt)

    # process and save
    process_and_save(raw_frames, OUTPUT_FILE_UNFILTERED, OUTPUT_FILE_UNFILTERED, "unfiltered")
    process_and_save(filtered_frames, OUTPUT_FILE_FILTERED, OUTPUT_FILE_FILTERED, "filtered")

# ------------------ Scheduler Entrypoint ------------------

def main():
    run_scrape_cycle()
    schedule.every(10).minutes.do(run_scrape_cycle)
    logger.info("Scheduler started: scraping every 20 minutes...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()