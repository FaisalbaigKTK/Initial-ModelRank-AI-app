import pandas as pd

from analyze_service import filter_ranked_results
from hf_live_service import analyze_live_hf_query


def run_pipeline(
    query: str,
    domain: str = "All",
    family: str = "All",
    label: str = "All",
    only_representatives: bool = False,
    top_n: int = 50,
    mode: str = "live",
) -> pd.DataFrame:
    if mode == "live":
        live_df = analyze_live_hf_query(
            query=query,
            limit_per_query=max(8, min(top_n, 20)),
        )
        if live_df.empty:
            return live_df

        return filter_ranked_results(
            df=live_df,
            query="",
            domain=domain,
            family=family,
            label=label,
            only_representatives=only_representatives,
            top_n=top_n,
        )

    from analyze_service import analyze_repos
    return analyze_repos(
        query=query,
        domain=domain,
        family=family,
        label=label,
        only_representatives=only_representatives,
        top_n=top_n,
    )
