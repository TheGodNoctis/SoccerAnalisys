from collections import defaultdict

def summarize_passes(passes):
    summary = defaultdict(lambda: {
        "total": 0,
        "certos": 0,
        "interceptados": 0,
        "percentual_acerto": 0.0
    })

    for p in passes:
        team = p["team"]
        summary[team]["total"] += 1
        if p["pass_type"] == "normal":
            summary[team]["certos"] += 1
        elif p["pass_type"] == "interceptado":
            summary[team]["interceptados"] += 1

    for team, stats in summary.items():
        total = stats["total"]
        certos = stats["certos"]
        stats["percentual_acerto"] = round((certos / total) * 100, 2) if total > 0 else 0.0

    return dict(summary)
