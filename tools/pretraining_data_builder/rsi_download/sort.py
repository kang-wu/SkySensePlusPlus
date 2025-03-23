def sort_by_cloudcover(search_result):
    entries = search_result["feed"]["entry"]
    sorted_entries = []
    for entry in entries:
        sorted_entries.append(
            [
                float(e["content"])
                for e in entry["double"]
                if e["name"] == "cloudcoverpercentage"
            ][0]
        )
    return sorted(sorted_entries, key=float)
