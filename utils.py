def parse_video_source(video_arg: str) -> str | int:
    """If user provides "--source 0" """
    if video_arg.isdigit():
        return int(video_arg)
    return video_arg