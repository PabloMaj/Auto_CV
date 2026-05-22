VERIFIED_PROBLEMS_START = "<VERIFIED_PROBLEMS>"
VERIFIED_PROBLEMS_END = "</VERIFIED_PROBLEMS>"
IMPROVEMENT_SUGGESTIONS_START = "<IMPROVEMENT_SUGGESTIONS>"
IMPROVEMENT_SUGGESTIONS_END = "</IMPROVEMENT_SUGGESTIONS>"


def parse_section(text: str, start_token: str, end_token: str):

    if start_token not in text:
        return ""
    if end_token not in text:
        return ""

    start_idx = text.index(start_token) + len(start_token)
    end_idx = text.index(end_token)

    return text[start_idx:end_idx].strip()


def parse_improvement_response(response: str):

    verified_problems = parse_section(response, VERIFIED_PROBLEMS_START, VERIFIED_PROBLEMS_END)
    improvement_suggestions = parse_section(response, IMPROVEMENT_SUGGESTIONS_START, IMPROVEMENT_SUGGESTIONS_END)

    return {"verified_problems": verified_problems, "improvement_suggestions": improvement_suggestions}
