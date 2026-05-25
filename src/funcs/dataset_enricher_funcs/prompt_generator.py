from typing import List


class PromptGenerator:
    # --------------------------------------------------
    # Hard constraints (shared)
    # --------------------------------------------------
    HARD_RULES = (
        "OUTPUT FORMAT RULES (MANDATORY):\n"
        "- Output EXACTLY one noun as the head word\n"
        "- You MAY add optional adjectives BEFORE the noun\n"
        "- Total number of words MUST be <= {max_words}\n"
        "- Do NOT use verbs\n"
        "- Do NOT use commas, periods, or punctuation\n"
        "- Do NOT mention professions, roles, or activities\n"
        "- Do NOT mention background, scene, or context\n"
        "- Describe ONLY what is visually inside the marked region (red rectangle or polygon)\n"
        "- Use singular form only\n"
        "- Use lowercase only\n"
        "- Do NOT include any scores, metrics, or previous text\n"
    )

    # --------------------------------------------------
    # Initial prompt
    # --------------------------------------------------
    INITIAL_PROMPT = (
        "You are given MULTIPLE images.\n"
        "Each image contains a red-marked region indicating the target object.\n\n"

        "The description must:\n"
        "- use EXACTLY one noun with optional adjectives\n"
        "- describe a common visual characteristic shared by all objects\n"
        "- be valid for every marked region\n\n"

        "Focus on:\n"
        "- common definition (name of food, animal or other)\n"
        "- common shape or silhouette\n"
        "- common clothing or worn item\n"
        "- common material or texture\n"
        "- common color or color placement\n\n"

        "Use the following images: [IMAGES]\n\n"
        "{rules}"

        "In the response, include only the description without any additional explanation."

        "Examples of good generated descriptions: grey pill, red jacket, rounded object,  bird, parrot, yellow rounded cluster"
    )

    # --------------------------------------------------
    # Iterative prompt
    # --------------------------------------------------
    ITER_PROMPT = (
        "You are given MULTIPLE images.\n"
        "Each image contains a red-marked region indicating the target object.\n\n"

        "The description must:\n"
        "- contain EXACTLY one noun with optional adjectives\n"
        "- use at most {max_words} words\n"
        "- be valid for every marked region\n\n"

        "IMPORTANT:\n"
        "- If a feature appears only in some images, do NOT use it\n"
        "- Prefer properties that are consistently visible\n\n"

        "Choose ONE new shared focus:\n"
        "- overall silhouette or contour\n"
        "- consistently worn clothing or equipment\n"
        "- dominant material common to all objects\n"
        "- dominant color present in all marked regions\n\n"

        "Previous descriptions with AP50 (filtered segments with given description versus ground truth):\n"
        "{history}\n\n"
        "Use top description to find similar concepts."

        "Use the following images: [IMAGES]\n\n"
        "{rules}"

        "In the response, include only the description without any additional explanation."
        "Don't repeat previous descriptions in output."

        "Examples of good generated descriptions: grey pill, jacket, rounded object, bird, parrot, yellow rounded cluster"
    )

    # --------------------------------------------------
    # Builders
    # --------------------------------------------------
    @staticmethod
    def build_initial(
        images: List[str],
        max_words: int,
    ) -> str:
        return (
            PromptGenerator.INITIAL_PROMPT
            .format(
                rules=PromptGenerator.HARD_RULES.format(
                    max_words=max_words
                )
            )
            .replace("[IMAGES]", ", ".join(images))
        )

    @staticmethod
    def build_iterative(
        images: List[str],
        history: List[dict],
        max_words: int,
    ) -> str:
        hist_str = "\n".join(
            f"description: {h['desc']} | AP50: {h['AP50']} | iter: {h["iter"]}"
            for h in history
        )

        return (
            PromptGenerator.ITER_PROMPT
            .format(
                history=hist_str,
                max_words=max_words,
                rules=PromptGenerator.HARD_RULES.format(
                    max_words=max_words
                ),
            )
            .replace("[IMAGES]", ", ".join(images))
        )
