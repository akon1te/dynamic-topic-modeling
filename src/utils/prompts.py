

ZERO_SHOT_TOPIC_EXTRACTION_PROMPTS = {
    'phi2': {
        "query": "Instruct: <text>\nFrom dialogue between two people extract 4 or 5 key words in one line separated by commas.\nOutput:",
        "answer": "<keywords>",
    },
    'gemma': {
        "query":  "<start_of_turn>user\nFrom given dialogue between two people extract 4 or 5 key words in one line separated by commas:\n\n<text><end_of_turn>\n<start_of_turn>model",
        "answer": "<keywords>",
    }
}       