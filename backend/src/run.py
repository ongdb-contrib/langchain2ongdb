import contextlib
import io

from logger import logger


def get_result_and_thought_using_graph(
    langchain_object,
    message: str,
):
    """Get result and thought from extracted json"""
    try:
        if hasattr(langchain_object, "verbose"):
            langchain_object.verbose = True
        chat_input = None
        memory_key = ""
        if hasattr(langchain_object, "memory") and langchain_object.memory is not None:
            memory_key = langchain_object.memory.memory_key

        for key in langchain_object.input_keys:
            if key not in [memory_key, "chat_history"]:
                chat_input = {key: message}

        with io.StringIO() as output_buffer, contextlib.redirect_stdout(output_buffer):
            try:
                output = langchain_object(chat_input)
            except ValueError as exc:
                # make the error message more informative
                logger.debug(f"Error: {str(exc)}")
                output = langchain_object.run(chat_input)
            thought = output_buffer.getvalue().strip()
            logger.info(thought)

    except Exception as exc:
        raise ValueError(f"Error: {str(exc)}") from exc

    return {"response": output["output"], "thought": thought}
