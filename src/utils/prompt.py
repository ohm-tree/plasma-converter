# prompt.py
from typing import Callable, Optional


class FewShotPrompter:
    def __init__(
            self,
            system_prompt: str,
            function_prompt: Optional[Callable[[], list[dict]]] = None,
            fixed_prompt: Optional[str] = None,
            fstring_prompt: Optional[str] = None,
            few_shot_examples: list[dict] = [],
            few_shot_responses: list[str] = []
    ):
        """
        Initializes the few-shot prompter. One of `function_prompt` or `fixed_prompt` or `fstring_prompt` must be provided.

        Parameters
        ----------
        system_prompt : str
            The system prompt for the LLM.
        function_prompt : Optional[Callable[[], list[dict]]]
            A function that creates the prompt for the LLM.
            Useful for complex prompts that require additional logic
            or multiple messages.
        fixed_prompt : Optional[str]
            A fixed prompt for the LLM.
        fstring_prompt : Optional[str]
            A fstring prompt for the LLM.
        few_shot_examples : list[dict]
            A list of few-shot examples.
            Each one should contain the kwargs for `create_prompt`.
        few_shot_responses : list[str]
            A list of few-shot responses.
        """
        self.system_prompt: str = system_prompt
        self.function_prompt: Optional[Callable[[],
                                                list[dict]]] = function_prompt
        self.fixed_prompt: Optional[str] = fixed_prompt
        self.fstring_prompt: Optional[str] = fstring_prompt
        self.few_shot_examples: list[dict] = few_shot_examples
        self.few_shot_responses: list[str] = few_shot_responses
        assert len(self.few_shot_examples) == len(self.few_shot_responses), \
            "The number of few-shot examples and responses must be the same."
        assert self.function_prompt is not None or self.fixed_prompt is not None or self.fstring_prompt is not None, \
            "One of `function_prompt`, `fixed_prompt`, or `fstring_prompt` must be provided."

    def create_prompt(self, **kwargs) -> list[dict]:
        """
        Creates the prompt for the LLM.
        """
        if self.function_prompt is not None:
            return self.function_prompt(**kwargs)
        elif self.fixed_prompt is not None:
            return [{"role": "user", "content": self.fixed_prompt}]
        elif self.fstring_prompt is not None:
            return [{"role": "user", "content": self.fstring_prompt.format(**kwargs)}]
        else:
            raise ValueError("No prompt creation method provided.")

    def few_shot_prompt_header(self) -> list[dict]:
        """
        Creates the few-shot prompt header for the LLM.
        """
        few_shot_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for example, response in zip(self.few_shot_examples, self.few_shot_responses):
            few_shot_messages += self.create_prompt(**example)
            few_shot_messages.append(
                {"role": "assistant", "content": response})
        return few_shot_messages

    def few_shot_prompt(self, **kwargs) -> list[dict]:
        """
        Creates the few-shot prompt for the LLM.
        """
        return self.few_shot_prompt_header() + self.create_prompt(**kwargs)
